"""The TMR 2D zero-pressure-gradient flat plate (2DZP) for SU2: grids, configuration, results.

The grids are TMR's PLOT3D files from the case library (`tmr-2dzp-flatplate`), converted to SU2's
native 2D format with the boundaries of the TMR case (flatplate.html): `inlet` at the first i line,
`outlet` at the last, `farfield` at the last j line, and on the first j line `symmetry` ahead of
the plate (x < 0) and `wall` along it (0 <= x <= 2). The configuration is SU2's published
`turb_SA_flatplate.cfg` with only the mesh, iteration cap, convergence fields and outputs changed.
Results are cf at x = 0.970084071 and CD, read from SU2's surface file and history.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError
from autan_vv.su2.config import options
from autan_vv.su2.mesh import VTK_LINE, VTK_QUAD, _write

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.su2.msh import FloatArray, IntArray

RECORD_SCHEMA = "autan.su2-flatplate-mesh/1"
PLATE = (0.0, 2.0)
CF_STATION = 0.970084071
# What `configure` changes in SU2's published configuration; nothing that sets the model, the
# scheme or the solver.
ADOPTED_KEYS = ("MESH_FILENAME", "MESH_FORMAT", "ITER", "CONV_FIELD", "CONV_RESIDUAL_MINVAL",
                "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME", "OUTPUT_FILES",
                "OUTPUT_WRT_FREQ")  # fmt: skip


@dataclass(frozen=True)
class Record:
    """What `convert` wrote and from what (schema autan.su2-flatplate-mesh/1)."""

    schema: str
    source: str
    source_sha256: str
    file: str
    file_sha256: str
    idim: int
    jdim: int
    markers: dict[str, int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plane(x: FloatArray, y: FloatArray) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """Points, counterclockwise quadrangles and boundary edges (domain on their left) of one
    structured block x[j, i], y[j, i]. V0902 if the block is not the 2DZP plate's: a wall that is
    not y = 0 from x = 0 to 2, a symmetry plane that is not y = 0, or inverted cells."""
    jd, id_ = x.shape
    points = np.column_stack([x.ravel(), y.ravel()])

    def node(j: IntArray | int, i: IntArray | int) -> IntArray:
        return np.asarray(j) * id_ + np.asarray(i)

    jj, ii = np.meshgrid(np.arange(jd - 1), np.arange(id_ - 1), indexing="ij")
    quads = np.stack(
        [node(jj, ii), node(jj, ii + 1), node(jj + 1, ii + 1), node(jj + 1, ii)], axis=-1
    ).reshape(-1, 4)
    a, b, c, d = (points[quads[:, k]] for k in range(4))
    u, v = c - a, d - b
    area2 = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    if np.any(area2 <= 0.0):
        msg = "the PLOT3D block has cells that are not counterclockwise in (i, j)"
        raise VVError(msg, code="V0902")
    i = np.arange(id_ - 1)
    j = np.arange(jd - 1)
    # Counterclockwise around the domain: bottom left to right, right side up, top right to left,
    # left side down; each edge keeps the domain on its left.
    bottom = np.stack([node(0, i), node(0, i + 1)], axis=1)
    right = np.stack([node(j, id_ - 1), node(j + 1, id_ - 1)], axis=1)
    top = np.stack([node(jd - 1, i + 1), node(jd - 1, i)], axis=1)
    left = np.stack([node(j + 1, 0), node(j, 0)], axis=1)
    mid = 0.5 * (points[bottom[:, 0], 0] + points[bottom[:, 1], 0])
    on_plate = mid > PLATE[0]
    if not np.all(points[bottom.ravel(), 1] == 0.0):
        msg = "the first j line of the block is not y = 0"
        raise VVError(msg, code="V0902")
    wall_x = points[bottom[on_plate].ravel(), 0]
    if not (wall_x.min() == PLATE[0] and np.isclose(wall_x.max(), PLATE[1], rtol=0, atol=1e-12)):
        msg = f"the wall runs from x = {wall_x.min()} to {wall_x.max()}, not the plate {PLATE}"
        raise VVError(msg, code="V0902")
    markers = {
        "inlet": left,
        "outlet": right,
        "farfield": top,
        "symmetry": bottom[~on_plate],
        "wall": bottom[on_plate],
    }
    return points, quads, markers


def convert(grid: Path, out: Path) -> Record:
    """Write the single-block PLOT3D 2D `grid` (TMR layout) to `out` in SU2 format, with `out`.json
    holding its Record. V0902 for a multi-block grid or one that is not the 2DZP plate."""
    from autan_vv.cases import read_plot3d_2d  # noqa: PLC0415

    blocks = read_plot3d_2d(grid)
    if len(blocks) != 1:
        msg = f"{grid}: {len(blocks)} blocks, the 2DZP grids have one"
        raise VVError(msg, code="V0902")
    x, y = blocks[0]
    points, quads, markers = plane(x, y)
    _write(
        out,
        2,
        points,
        (VTK_QUAD, quads),
        {name: (VTK_LINE, edges) for name, edges in markers.items()},
    )
    record = Record(
        schema=RECORD_SCHEMA,
        source=grid.name,
        source_sha256=_sha256(grid.read_bytes()),
        file=out.name,
        file_sha256=_sha256(out.read_bytes()),
        idim=x.shape[1],
        jdim=x.shape[0],
        markers={name: len(edges) for name, edges in markers.items()},
    )
    out.with_suffix(".json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    return record


def configure(published: str, mesh_file: str, max_iterations: int = 50000) -> str:
    """SU2's published flat-plate configuration with only ADOPTED_KEYS changed: the mesh, the
    iteration cap, every residual's relative drop as SU2's own stop (two orders beyond the
    protocol's ten, the runner stopping it at ten), and the outputs the results are read from.
    V0903 if the published file does not have the markers of `convert`."""
    theirs = options(published)
    for key, marker in (("MARKER_HEATFLUX", "wall"), ("MARKER_SYM", "symmetry"),
                        ("MARKER_INLET", "inlet"), ("MARKER_OUTLET", "outlet")):  # fmt: skip
        if marker not in theirs.get(key, ""):
            msg = f"published {key} = {theirs.get(key)!r} does not name {marker!r}"
            raise VVError(msg, code="V0903")
    ours = {
        "MESH_FILENAME": mesh_file,
        "MESH_FORMAT": "SU2",
        "ITER": str(max_iterations),
        "CONV_FIELD": "( REL_RMS_DENSITY, REL_RMS_MOMENTUM-X, REL_RMS_MOMENTUM-Y, REL_RMS_ENERGY,"
        " REL_RMS_NU_TILDE )",
        "CONV_RESIDUAL_MINVAL": "-12.0",
        "HISTORY_OUTPUT": "( ITER, RMS_RES, REL_RMS_RES, AERO_COEFF, WALL_TIME )",
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "history",
        "OUTPUT_FILES": "( RESTART_ASCII, SURFACE_CSV )",
        "OUTPUT_WRT_FREQ": str(max_iterations),
    }
    kept = [
        line for line in published.splitlines()
        if line.split("%", 1)[0].split("=", 1)[0].strip() not in ADOPTED_KEYS
    ]  # fmt: skip
    return "\n".join([*kept, *(f"{k}= {ours[k]}" for k in ADOPTED_KEYS)]) + "\n"


def skin_friction(surface: Path, x: float = CF_STATION) -> float:
    """SU2's skin-friction coefficient (the x component, `Skin_Friction_Coefficient_x`) at `x` on
    the wall, linear between the two wall nodes around it. V0904 if the file has no such column or
    no wall node on either side of x."""
    with surface.open(newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    head = [h.strip().strip('"') for h in rows[0]]
    try:
        ix, icf = head.index("x"), head.index("Skin_Friction_Coefficient_x")
    except ValueError:
        msg = f"{surface}: no x or Skin_Friction_Coefficient_x column"
        raise VVError(msg, code="V0904") from None
    data = np.array([[float(r[ix]), float(r[icf])] for r in rows[1:]])
    data = data[np.argsort(data[:, 0])]
    if not data[0, 0] <= x <= data[-1, 0]:
        msg = f"{surface}: no wall node on both sides of x = {x}"
        raise VVError(msg, code="V0904")
    return float(np.interp(x, data[:, 0], data[:, 1]))
