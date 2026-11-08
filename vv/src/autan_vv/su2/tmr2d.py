"""TMR's 2D bump-in-channel (2DBC) and NACA 0012 (2DN00) grids in SU2's native format.

Both are single-block PLOT3D files of the case library (`tmr-2dbc-bump`, `tmr-2dn00-naca0012`).
The boundaries are those of the TMR pages and of SU2's published configurations: for the bump
(bump.html; SU2 `vandv/rans/bump_in_channel/turb_bump_sst.cfg`) `inlet` at the first i line,
`outlet` at the last, `upper` at the last j line, and on the first j line `lower_upstream` ahead of
the wall (x < 0), `bump` along it (0 <= x <= 1.5) and `lower_downstream` behind it (x > 1.5); for
the NACA 0012 C-grid (SU2 `rans/naca0012/turb_NACA0012_sa.cfg`) `airfoil` on the first j line
between the trailing-edge nodes and `farfield` on every other boundary, the wake cut's coincident
nodes merged into one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError
from autan_vv.su2.mesh import VTK_LINE, VTK_QUAD, _write

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.su2.msh import FloatArray, IntArray

RECORD_SCHEMA = "autan.su2-tmr2d-mesh/1"
BUMP_WALL = (0.0, 1.5)


@dataclass(frozen=True)
class Record:
    """What `convert` wrote and from what (schema autan.su2-tmr2d-mesh/1)."""

    schema: str
    case: str
    source: str
    source_sha256: str
    file: str
    file_sha256: str
    idim: int
    jdim: int
    cells: int
    points: int
    markers: dict[str, int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def block(x: FloatArray, y: FloatArray) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """Points, counterclockwise quadrangles and the four sides of one structured block x[j, i],
    y[j, i] as edges with the domain on their left: `j0` (first j line), `i1` (last i line), `j1`
    (last j line) and `i0` (first i line). A block numbered clockwise has its cells and edges
    reversed; V0902 if its cells have both orientations or none."""
    jd, id_ = x.shape
    points = np.column_stack([x.ravel(), y.ravel()])

    def node(j: IntArray | int, i: IntArray | int) -> IntArray:
        return np.asarray(np.asarray(j) * id_ + np.asarray(i), dtype=np.int64)

    jj, ii = np.meshgrid(np.arange(jd - 1), np.arange(id_ - 1), indexing="ij")
    quads = np.stack(
        [node(jj, ii), node(jj, ii + 1), node(jj + 1, ii + 1), node(jj + 1, ii)], axis=-1
    ).reshape(-1, 4)
    i = np.arange(id_ - 1)
    j = np.arange(jd - 1)
    sides = {
        "j0": np.stack([node(0, i), node(0, i + 1)], axis=1),
        "i1": np.stack([node(j, id_ - 1), node(j + 1, id_ - 1)], axis=1),
        "j1": np.stack([node(jd - 1, i + 1), node(jd - 1, i)], axis=1),
        "i0": np.stack([node(j + 1, 0), node(j, 0)], axis=1),
    }
    area2 = _area2(points, quads)
    if np.all(area2 < 0.0):
        quads = quads[:, ::-1]
        sides = {name: edges[:, ::-1] for name, edges in sides.items()}
    elif not np.all(area2 > 0.0):
        msg = "the PLOT3D block has degenerate cells or cells of both orientations"
        raise VVError(msg, code="V0902")
    if not np.all(_simple(points, quads)):
        msg = "the PLOT3D block has self-intersecting or collapsed quadrangles"
        raise VVError(msg, code="V0902")
    return points, quads, sides


def _cross(o: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    return np.asarray(
        (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])
    )


def _simple(points: FloatArray, quads: IntArray) -> FloatArray:
    """Whether each counterclockwise quadrangle splits along a diagonal into two counterclockwise
    triangles of positive area: false for a bow-tie and for a collapsed corner."""
    a, b, c, d = (points[quads[:, k]] for k in range(4))
    one = (_cross(a, b, c) > 0.0) & (_cross(a, c, d) > 0.0)
    other = (_cross(b, c, d) > 0.0) & (_cross(b, d, a) > 0.0)
    return np.asarray(one | other)


def _area2(points: FloatArray, quads: IntArray) -> FloatArray:
    a, b, c, d = (points[quads[:, k]] for k in range(4))
    u, v = c - a, d - b
    return np.asarray(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])


def bump(x: FloatArray, y: FloatArray) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """The 2DBC block with SU2's bump markers. V0902 if the first j line's wall nodes do not span
    0 <= x <= 1.5 or its symmetry parts are not y = 0, both to 1e-12."""
    points, quads, sides = block(x, y)
    bottom = sides["j0"]
    mid = 0.5 * (points[bottom[:, 0], 0] + points[bottom[:, 1], 0])
    wall = (mid > BUMP_WALL[0]) & (mid < BUMP_WALL[1])
    wall_x = points[bottom[wall].ravel(), 0]
    # TMR's grids place the wall's ends within round-off of 0 and 1.5 (x = 2.78e-16 on L0-L4).
    ends = (wall_x.min(), wall_x.max())
    if not np.allclose(ends, BUMP_WALL, rtol=0.0, atol=1e-12):
        msg = f"the wall runs from x = {ends[0]} to {ends[1]}, not {BUMP_WALL}"
        raise VVError(msg, code="V0902")
    if not np.allclose(points[bottom[~wall].ravel(), 1], 0.0, rtol=0.0, atol=1e-12):
        msg = "the symmetry parts of the first j line are not y = 0"
        raise VVError(msg, code="V0902")
    markers = {
        "inlet": sides["i0"],
        "outlet": sides["i1"],
        "upper": sides["j1"],
        "lower_upstream": bottom[(~wall) & (mid < BUMP_WALL[0])],
        "bump": bottom[wall],
        "lower_downstream": bottom[(~wall) & (mid > BUMP_WALL[1])],
    }
    return points, quads, markers


def c_grid(x: FloatArray, y: FloatArray) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """The 2DN00 C-grid with the wake cut closed: on the first j line, node i and node idim-1-i
    coincide (to 1e-12 of the grid's extent) for i up to the trailing edge; they become one node
    (the lower-index one), the cut's edges vanish and the airfoil is the first j line between the
    trailing-edge nodes. V0902 if no node pair coincides (not a C-grid) or if the airfoil so found
    does not span x = 0 to 1, the sharp-trailing-edge NACA 0012 of TMR's grids
    (naca0012numerics_val.html: "closes at chord=1 with a sharp trailing edge"), as a blunt or
    open wake cut would give."""
    points, quads, sides = block(x, y)
    id_ = x.shape[1]
    first = np.arange(id_)
    tol = 1e-12 * float(np.max(np.abs(points)))
    k = 0
    while k < id_ // 2 and np.max(np.abs(points[first[k]] - points[first[id_ - 1 - k]])) <= tol:
        k += 1
    if k == 0:
        msg = "no coincident nodes on the first j line: not a C-grid"
        raise VVError(msg, code="V0902")
    te = k - 1  # the last coincident pair is the trailing edge
    chord_x = points[first[te : id_ - te], 0]
    if not np.allclose((chord_x.min(), chord_x.max()), (0.0, 1.0), rtol=0.0, atol=1e-12):
        msg = (
            f"the airfoil found between the wake-cut nodes spans x = {chord_x.min()} to "
            f"{chord_x.max()}, not TMR's sharp-edged chord 0 to 1"
        )
        raise VVError(msg, code="V0902")
    remap = np.arange(len(points))
    remap[first[id_ - 1 - np.arange(k)]] = first[np.arange(k)]
    keep = np.setdiff1d(np.arange(len(points)), first[id_ - 1 - np.arange(k)])
    new = np.full(len(points), -1, dtype=np.int64)
    new[keep] = np.arange(len(keep))
    index = new[remap]
    bottom = sides["j0"]  # node ids of the first j line are its i indices
    lo, hi = bottom.min(axis=1), bottom.max(axis=1)
    on_airfoil = (lo >= te) & (hi <= id_ - 1 - te)
    markers = {
        "airfoil": index[bottom[on_airfoil]],
        "farfield": index[np.concatenate([sides["i1"], sides["j1"], sides["i0"]])],
    }
    return points[keep], index[quads], markers


CONVERTERS = {"tmr-2dbc-bump": bump, "tmr-2dn00-naca0012": c_grid}


def convert(case: str, grid: Path, out: Path) -> Record:
    """Write the single-block PLOT3D 2D `grid` of `case` (a key of CONVERTERS) to `out` in SU2
    format, with `out`.json holding its Record. V0902 for a multi-block grid or one that does not
    have its case's boundaries."""
    from autan_vv.cases import read_plot3d_2d  # noqa: PLC0415

    blocks = read_plot3d_2d(grid)
    if len(blocks) != 1:
        msg = f"{grid}: {len(blocks)} blocks, TMR's {case} grids have one"
        raise VVError(msg, code="V0902")
    x, y = blocks[0]
    points, quads, markers = CONVERTERS[case](x, y)
    _write(
        out,
        2,
        points,
        (VTK_QUAD, quads),
        {name: (VTK_LINE, edges) for name, edges in markers.items()},
    )
    record = Record(
        schema=RECORD_SCHEMA,
        case=case,
        source=grid.name,
        source_sha256=_sha256(grid.read_bytes()),
        file=out.name,
        file_sha256=_sha256(out.read_bytes()),
        idim=x.shape[1],
        jdim=x.shape[0],
        cells=len(quads),
        points=len(points),
        markers={name: len(edges) for name, edges in markers.items()},
    )
    out.with_suffix(".json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    return record
