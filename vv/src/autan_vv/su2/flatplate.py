"""The TMR 2D zero-pressure-gradient flat plate (2DZP) for SU2: grids, configuration, results.

The grids are TMR's PLOT3D files from the case library (`tmr-2dzp-flatplate`), converted to SU2's
native 2D format with the boundaries of the TMR case (flatplate.html): `inlet` at the first i line,
`outlet` at the last, `farfield` at the last j line, and on the first j line `symmetry` ahead of
the plate (x < 0) and `wall` along it (0 <= x <= 2). The configuration is SU2's published
`turb_SA_flatplate.cfg` with only the mesh, iteration cap, convergence fields and outputs changed;
the secondary model adds `SA_OPTIONS= WITHFT2`. Results are cf at x = 0.970084071 and CD, read from
SU2's surface file and history, against TMR's CFL3D and FUN3D on the same grids.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.errors import VVError
from autan_vv.su2 import history
from autan_vv.su2.config import options
from autan_vv.su2.mesh import VTK_LINE, VTK_QUAD, _write

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.su2.msh import FloatArray, IntArray

CASE = "tmr-2dzp-flatplate"
RECORD_SCHEMA = "autan.su2-flatplate-mesh/1"
LEVEL_SCHEMA = "autan.su2-flatplate-level/1"
REPORT_SCHEMA = "autan.su2-flatplate/1"
PLATE = (0.0, 2.0)
CF_STATION = 0.970084071
# acceptance.md, M3: the gate's centres and its 0.5 % half-width.
GATE = {"cf": 2.7056e-3, "cd": 2.8589e-3}
GATE_TOLERANCE = 0.005
# Primary: SU2's published configuration (SA-noft2 in v8.6.0); secondary: standard SA, TMR's model.
MODELS: dict[str, str | None] = {"sa-noft2": None, "sa": "WITHFT2"}
# What `configure` changes in SU2's published configuration; nothing that sets the model, the
# scheme or the solver.
ADOPTED_KEYS = ("MESH_FILENAME", "MESH_FORMAT", "ITER", "CONV_FIELD", "CONV_RESIDUAL_MINVAL",
                "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME", "OUTPUT_FILES",
                "OUTPUT_WRT_FREQ", "SURFACE_FILENAME", "VOLUME_OUTPUT", "WRT_RESTART_COMPACT")  # fmt: skip
SURFACE = "surface_flow"


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
        return np.asarray(np.asarray(j) * id_ + np.asarray(i), dtype=np.int64)

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


def read_su2_2d(path: Path) -> tuple[FloatArray, int, dict[str, IntArray]]:
    """Points, number of elements and marker edges of an SU2 native 2D mesh (SU2's format: NDIME,
    NELEM, NPOIN with an optional trailing index, NMARK). V0902 if it is not 2D."""
    lines = iter(path.read_text().splitlines())

    def value(key: str) -> str:
        for line in lines:
            if line.split("%", 1)[0].strip().startswith(key + "="):
                return line.split("=", 1)[1].strip()
        msg = f"{path}: no {key}"
        raise VVError(msg, code="V0902")

    if int(value("NDIME")) != 2:
        msg = f"{path}: not a 2D SU2 mesh"
        raise VVError(msg, code="V0902")
    nelem = int(value("NELEM"))
    for _ in range(nelem):
        next(lines)
    npoin = int(value("NPOIN").split()[0])
    points = np.array([[float(t) for t in next(lines).split()[:2]] for _ in range(npoin)])
    markers = {}
    for _ in range(int(value("NMARK"))):
        tag = value("MARKER_TAG")
        n = int(value("MARKER_ELEMS"))
        markers[tag] = np.array([[int(t) for t in next(lines).split()[1:3]] for _ in range(n)])
    return points, nelem, markers


def same_grid(ours: Path, theirs: Path) -> dict[str, Any]:
    """Compare two SU2 2D meshes of one grid as geometry: the point sets matched by sorted
    coordinates, the element counts, and each marker's edges as unordered pairs of coordinates
    (SU2 does not orient boundary edges). Returns the largest coordinate difference over matched
    points, the markers that differ, and those whose edges run the other way."""
    p1, e1, m1 = read_su2_2d(ours)
    p2, e2, m2 = read_su2_2d(theirs)
    if len(p1) != len(p2) or e1 != e2 or set(m1) != set(m2):
        return {"same": False, "points": [len(p1), len(p2)], "elements": [e1, e2],
                "markers": [sorted(m1), sorted(m2)]}  # fmt: skip
    o1, o2 = np.lexsort(p1.T[::-1]), np.lexsort(p2.T[::-1])
    gap = float(np.max(np.abs(p1[o1] - p2[o2])))
    tol = 1e-12 * float(np.max(np.abs(p1)))

    def edges(points: FloatArray, conn: IntArray, oriented: bool) -> FloatArray:
        e = points[conn]
        if not oriented:
            first = (e[:, 0, 0] > e[:, 1, 0]) | (
                (e[:, 0, 0] == e[:, 1, 0]) & (e[:, 0, 1] > e[:, 1, 1])
            )
            e = np.where(first[:, None, None], e[:, ::-1], e)
        flat = e.reshape(-1, 4)
        return flat[np.lexsort(flat.T[::-1])]

    def differ(oriented: bool) -> list[str]:
        return [
            tag for tag in m1
            if len(m1[tag]) != len(m2[tag])
            or float(np.max(np.abs(edges(p1, m1[tag], oriented) - edges(p2, m2[tag], oriented))))
            > tol
        ]  # fmt: skip

    unoriented = differ(oriented=False)
    return {"same": gap <= tol and not unoriented, "max_coordinate_difference": gap,
            "tolerance": tol, "markers_differing": unoriented,
            "markers_oriented_otherwise": differ(oriented=True)}  # fmt: skip


def configure(
    published: str, mesh_file: str, max_iterations: int = 50000, sa_options: str | None = None
) -> str:
    """SU2's published flat-plate configuration with only ADOPTED_KEYS changed, and SA_OPTIONS
    when `sa_options` is given: the mesh, the iteration cap, every residual's relative drop as
    SU2's own stop (two orders beyond the protocol's ten, the runner stopping it at ten), and the
    outputs the results are read from. V0903 if the published file does not have the markers of
    `convert`."""
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
        "SURFACE_FILENAME": SURFACE,
        # Skin friction is in SU2's PRIMITIVE output group, which a compact restart leaves out of
        # the surface file too.
        "VOLUME_OUTPUT": "( COORDINATES, SOLUTION, PRIMITIVE )",
        "WRT_RESTART_COMPACT": "NO",
    }
    changed: tuple[str, ...] = ADOPTED_KEYS
    if sa_options is not None:
        ours["SA_OPTIONS"] = sa_options
        changed = (*ADOPTED_KEYS, "SA_OPTIONS")
    kept = [
        line for line in published.splitlines()
        if line.split("%", 1)[0].split("=", 1)[0].strip() not in changed
    ]  # fmt: skip
    return "\n".join([*kept, *(f"{k}= {ours[k]}" for k in changed)]) + "\n"


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


def prepare(
    cache: Path,
    published: Path,
    out: Path,
    su2_mesh: Path | None = None,
    max_iterations: int = 50000,
) -> list[Path]:
    """Fetch and verify the five TMR grids into `cache`, convert them to OUT/meshes/, and write
    OUT/<model>/<idim>x<jdim>/{case.cfg, level.json} for each model of MODELS; returns the case
    directories, coarsest first. With `su2_mesh` (SU2's own 137x97 mesh), OUT/meshes/crosscheck.json
    records `same_grid` against our 137x97."""
    from autan_vv.cases import fetch, get_case  # noqa: PLC0415

    grids = fetch(get_case(CASE), cache, roles=("grid",))
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    text = published.read_text()
    published_sha = _sha256(published.read_bytes())
    records = []
    for grid in grids:
        stem = grid.name.removesuffix(".gz").removesuffix(".p2dfmt")
        records.append(convert(grid, out / "meshes" / f"{stem}.su2"))
    records.sort(key=lambda r: r.idim * r.jdim)
    if su2_mesh is not None:
        ours = next(r for r in records if (r.idim, r.jdim) == (137, 97))
        check = {"ours": ours.file, "theirs": su2_mesh.name,
                 "theirs_sha256": _sha256(su2_mesh.read_bytes()),
                 **same_grid(out / "meshes" / ours.file, su2_mesh)}  # fmt: skip
        (out / "meshes" / "crosscheck.json").write_text(json.dumps(check, indent=2) + "\n")
    dirs = []
    for model, sa_options in MODELS.items():
        for record in records:
            run = out / model / f"{record.idim}x{record.jdim}"
            run.mkdir(parents=True, exist_ok=True)
            cfg = configure(text, f"../../meshes/{record.file}", max_iterations, sa_options)
            (run / "case.cfg").write_text(cfg)
            level = {
                "schema": LEVEL_SCHEMA,
                "model": model,
                "published_sha256": published_sha,
                "case_sha256": _sha256(cfg.encode()),
                "cells": (record.idim - 1) * (record.jdim - 1),
                "mesh": asdict(record),
            }
            (run / "level.json").write_text(json.dumps(level, indent=2) + "\n")
            dirs.append(run)
    return dirs


def _richardson(h: list[float], values: list[float]) -> dict[str, Any]:
    from autan_vv.order import richardson  # noqa: PLC0415

    try:
        r = richardson(h, values)
    except VVError as err:
        return {"error": err.code}
    return {"estimate": r.extrapolated, "order": r.order, "gci_fine": r.gci_fine}


def references(cache: Path) -> dict[str, dict[str, dict[int, float]]]:
    """TMR's SA cf(0.970084071) and CD per number of cells, per code (CFL3D, FUN3D), from the
    case library's cf_convergence.dat and drag_convergence.dat."""
    from autan_vv.cases import fetch, get_case, parse  # noqa: PLC0415

    case = get_case(CASE)
    paths = dict(zip(
        [f.name for f in case.files if f.role == "reference"],
        fetch(case, cache, roles=("reference",)), strict=True,
    ))  # fmt: skip
    out: dict[str, dict[str, dict[int, float]]] = {}
    for key, name, column in (("cf", "cf_convergence.dat", "C_f,x=0.97"),
                              ("cd", "drag_convergence.dat", "C_D")):  # fmt: skip
        for table in parse(case.file(name), paths[name]):
            cells = table.column("N").astype(int)
            out.setdefault(table.title, {})[key] = dict(
                zip(cells.tolist(), table.column(column).tolist(), strict=True)
            )
    return out


def results(root: Path, cache: Path, orders: float = 10.0) -> dict[str, Any]:
    """The report (autan.su2-flatplate/1) on the runs `prepare` wrote under `root`: per model and
    grid, convergence by the protocol's criterion, cf at CF_STATION and CD at the converged
    iteration and their gaps to CFL3D and FUN3D on the same grid; the continuum from the three
    finest converged grids (h = N^-1/2) for SU2 and for both references; the M3 gate on each model.
    V0904 if a run's configuration or mesh is not the prepared one or SU2 failed."""
    refs = references(cache)
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "orders": orders, "models": {}}
    report["references"] = {
        code: {key: _richardson([n**-0.5 for n in sorted(v)[-3:]],
                                [v[n] for n in sorted(v)[-3:]]) for key, v in data.items()}
        for code, data in refs.items()
    }  # fmt: skip
    for model in MODELS:
        levels = []
        for run in sorted((root / model).iterdir(), key=lambda p: json.loads(
                (p / "level.json").read_text())["cells"]):  # fmt: skip
            level = json.loads((run / "level.json").read_text())
            prov: dict[str, Any] = history.provenance(run)["provenance"]  # type: ignore[assignment]
            mesh_sha = level["mesh"]["file_sha256"]
            if prov["case_sha256"] != level["case_sha256"] or prov["mesh_sha256"] != mesh_sha:
                msg = f"{run}: the run's configuration or mesh is not the prepared one"
                raise VVError(msg, code="V0904")
            if prov["exit_status"] != 0:
                msg = f"{run}: SU2 exited with {prov['exit_status']}"
                raise VVError(msg, code="V0904")
            conv = history.convergence(history.read_su2(run / "history.csv"), orders)
            cells = level["cells"]
            row: dict[str, Any] = {
                "grid": run.name, "cells": cells, "converged": conv.converged,
                "iterations": conv.iterations, "drop": conv.drop, "cd": conv.cd,
                "cf": skin_friction(run / f"{SURFACE}.csv"),
                "image_id": prov["image_id"], "case_sha256": level["case_sha256"],
                "mesh_sha256": mesh_sha, "source_sha256": level["mesh"]["source_sha256"],
            }  # fmt: skip
            row["gap"] = {
                code: {k: row[k] / data[k][cells] - 1.0 for k in ("cf", "cd")}
                for code, data in refs.items() if cells in data["cf"]
            }  # fmt: skip
            levels.append(row)
        fine = [lv for lv in levels if lv["converged"]][-3:]
        continuum = (
            {k: _richardson([lv["cells"] ** -0.5 for lv in fine], [lv[k] for lv in fine])
             for k in ("cf", "cd")} if len(fine) == 3 else {}
        )  # fmt: skip
        verdicts = {
            k: {"pass": "estimate" in est
                and abs(est["estimate"] / GATE[k] - 1.0) <= GATE_TOLERANCE,
                "relative": est["estimate"] / GATE[k] - 1.0 if "estimate" in est else None}
            for k, est in continuum.items()
        }  # fmt: skip
        report["models"][model] = {
            "levels": levels, "continuum": continuum, "gate": verdicts,
            "pass": bool(verdicts) and all(v["pass"] for v in verdicts.values()),
        }  # fmt: skip
    return report
