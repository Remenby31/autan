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
    from collections.abc import Callable
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
# case-schema.md (Time): the criterion the gate is registered with.
ORDERS = 10.0
# Primary: SU2's published configuration (SA-noft2 in v8.6.0); secondary: standard SA, TMR's model.
MODELS: dict[str, str | None] = {"sa-noft2": None, "sa": "WITHFT2"}
MODEL_REFERENCE = "sa-noft2"
# SU2 v8.6.0, TestCases/rans/flatplate/turb_SA_flatplate.cfg.
PUBLISHED_SHA256 = "8d5f6f7cca2606f87453858a190c0561e7aa36220c9913c18effa20da892e917"
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


def read_su2_2d(path: Path) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    """Points, quadrangles and marker edges of an SU2 native 2D mesh of quadrangles (SU2's format:
    NDIME, NELEM, NPOIN with an optional trailing index, NMARK). V0902 otherwise."""
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
    cells = np.array(
        [[int(t) for t in next(lines).split()[:5]] for _ in range(int(value("NELEM")))]
    )
    if np.any(cells[:, 0] != VTK_QUAD):
        msg = f"{path}: elements other than quadrangles"
        raise VVError(msg, code="V0902")
    npoin = int(value("NPOIN").split()[0])
    points = np.array([[float(t) for t in next(lines).split()[:2]] for _ in range(npoin)])
    markers = {}
    for _ in range(int(value("NMARK"))):
        tag = value("MARKER_TAG")
        n = int(value("MARKER_ELEMS"))
        markers[tag] = np.array([[int(t) for t in next(lines).split()[1:3]] for _ in range(n)])
    return points, cells[:, 1:], markers


def same_grid(ours: Path, theirs: Path) -> dict[str, Any]:
    """Compare two SU2 2D meshes of one grid as geometry: the point sets matched by sorted
    coordinates, the cells as sets of matched points, and each marker's edges as unordered pairs
    of coordinates (SU2 does not orient boundary edges). Returns the largest coordinate difference
    over matched points, whether the cells agree, the markers that differ, and those whose edges
    run the other way."""
    p1, c1, m1 = read_su2_2d(ours)
    p2, c2, m2 = read_su2_2d(theirs)
    if len(p1) != len(p2) or len(c1) != len(c2) or set(m1) != set(m2):
        return {"same": False, "points": [len(p1), len(p2)], "cells": [len(c1), len(c2)],
                "markers": [sorted(m1), sorted(m2)]}  # fmt: skip
    o1, o2 = np.lexsort(p1.T[::-1]), np.lexsort(p2.T[::-1])
    gap = float(np.max(np.abs(p1[o1] - p2[o2])))
    tol = 1e-12 * float(np.max(np.abs(p1)))

    def cells(order: IntArray, conn: IntArray) -> IntArray:
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        c = np.sort(rank[conn], axis=1)
        return c[np.lexsort(c.T[::-1])]

    same_cells = bool(np.array_equal(cells(o1, c1), cells(o2, c2)))

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
    return {"same": gap <= tol and same_cells and not unoriented,
            "max_coordinate_difference": gap, "tolerance": tol, "same_cells": same_cells,
            "markers_differing": unoriented,
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


def _history(run: Path, model: str) -> tuple[dict[str, Any], history.History]:
    level = json.loads((run / "level.json").read_text())
    prov: dict[str, Any] = history.provenance(run)["provenance"]  # type: ignore[assignment]
    mesh = run / options((run / "case.cfg").read_text())["MESH_FILENAME"]
    on_disk = (_sha256((run / "case.cfg").read_bytes()), _sha256(mesh.read_bytes()))
    prepared = (level["case_sha256"], level["mesh"]["file_sha256"])
    ran = (prov["case_sha256"], prov["mesh_sha256"])
    if level["model"] != model or not on_disk == prepared == ran:
        msg = f"{run}: the run's model, configuration or mesh is not the prepared one"
        raise VVError(msg, code="V0904")
    if prov["exit_status"] != 0:
        msg = f"{run}: SU2 exited with {prov['exit_status']}"
        raise VVError(msg, code="V0904")
    runs = history.read_su2(run / "history.csv")
    if "nu-tilde" not in runs.residuals:
        msg = f"{run}: history.csv has no SA residual"
        raise VVError(msg, code="V0904")
    return {"level": level, "provenance": prov}, runs


def _run(run: Path, model: str, reference: Path | None) -> dict[str, Any]:
    meta, runs = _history(run, model)
    level, prov = meta["level"], meta["provenance"]
    conv = history.convergence(runs, ORDERS)
    # The surface file is written at the run's last iteration, so CD is read there too.
    row = {
        "grid": run.name, "cells": level["cells"], "converged": conv.converged,
        "iterations": conv.iterations, "last_iteration": int(runs.iterations[-1]),
        "drop": conv.drop, "cd": float(runs.cd[-1]),
        "cf": skin_friction(run / f"{SURFACE}.csv"),
        "su2": prov["su2"], "image_id": prov["image_id"], "precision": "fp64",
        "case_sha256": level["case_sha256"], "mesh_sha256": level["mesh"]["file_sha256"],
        "source_sha256": level["mesh"]["source_sha256"],
    }  # fmt: skip
    if reference is not None:
        meta, theirs = _history(reference, MODEL_REFERENCE)
        published = meta["level"]
        if (
            published.get("published_sha256") != PUBLISHED_SHA256
            or published["mesh"]["file_sha256"] != level["mesh"]["file_sha256"]
            or set(theirs.residuals) != set(runs.residuals)
        ):
            msg = f"{reference}: not SU2's published configuration on the mesh of {run}"
            raise VVError(msg, code="V0905")
        at = history.converged_at(theirs, ORDERS)
        if at is None:
            msg = f"{reference}: the reference run did not meet the criterion"
            raise VVError(msg, code="V0904")
        thresholds = {eq: float(r[at]) for eq, r in theirs.residuals.items()}
        met = np.ones(len(runs.iterations), dtype=bool)
        for eq, r in runs.residuals.items():
            met &= r <= thresholds[eq]
        hits = np.flatnonzero(met)
        row["amended"] = {
            "converged": bool(hits.size),
            "iterations": int(runs.iterations[hits[0]] - runs.iterations[0]) + 1
            if hits.size else None,
            "reference": {
                "case_sha256": published["case_sha256"], "su2": meta["provenance"]["su2"],
                "image_id": meta["provenance"]["image_id"],
                "history_sha256": _sha256((reference / "history.csv").read_bytes()),
                "iteration": int(theirs.iterations[at]), "thresholds": thresholds,
            },
        }  # fmt: skip
    return row


def _assess(
    levels: list[dict[str, Any]], converged: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    fine = levels[-3:]
    continuum: dict[str, dict[str, Any]] = {
        k: _richardson([lv["cells"] ** -0.5 for lv in fine], [lv[k] for lv in fine])
        if all(converged(lv) for lv in fine) else {"error": "not converged"}
        for k in ("cf", "cd")
    }  # fmt: skip
    verdicts = {
        k: {"pass": "estimate" in est
            and abs(est["estimate"] / GATE[k] - 1.0) <= GATE_TOLERANCE,
            "relative": est["estimate"] / GATE[k] - 1.0 if "estimate" in est else None}
        for k, est in continuum.items()
    }  # fmt: skip
    return {"continuum": continuum, "gate": verdicts,
            "pass": all(v["pass"] for v in verdicts.values())}  # fmt: skip


def results(
    root: Path, cache: Path, models: tuple[str, ...] = tuple(MODELS), reference: Path | None = None
) -> dict[str, Any]:
    """The report (autan.su2-flatplate/1) on the runs `prepare` wrote under `root`, for `models`:
    per model and grid, convergence by the protocol's criterion (ORDERS), cf at CF_STATION and CD
    at the run's last iteration, and their gaps to CFL3D and FUN3D on the same grid; the continuum
    from the three finest of the five grids (h = N^-1/2), for SU2 and for both references, none
    for a model one of whose three finest runs did not converge; the M3 gate on each model. With
    `reference`, the root of the published configuration's runs, each run is also judged by the
    amended criterion of #195 (`amended`): converged once every residual is at or below the one
    the published SA-noft2 run on the same grid had when it met the protocol's criterion; the
    continuum and gate are then given under both readings. V0904 if a run's model, configuration
    or mesh, as prepared, as run and as on disk, differ, if SU2 failed, if the SA residual is
    missing or a reference run did not converge; V0905 if a model does not have the five grids or
    mixes SU2 builds, or if a reference run is not SU2's published configuration (PUBLISHED_SHA256)
    on the same mesh with the same equations.
    """
    refs = references(cache)
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "orders": ORDERS, "models": {}}
    report["references"] = {
        code: {key: _richardson([n**-0.5 for n in sorted(v)[-3:]],
                                [v[n] for n in sorted(v)[-3:]]) for key, v in data.items()}
        for code, data in refs.items()
    }  # fmt: skip
    for model in models:
        levels = sorted(
            (_run(run, model, None if reference is None else reference / MODEL_REFERENCE / run.name)
             for run in (root / model).iterdir() if run.is_dir()),
            key=lambda lv: lv["cells"],
        )  # fmt: skip
        grids = sorted(refs["CFL3D"]["cf"])
        if [lv["cells"] for lv in levels] != grids or len({(lv["su2"], lv["image_id"])
                                                           for lv in levels}) != 1:  # fmt: skip
            msg = f"{root / model}: not the five TMR grids run by one SU2 build"
            raise VVError(msg, code="V0905")
        for lv in levels:
            lv["gap"] = {code: {k: lv[k] / data[k][lv["cells"]] - 1.0 for k in ("cf", "cd")}
                         for code, data in refs.items()}  # fmt: skip
        report["models"][model] = {"levels": levels,
                                   **_assess(levels, lambda lv: lv["converged"])}  # fmt: skip
        if reference is not None:
            report["models"][model]["amended"] = _assess(
                levels, lambda lv: lv["amended"]["converged"]
            )
    return report
