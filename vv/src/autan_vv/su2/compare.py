"""Forces of Autan and SU2 on the same NACA 0012 levels and condition (docs/validation/su2.md).

Autan's runs are read from their `run.json` and `history.jsonl` (results.md), so that runs written
by any solver commit can be compared; SU2's from the `autan.su2-study/1` report of `naca0012`. Both
are taken at the iteration that meets the protocol's criterion (`history.convergence`), and both
families are extrapolated with the same Richardson estimate as the gates
(`autan_vv.cases.naca0012.continuum`).
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.cases import naca0012 as cases
from autan_vv.errors import VVError
from autan_vv.su2 import history

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

COMPARISON_SCHEMA = "autan.su2-comparison/1"
TRAILING_EDGE_FRACTION = 0.95


@dataclass(frozen=True)
class AutanLevel:
    """One Autan run at the protocol's criterion, with what identifies it."""

    run: str
    n: int
    case: str
    point_vortex: bool
    converged: bool
    iterations: int
    cl: float
    cd: float
    cmy: float
    status: str
    commit: str
    precision: str
    backend: str
    case_sha256: str
    mesh_sha256: str


def read_autan(run: Path) -> AutanLevel:
    """An Autan run directory: its forces at the criterion and its provenance; a `continue` run is
    judged against the residuals of its original run. V0904 if a file the results contract
    requires is missing or the run is not a one-layer NACA 0012 slab."""
    try:
        info = json.loads((run / "run.json").read_text())
        mesh = json.loads((run / "mesh-info.json").read_text())
        spec = tomllib.loads((run / "case.toml").read_text())
    except FileNotFoundError as err:
        msg = f"{run}: {err.filename} is missing"
        raise VVError(msg, code="V0904") from None
    cells = int(mesh["counts"]["cells"])
    n = math.isqrt(cells)
    if n * n != cells:
        msg = f"{run}: {cells} cells is not an NC x NC slab"
        raise VVError(msg, code="V0904")
    series = history.read_autan(run / "history.jsonl")
    start = info.get("start") or {}
    if start.get("residual_reference"):
        series = history.continued(series, start["residual_reference"])
    conv = history.convergence(series, cases.RESIDUAL_ORDERS)
    vortex = any("point_vortex" in rule["condition"] for rule in spec["boundaries"])
    return AutanLevel(
        run=info["run_id"],
        n=n,
        case=info["case"]["name"],
        point_vortex=vortex,
        converged=conv.converged and info["status"] == "converged",
        iterations=conv.iterations,
        cl=conv.cl,
        cd=conv.cd,
        cmy=conv.cmy,
        status=info["status"],
        commit=info["solver"]["commit"],
        precision=info["precision"],
        backend=info["solver"]["backend"],
        case_sha256=info["case"]["sha256"],
        mesh_sha256=info["mesh"]["sha256"],
    )


def _forces(n: int, cl: float, cd: float, cmy: float, label: str) -> cases.LevelForces:
    return cases.LevelForces(
        run=label, n_cells=n * n, h=1.0 / n, cl=cl, cd=cd, cmy=cmy, status="converged",
        backend="", commit="", precision="", case_sha256="", mesh_sha256="",
    )  # fmt: skip


def compare(
    autan_runs: Sequence[Path], su2_report: Path, su2_root: Path | None = None
) -> dict[str, Any]:
    """Level by level and in the continuum, Autan against the SU2 study `su2_report` of the same
    condition. The Autan runs must be one family: one case name, commit, precision and backend,
    one point-vortex choice, distinct levels (V0905). Levels missing on either side, or not
    converged on either side, are listed and left out of the differences and estimates. With
    `su2_root`, the directory SU2's runs were prepared in, each level compared also gets its wall
    Cp comparison (`compare_cp`) when the Autan run has its solution.cgns."""
    read = [(read_autan(r), r) for r in autan_runs]
    paths = {level.run: r for level, r in read}
    autan = sorted((level for level, _ in read), key=lambda lv: lv.n)
    su2 = json.loads(su2_report.read_text())
    identity = {(a.case, a.commit, a.precision, a.backend, a.point_vortex) for a in autan}
    if len(identity) != 1 or len({a.n for a in autan}) != len(autan):
        msg = "the Autan runs are not one family (case, commit, precision, backend, vortex, level)"
        raise VVError(msg, code="V0905")
    condition = su2["condition"]
    if not autan[0].case.startswith(f"naca0012-{condition}"):
        msg = f"Autan case {autan[0].case} is not the condition {condition} of the SU2 report"
        raise VVError(msg, code="V0905")
    by_n = {lv["n"]: lv for lv in su2["levels"]}
    rows = []
    both: list[tuple[AutanLevel, dict[str, Any]]] = []
    for a in autan:
        s = by_n.get(a.n)
        row: dict[str, Any] = {"n": a.n, "autan": asdict(a), "su2": s}
        if s is not None and a.converged and s["converged"]:
            row["difference"] = {k: a.__dict__[k] - s[k] for k in ("cl", "cd", "cmy")}
            both.append((a, s))
            surface = None if su2_root is None else su2_root / condition / s["run"] / "surface.csv"
            if (
                surface is not None
                and surface.exists()
                and (paths[a.run] / "solution.cgns").exists()
            ):
                row["cp"] = compare_cp(paths[a.run], surface)
        rows.append(row)
    continuum = {}
    if len(both) >= cases.MIN_LEVELS:
        continuum = {
            "autan": cases.continuum([_forces(a.n, a.cl, a.cd, a.cmy, a.run) for a, _ in both]),
            "su2": cases.continuum(
                [_forces(s["n"], s["cl"], s["cd"], s["cmy"], s["run"]) for _, s in both]
            ),
        }
    return {
        "schema": COMPARISON_SCHEMA,
        "case": "naca0012",
        "condition": condition,
        "autan": {
            "case": autan[0].case,
            "commit": autan[0].commit,
            "precision": autan[0].precision,
            "backend": autan[0].backend,
            "point_vortex": autan[0].point_vortex,
        },
        "su2": {"scheme": su2["scheme"], "su2": su2["su2"], "image_id": su2["image_id"]},
        "levels": rows,
        "continuum": continuum,
    }


# SU2 writes surface coordinates with 16 significant digits, the mesh holds 17: nodes are matched
# on coordinates rounded to 1e-11 (wall coordinates are of order 1, nodes 1e-4 apart at 1024²).
def _node(x: float, z: float) -> tuple[float, float]:
    return (round(x, 11), round(z, 11))


def _edge_key(a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    return tuple(sorted((_node(*a), _node(*b))))


def autan_wall_cp(run: Path) -> dict[tuple[tuple[float, float], ...], float]:
    """Cp on each wall face across the layer of an Autan slab run (the airfoil), keyed by the
    (x, z) of the face's two nodes on the lower plane (solution.cgns `WallSolution`, results.md),
    with the freestream of its case."""
    from autan import cgns  # noqa: PLC0415
    from autan.units import si, ureg  # noqa: PLC0415

    spec = tomllib.loads((run / "case.toml").read_text())
    fs = spec["freestream"]
    mach, p_inf = float(fs["mach"]), float(si(ureg(fs["pressure"]), "Pa"))
    gamma = float(spec["physics"].get("gas", {}).get("gamma", 1.4))
    zone = cgns.read(run / "solution.cgns").child("Base").child("Zone")
    xyz = np.stack([zone.child("GridCoordinates").child(f"Coordinate{c}").data for c in "XYZ"], 1)
    ngon = zone.child("NGonElements")
    offset = np.asarray(ngon.child("ElementStartOffset").data, dtype=np.int64)
    conn = np.asarray(ngon.child("ElementConnectivity").data, dtype=np.int64) - 1
    wall = zone.child("WallSolution")
    faces = np.asarray(wall.child("PointList").data, dtype=np.int64).ravel() - 1
    pressure = np.asarray(wall.child("Pressure").data, dtype=np.float64)
    q_inf = 0.5 * gamma * p_inf * mach * mach
    y0 = float(xyz[:, 1].min())
    out = {}
    for face, p in zip(faces, pressure, strict=True):
        nodes = conn[offset[face] : offset[face + 1]]
        low = [(float(xyz[k, 0]), float(xyz[k, 2])) for k in nodes if xyz[k, 1] == y0]
        if len(low) == 2:  # a face across the layer; faces on its planes are not the airfoil
            out[_edge_key(low[0], low[1])] = (p - p_inf) / q_inf
    if not out:
        msg = f"{run}: no wall face crosses the layer"
        raise VVError(msg, code="V0904")
    return out


def su2_wall_cp(surface: Path, mach: float, gamma: float) -> dict[tuple[float, float], float]:
    """Cp at each wall node of a 2D SU2 run's surface.csv (conservative variables in units where
    the freestream pressure and density are 1, REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE)."""
    import csv  # noqa: PLC0415

    with surface.open(newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    head = [h.strip().strip('"') for h in rows[0]]
    col = {
        name: head.index(name)
        for name in ("x", "y", "Density", "Momentum_x", "Momentum_y", "Energy")
    }
    out = {}
    for row in rows[1:]:
        v = {k: float(row[i]) for k, i in col.items()}
        p = (gamma - 1.0) * (
            v["Energy"] - 0.5 * (v["Momentum_x"] ** 2 + v["Momentum_y"] ** 2) / v["Density"]
        )
        out[_node(v["x"], v["y"])] = (p - 1.0) / (0.5 * gamma * mach * mach)
    return out


def _shock(xs: np.ndarray, cp: np.ndarray) -> float | None:
    """x of the steepest pressure rise (largest dCp/dx) for 0.2 < x < 0.95, or None if none rises
    by more than 0.1 between two faces (no shock)."""
    order = np.argsort(xs)
    x, c = xs[order], cp[order]
    keep = (x[1:] > 0.2) & (x[:-1] < 0.95)
    rise = np.diff(c)[keep]
    if rise.size == 0 or rise.max() <= 0.1:
        return None
    i = int(np.argmax(rise))
    return float(0.5 * (x[1:][keep][i] + x[:-1][keep][i]))


def compare_cp(autan_run: Path, su2_surface: Path) -> dict[str, Any]:
    """Wall Cp of one Autan run against SU2's on the same level: per face, Autan's face value
    against the mean of SU2's at the face's two nodes (the face centre, linear along the edge);
    the largest and RMS difference over all faces, over the faces at least two faces away from
    either code's shock, and over those that also lie ahead of the last 5 % of chord (`body`); and
    the shock's x on each side. V0904 if a face has no SU2 node pair."""
    spec = tomllib.loads((autan_run / "case.toml").read_text())
    mach = float(spec["freestream"]["mach"])
    gamma = float(spec["physics"].get("gas", {}).get("gamma", 1.4))
    a = autan_wall_cp(autan_run)
    s = su2_wall_cp(su2_surface, mach, gamma)
    keys = list(a)
    # A closed wall has as many nodes as faces; a nested coarser or finer level would match nodes.
    if len(s) != len(a):
        msg = f"{su2_surface}: {len(s)} wall nodes for {len(a)} Autan wall faces: not the same grid"
        raise VVError(msg, code="V0904")
    try:
        su2 = np.array([0.5 * (s[k[0]] + s[k[1]]) for k in keys])
    except KeyError as err:
        msg = f"{su2_surface}: no SU2 wall node at {err}: not the same grid"
        raise VVError(msg, code="V0904") from None
    autan = np.array([a[k] for k in keys])
    xc = np.array([0.5 * (k[0][0] + k[1][0]) for k in keys])
    zc = np.array([0.5 * (k[0][1] + k[1][1]) for k in keys])
    diff = autan - su2
    result: dict[str, Any] = {"faces": len(keys), "all": _stats(diff)}
    smooth = np.ones(len(keys), dtype=bool)
    for side, mask in (("upper", zc >= 0.0), ("lower", zc < 0.0)):
        shocks = {"autan": _shock(xc[mask], autan[mask]), "su2": _shock(xc[mask], su2[mask])}
        result[f"shock_{side}"] = shocks
        spacing = np.median(np.diff(np.sort(xc[mask])))
        for x in shocks.values():
            if x is not None:
                smooth &= ~(mask & (np.abs(xc - x) <= 2.5 * spacing))
    result["outside_shocks"] = _stats(diff[smooth])
    # The sharp trailing edge is a stagnation point: SU2 has a node on it, Autan a face beside it,
    # and their values differ by O(1) there at every level. `body` leaves out its last 5 % of chord.
    body = smooth & (xc <= TRAILING_EDGE_FRACTION * xc.max())
    result["body"] = {**_stats(diff[body]), "faces": int(body.sum())}
    return result


def _stats(diff: np.ndarray) -> dict[str, float]:
    return {"max_abs": float(np.abs(diff).max()), "rms": float(np.sqrt(np.mean(diff**2)))}
