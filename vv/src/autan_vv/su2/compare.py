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

from autan_vv.cases import naca0012 as cases
from autan_vv.errors import VVError
from autan_vv.su2 import history

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

COMPARISON_SCHEMA = "autan.su2-comparison/1"


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
    """An Autan run directory: its forces at the criterion and its provenance. V0904 if a file the
    results contract requires is missing or the run is not a one-layer NACA 0012 slab."""
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
    conv = history.convergence(history.read_autan(run / "history.jsonl"), cases.RESIDUAL_ORDERS)
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


def compare(autan_runs: Sequence[Path], su2_report: Path) -> dict[str, Any]:
    """Level by level and in the continuum, Autan against the SU2 study `su2_report` of the same
    condition. The Autan runs must be one family: one case name, commit, precision and backend,
    one point-vortex choice, distinct levels (V0905). Levels missing on either side, or not
    converged on either side, are listed and left out of the differences and estimates."""
    autan = sorted((read_autan(r) for r in autan_runs), key=lambda lv: lv.n)
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
