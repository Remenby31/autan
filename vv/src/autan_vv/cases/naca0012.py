"""NACA 0012 inviscid cases of Vassberg and Jameson (2010) and their force gates (acceptance.md, M2).

`case` builds the autan.case/1 case of a flow condition on one level of the `naca0012` mesh family
(`autan_vv.meshes.naca0012`); `study` reads the runs of one condition over the family, extrapolates
CL, CD and CM to the continuum from the three finest levels (Richardson with Roache's GCI,
`autan_vv.order.richardson`, as the paper's eqs. 20-24 for a ratio of 2) and checks the gates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import autan
from autan.units import deg, m, si

from autan_vv.errors import VVError
from autan_vv.meshes.naca0012 import SPAN
from autan_vv.order import richardson

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

STUDY_SCHEMA = "autan.vv-force-study/1"
COEFFICIENTS = ("cl", "cd", "cmy")
COUNT = 1e-4  # one drag count
# acceptance.md: continuum estimates from the three finest of at least four levels, on runs
# converged by at least this many orders.
MIN_LEVELS = 4
RESIDUAL_ORDERS = 10.0


@dataclass(frozen=True)
class Condition:
    """A flow condition of the paper's Table 1."""

    mach: float
    alpha_deg: float


CONDITIONS = {
    "m050-a125": Condition(0.5, 1.25),
    "m080-a000": Condition(0.8, 0.0),
    "m080-a125": Condition(0.8, 1.25),
}


@dataclass(frozen=True)
class ForceGate:
    """A force gate of acceptance.md on the continuum estimate of `coefficient`.

    `within`: |estimate - target| <= tolerance. `band`: the estimate lies in [lower, upper] widened
    by its own GCI. `vanishing`: the magnitude of the coefficient decreases strictly from every level
    to the next finer one and |estimate| <= tolerance.
    """

    id: str
    condition: str
    coefficient: str
    measure: str
    target: float = 0.0
    tolerance: float = 0.0
    lower: float = 0.0
    upper: float = 0.0


# Vassberg and Jameson (2010), Tables 10-11; the point-vortex far-field band for CL (datasets §M2).
GATES = {
    g.id: g
    for g in (
        ForceGate("naca0012-m080-a000-cd", "m080-a000", "cd", "within", 0.008342, COUNT),
        ForceGate("naca0012-m080-a125-cl", "m080-a125", "cl", "band", lower=0.3516, upper=0.3562),
        ForceGate("naca0012-m080-a125-cd", "m080-a125", "cd", "within", 0.02268, COUNT),
        ForceGate("naca0012-m050-a125-cd", "m050-a125", "cd", "vanishing", tolerance=0.5 * COUNT),
    )
}


def case(
    condition: str,
    mesh: str,
    *,
    precision: Literal["fp64", "fp32+acc64"] = "fp64",
    max_iterations: int = 200000,
    residual_orders: float = 10.0,
) -> autan.Case:
    """The case of `condition` (a key of CONDITIONS) on the NACA 0012 slab `mesh`.

    Slip airfoil, characteristic far field, symmetry planes on the two faces of the layer; forces on
    the airfoil with reference chord 1 m (the paper's, although the sharp-edged chord is 1.0089),
    reference area chord x span and moments about the quarter chord.
    """
    if condition not in CONDITIONS:
        msg = f"unknown NACA 0012 condition {condition!r}; expected one of {sorted(CONDITIONS)}"
        raise VVError(msg, code="V0801")
    c = CONDITIONS[condition]
    result = autan.Case(f"naca0012-{condition}", mesh=mesh)
    result.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision)
    result.freestream(
        mach=c.mach,
        aoa=c.alpha_deg * deg,
        pressure=101325.0 * autan.units.ureg.Pa,
        ref_length=1.0 * m,
    )
    result.time = autan.Steady(max_iterations=max_iterations, residual_orders=residual_orders)
    result.boundary("airfoil").slip()
    result.boundary("farfield").farfield()
    result.boundary("back").symmetry()
    result.boundary("front").symmetry()
    result.monitor(
        autan.forces("airfoil", ref_area=SPAN * m * m, moment_center=(0.25 * m, 0.0 * m, 0.0 * m))
    )
    return result


@dataclass(frozen=True)
class LevelForces:
    """Final force coefficients of one converged run, with its in-plane mesh size."""

    run: str
    n_cells: int
    h: float
    cl: float
    cd: float
    cmy: float
    status: str
    backend: str
    commit: str
    precision: str
    case_sha256: str
    mesh_sha256: str


def read_run(path: Path) -> tuple[LevelForces, dict[str, Any]]:
    """The final coefficients of the run at `path` and the identity its family must share."""
    results = autan.Results.open(path)
    run = results.run
    spec = results.case
    forces = results.forces
    n_cells = results.mesh_info.counts.cells
    level = LevelForces(
        run=run.run_id,
        n_cells=n_cells,
        # One layer of cells: h = sqrt(area / N) scales as 1/NC; the area is the same at every level.
        h=1.0 / math.sqrt(n_cells),
        cl=float(forces.cl[-1]),
        cd=float(forces.cd[-1]),
        cmy=float(forces.cmy[-1]),
        status=run.status,
        backend=run.solver.backend,
        commit=run.solver.commit,
        precision=run.precision,
        case_sha256=run.case.sha256,
        mesh_sha256=run.mesh.sha256,
    )
    document = spec.model_dump(mode="json", by_alias=True)
    family = {
        "case": {k: v for k, v in document.items() if k not in ("name", "mesh")},
        "commit": run.solver.commit,
        "backend": run.solver.backend,
        "precision": run.precision,
        "mach": spec.freestream.mach,
        "alpha_deg": float(si(spec.freestream.aoa, "rad")) * 180.0 / math.pi,
    }
    _check_case(path, document, family)
    return level, family


def _check_case(path: Path, document: dict[str, Any], family: dict[str, Any]) -> None:
    """The run's case must be `case` of its condition (V0801) up to its mesh, precision and
    iteration budget, and converge at least RESIDUAL_ORDERS orders."""
    condition = _condition_of(family)
    expected = case(condition, document["mesh"], precision=family["precision"])
    reference = expected.spec.model_dump(mode="json", by_alias=True)
    time = document["time"]
    if time.get("residual_orders", 0.0) < RESIDUAL_ORDERS:
        msg = f"run {path} stops at {time.get('residual_orders')} orders of residual drop"
        raise VVError(msg, code="V0801")
    strip = {"time": None}
    if {**document, **strip} != {**reference, **strip} or time["kind"] != "steady":
        msg = f"run {path} is not the NACA 0012 case {condition} (physics, boundaries or monitor)"
        raise VVError(msg, code="V0801")


def _condition_of(family: dict[str, Any]) -> str:
    if family["mach"] is None:
        msg = "the NACA 0012 cases are compressible: the run has no freestream Mach number"
        raise VVError(msg, code="V0801")
    for name, c in CONDITIONS.items():
        if math.isclose(family["mach"], c.mach, abs_tol=1e-12) and math.isclose(
            family["alpha_deg"], c.alpha_deg, abs_tol=1e-9
        ):
            return name
    msg = f"no NACA 0012 condition has M = {family['mach']} and alpha = {family['alpha_deg']} deg"
    raise VVError(msg, code="V0801")


def continuum(levels: Sequence[LevelForces]) -> dict[str, Any]:
    """Richardson extrapolation of each coefficient from the three finest levels.

    A coefficient whose three values do not converge monotonically gets its error instead of an
    estimate, with the code richardson raised (V0106-V0109).
    """
    finest = sorted(levels, key=lambda lv: lv.h)[:3]
    out: dict[str, Any] = {}
    for name in COEFFICIENTS:
        try:
            r = richardson([lv.h for lv in finest], [getattr(lv, name) for lv in finest])
        except VVError as err:
            out[name] = {"error": err.code}
            continue
        out[name] = {
            "estimate": r.extrapolated,
            "order": r.order,
            "gci": r.gci_fine,
            # Eq. 25 of the paper: the order if the limit were zero, meaningful for subcritical CD.
            "order_if_zero": _order_if_zero([getattr(lv, name) for lv in finest[:2]], finest),
        }
    return out


def _order_if_zero(values: Sequence[float], finest: Sequence[LevelForces]) -> float | None:
    fine, medium = values
    if fine == 0.0 or medium / fine <= 0.0:
        return None
    return math.log(medium / fine) / math.log(finest[1].h / finest[0].h)


def check(
    gate: ForceGate, levels: Sequence[LevelForces], estimates: dict[str, Any]
) -> dict[str, Any]:
    """Verdict of `gate` on the levels of its condition and their continuum estimates."""
    est = estimates[gate.coefficient]
    if "error" in est:
        return {"gate": gate.id, "pass": False, "reason": f"no continuum estimate ({est['error']})"}
    value, gci = est["estimate"], est["gci"]
    verdict: dict[str, Any] = {"gate": gate.id, "value": value, "gci": gci}
    if gate.measure == "within":
        verdict["bounds"] = (gate.target - gate.tolerance, gate.target + gate.tolerance)
    elif gate.measure == "band":
        verdict["bounds"] = (gate.lower - gci, gate.upper + gci)
    else:
        verdict["bounds"] = (-gate.tolerance, gate.tolerance)
        series = [getattr(lv, gate.coefficient) for lv in sorted(levels, key=lambda lv: -lv.h)]
        if not all(abs(b) < abs(a) for a, b in itertools.pairwise(series)):
            return {**verdict, "pass": False, "reason": "magnitude not decreasing level to level"}
    lower, upper = verdict["bounds"]
    return {**verdict, "pass": lower <= value <= upper}


def study(runs: Sequence[Path]) -> dict[str, Any]:
    """Continuum CL, CD, CM of a NACA 0012 family and the verdicts of its condition's gates.

    Each run must be the case of its condition, converging 10 orders (V0801); the runs must be one
    refinement family (V0802: the same case apart from its mesh, solver commit, backend and
    precision, distinct meshes), at least four levels, all `converged` (V0803). The verdicts are
    evidence only when no run comes from the fake backend.
    """
    if len(runs) < MIN_LEVELS:
        msg = f"a continuum estimate needs at least {MIN_LEVELS} levels, got {len(runs)}"
        raise VVError(msg, code="V0803")
    read = [read_run(p) for p in runs]
    levels = [lv for lv, _ in read]
    reference = read[0][1]
    for (lv, family), path in zip(read, runs, strict=True):
        if family != reference:
            msg = f"run {path} is not of the same family as {runs[0]}"
            raise VVError(msg, code="V0802")
        if lv.status != "converged":
            msg = f"run {path} ended {lv.status!r}; forces are compared on converged runs only"
            raise VVError(msg, code="V0803")
    if len({lv.n_cells for lv in levels}) != len(levels):
        msg = "two runs use meshes with the same cell count"
        raise VVError(msg, code="V0802")
    condition = _condition_of(reference)
    estimates = continuum(levels)
    verdicts = [check(g, levels, estimates) for g in GATES.values() if g.condition == condition]
    fake = any(lv.backend == "fake" for lv in levels)
    return {
        "schema": STUDY_SCHEMA,
        "case": "naca0012",
        "condition": condition,
        "levels": [asdict(lv) for lv in sorted(levels, key=lambda lv: -lv.h)],
        "continuum": estimates,
        "verdicts": verdicts,
        "pass": all(v["pass"] for v in verdicts),
        "evidence": not fake,
    }
