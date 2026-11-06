"""Residual and force histories of SU2 and Autan runs on one footing, and the protocol's
convergence criterion applied to both.

The criterion is the one case-schema.md (Time) gives a steady Autan run: every residual has dropped
`orders` orders of magnitude below its value at the first iteration of the history, an equation
whose first residual is below 1e-12 times the largest one being exempt while it stays there. Both
codes define a residual the same way, the RMS over the control volumes of the integrated flux
balance (results.md, History; SU2's rms[...] columns), so a drop is compared with a drop.

Pre-registrations from #493 on use an absolute criterion instead, independent of the initial state:
every residual at or below epsilon times its freestream flux scale, epsilon taken from a baseline
run where its outputs settle, and the outputs stationary over a window.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

FloatArray = npt.NDArray[np.float64]
IDENTICALLY_ZERO = 1e-12

# SU2 2D column, Autan equation of the slab (airfoil in x-z): SU2's y is Autan's z.
SU2_2D = {"rms[Rho]": "density", "rms[RhoU]": "momentum-x", "rms[RhoV]": "momentum-z",
          "rms[RhoE]": "energy"}  # fmt: skip
SU2_3D = {"rms[Rho]": "density", "rms[RhoU]": "momentum-x", "rms[RhoV]": "momentum-y",
          "rms[RhoW]": "momentum-z", "rms[RhoE]": "energy"}  # fmt: skip
SU2_TURBULENCE = {"rms[nu]": "nu-tilde"}


@dataclass(frozen=True)
class History:
    """Iterations, residuals per equation (linear, not log10), CL, CD and the nose-up pitching
    moment `cmy` of Autan's convention, and, when the code records it, the cumulative wall time.
    `woken`: identically zero equations that already rose above their floor before this history
    (a continued run's `start.woken`), which are no longer exempt."""

    iterations: npt.NDArray[np.int64]
    residuals: dict[str, FloatArray]
    cl: FloatArray
    cd: FloatArray
    cmy: FloatArray
    wall_seconds: FloatArray | None
    woken: frozenset[str] = frozenset()


def read_su2(path: Path) -> History:
    """SU2's history.csv (HISTORY_OUTPUT with ITER, RMS_RES, AERO_COEFF and WALL_TIME), with the
    SA equation's residual (`nu-tilde`) when the run solves it.

    In 2D the plane (x, y) of SU2 is Autan's (x, z), so the nose-up moment is -CMz; in 3D it is
    CMy. The cumulative wall time is SU2's average time per iteration times the iterations done.
    V0904 if a column is missing or the file is empty.
    """
    with path.open(newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    if len(rows) < 2:
        msg = f"{path}: no iteration recorded"
        raise VVError(msg, code="V0904")
    header = [h.strip().strip('"') for h in rows[0]]
    data = np.array([[float(v) for v in r[: len(header)]] for r in rows[1:]])

    def column(name: str, occurrence: int = 0) -> FloatArray:
        found = [i for i, h in enumerate(header) if h == name]
        if len(found) <= occurrence:
            msg = f"{path}: no column {name!r}"
            raise VVError(msg, code="V0904")
        return np.asarray(data[:, found[occurrence]])

    three_d = "rms[RhoW]" in header
    names = {**(SU2_3D if three_d else SU2_2D),
             **{col: eq for col, eq in SU2_TURBULENCE.items() if col in header}}  # fmt: skip
    iterations = column("Inner_Iter").astype(np.int64)
    # WALL_TIME's two columns are both "Time(sec)": this iteration's, then the running average.
    average = column("Time(sec)", 1)
    return History(
        iterations=iterations,
        residuals={eq: 10.0 ** column(col) for col, eq in names.items()},
        cl=column("CL"),
        cd=column("CD"),
        cmy=column("CMy") if three_d else -column("CMz"),
        wall_seconds=average * (iterations - iterations[0] + 1),
    )


def read_autan(path: Path, monitor: str = "forces", text: str | None = None) -> History:
    """An Autan run's history.jsonl (results.md, History), forces from `monitor`; `text`, when
    given, is the file's content already read."""
    content = path.read_text() if text is None else text
    lines = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not lines:
        msg = f"{path}: no iteration recorded"
        raise VVError(msg, code="V0904")
    try:
        equations = list(lines[0]["residuals"])
        forces = [line["monitors"][monitor] for line in lines]
        return History(
            iterations=np.array([line["iteration"] for line in lines], dtype=np.int64),
            residuals={eq: np.array([line["residuals"][eq] for line in lines]) for eq in equations},
            cl=np.array([f["cl"] for f in forces]),
            cd=np.array([f["cd"] for f in forces]),
            cmy=np.array([f["cmy"] for f in forces]),
            wall_seconds=None,
        )
    except KeyError as err:
        msg = f"{path}: missing {err} in a history line"
        raise VVError(msg, code="V0904") from None


def continued(
    history: History, residual_reference: dict[str, float], woken: Sequence[str] = ()
) -> History:
    """The history of a `continue` run with the residuals at iteration 1 of its original run
    (run.json `start.residual_reference`, results.md) as its first line and the equations of
    `start.woken` no longer exempt, so that the criterion is judged as for the uninterrupted run
    and iterations count from the original's first; forces of that line are not known (NaN)."""
    nan = np.array([np.nan])
    return History(
        iterations=np.concatenate([[1], history.iterations]).astype(np.int64),
        residuals={
            eq: np.concatenate([[residual_reference[eq]], r]) for eq, r in history.residuals.items()
        },
        cl=np.concatenate([nan, history.cl]),
        cd=np.concatenate([nan, history.cd]),
        cmy=np.concatenate([nan, history.cmy]),
        wall_seconds=None,
        woken=frozenset(woken),
    )


def converged_at(history: History, orders: float) -> int | None:
    """Index into `history` of the first iteration meeting the criterion, or None."""
    first = {eq: r[0] for eq, r in history.residuals.items()}
    largest = max(first.values())
    met = np.ones(len(history.iterations), dtype=bool)
    for eq, r in history.residuals.items():
        if first[eq] < IDENTICALLY_ZERO * largest:
            floor = IDENTICALLY_ZERO * largest
            # Exempt until it first rises above the floor; from then on it must drop like the rest.
            risen = np.maximum.accumulate(r > floor) | (eq in history.woken)
            met &= ~risen | (r <= 10.0**-orders * first[eq])
        else:
            met &= r <= 10.0**-orders * first[eq]
    hits = np.flatnonzero(met)
    return int(hits[0]) if hits.size else None


@dataclass(frozen=True)
class Convergence:
    """What the criterion gives on a history: the iterations run up to and including the one whose
    residual meets it (a run given that many iterations ends on the criterion), the cumulative wall
    time there when known, the forces there, and the orders each equation dropped."""

    converged: bool
    iterations: int
    wall_seconds: float | None
    cl: float
    cd: float
    cmy: float
    drop: dict[str, float]


def convergence(history: History, orders: float) -> Convergence:
    """The criterion on `history`. Forces are those at the converged iteration, or at the last one
    when the run did not converge."""
    at = converged_at(history, orders)
    i = len(history.iterations) - 1 if at is None else at
    return Convergence(
        converged=at is not None,
        iterations=int(history.iterations[i] - history.iterations[0]) + 1,
        wall_seconds=None if history.wall_seconds is None else float(history.wall_seconds[i]),
        cl=float(history.cl[i]),
        cd=float(history.cd[i]),
        cmy=float(history.cmy[i]),
        drop={eq: float(np.log10(r[0] / r[i])) for eq, r in history.residuals.items()},
    )


# The absolute criterion (#493): every residual at or below EPSILON times its freestream flux scale
# and the case's outputs stationary to RTOL over WINDOW iterations, whatever the initial state.
WINDOW = 500
RTOL = 1e-7
SU2_FREESTREAM = {"Static Pressure": "pressure", "Density": "density", "Total Energy": "energy",
                  "Velocity Magnitude": "speed", "Viscosity": "viscosity"}  # fmt: skip


def su2_freestream(log: str) -> dict[str, float]:
    """The freestream SU2 prints in its log ("Initial and free-stream conditions", dimensional
    values, SI units): pressure, density, total energy per unit mass, speed and dynamic viscosity.
    V0904 if one is missing."""
    found: dict[str, float] = {}
    section = log.split("-- Initial and free-stream conditions:", 1)[-1]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] in SU2_FREESTREAM and SU2_FREESTREAM[cells[0]] not in found:
            found[SU2_FREESTREAM[cells[0]]] = float(cells[1])
    missing = set(SU2_FREESTREAM.values()) - set(found)
    if missing:
        msg = f"SU2 log: no freestream {sorted(missing)}"
        raise VVError(msg, code="V0904")
    return found


def flux_scales(
    freestream: dict[str, float], length: float, nu_factor: float = 3.0
) -> dict[str, float]:
    """Scale of each equation's integrated flux balance per unit span: its freestream flux times
    `length` (rho U, rho U^2, rho U H, nu_tilde U, with H = E + p / rho and nu_tilde = `nu_factor`
    times the kinematic viscosity, SU2's FREESTREAM_NU_FACTOR)."""
    rho, u = freestream["density"], freestream["speed"]
    enthalpy = freestream["energy"] + freestream["pressure"] / rho
    nu_tilde = nu_factor * freestream["viscosity"] / rho
    momentum = rho * u * u * length
    return {
        "density": rho * u * length, "momentum-x": momentum, "momentum-y": momentum,
        "momentum-z": momentum, "energy": rho * u * enthalpy * length,
        "nu-tilde": nu_tilde * u * length,
    }  # fmt: skip


def stationary(
    values: FloatArray, window: int = WINDOW, rtol: float = RTOL
) -> npt.NDArray[np.bool_]:
    """Per iteration, whether `values` stayed within `rtol` of its current value, relative, over
    the `window` iterations before it (False until there are that many)."""
    out = np.zeros(len(values), dtype=bool)
    for i in range(window, len(values)):
        out[i] = np.max(np.abs(values[i - window : i] - values[i])) <= rtol * abs(values[i])
    return out


def epsilon(
    history: History, scales: dict[str, float], outputs: Sequence[str], rtol: float = RTOL
) -> float:
    """The baseline's level of #493: the largest scaled residual at the first iteration from which
    every output in `outputs` (History field names, e.g. "cd") stays within `rtol` of its final
    value. V0904 if the history never settles."""
    settled = np.ones(len(history.iterations), dtype=bool)
    for name in outputs:
        f = getattr(history, name)
        within = np.abs(f - f[-1]) <= rtol * abs(f[-1])
        # Within from that iteration to the end: a suffix of True.
        settled &= np.flip(np.logical_and.accumulate(np.flip(within)))
    hits = np.flatnonzero(settled)
    if hits.size == 0:
        msg = "the baseline history never settles"
        raise VVError(msg, code="V0904")
    k = int(hits[0])
    return max(float(r[k]) / scales[eq] for eq, r in history.residuals.items())


def absolute_converged_at(
    history: History,
    thresholds: dict[str, float],
    outputs: Sequence[str],
    window: int = WINDOW,
    rtol: float = RTOL,
) -> int | None:
    """Index of the first iteration where every residual is at or below its threshold (epsilon
    times its flux scale) and every output has been stationary over `window` iterations, or None.
    V0904 if an equation has no threshold."""
    if set(history.residuals) - set(thresholds):
        msg = f"no threshold for {sorted(set(history.residuals) - set(thresholds))}"
        raise VVError(msg, code="V0904")
    met = np.ones(len(history.iterations), dtype=bool)
    for eq, r in history.residuals.items():
        met &= r <= thresholds[eq]
    for name in outputs:
        met &= stationary(getattr(history, name), window, rtol)
    hits = np.flatnonzero(met)
    return int(hits[0]) if hits.size else None


# Configuration keys that set the scheme, recorded with a run's convergence.
SCHEME_KEYS = ("CONV_NUM_METHOD_FLOW", "MUSCL_FLOW", "SLOPE_LIMITER_FLOW", "VENKAT_LIMITER_COEFF",
               "ENTROPY_FIX_COEFF", "NUM_METHOD_GRAD")  # fmt: skip


def provenance(run: Path) -> dict[str, object]:
    """What identifies a run directory's result: its provenance.json (image, SU2 version, threads,
    hashes of the configuration and mesh it ran), the precision (SU2 runs in FP64) and the scheme
    keys of its case.cfg. V0904 if either file is missing."""
    try:
        prov = json.loads((run / "provenance.json").read_text())
        text = (run / "case.cfg").read_text()
    except FileNotFoundError as err:
        msg = f"{run}: {err.filename} is missing"
        raise VVError(msg, code="V0904") from None
    cfg = dict(line.split("= ", 1) for line in text.splitlines() if "= " in line)
    return {
        "provenance": prov,
        "precision": "fp64",
        "scheme": {k: cfg[k] for k in SCHEME_KEYS if k in cfg},
    }
