"""Residual and force histories of SU2 and Autan runs on one footing, and the protocol's
convergence criterion applied to both.

The criterion is the one case-schema.md (Time) gives a steady Autan run: every residual has dropped
`orders` orders of magnitude below its value at the first iteration of the history, an equation
whose first residual is below 1e-12 times the largest one being exempt while it stays there. Both
codes define a residual the same way, the RMS over the control volumes of the integrated flux
balance (results.md, History; SU2's rms[...] columns), so a drop is compared with a drop.
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
    from pathlib import Path

FloatArray = npt.NDArray[np.float64]
IDENTICALLY_ZERO = 1e-12

# SU2 2D column, Autan equation of the slab (airfoil in x-z): SU2's y is Autan's z.
SU2_2D = {"rms[Rho]": "density", "rms[RhoU]": "momentum-x", "rms[RhoV]": "momentum-z",
          "rms[RhoE]": "energy"}  # fmt: skip
SU2_3D = {"rms[Rho]": "density", "rms[RhoU]": "momentum-x", "rms[RhoV]": "momentum-y",
          "rms[RhoW]": "momentum-z", "rms[RhoE]": "energy"}  # fmt: skip


@dataclass(frozen=True)
class History:
    """Iterations, residuals per equation (linear, not log10), CL, CD and the nose-up pitching
    moment `cmy` of Autan's convention, and, when the code records it, the cumulative wall time."""

    iterations: npt.NDArray[np.int64]
    residuals: dict[str, FloatArray]
    cl: FloatArray
    cd: FloatArray
    cmy: FloatArray
    wall_seconds: FloatArray | None


def read_su2(path: Path) -> History:
    """SU2's history.csv (HISTORY_OUTPUT with ITER, RMS_RES, AERO_COEFF and WALL_TIME).

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
    names = SU2_3D if three_d else SU2_2D
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


def read_autan(path: Path, monitor: str = "forces") -> History:
    """An Autan run's history.jsonl (results.md, History), forces from `monitor`."""
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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


def converged_at(history: History, orders: float) -> int | None:
    """Index into `history` of the first iteration meeting the criterion, or None."""
    first = {eq: r[0] for eq, r in history.residuals.items()}
    largest = max(first.values())
    met = np.ones(len(history.iterations), dtype=bool)
    for eq, r in history.residuals.items():
        if first[eq] < IDENTICALLY_ZERO * largest:
            floor = IDENTICALLY_ZERO * largest
            # Exempt until it first rises above the floor; from then on it must drop like the rest.
            risen = np.maximum.accumulate(r > floor)
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
