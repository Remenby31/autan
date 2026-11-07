"""The stop verdict of a steady run recomputed from its history (case-schema.md#time)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from autan.schema import Steady

from autan_vv.errors import VVError

if TYPE_CHECKING:
    import autan
    from autan.results import RunInfo

# case-schema.md#time.
ZERO_FLOOR = 1e-12
PLATEAU_RATIO = 0.9
PLATEAU_STEPS = 100
BAND = 10.0
# euler.md#time-stepping: the first-order start, the only iterations a physics marks provisional,
# ends with its 201st step at the latest.
MAX_PROVISIONAL = 201


@dataclass(frozen=True)
class _History:
    equations: list[str]
    iterations: list[int]
    residuals: list[list[float]]  # one line per iteration, one value per equation
    reference: list[float]
    exempt_until: list[int]  # an exempt equation is exempt on the lines before this index
    drop: float
    floors: list[float] | None


@dataclass(frozen=True)
class _Disagreement:
    why: str
    iteration: int
    equation: str | None = None


def _first_stop(h: _History, provisional: int) -> tuple[int | None, list[int]]:
    """The first line meeting the rule and the equations met there on a plateau, the first
    `provisional` lines counting towards no plateau."""
    n = len(h.reference)
    plateau_ref, since = [math.inf] * n, [0] * n
    for i, line in enumerate(h.residuals):
        counted = i - provisional + 1
        met_all, on_plateau = True, []
        for e, r in enumerate(line):
            flat = False
            if h.floors is not None and counted >= 1:
                if r < PLATEAU_RATIO * plateau_ref[e]:
                    plateau_ref[e], since[e] = r, counted
                flat = counted - since[e] >= PLATEAU_STEPS
            met = i < h.exempt_until[e] or r <= h.reference[e] * h.drop
            plateau = h.floors is not None and flat and r <= BAND * h.floors[e]
            if not met and plateau:
                on_plateau.append(e)
            met_all = met_all and (met or plateau)
        if met_all:
            return i, on_plateau
    return None, []


def _disagreement(h: _History, run: RunInfo, budget: int, provisional: int) -> _Disagreement | None:
    stop, on_plateau = _first_stop(h, provisional)
    last = len(h.residuals) - 1
    if stop is not None and stop < last:
        return _Disagreement(
            f"every equation meets the rule at iteration {h.iterations[stop]}, yet the run "
            f"went on to {h.iterations[last]}",
            h.iterations[stop],
        )
    if h.iterations[last] > budget:
        return _Disagreement(
            f"the run went on to iteration {h.iterations[last]}, past max_iterations {budget}",
            h.iterations[last],
        )
    if run.status == "completed":
        if stop is not None:
            return _Disagreement(
                f"completed, but every equation meets the rule at iteration {h.iterations[last]}",
                h.iterations[last],
            )
        if h.iterations[last] != budget:
            return _Disagreement(
                f"completed at iteration {h.iterations[last]}, not at max_iterations {budget}",
                h.iterations[last],
            )
        return None
    if stop is None:
        line = h.residuals[last]
        failing = [
            e
            for e, r in enumerate(line)
            if last >= h.exempt_until[e] and r > h.reference[e] * h.drop
        ]
        # Name one outside its plateau band if any: one inside may only lack its 100 iterations.
        outside = [e for e in failing if h.floors is None or line[e] > BAND * h.floors[e]]
        e = (outside or failing)[0]
        return _Disagreement(
            f"at iteration {h.iterations[last]} {h.equations[e]} is {line[e]:.6g}, above its "
            f"reference {h.reference[e]:.6g} dropped {-math.log10(h.drop):g} orders and on no "
            f"round-off plateau within {BAND:g} floors",
            h.iterations[last],
            h.equations[e],
        )
    plateau = [h.equations[e] for e in on_plateau]
    recorded = run.convergence.plateau if run.convergence is not None else []
    if plateau != recorded:
        return _Disagreement(
            f"convergence.plateau is {recorded}, the history gives {plateau}",
            h.iterations[last],
        )
    return None


def check_stop(results: autan.Results) -> None:
    """Refuse a steady run whose run.json status is not the one its history gives (V0730).

    The rule is case-schema.md#time, implemented here from that text and not by the solver: a
    `converged` run's last history line is the first one where every equation has dropped
    `residual_orders` from its reference (iteration 1, or `start.residual_reference` of a
    continuation; for one identically zero there, the largest of its kind), is identically zero
    and still at its floor, or in `fp32+acc64` sits on a
    round-off plateau within 10 floors, at most at `max_iterations`; a `completed` run has no
    such line and ends at `max_iterations`. `convergence.plateau` and the `W0710` diagnostics must be the ones the
    history gives. Other statuses and unsteady runs are not judged.

    Not guaranteed: the history does not say which iterations were provisional (a first-order
    start), so a verdict is accepted when some provisional prefix of 0 to 201 iterations gives
    it. The drop, the exemption, W0710 and the budget are exact; a plateau is checked as the
    band and 100 iterations without a new reference, not from its exact start. The floors are
    run.json's `convergence.floor`, not recomputed from the mesh.
    """
    run, time = results.run, results.case.time
    if not isinstance(time, Steady) or run.status not in ("converged", "completed"):
        return
    history = results.history
    equations = history.equations
    iterations = [int(i) for i in history.iterations]
    if not iterations:
        _refuse(results, _Disagreement(f"{run.status} with an empty history", 0))
    series = [history.residual(e) for e in equations]
    residuals = [[float(s[i]) for s in series] for i in range(len(iterations))]

    start = run.start
    if start is not None and start.residual_reference is not None:
        reference = [start.residual_reference[e] for e in equations]
        woken = set(start.woken or [])
    else:
        reference, woken = residuals[0], set()
    zero = ZERO_FLOOR * max(reference)
    exempt_until = [0] * len(equations)
    wakes: set[tuple[str, int]] = set()
    for e, name in enumerate(equations):
        if reference[e] < zero and name not in woken:
            above = next((i for i, line in enumerate(residuals) if line[e] > zero), None)
            exempt_until[e] = len(residuals) if above is None else above
            if above is not None:
                wakes.add((name, iterations[above]))
    # A woken equation drops from the largest first residual of its kind: the momentum components
    # share units, every other equation takes the largest of all (case-schema.md#time).
    reference = [
        max(
            r
            for k, r in enumerate(reference)
            if not name.startswith("momentum-") or equations[k].startswith("momentum-")
        )
        if reference[e] < zero
        else reference[e]
        for e, name in enumerate(equations)
    ]
    reported = {
        (str(d.data.get("equation")), int(d.data.get("iteration", -1)))
        for d in run.diagnostics
        if d.code == "W0710"
    }
    if reported != wakes:
        name, iteration = min(reported ^ wakes, key=lambda w: w[1])
        _refuse(
            results,
            _Disagreement(
                f"W0710 for {name} at iteration {iteration} is "
                f"{'reported' if (name, iteration) in reported else 'missing'}: the history "
                f"leaves the zero floor {zero:.6g} at {sorted(wakes, key=lambda w: w[1])}",
                iteration,
                name,
            ),
        )

    floors = run.convergence.floor if run.convergence is not None else None
    h = _History(
        equations,
        iterations,
        residuals,
        reference,
        exempt_until,
        10.0 ** (-time.residual_orders),
        [floors[e] for e in equations] if floors is not None else None,
    )
    prefixes = range(min(MAX_PROVISIONAL, len(residuals)) + 1) if floors is not None else [0]
    first: _Disagreement | None = None
    for provisional in prefixes:
        found = _disagreement(h, run, time.max_iterations, provisional)
        if found is None:
            return
        first = first or found
    assert first is not None
    _refuse(results, first)


def _refuse(results: autan.Results, found: _Disagreement) -> None:
    msg = f"run {results.run.run_id}: run.json says {results.run.status}, but {found.why}"
    raise VVError(
        msg, code="V0730", data={"iteration": found.iteration, "equation": found.equation}
    )
