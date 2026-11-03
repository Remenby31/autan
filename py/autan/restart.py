"""Restarting from a previous run (docs/contracts/solver-input.md#start)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- dataclass field type, resolved at runtime
from typing import TYPE_CHECKING, Any, Literal

from autan.errors import CaseError, Diagnostic, diagnostic
from autan.results import ResumePoint, StartInfo, residual_names
from autan.schema import Steady

if TYPE_CHECKING:
    from autan.meshinfo import MeshInfo
    from autan.results import Results
    from autan.schema import CaseSpec

Mode = Literal["continue", "warm"]
RESUMABLE = frozenset({"converged", "completed", "stopped"})


def _budget(spec: CaseSpec) -> int:
    return spec.time.max_iterations if isinstance(spec.time, Steady) else spec.time.steps


def _without_budget(spec: CaseSpec) -> dict[str, Any]:
    document = spec.document()
    document["time"] = {
        k: v
        for k, v in document["time"].items()
        if k not in ("max_iterations", "steps", "checkpoint")
    }
    return document


def _resume_point(previous: Results) -> ResumePoint | None:
    """Where ``previous`` can be continued from: its final solution when it ended converged,
    completed or stopped, else its last checkpoint (results.md#checkpointcgns); ``None`` without
    either. Raises ``CaseError`` when that file does not conform."""
    run = previous.run
    if run.status in RESUMABLE:
        path = previous.path / "solution.cgns"
        if not path.is_file():
            return None
        previous.field("Density")  # reads the solution and checks it against run.json
        return ResumePoint(
            iteration=run.iterations,
            time=run.time,
            averaging=run.averaging,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    if not (previous.path / "checkpoint.cgns").is_file():
        return None
    return previous.checkpoint()


def problems(
    spec: CaseSpec, mesh_info: MeshInfo, previous: Results, mode: Mode
) -> list[Diagnostic]:
    """Why ``spec`` on ``mesh_info`` cannot start from ``previous`` in ``mode``."""
    run = previous.run

    def problem(code: str, message: str) -> Diagnostic:
        return diagnostic(code, message, run=run.run_id)

    found = []
    if spec.initial_state is not None:
        found.append(
            problem("E0512", "a case with an initial-state file cannot start from a previous run")
        )
    point = None
    try:
        point = _resume_point(previous)
    except CaseError as error:
        outside = [d for d in error.diagnostics if d.code == "E0134"]
        found.extend(
            outside
            or [problem("E0707", f"the state of run {run.run_id} does not conform: {error.codes}")]
        )
    else:
        if point is None and run.status in RESUMABLE:
            found.append(problem("E0707", f"run {run.run_id} is {run.status} but has no solution"))
        elif point is None:
            found.append(
                problem(
                    "E0707",
                    f"run {run.run_id} is {run.status} and wrote no checkpoint before it ended: "
                    "there is no state to start from",
                )
            )
        elif point.iteration == 0:
            found.append(problem("E0707", f"run {run.run_id} stopped before its first iteration"))
    if run.mesh.sha256 != mesh_info.source.sha256:
        found.append(problem("E0203", f"run {run.run_id} was computed on another mesh"))
    old = previous.case
    if mode == "continue":
        if _without_budget(old) != _without_budget(spec):
            found.append(
                problem(
                    "E0708",
                    "continue needs the same case as the previous run, "
                    "except its iteration or step budget and its checkpoint cadence",
                )
            )
        elif point is not None and _budget(spec) <= point.iteration:
            found.append(
                problem(
                    "E0708",
                    f"the budget ({_budget(spec)}) must exceed the "
                    f"{point.iteration} iterations already done",
                )
            )
    elif (old.physics.kind, residual_names(old)) != (spec.physics.kind, residual_names(spec)):
        found.append(
            problem(
                "E0708",
                "warm start needs the same physics and transported variables as the previous run",
            )
        )
    return found


def start_info(previous: Results, mode: Mode) -> StartInfo:
    """The provenance the new run records, with the hash of the file it reads: the final
    solution, or the checkpoint of a run that did not end converged, completed or stopped."""
    run = previous.run
    final = run.status in RESUMABLE
    point = _resume_point(previous)
    if point is None:
        raise CaseError([diagnostic("E0707", f"run {run.run_id} has no state to start from")])
    file: Literal["solution", "checkpoint"] = "solution" if final else "checkpoint"
    # A checkpoint carries its own convergence state; a final solution's is the history's.
    reference: dict[str, float] | None = None
    found_woken: list[str] | None = None
    if mode == "continue" and isinstance(previous.case.time, Steady):
        if final:
            reference = _reference(previous)
            found_woken = None if reference is None else woken(previous, reference, point.iteration)
        else:
            reference, found_woken = point.residual_reference, point.woken
    return StartInfo(
        run_id=run.run_id,
        identity=run.identity,
        mode=mode,
        iterations=point.iteration,
        time=point.time,
        solution_sha256=point.sha256,
        file=file,
        residual_reference=reference,
        woken=found_woken,
    )


def _reference(previous: Results) -> dict[str, float] | None:
    """Residuals at iteration 1 of the original run, carried along a chain of continuations:
    the steady convergence criterion is relative to them (case-schema.md#time)."""
    if not isinstance(previous.case.time, Steady):
        return None
    start = previous.run.start
    if start is not None and start.mode == "continue":
        return start.residual_reference
    history = previous.history
    return {name: float(history.residual(name)[0]) for name in history.equations}


# An equation whose first residual is below FLOOR times the largest is identically zero
# (case-schema.md#time).
FLOOR = 1e-12


def woken(previous: Results, reference: dict[str, float], up_to: int) -> list[str]:
    """The identically zero equations that rose above their floor during ``previous``, up to
    iteration ``up_to``, or the runs it continues: they no longer have the exemption
    (case-schema.md#time)."""
    floor = FLOOR * max(reference.values(), default=0.0)
    start = previous.run.start
    found = set(start.woken or []) if start is not None and start.mode == "continue" else set()
    history = previous.history
    upto = history.iterations <= up_to
    for name, first in reference.items():
        if first < floor and bool((history.residual(name)[upto] > floor).any()):
            found.add(name)
    return sorted(found)


@dataclass(frozen=True)
class Start:
    """A checked start: the provenance to record and the run directory to read."""

    info: StartInfo
    run: Path


def resolve(
    spec: CaseSpec, mesh_info: MeshInfo, previous: Results | None, mode: Mode
) -> Start | None:
    """Check a start and return it; ``None`` without a previous run."""
    if previous is None:
        return None
    found = problems(spec, mesh_info, previous, mode)
    if found:
        raise CaseError(found)
    return Start(info=start_info(previous, mode), run=previous.path.absolute())
