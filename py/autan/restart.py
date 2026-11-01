"""Restarting from a previous run (docs/contracts/solver-input.md#start)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- dataclass field type, resolved at runtime
from typing import TYPE_CHECKING, Any, Literal

from autan.errors import CaseError, Diagnostic, diagnostic
from autan.results import StartInfo, residual_names
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
        k: v for k, v in document["time"].items() if k not in ("max_iterations", "steps")
    }
    return document


def problems(
    spec: CaseSpec, mesh_info: MeshInfo, previous: Results, mode: Mode
) -> list[Diagnostic]:
    """Why ``spec`` on ``mesh_info`` cannot start from ``previous`` in ``mode``."""
    run = previous.run

    def problem(code: str, message: str) -> Diagnostic:
        return diagnostic(code, message, run=run.run_id)

    found = []
    if run.status not in RESUMABLE or not (previous.path / "solution.cgns").is_file():
        found.append(
            problem(
                "E0707",
                f"run {run.run_id} is {run.status}; a start needs a "
                "converged, completed or stopped run with its solution",
            )
        )
    elif run.iterations == 0:
        found.append(problem("E0707", f"run {run.run_id} stopped before its first iteration"))
    else:
        try:
            previous.field("Density")  # reads the solution and checks it against run.json
        except CaseError as error:
            found.append(
                problem(
                    "E0707", f"the solution of run {run.run_id} does not conform: {error.codes}"
                )
            )
    if run.mesh.sha256 != mesh_info.source.sha256:
        found.append(problem("E0203", f"run {run.run_id} was computed on another mesh"))
    old = previous.case
    if mode == "continue":
        if _without_budget(old) != _without_budget(spec):
            found.append(
                problem(
                    "E0708",
                    "continue needs the same case as the previous run, "
                    "except its iteration or step budget",
                )
            )
        elif _budget(spec) <= run.iterations:
            found.append(
                problem(
                    "E0708",
                    f"the budget ({_budget(spec)}) must exceed the "
                    f"{run.iterations} iterations already done",
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
    """The provenance the new run records, with the hash of the solution it reads."""
    run = previous.run
    digest = hashlib.sha256((previous.path / "solution.cgns").read_bytes()).hexdigest()
    return StartInfo(
        run_id=run.run_id,
        identity=run.identity,
        mode=mode,
        iterations=run.iterations,
        time=run.time,
        solution_sha256=digest,
        residual_reference=_reference(previous) if mode == "continue" else None,
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
