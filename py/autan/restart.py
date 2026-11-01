"""Restarting from a previous run (docs/contracts/solver-input.md#start)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- dataclass field type, resolved at runtime
from typing import TYPE_CHECKING, Any, Literal

from autan.errors import CaseError, Diagnostic, diagnostic
from autan.results import StartInfo
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
    found = []
    if run.status not in RESUMABLE or not (previous.path / "solution.cgns").is_file():
        found.append(
            diagnostic(
                "E0707",
                f"run {run.run_id} is {run.status}; a start needs a converged, completed "
                "or stopped run with its solution",
                run=run.run_id,
            )
        )
    if run.mesh.sha256 != mesh_info.source.sha256:
        found.append(
            diagnostic("E0203", f"run {run.run_id} was computed on another mesh", run=run.run_id)
        )
    old = previous.case
    if mode == "continue":
        if _without_budget(old) != _without_budget(spec):
            found.append(
                diagnostic(
                    "E0708",
                    "continue needs the same case as the previous run, except its "
                    "iteration or step budget",
                    run=run.run_id,
                )
            )
        elif _budget(spec) <= run.iterations:
            found.append(
                diagnostic(
                    "E0708",
                    f"the budget ({_budget(spec)}) must exceed the {run.iterations} "
                    "iterations already done",
                    run=run.run_id,
                )
            )
    elif (old.physics.kind, old.physics.turbulence.model) != (
        spec.physics.kind,
        spec.physics.turbulence.model,
    ):
        found.append(
            diagnostic(
                "E0708",
                "warm start needs the same physics and turbulence model as the previous run",
                run=run.run_id,
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
    )


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
