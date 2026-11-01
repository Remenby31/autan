"""``autan.SolverBackend``: run a case with the solver runtime ``autan-run``.

The SDK plans the case, writes its solver input (docs/contracts/solver-input.md) and
calls ``autan-run --input FILE --runs DIR``; the runtime writes the run directory of
docs/contracts/results.md, which ``autan.Results`` reads back.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from autan.errors import CaseError, Diagnostic, diagnostic
from autan.results import Devices, Results
from autan.solver_input import build

if TYPE_CHECKING:
    from autan.case import Case
    from autan.meshinfo import MeshInfo

EXECUTABLE_VARIABLE = "AUTAN_RUN"


def find_executable() -> Path | None:
    """``$AUTAN_RUN`` if set, else ``autan-run`` on the PATH."""
    explicit = os.environ.get(EXECUTABLE_VARIABLE)
    if explicit:
        return Path(explicit)
    found = shutil.which("autan-run")
    return Path(found) if found else None


@dataclass(frozen=True)
class SolverBackend:
    """Runs a case with ``autan-run`` on ``devices``; ``executable`` defaults to
    :func:`find_executable`. Input errors of the runtime raise ``CaseError`` with its
    coded diagnostics (``E0705``, ``E0706``, ``E0203`` …); a stopped run raises
    ``KeyboardInterrupt`` after the runtime has finalized it."""

    devices: Devices = field(default_factory=lambda: Devices(kind="cpu", count=1))
    deterministic: bool = False
    executable: Path | None = None

    def run(self, case: Case, mesh_info: MeshInfo, runs: Path) -> Results:
        executable = self.executable or find_executable()
        if executable is None or not executable.is_file():
            missing = diagnostic(
                "E0804",
                f"the solver runtime autan-run is not installed (set ${EXECUTABLE_VARIABLE} "
                "or put it on the PATH)",
            )
            raise CaseError([missing])
        document = build(
            case, mesh_info, devices=self.devices, deterministic=self.deterministic
        ).document()
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="autan-input-") as folder:
            source = Path(folder) / "solver-input.json"
            source.write_text(json.dumps(document, indent=2) + "\n")
            process = subprocess.Popen(
                [str(executable), "--input", str(source), "--runs", str(runs)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate()
            except KeyboardInterrupt:
                # Let the runtime finalize the run as stopped, with its solution.
                process.send_signal(signal.SIGINT)
                process.communicate()
                raise
        if process.returncode == 130:
            raise KeyboardInterrupt
        if process.returncode == 2:
            raise CaseError(_diagnostics(stderr))
        lines = stdout.strip().splitlines()
        if process.returncode not in (0, 1) or not lines:
            failure = diagnostic(
                "E0704", f"autan-run exited with {process.returncode}: {stderr.strip()}"
            )
            raise CaseError([failure])
        return Results.open(lines[-1])


def _diagnostics(stderr: str) -> list[Diagnostic]:
    found = []
    for line in stderr.splitlines():
        try:
            found.append(Diagnostic.model_validate(json.loads(line)))
        except ValueError:
            continue
    if not found:
        found.append(diagnostic("E0705", f"autan-run rejected its input: {stderr.strip()}"))
    return found
