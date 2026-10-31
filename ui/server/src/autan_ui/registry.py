import json
import logging
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import autan
from autan.errors import CaseError

from autan_ui.summary import Summaries

log = logging.getLogger(__name__)

RUN_ID = re.compile(r"run-\d{8}T\d{6}Z-[0-9a-f]{8}")


class RunNotFoundError(LookupError):
    pass


class RunRejectedError(ValueError):
    pass


# Files whose change can change the verdict; history.jsonl grows during a run and is checked by
# the history stream instead.
CHECKED_FILES = ("run.json", "case.toml", "mesh-info.json", "solution.cgns")


@dataclass(frozen=True)
class Snapshot:
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        runs = sorted(self.runs.values(), key=lambda run: run["started"], reverse=True)
        rejected = [{"dir": name, "reason": reason} for name, reason in self.rejected.items()]
        return {"runs": runs, "rejected": rejected, "summaries": self.summaries}


class Registry:
    """Every run directory directly under `root`, whoever wrote it (UI, SDK, CLI, agent).

    A directory is checked by the SDK's reader (`autan.Results.open`) and checked again only
    when one of `CHECKED_FILES` changes, so rescanning costs a few stats per directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._checked: dict[str, tuple[tuple[Any, ...], dict[str, Any] | str]] = {}
        self._summaries = Summaries()
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        """The directory of an accepted run."""
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id).is_dir():
            raise RunNotFoundError(run_id)
        with self._lock:
            outcome = self._check(self.root / run_id)
        if isinstance(outcome, str):
            raise RunRejectedError(outcome)
        return self.root / run_id

    def scan(self) -> Snapshot:
        snapshot = Snapshot()
        with self._lock:
            names = set()
            if self.root.is_dir():
                for child in sorted(self.root.iterdir()):
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    names.add(child.name)
                    outcome = self._check(child)
                    if isinstance(outcome, str):
                        snapshot.rejected[child.name] = outcome
                    else:
                        snapshot.runs[child.name] = outcome
                        live = outcome["status"] == "running"
                        summary = self._summaries.update(child, live=live)
                        if summary is not None:
                            snapshot.summaries[child.name] = summary
            gone = self._checked.keys() - names
            for name in gone:
                del self._checked[name]
            self._summaries.forget(gone)
        return snapshot

    def _check(self, directory: Path) -> dict[str, Any] | str:
        key = tuple(_stat(directory / name) for name in CHECKED_FILES)
        cached = self._checked.get(directory.name)
        if cached and cached[0] == key:
            return cached[1]
        outcome: dict[str, Any] | str
        if key[0] is None:
            outcome = "no run.json"
        else:
            try:
                outcome = autan.Results.open(directory).run.document()
            except CaseError as error:
                outcome = "; ".join(f"{d.code} {d.message}" for d in error.diagnostics)
            except (OSError, ValueError) as error:
                outcome = f"cannot read the run directory: {error}"
        if isinstance(outcome, str):
            log.warning("run directory %s rejected: %s", directory, outcome)
        self._checked[directory.name] = (key, outcome)
        return outcome

    def details(self, directory: Path) -> dict[str, Any]:
        return {
            "run": read_run(directory),
            "case": tomllib.loads((directory / "case.toml").read_text(encoding="utf-8")),
            "mesh_info": json.loads((directory / "mesh-info.json").read_text(encoding="utf-8")),
        }


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def read_run(directory: Path) -> dict[str, Any]:
    """run.json, checked against the SDK's model of its fields; ValueError otherwise."""
    try:
        document = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        return autan.RunInfo.model_validate(document).document()
    except (OSError, ValueError) as error:
        msg = f"run.json: {error}"
        raise ValueError(msg) from error
