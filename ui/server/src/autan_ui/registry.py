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
    progress: dict[str, int] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        runs = sorted(self.runs.values(), key=lambda run: run["started"], reverse=True)
        rejected = [{"dir": name, "reason": reason} for name, reason in self.rejected.items()]
        return {"runs": runs, "rejected": rejected, "progress": self.progress}


@dataclass(frozen=True)
class HistoryChunk:
    records: list[dict[str, Any]]
    offset: int


class Registry:
    """Every run directory directly under `root`, whoever wrote it (UI, SDK, CLI, agent).

    A directory is checked by the SDK's reader (`autan.Results.open`) and checked again only
    when one of `CHECKED_FILES` changes, so rescanning costs a few stats per directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._checked: dict[str, tuple[tuple[Any, ...], dict[str, Any] | str]] = {}
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
                        if outcome["status"] == "running":
                            last = last_iteration(child)
                            if last is not None:
                                snapshot.progress[child.name] = last
            for gone in self._checked.keys() - names:
                del self._checked[gone]
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


def last_iteration(directory: Path) -> int | None:
    """Iteration of the last complete history line: run.json is rewritten only when the status
    changes, so this is how far a running run has got. Reads backwards until it has the whole
    line, however long."""
    try:
        with (directory / "history.jsonl").open("rb") as file:
            end = file.seek(0, 2)
            tail = b""
            while end > 0:
                start = max(end - 65536, 0)
                file.seek(start)
                tail = file.read(end - start) + tail
                end = start
                complete = tail[: tail.rfind(b"\n") + 1]
                if complete.count(b"\n") >= 2 or (end == 0 and complete):
                    break
    except OSError:
        return None
    lines = tail[: tail.rfind(b"\n") + 1].splitlines()
    if not lines:
        return None
    try:
        iteration = json.loads(lines[-1]).get("iteration")
    except (ValueError, AttributeError):
        return None
    return iteration if type(iteration) is int else None


def read_history(directory: Path, offset: int, *, live: bool) -> HistoryChunk:
    """Complete lines of history.jsonl after byte `offset`.

    While the run is live its last line may still be being written and waits; once the run has
    ended, an unterminated last line violates autan.results/1 and raises ValueError.
    """
    path = directory / "history.jsonl"
    if not path.is_file():
        return HistoryChunk([], offset)
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read()
    end = data.rfind(b"\n") + 1
    if end < len(data) and not live:
        msg = "history.jsonl: the last line is not terminated"
        raise ValueError(msg)
    records = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return HistoryChunk(records, offset + end)
