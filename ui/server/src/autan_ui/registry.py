import json
import logging
import os
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import autan
from autan.errors import CaseError

from autan_ui.confine import Folder, is_link

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
        if not RUN_ID.fullmatch(run_id):
            raise RunNotFoundError(run_id)
        # The path comes from listing the runs directory, never from the requested string.
        directory = next((child for child in self.root.iterdir() if child.name == run_id), None)
        if directory is None or not _inside(self.root, directory):
            raise RunNotFoundError(run_id)
        with self._lock:
            outcome = self._check(directory)
        if isinstance(outcome, str):
            raise RunRejectedError(outcome)
        return directory

    def scan(self) -> Snapshot:
        snapshot = Snapshot()
        with self._lock:
            names = set()
            if self.root.is_dir():
                for child in sorted(self.root.iterdir()):
                    if child.name.startswith(".") or not child.is_dir():
                        continue
                    if not _inside(self.root, child):
                        snapshot.rejected[child.name] = "is or contains a symbolic link"
                        continue
                    names.add(child.name)
                    outcome = self._check(child)
                    if isinstance(outcome, str):
                        snapshot.rejected[child.name] = outcome
                    else:
                        snapshot.runs[child.name] = outcome
                        if outcome["status"] == "running":
                            try:
                                with Folder(child) as folder:
                                    last = last_iteration(folder)
                            except OSError:
                                last = None
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
        with Folder(directory) as folder:
            return {
                "run": read_run(folder),
                "case": tomllib.loads(folder.read("case.toml").decode()),
                "mesh_info": json.loads(folder.read("mesh-info.json")),
            }


RUN_FILES = (*CHECKED_FILES, "history.jsonl")


def _inside(root: Path, directory: Path) -> bool:
    """A real directory directly under `root` whose run files are not links: every file the
    server reads for a run then lies under the runs directory."""
    return (
        directory.is_dir()
        and not is_link(directory)
        and directory.resolve().parent == root
        and not any(is_link(directory / name) for name in RUN_FILES)
    )


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def read_run(folder: Folder) -> dict[str, Any]:
    """run.json, checked against the SDK's model of its fields; ValueError otherwise."""
    try:
        document = json.loads(folder.read("run.json"))
        return autan.RunInfo.model_validate(document).document()
    except (OSError, ValueError) as error:
        msg = f"run.json: {error}"
        raise ValueError(msg) from error


def last_iteration(folder: Folder) -> int | None:
    """Iteration of the last complete history line: run.json is rewritten only when the status
    changes, so this is how far a running run has got. Reads backwards until it has the whole
    line, however long."""
    try:
        with os.fdopen(folder.open("history.jsonl"), "rb") as file:
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


def read_history(folder: Folder, offset: int, *, live: bool) -> HistoryChunk:
    """Complete lines of history.jsonl after byte `offset`.

    While the run is live its last line may still be being written and waits; once the run has
    ended, an unterminated last line violates autan.results/1 and raises ValueError.
    """
    if folder.stat("history.jsonl") is None:
        return HistoryChunk([], offset)
    data = folder.read("history.jsonl", offset)
    end = data.rfind(b"\n") + 1
    if end < len(data) and not live:
        msg = "history.jsonl: the last line is not terminated"
        raise ValueError(msg)
    records = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return HistoryChunk(records, offset + end)
