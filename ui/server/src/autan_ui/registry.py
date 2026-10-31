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


@dataclass(frozen=True)
class Snapshot:
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        runs = sorted(self.runs.values(), key=lambda run: run["started"], reverse=True)
        rejected = [{"dir": name, "reason": reason} for name, reason in self.rejected.items()]
        return {"runs": runs, "rejected": rejected}


@dataclass(frozen=True)
class HistoryChunk:
    records: list[dict[str, Any]]
    offset: int


class Registry:
    """Every run directory directly under `root`, whoever wrote it (UI, SDK, CLI, agent).

    A directory is checked by the SDK's reader (`autan.Results.open`) and checked again only
    when its run.json changes, so rescanning costs one stat per directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._checked: dict[str, tuple[tuple[int, int] | None, dict[str, Any] | str]] = {}
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").is_file():
            raise RunNotFoundError(run_id)
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
            for gone in self._checked.keys() - names:
                del self._checked[gone]
        return snapshot

    def _check(self, directory: Path) -> dict[str, Any] | str:
        try:
            stat = (directory / "run.json").stat()
            key: tuple[int, int] | None = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = None
        cached = self._checked.get(directory.name)
        if cached and cached[0] == key:
            return cached[1]
        outcome: dict[str, Any] | str
        if key is None:
            outcome = "no run.json"
        else:
            try:
                outcome = autan.Results.open(directory).run.document()
            except CaseError as error:
                outcome = "; ".join(f"{d.code} {d.message}" for d in error.diagnostics)
                log.warning("run directory %s rejected: %s", directory, outcome)
        self._checked[directory.name] = (key, outcome)
        return outcome

    def details(self, run_id: str) -> dict[str, Any]:
        directory = self.run_dir(run_id)
        return {
            "run": read_run(directory),
            "case": tomllib.loads((directory / "case.toml").read_text(encoding="utf-8")),
            "mesh_info": json.loads((directory / "mesh-info.json").read_text(encoding="utf-8")),
        }


def read_run(directory: Path) -> dict[str, Any]:
    run: dict[str, Any] = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    return run


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
