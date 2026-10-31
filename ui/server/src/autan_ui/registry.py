import json
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RESULTS_SCHEMA = "autan.results/1"
RUN_ID = re.compile(r"run-\d{8}T\d{6}Z-[0-9a-f]{8}")


class RunNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class Rejected:
    dir: str
    reason: str


@dataclass(frozen=True)
class HistoryChunk:
    records: list[dict[str, Any]]
    offset: int


class Registry:
    """Every run directory directly under `root`, whoever wrote it (UI, SDK, CLI, agent)."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").is_file():
            raise RunNotFoundError(run_id)
        return self.root / run_id

    def scan(self) -> tuple[list[dict[str, Any]], list[Rejected]]:
        runs: list[dict[str, Any]] = []
        rejected: list[Rejected] = []
        if not self.root.is_dir():
            return runs, rejected
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                runs.append(read_run(child))
            except (OSError, ValueError) as error:
                log.warning("run directory %s rejected: %s", child, error)
                rejected.append(Rejected(child.name, str(error)))
        runs.sort(key=lambda run: run["started"], reverse=True)
        return runs, rejected

    def details(self, run_id: str) -> dict[str, Any]:
        directory = self.run_dir(run_id)
        return {
            "run": read_run(directory),
            "case": tomllib.loads((directory / "case.toml").read_text(encoding="utf-8")),
            "mesh_info": json.loads((directory / "mesh-info.json").read_text(encoding="utf-8")),
        }


def read_run(directory: Path) -> dict[str, Any]:
    run_json = directory / "run.json"
    if not run_json.is_file():
        msg = "no run.json"
        raise ValueError(msg)
    run = json.loads(run_json.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or run.get("schema") != RESULTS_SCHEMA:
        msg = f"run.json schema is not {RESULTS_SCHEMA}"
        raise ValueError(msg)
    if run.get("run_id") != directory.name:
        msg = "run.json run_id differs from the directory name"
        raise ValueError(msg)
    return run


def read_history(directory: Path, offset: int) -> HistoryChunk:
    """Complete lines of history.jsonl after byte `offset`; a line still being written waits."""
    path = directory / "history.jsonl"
    if not path.is_file():
        return HistoryChunk([], offset)
    with path.open("rb") as file:
        file.seek(offset)
        data = file.read()
    end = data.rfind(b"\n") + 1
    records = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return HistoryChunk(records, offset + end)
