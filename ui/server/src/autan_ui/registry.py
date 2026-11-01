import json
import logging
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import Any

import autan
from autan.errors import CaseError

from autan_ui.confine import Folder, is_link
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
        self._budgets: dict[str, dict[str, Any]] = {}
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
                        try:
                            with Folder(child) as folder:
                                live = outcome["status"] == "running"
                                summary = self._summaries.update(child.name, folder, live=live)
                        except (OSError, ValueError, ArithmeticError) as error:
                            log.warning("no summary for %s: %s", child, error)
                            summary = None
                        if summary is not None:
                            budget = self._budgets.get(child.name, {})
                            snapshot.summaries[child.name] = summary | budget
            gone = self._checked.keys() - names
            for name in gone:
                del self._checked[name]
                self._budgets.pop(name, None)
            self._summaries.forget(gone)
        return snapshot

    def _check(self, directory: Path) -> dict[str, Any] | str:
        # log.jsonl grows during a run like history.jsonl: only its presence can change the
        # verdict, so its size and time do not trigger a new check.
        key = (
            *(_stat(directory / name) for name in CHECKED_FILES),
            (directory / "log.jsonl").is_file(),
        )
        cached = self._checked.get(directory.name)
        if cached and cached[0] == key:
            return cached[1]
        outcome: dict[str, Any] | str
        if key[0] is None:
            outcome = "no run.json"
        else:
            try:
                results = autan.Results.open(directory)
                outcome = results.run.document()
                time = results.case.document()["time"]
                case = results.case.document()
                self._budgets[directory.name] = {
                    "target_orders": time.get("residual_orders"),
                    "max_iterations": time.get("max_iterations", time.get("steps")),
                    "settings": _settings(case),
                }
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
            case_toml = folder.read("case.toml").decode()
            outputs = []
            for name in sorted(path.name for path in folder.path.iterdir()):
                stat = folder.stat(name)
                if stat is not None and S_ISREG(stat.st_mode) and not name.startswith("."):
                    outputs.append({"name": name, "bytes": stat.st_size, "modified": stat.st_mtime})
            return {
                "run": read_run(folder),
                "case": tomllib.loads(case_toml),
                "case_toml": case_toml,
                "mesh_info": json.loads(folder.read("mesh-info.json")),
                "outputs": outputs,
            }


RUN_FILES = (*CHECKED_FILES, "history.jsonl", "log.jsonl", "manufactured.json")


def _inside(root: Path, directory: Path) -> bool:
    """A real directory directly under `root` whose run files are not links: every file the
    server reads for a run then lies under the runs directory."""
    return (
        directory.is_dir()
        and not is_link(directory)
        and directory.resolve().parent == root
        and not any(is_link(directory / name) for name in RUN_FILES)
    )


def _settings(case: dict[str, Any]) -> dict[str, Any]:
    """The settings a study usually sweeps, as the case writes them, for labels and polars."""
    freestream = case.get("freestream", {})
    physics = case.get("physics", {})
    turbulence = physics.get("turbulence", {})
    return {
        "aoa": freestream.get("aoa"),
        "sideslip": freestream.get("sideslip"),
        "mach": freestream.get("mach"),
        "re": freestream.get("re"),
        "turbulence": turbulence.get("model") if isinstance(turbulence, dict) else None,
        "precision": physics.get("precision"),
    }


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
