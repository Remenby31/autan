"""A pytest plugin for ADR 0003's auxiliary suite, loaded with `-p oracle.freeze.nodeid_report` by
`heldout aux`. It appends JSON lines to the file named by AUTAN_NODEID_REPORT:
- one per test phase, with the exact `report.nodeid` pytest uses (the same ids as
  `pytest --collect-only -q`), the phase and its outcome;
- the SHA-256 of the spike library this pytest process loads (AUTAN_SPIKE_LIB), measured in this
  process at session start and again at session end, so the library is attested by the process that
  ran the tests, not by the wrapper."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _emit(entry: dict[str, Any]) -> None:
    path = os.environ.get("AUTAN_NODEID_REPORT")
    if path:
        with Path(path).open("a") as f:
            f.write(json.dumps(entry) + "\n")


def _library() -> str | None:
    lib = os.environ.get("AUTAN_SPIKE_LIB")
    return (
        hashlib.sha256(Path(lib).read_bytes()).hexdigest() if lib and Path(lib).is_file() else None
    )


def pytest_sessionstart(session: Any) -> None:  # noqa: ARG001
    _emit({"library_sha256": _library(), "when": "sessionstart"})


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # noqa: ARG001
    _emit({"library_sha256": _library(), "when": "sessionfinish", "exitstatus": int(exitstatus)})


def pytest_runtest_logreport(report: Any) -> None:
    _emit({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})
