"""A run's ``log.jsonl``, read as results.md#log defines it, while the run writes it too."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from autan.errors import CaseError, Location, diagnostic
from autan.results import LogLevel, LogLine, Results, log_line_problems

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_RANK: dict[LogLevel, int] = {"info": 0, "warning": 1, "error": 2}
# How often a follower looks for new lines and for the run's end (docs/sdk/logs.md).
POLL_SECONDS = 0.2


def _broken(path: Path, message: str) -> CaseError:
    return CaseError([diagnostic("E0701", message, location=Location(file=str(path)))])


def _parsed(path: Path, text: str) -> LogLine:
    try:
        entry = LogLine.model_validate(json.loads(text))
    except (ValueError, ValidationError) as error:
        raise _broken(path, f"a log line is not in the results.md shape: {error}") from error
    problems = log_line_problems(entry)
    if problems:
        raise _broken(path, f"a log line breaks results.md#log: {problems[0]}")
    return entry


def _status(run: Path) -> str:
    try:
        status = json.loads((run / "run.json").read_text())["status"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise _broken(run / "run.json", f"cannot read the run's status: {error}") from error
    return str(status)


def read(
    run: Path, *, follow: bool = False, level: LogLevel = "info"
) -> Iterator[tuple[str, LogLine]]:
    """Each line of the run directory ``run``'s ``log.jsonl`` at ``level`` or above, as its
    text (without the line feed) and its parsed form, in the order written. With ``follow``,
    waits for the lines a running run appends and returns once ``run.json`` holds a final
    status, after the last line written before it. A run directory that is not one, or a line
    outside results.md#log, raises ``CaseError`` ``E0701``."""
    Results.open(run)
    path = run / "log.jsonl"
    offset = 0
    pending = b""
    while True:
        # The status before the lines: both writers log their last line, then rewrite
        # run.json, so the lines read after a final status include every one before it.
        done = _status(run) != "running"
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
        except OSError as error:
            raise _broken(path, f"cannot read log.jsonl: {error}") from error
        offset += len(chunk)
        *complete, pending = (pending + chunk).split(b"\n")
        for raw in complete:
            text = raw.decode("utf-8")
            entry = _parsed(path, text)
            if _RANK[entry.level] >= _RANK[level]:
                yield text, entry
        if done and pending:
            raise _broken(path, "the last line of a finished run's log.jsonl is incomplete")
        if done or not follow:
            return
        time.sleep(POLL_SECONDS)


def text(entry: LogLine) -> str:
    """One line for a reader: time, level, source, iteration, code and message."""
    iteration = "-" if entry.iteration is None else str(entry.iteration)
    code = f" {entry.code}" if entry.code else ""
    return f"{entry.time} {entry.level:<7} {entry.source:<8} {iteration:>7}{code} {entry.message}"
