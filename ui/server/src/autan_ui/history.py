import json
from dataclasses import dataclass
from typing import Any

from autan_ui.confine import Folder


@dataclass(frozen=True)
class HistoryChunk:
    records: list[dict[str, Any]]
    offset: int


def read_history(folder: Folder, offset: int, *, live: bool) -> HistoryChunk:
    """Complete lines of history.jsonl after byte `offset` (see `read_lines`)."""
    return read_lines(folder, "history.jsonl", offset, live=live)


def read_lines(folder: Folder, name: str, offset: int, *, live: bool) -> HistoryChunk:
    """Complete JSON lines of `name` (history.jsonl, log.jsonl) after byte `offset`.

    While the run is live its last line may still be being written and waits; once the run has
    ended, an unterminated last line violates autan.results/1 and raises ValueError.
    """
    if folder.stat(name) is None:
        return HistoryChunk([], offset)
    data = folder.read(name, offset)
    end = data.rfind(b"\n") + 1
    if end < len(data) and not live:
        msg = f"{name}: the last line is not terminated"
        raise ValueError(msg)
    records = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return HistoryChunk(records, offset + end)
