import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HistoryChunk:
    records: list[dict[str, Any]]
    offset: int


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
