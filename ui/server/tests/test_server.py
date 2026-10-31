import json
import socket
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from autan_ui import create_app
from autan_ui.demo import write_run
from autan_ui.registry import Registry, read_history
from autan_ui.surface import synthetic_wing

STARTED = datetime(2026, 10, 31, 21, 40, 12, tzinfo=UTC)


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    write_run(tmp_path, started=STARTED, aoa=19.57, status="converged", iterations=30)
    return tmp_path


@pytest.fixture
def server(runs: Path) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(create_app(runs, static=None), port=port, log_level="warning")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    while not instance.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join()


def sse_events(response: httpx.Response) -> Iterator[tuple[str, Any]]:
    event = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: ") and event:
            yield event, json.loads(line.removeprefix("data: "))
            event = None


def test_registry_lists_conforming_runs_and_rejects_others(runs: Path) -> None:
    (runs / "notes").mkdir()
    foreign = runs / "run-20261031T000000Z-00000000"
    foreign.mkdir()
    (foreign / "run.json").write_text('{"schema": "other/1"}')

    listed, rejected = Registry(runs).scan()

    assert [run["status"] for run in listed] == ["converged"]
    assert listed[0]["run_id"].startswith("run-20261031T214012Z-")
    assert sorted(r.dir for r in rejected) == ["notes", foreign.name]


def test_history_waits_for_a_complete_line(runs: Path) -> None:
    directory = next(runs.iterdir())
    first = read_history(directory, 0)
    assert [r["iteration"] for r in first.records] == list(range(1, 31))

    with (directory / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    assert read_history(directory, first.offset).records == []

    with (directory / "history.jsonl").open("a") as history:
        history.write('uals": {"density": 1e-9}}\n')
    assert [r["iteration"] for r in read_history(directory, first.offset).records] == [31]


def test_history_stream_delivers_appended_lines_and_status(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    run_id = directory.name
    with httpx.stream("GET", f"{server}/api/runs/{run_id}/history/events", timeout=10) as response:
        events = sse_events(response)
        assert next(events)[0] == "run"
        kind, records = next(events)
        assert kind == "history"
        assert len(records) == 30

        with (directory / "history.jsonl").open("a") as history:
            history.write(json.dumps({"iteration": 31, "time": None, "residuals": {}}) + "\n")
        kind, records = next(events)
        assert (kind, [r["iteration"] for r in records]) == ("history", [31])

        run = json.loads((directory / "run.json").read_text())
        run["status"] = "stopped"
        (directory / "run.json").write_text(json.dumps(run))
        kind, data = next(events)
        assert (kind, data["status"]) == ("run", "stopped")


def test_unknown_run_is_404(server: str) -> None:
    assert httpx.get(f"{server}/api/runs/run-20261031T000000Z-deadbeef").status_code == 404
    assert httpx.get(f"{server}/api/runs/..%2F..%2Fetc").status_code == 404


def test_synthetic_surface_layout() -> None:
    message = synthetic_wing(10_000)
    length = int.from_bytes(message[:4], "little")
    header = json.loads(message[4 : 4 + length])
    body = memoryview(message)[4 + length :]
    buffers = {b["name"]: b for b in header["buffers"]}
    indices = np.frombuffer(
        body, np.uint32, buffers["indices"]["length"], buffers["indices"]["offset"]
    )

    assert (4 + length) % 4 == 0
    assert sum(zone["triangles"] for zone in header["zones"]) == header["triangles"]
    assert indices.size == 3 * header["triangles"]
    assert int(indices.max()) < header["nodes"]
    assert [zone["name"] for zone in header["zones"]] == ["wing.upper", "wing.lower", "wing.tip"]
