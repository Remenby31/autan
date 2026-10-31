import json
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import autan
import httpx
import numpy as np
import pytest
import uvicorn
from autan_ui import create_app
from autan_ui.registry import Registry, read_history
from autan_ui.surface import synthetic_wing

PY = Path(__file__).parents[3] / "py"


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    case.run(mesh_info, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
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
    source = next(runs.iterdir())
    (runs / "notes").mkdir()
    foreign = runs / "run-20261031T000000Z-00000000"
    foreign.mkdir()
    (foreign / "run.json").write_text('{"schema": "other/1"}')
    bad_time = shutil.copytree(source, runs / "run-20261031T000001Z-00000001")
    run = json.loads((bad_time / "run.json").read_text())
    run |= {"run_id": bad_time.name, "started": "not-a-timestamp"}
    (bad_time / "run.json").write_text(json.dumps(run))
    (runs / ".run-20261031T000002Z-00000002.tmp").mkdir()

    snapshot = Registry(runs).scan()

    assert [(r["case"]["name"], r["status"]) for r in snapshot.runs.values()] == [
        ("crm-hl", "stopped")
    ]
    assert sorted(snapshot.rejected) == sorted(["notes", foreign.name, bad_time.name])
    assert snapshot.rejected[foreign.name].startswith("E0701")


def test_registry_rechecks_a_run_when_any_contract_file_changes(runs: Path) -> None:
    source = next(runs.iterdir())
    registry = Registry(runs)
    case = (source / "case.toml").read_text()

    (source / "case.toml").write_text("schema = 1\n")
    assert source.name in registry.scan().rejected
    (source / "case.toml").write_text(case)
    assert source.name in registry.scan().runs


def test_an_unreadable_run_is_rejected_not_fatal(runs: Path) -> None:
    source = next(runs.iterdir())
    (source / "history.jsonl").unlink()
    (source / "history.jsonl").mkdir()
    assert source.name in Registry(runs).scan().rejected


def test_history_holds_a_partial_line_only_while_live(runs: Path) -> None:
    directory = next(runs.iterdir())
    first = read_history(directory, 0, live=False)
    assert [r["iteration"] for r in first.records] == list(range(1, 31))

    with (directory / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    assert read_history(directory, first.offset, live=True).records == []
    with pytest.raises(ValueError, match="not terminated"):
        read_history(directory, first.offset, live=False)

    with (directory / "history.jsonl").open("a") as history:
        history.write('uals": {"density": 1e-9}}\n')
    chunk = read_history(directory, first.offset, live=True)
    assert [r["iteration"] for r in chunk.records] == [31]


def test_registry_stream_reports_runs_and_rejections_as_they_appear(
    runs: Path, server: str
) -> None:
    source = next(runs.iterdir())
    with httpx.stream("GET", f"{server}/api/runs/events", timeout=10) as response:
        events = sse_events(response)
        kind, snapshot = next(events)
        assert (kind, len(snapshot["runs"])) == ("runs", 1)

        staging = shutil.copytree(source, runs / ".staging")
        run = json.loads((staging / "run.json").read_text())
        run |= {"run_id": "run-20261031T235959Z-0123abcd", "started": "2026-10-31T23:59:59Z"}
        (staging / "run.json").write_text(json.dumps(run))
        copy = staging.rename(runs / run["run_id"])
        kind, data = next(events)
        assert (kind, data["run_id"]) == ("run", copy.name)

        (copy / "run.json").write_text("{")
        assert [next(events), next(events)] == [
            ("removed", copy.name),
            ("rejected", [{"dir": copy.name, "reason": ANY}]),
        ]


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
        run["status"] = "failed"
        (directory / "run.json").write_text(json.dumps(run))
        kind, data = next(events)
        assert (kind, data["status"]) == ("run", "failed")


def test_unknown_run_is_404_before_any_stream(server: str) -> None:
    unknown = "run-20261031T000000Z-deadbeef"
    assert httpx.get(f"{server}/api/runs/{unknown}").status_code == 404
    assert httpx.get(f"{server}/api/runs/{unknown}/history/events").status_code == 404
    assert httpx.get(f"{server}/api/runs/..%2F..%2Fetc").status_code == 404


def test_a_rejected_run_is_409_with_its_reason(runs: Path, server: str) -> None:
    source = next(runs.iterdir())
    (source / "mesh-info.json").write_text("{}")
    response = httpx.get(f"{server}/api/runs/{source.name}")
    assert (response.status_code, response.json()["detail"][:5]) == (409, "E0701")
    assert httpx.get(f"{server}/api/runs/{source.name}/history/events").status_code == 409


def test_foreign_host_header_is_refused(server: str) -> None:
    assert httpx.get(f"{server}/api/runs", headers={"Host": "evil.example"}).status_code == 400


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
