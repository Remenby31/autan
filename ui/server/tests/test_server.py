import hashlib
import http.client
import json
import math
import shutil
import socket
import threading
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import ANY

import autan
import httpx
import numpy as np
import numpy.typing as npt
import pytest
import uvicorn
from autan_ui import create_app
from autan_ui.confine import Folder
from autan_ui.history import read_history
from autan_ui.registry import Registry
from autan_ui.summary import residual_orders
from autan_ui.surface import synthetic_wing
from fastapi import FastAPI

PY = Path(__file__).parents[3] / "py"


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    case.run(mesh_info, backend=autan.FakeBackend(iterations=30), runs=tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def static(tmp_path: Path) -> Path:
    """A built app next to a file it must never serve, with a link pointing at that file."""
    (tmp_path / "secret.txt").write_text("SECRET")
    app = tmp_path / "static"
    (app / "assets").mkdir(parents=True)
    (app / "index.html").write_text("INDEX")
    (app / "assets" / "app.js").write_text("APP")
    (app / "leak").symlink_to(tmp_path / "secret.txt")
    return app


@contextmanager
def serving(app: FastAPI) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    while not instance.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        instance.should_exit = True
        thread.join()


@pytest.fixture
def server(runs: Path, static: Path) -> Iterator[str]:
    with serving(create_app(runs, static=static)) as url:
        yield url


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
    folder = Folder(directory)
    first = read_history(folder, 0, live=False)
    assert [r["iteration"] for r in first.records] == list(range(1, 31))

    with (directory / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    assert read_history(folder, first.offset, live=True).records == []
    with pytest.raises(ValueError, match="not terminated"):
        read_history(folder, first.offset, live=False)

    with (directory / "history.jsonl").open("a") as history:
        history.write('uals": {"density": 1e-9}}\n')
    chunk = read_history(folder, first.offset, live=True)
    assert [r["iteration"] for r in chunk.records] == [31]
    folder.close()


def test_registry_stream_reports_runs_and_rejections_as_they_appear(
    runs: Path, server: str
) -> None:
    source = next(runs.iterdir())
    with httpx.stream("GET", f"{server}/api/runs/events", timeout=10) as response:
        events = (e for e in sse_events(response) if e[0] != "summaries")
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
        events = (e for e in sse_events(response) if e[0] != "log")
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


def test_every_run_reports_its_iteration_and_residual_drop(runs: Path) -> None:
    source = next(runs.iterdir())
    run = json.loads((source / "run.json").read_text())
    (source / "run.json").write_text(json.dumps(run | {"status": "running", "finished": None}))
    records = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
    with (source / "history.jsonl").open("a") as history:
        history.write('{"iteration": 31, "time": null, "resid')
    registry = Registry(runs)

    summary = registry.scan().summaries[source.name]

    first, last = records[0]["residuals"], records[-1]["residuals"]
    slowest = min(math.log10(first[name] / last[name]) for name in first)
    assert summary["iteration"] == 30
    assert summary["orders"] == pytest.approx(slowest, abs=1e-3)
    assert len(summary["trend"]) == 30
    assert (summary["target_orders"], summary["max_iterations"]) == (8.0, 10000)
    case = tomllib.loads((source / "case.toml").read_text())
    assert summary["settings"]["aoa"] == case["freestream"]["aoa"]
    assert summary["settings"]["turbulence"] == case["physics"]["turbulence"]["model"]

    # Lines appended after the directory was checked are not re-validated by the registry.
    with (source / "history.jsonl").open("a") as history:
        history.write('uals": {}, "padding": "' + "x" * 200_000 + '"}\n')
        history.write('{"iteration": 1e400}\n')
    snapshot = registry.scan()
    assert source.name in snapshot.runs
    assert snapshot.summaries[source.name]["iteration"] == 31


def test_run_details_and_plan(runs: Path, server: str) -> None:
    run_id = next(runs.iterdir()).name
    details = httpx.get(f"{server}/api/runs/{run_id}").json()
    assert {o["name"] for o in details["outputs"]} >= {
        "run.json",
        "case.toml",
        "mesh-info.json",
        "history.jsonl",
        "solution.cgns",
    }
    assert details["case_toml"] == (runs / run_id / "case.toml").read_text()
    mesh_info = json.loads((runs / run_id / "mesh-info.json").read_text())

    plan = httpx.get(f"{server}/api/runs/{run_id}/plan").json()

    assert plan["ok"] is True
    assert {b["zone"] for b in plan["boundaries"]} == {
        z["name"] for z in mesh_info["boundary_zones"]
    }


@pytest.mark.parametrize(
    "target",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/..%2f..%2fsecret.txt",
        "//etc/passwd",
        "/leak",
    ],
)
def test_static_files_never_leave_the_build(server: str, target: str) -> None:
    host, port = server.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port))
    connection.putrequest("GET", target, skip_accept_encoding=True)
    connection.endheaders()
    body = connection.getresponse().read()
    connection.close()
    assert body == b"INDEX"
    assert httpx.get(f"{server}/assets/app.js").text == "APP"


def test_static_files_are_checked_when_served(server: str, static: Path, tmp_path: Path) -> None:
    (static / "added.js").write_text("ADDED")
    (static / "assets" / "app.js").unlink()
    (static / "assets" / "app.js").symlink_to(tmp_path / "secret.txt")

    assert httpx.get(f"{server}/added.js").text == "INDEX"
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"
    (static / "assets" / "app.js").unlink()
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"

    elsewhere = tmp_path / "elsewhere-assets"
    elsewhere.mkdir()
    (elsewhere / "app.js").write_text("SECRET")
    shutil.rmtree(static / "assets")
    (static / "assets").symlink_to(elsewhere, target_is_directory=True)
    assert httpx.get(f"{server}/assets/app.js").text == "INDEX"


def test_a_linked_index_disables_static_files(runs: Path, static: Path, tmp_path: Path) -> None:
    (static / "index.html").unlink()
    (static / "index.html").symlink_to(tmp_path / "secret.txt")
    with serving(create_app(runs, static=static)) as url:
        assert httpx.get(f"{url}/").status_code == 404
        assert httpx.get(f"{url}/api/runs").status_code == 200


@pytest.mark.parametrize("linked", ["directory", "run.json", "history.jsonl", "log.jsonl"])
def test_a_run_reached_through_a_link_is_refused(
    runs: Path, server: str, tmp_path: Path, linked: str
) -> None:
    source = next(runs.iterdir())
    outside = shutil.copytree(source, tmp_path / "elsewhere" / source.name)
    if linked == "directory":
        shutil.rmtree(source)
        source.symlink_to(outside, target_is_directory=True)
    else:
        (source / linked).unlink()
        (source / linked).symlink_to(outside / linked)

    assert httpx.get(f"{server}/api/runs/{source.name}").status_code == 404
    assert httpx.get(f"{server}/api/runs/{source.name}/history/events").status_code == 404
    assert Registry(runs).scan().rejected == {source.name: "is or contains a symbolic link"}


def test_a_stream_keeps_reading_the_run_it_opened(runs: Path, server: str, tmp_path: Path) -> None:
    source = next(runs.iterdir())
    outside = shutil.copytree(source, tmp_path / "elsewhere" / source.name)
    url = f"{server}/api/runs/{source.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        events = sse_events(response)
        assert next(events)[0] == "run"
        assert next(events)[0] == "history"

        moved = source.rename(tmp_path / "moved")
        source.symlink_to(outside, target_is_directory=True)
        for directory, iteration in ((outside, 999), (moved, 31)):
            with (directory / "history.jsonl").open("a") as history:
                record: dict[str, Any] = {"iteration": iteration, "time": None, "residuals": {}}
                history.write(json.dumps(record) + "\n")

        _, records = next(e for e in events if e[0] == "history")
        assert [r["iteration"] for r in records] == [31]


def test_a_manufactured_solution_run_plans_against_its_archive(
    runs: Path, server: str, tmp_path: Path
) -> None:
    fixture = PY / "tests" / "fixtures" / "euler.mms.json"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    case = tmp_path / "mms.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa, kg, mol, s\n"
        f"gas = autan.IdealGas(name='mms', molar_mass={8.314462618 / 287.0!r} * kg / mol,"
        " gamma=1.4, prandtl=0.72, viscosity=autan.ConstantViscosity(mu=1e-3 * Pa * s))\n"
        "case = autan.Case('mms', mesh='crm-hl.cgns')\n"
        "case.physics = autan.Compressible(turbulence=autan.Inviscid(), gas=gas)\n"
        "case.freestream(mach=0.3, pressure=101325 * Pa, temperature=300 * K)\n"
        f"case.verification = autan.Manufactured(path='euler.mms.json', sha256={digest!r})\n"
    )
    shutil.copy(fixture, tmp_path / "euler.mms.json")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    result = autan.load(case).run(mesh_info, backend=autan.FakeBackend(iterations=2), runs=runs)

    plan = httpx.get(f"{server}/api/runs/{result.path.name}/plan").json()

    assert plan["ok"] is True, plan


@pytest.mark.parametrize(
    ("first", "now", "expected"),
    [
        ({"density": 1e-300, "energy": 1.0}, {"density": 1e-310, "energy": 1e-8}, 8.0),
        ({"density": 1.0, "energy": 0.0}, {"density": 1e-8, "energy": 0.0}, 8.0),
        ({"density": 1.0, "energy": 0.0}, {"density": 1e-8, "energy": 1.0}, None),
        ({"density": 1.0, "energy": 1.0}, {"density": 0.0, "energy": 1e-8}, 8.0),
    ],
)
def test_residual_drop_is_the_slowest_equation(
    first: dict[str, float], now: dict[str, float], expected: float | None
) -> None:
    orders = residual_orders(first, now)
    assert orders == (None if expected is None else pytest.approx(expected))


def test_an_oracle_state_run_plans_against_its_archived_face_states(
    runs: Path, server: str, tmp_path: Path
) -> None:
    mesh = json.loads((PY / "tests" / "fixtures" / "crm-hl.meshinfo.json").read_text())
    for zone in mesh["boundary_zones"]:
        if zone["name"] == "farfield":
            zone.update(faces=2, type=None, native_type=None)
    (tmp_path / "mesh.meshinfo.json").write_text(json.dumps(mesh))
    states = json.dumps(
        {
            "schema": "autan.face-states/1",
            "oracle": "test",
            "mesh_sha256": mesh["source"]["sha256"],
            "variables": ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"],
            "zones": {
                "farfield": {
                    "faces": 2,
                    "centroids": [[0, 0, 0], [1, 0, 0]],
                    "values": [[1.2, 30, 0, 0, 1e5], [1.2, 30, 0, 0, 1e5]],
                }
            },
        }
    ).encode()
    (tmp_path / "ringleb.json").write_bytes(states)
    case = tmp_path / "oracle.py"
    case.write_text(
        (PY / "examples" / "crm_hl.py").read_text().replace("autan.SA()", "autan.Laminar()")
        + "case.boundary('farfield').oracle_state(path='ringleb.json', "
        + f"sha256={hashlib.sha256(states).hexdigest()!r})\n"
    )
    mesh_info = autan.MeshInfo.load(tmp_path / "mesh.meshinfo.json")
    result = autan.load(case).run(mesh_info, backend=autan.FakeBackend(iterations=2), runs=runs)
    (tmp_path / "ringleb.json").unlink()  # The plan reads the run's archive, not the source.

    plan = httpx.get(f"{server}/api/runs/{result.path.name}/plan").json()

    assert plan["ok"] is True, plan


def test_several_histories_share_one_stream(runs: Path, server: str) -> None:
    first = next(runs.iterdir())
    second = shutil.copytree(first, runs / ".staging")
    run = json.loads((second / "run.json").read_text())
    run |= {"run_id": "run-20261031T235959Z-0123abcd", "started": "2026-10-31T23:59:59Z"}
    (second / "run.json").write_text(json.dumps(run))
    second = second.rename(runs / run["run_id"])
    url = f"{server}/api/runs/history/events?run={first.name}&run={second.name}"
    resume = {"Last-Event-ID": json.dumps({first.name: 0, second.name: None})}
    with httpx.stream("GET", url, headers=resume, timeout=10) as response:
        events = sse_events(response)
        seen = [next(events) for _ in range(6)]
    assert sorted((kind, data["run_id"]) for kind, data in seen) == sorted(
        [(k, r) for k in ("run", "history", "log") for r in (first.name, second.name)]
    )
    assert (
        httpx.get(f"{server}/api/runs/history/events?run=run-20261031T000000Z-deadbeef").status_code
        == 404
    )


def test_the_log_streams_with_a_documentation_link_per_code(runs: Path, server: str) -> None:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    diverged = case.run(mesh_info, backend=autan.FakeBackend(diverge_at=20), runs=runs).path
    written = [json.loads(line) for line in (diverged / "log.jsonl").read_text().splitlines()]
    url = f"{server}/api/runs/{diverged.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        _, lines = next(e for e in sse_events(response) if e[0] == "log")

    assert [{k: v for k, v in line.items() if k != "doc"} for line in lines] == written
    coded = [line for line in lines if line["code"]]
    assert [line["code"] for line in coded if line["code"] == "D0107"] == ["D0107"]
    assert all(line["doc"].endswith(line["code"].lower()) for line in coded)
    assert all(line["doc"] is None for line in lines if not line["code"])


def test_removing_the_log_after_a_scan_rejects_the_run(runs: Path) -> None:
    source = next(runs.iterdir())
    registry = Registry(runs)
    assert source.name in registry.scan().runs
    (source / "log.jsonl").unlink()
    assert source.name in registry.scan().rejected


def test_stream_cursors_count_only_what_was_sent(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    url = f"{server}/api/runs/{directory.name}/history/events"
    ids = []
    with httpx.stream("GET", url, timeout=10) as response:
        for line in response.iter_lines():
            if line.startswith("id: "):
                ids.append(json.loads(line.removeprefix("id: ")))
            if len(ids) == 2:
                break
    history_size = (directory / "history.jsonl").stat().st_size
    log_size = (directory / "log.jsonl").stat().st_size
    assert ids == [
        {"history.jsonl": history_size, "log.jsonl": 0},
        {"history.jsonl": history_size, "log.jsonl": log_size},
    ]


def test_a_log_line_that_breaks_the_contract_ends_the_stream(runs: Path, server: str) -> None:
    directory = next(runs.iterdir())
    run = json.loads((directory / "run.json").read_text())
    (directory / "run.json").write_text(json.dumps(run | {"status": "running", "finished": None}))
    url = f"{server}/api/runs/{directory.name}/history/events"
    with httpx.stream("GET", url, timeout=10) as response:
        events = sse_events(response)
        assert [next(events)[0] for _ in range(3)] == ["run", "history", "log"]
        with (directory / "log.jsonl").open("a") as log:
            log.write('{"time": "yesterday", "iteration": null, "level": "info"}\n')
        kind, reason = next(e for e in events if e[0] != "history")
    assert kind == "invalid"
    assert reason.startswith("log.jsonl")


def test_the_wall_surface_carries_cp_and_cf_from_the_wall_solution(runs: Path, server: str) -> None:
    import h5py  # noqa: PLC0415 -- only this test edits a CGNS file
    from autan.solver_input import freestream_state  # noqa: PLC0415

    directory = next(runs.iterdir())
    state = freestream_state(autan.Results.open(directory).case)
    q_inf = 0.5 * state.density * state.speed**2
    with h5py.File(directory / "solution.cgns", "r+") as tree:
        zone = tree["Base"]["Zone"]
        wall = zone["WallSolution"]
        wall["Pressure"][" data"][...] = state.pressure + q_inf
        wall["WallShearStressX"][" data"][...] = 0.6 * q_inf
        wall["WallShearStressY"][" data"][...] = 0.8 * q_inf
        wall["WallShearStressZ"][" data"][...] = 0.0
        start = zone["NGonElements"]["ElementStartOffset"][" data"][()]
        walls = {
            name
            for name, bc in zone["ZoneBC"].items()
            if bytes(bc[" data"][()]).decode().startswith("BCWall")
        }
        faces = {name: zone["ZoneBC"][name]["PointList"][" data"][()].ravel() - 1 for name in walls}

    message = httpx.get(f"{server}/api/runs/{directory.name}/surface").content
    length = int.from_bytes(message[:4], "little")
    header = json.loads(message[4 : 4 + length])
    body = memoryview(message)[4 + length :]
    field = {b["name"]: b for b in header["buffers"]}

    def values(name: str) -> npt.NDArray[np.float32]:
        b = field[name]
        return np.frombuffer(body, np.float32, b["length"], b["offset"])

    assert {z["name"] for z in header["zones"]} == walls
    for z in header["zones"]:
        counts = start[faces[z["name"]] + 1] - start[faces[z["name"]]]
        assert z["triangles"] == int((counts - 2).sum())
    np.testing.assert_allclose(values("field:Cp"), 1.0, rtol=1e-5)
    np.testing.assert_allclose(values("field:Cf"), 1.0, rtol=1e-5)


def test_a_continued_run_measures_its_drop_from_the_original_first_line(runs: Path) -> None:
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    first = autan.Results.open(next(runs.iterdir()))
    continued = case.run(
        mesh_info, backend=autan.FakeBackend(iterations=60), runs=runs, start=first
    )
    run = json.loads((continued.path / "run.json").read_text())
    reference = run["start"]["residual_reference"]
    last = json.loads((continued.path / "history.jsonl").read_text().splitlines()[-1])["residuals"]

    summary = Registry(runs).scan().summaries[continued.path.name]

    expected = min(math.log10(reference[k] / last[k]) for k in reference)
    assert summary["orders"] == pytest.approx(expected, abs=1e-3)
    assert summary["trend"][0] > 0  # the first line of the continuation is already below
