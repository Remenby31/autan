"""`autan logs` through its output and exit status, on run directories the runtime wrote
(docs/sdk/logs.md; acceptance.md, M4). The runs use `autan-run-relaxation`, the shipped runtime
with the core's test physics, whose knobs end a run with each final status."""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from autan.cli import main
from autan.results import LogLine, log_line_problems

from tests.test_solver_backend import CASE, MESH, MESH_INFO, RELAXATION

pytestmark = pytest.mark.skipif(
    None in (RELAXATION, MESH_INFO),
    reason="autan-run-relaxation and autan-mesh-info not built",
)

RUN = (
    "import sys, autan\nfrom pathlib import Path\n"
    "case, mesh = autan.load(sys.argv[1]), autan.MeshInfo.load(sys.argv[2])\n"
    "backend = autan.SolverBackend(executable=Path(sys.argv[3]))\n"
    "try:\n    case.run(mesh, backend=backend, runs=Path(sys.argv[4]))\n"
    "except KeyboardInterrupt:\n    sys.exit(130)\n"
    "except Exception:\n    sys.exit(1)\n"
)
# The relaxation physics' knobs that end a run of at most 40 iterations with each final status;
# "stopped" is a stop asked of the SDK.
ENDINGS = {
    "converged": {},
    "completed": {"AUTAN_TEST_FACTOR": "1.0"},
    "diverged": {"AUTAN_TEST_DIVERGE_AT": "12"},
    "failed": {"AUTAN_TEST_THROW_AT": "12"},
    "stopped": {"AUTAN_TEST_FACTOR": "1.0"},
}


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    assert MESH_INFO is not None
    case = tmp_path / "hybrid.py"
    case.write_text(
        CASE + "case.time = autan.Steady(max_iterations=40, "
        "checkpoint=autan.Checkpoint(iterations=1))\n"
    )
    meshinfo = tmp_path / "hybrid.meshinfo.json"
    meshinfo.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
        ).stdout
    )
    return case, meshinfo


def _clean_env(**knobs: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTAN_TEST_")}
    return {**env, **knobs}


def _finished_run(tmp_path: Path, **knobs: str) -> Path:
    case, meshinfo = _inputs(tmp_path)
    runs = tmp_path / "runs"
    subprocess.run(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RELAXATION), str(runs)],
        env=_clean_env(**knobs),
        check=True,
    )
    [run] = runs.glob("run-*")
    return run


def test_logs_prints_a_runs_log_as_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _finished_run(tmp_path, AUTAN_TEST_WARN="1")
    written = (run / "log.jsonl").read_text()
    lines = written.splitlines()
    # Every line is in the results.md#log shape and keeps its rules.
    for line in lines:
        assert not log_line_problems(LogLine.model_validate(json.loads(line)))
    # --json: the log byte for byte.
    assert main(["logs", str(run), "--json"]) == 0
    assert capsys.readouterr().out == written
    # Readable: one line per event, with its level and message.
    assert main(["logs", str(run)]) == 0
    shown = capsys.readouterr().out.splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(shown) == len(entries)
    for out, entry in zip(shown, entries, strict=True):
        assert entry["level"] in out
        assert out.endswith(entry["message"])
    # --level keeps that level and above: here the relaxation physics' W0717.
    assert main(["logs", str(run), "--json", "--level", "warning"]) == 0
    kept = capsys.readouterr().out.splitlines()
    assert kept == [line for line, e in zip(lines, entries, strict=True) if e["level"] != "info"]
    assert any(json.loads(line)["code"] == "W0717" for line in kept)


def test_a_broken_run_directory_is_e0701(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["logs", str(tmp_path / "none"), "--json"]) == 1
    assert [d["code"] for d in json.loads(capsys.readouterr().out)["diagnostics"]] == ["E0701"]
    # A line outside the results.md#log shape, in a run directory that is otherwise whole.
    run = _finished_run(tmp_path)
    broken = tmp_path / "broken"
    shutil.copytree(run, broken)
    with (broken / "log.jsonl").open("a") as log:
        log.write('{"time": "not a time"}\n')
    assert main(["logs", str(broken), "--json"]) == 1
    out = capsys.readouterr().out
    assert [d["code"] for d in json.loads(out[out.index('{\n  "ok"') :])["diagnostics"]] == [
        "E0701"
    ]


@pytest.mark.parametrize("ending", list(ENDINGS))
def test_follow_prints_every_line_live_until_the_final_status(tmp_path: Path, ending: str) -> None:
    case, meshinfo = _inputs(tmp_path)
    runs = tmp_path / "runs"
    env = _clean_env(AUTAN_TEST_PAUSE_MS="100", **ENDINGS[ending])
    writer = subprocess.Popen(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RELAXATION), str(runs)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not list(runs.glob("run-*/run.json")):
            time.sleep(0.02)
        [run] = runs.glob("run-*")
        follower = subprocess.Popen(
            [sys.executable, "-m", "autan.cli", "logs", "-f", "--json", str(run)],
            stdout=subprocess.PIPE,
        )
        printed: list[tuple[float, bytes]] = []

        def collect() -> None:
            assert follower.stdout is not None
            # One item at a time, as the follower prints it: each with its arrival time.
            printed.extend((time.monotonic(), line) for line in follower.stdout)

        reader = threading.Thread(target=collect)
        reader.start()
        # When each complete line appears in the file, and whether the run was still running.
        appeared: list[tuple[float, bool]] = []
        while follower.poll() is None and time.monotonic() < deadline:
            seen = (run / "log.jsonl").read_bytes().count(b"\n")
            running = json.loads((run / "run.json").read_text())["status"] == "running"
            now = time.monotonic()
            appeared.extend((now, running) for _ in range(seen - len(appeared)))
            # Stop once two lines have appeared after the follower went live.
            live_lines = sum(1 for t, _ in appeared if printed and t > printed[0][0])
            if ending == "stopped" and live_lines >= 2 and writer.poll() is None:
                writer.send_signal(signal.SIGTERM)
            time.sleep(0.02)
        assert follower.wait(timeout=60) == 0
        reader.join(timeout=10)
    finally:
        writer.kill()
        writer.wait()
    assert json.loads((run / "run.json").read_text())["status"] == ending
    # Every line, in order and once each, the last ones included.
    assert b"".join(line for _, line in printed) == (run / "log.jsonl").read_bytes()
    # Live: once the follower prints, each line appended while the run runs is printed within
    # 5 s of its appearing in the file.
    started = printed[0][0]
    live = [
        (out, file)
        for (out, _), (file, running) in zip(printed, appeared, strict=False)
        if running and file > started
    ]
    assert live, "no line was appended while the run ran"
    assert all(out - file <= 5.0 for out, file in live)
