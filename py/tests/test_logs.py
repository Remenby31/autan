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
# Knobs of the relaxation physics that end a run of at most 60 iterations with each final
# status after about 50 iterations; "stopped" is a stop asked of the SDK. At 0.25 s an
# iteration, a run writes for more than 12 s.
ENDINGS = {
    "converged": {"AUTAN_TEST_FACTOR": "0.7"},
    "completed": {"AUTAN_TEST_FACTOR": "1.0"},
    "diverged": {"AUTAN_TEST_FACTOR": "1.0", "AUTAN_TEST_DIVERGE_AT": "50"},
    "failed": {"AUTAN_TEST_FACTOR": "1.0", "AUTAN_TEST_THROW_AT": "50"},
    "stopped": {"AUTAN_TEST_FACTOR": "1.0"},
}
PAUSE_MS = "250"
LIVE_SECONDS = 5.0
# Lines judged live must span longer than LIVE_SECONDS, so that printing them all at the end
# cannot pass.
SPAN_SECONDS = LIVE_SECONDS + 1.0


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    assert MESH_INFO is not None
    case = tmp_path / "hybrid.py"
    case.write_text(
        CASE + "case.time = autan.Steady(max_iterations=60, "
        "checkpoint=autan.Checkpoint(iterations=1))\n"
    )
    meshinfo = tmp_path / "hybrid.meshinfo.json"
    meshinfo.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
        ).stdout
    )
    return case, meshinfo


def _env(**knobs: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTAN_TEST_")}
    return {**env, **knobs}


def _finished_run(tmp_path: Path, **knobs: str) -> Path:
    case, meshinfo = _inputs(tmp_path)
    runs = tmp_path / "runs"
    subprocess.run(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RELAXATION), str(runs)],
        env=_env(**knobs),
        check=True,
    )
    [run] = runs.glob("run-*")
    return run


def _diagnostic_codes(out: str) -> list[str]:
    return [d["code"] for d in json.loads(out[out.index('{\n  "ok"') :])["diagnostics"]]


class Follower:
    """`autan logs -f --json RUN` in its own process, each printed chunk with its arrival."""

    def __init__(self, run: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "autan.cli", "logs", "-f", "--json", str(run)],
            stdout=subprocess.PIPE,
        )
        self.printed: list[tuple[float, bytes]] = []
        self.reader = threading.Thread(target=self._collect, daemon=True)
        self.reader.start()

    def _collect(self) -> None:
        assert self.process.stdout is not None
        while chunk := os.read(self.process.stdout.fileno(), 65536):
            self.printed.append((time.monotonic(), chunk))

    def output(self) -> bytes:
        return b"".join(chunk for _, chunk in self.printed)

    def wait_for(self, expected: bytes, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while self.output() != expected and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.output() == expected

    def finish(self) -> int:
        try:
            status = self.process.wait(timeout=60)
        finally:
            self.process.kill()
            self.reader.join(timeout=10)
        return status


def test_logs_prints_a_runs_log_as_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _finished_run(tmp_path, AUTAN_TEST_WARN="1")
    written = (run / "log.jsonl").read_text()
    lines = written.splitlines()
    for line in lines:
        assert not log_line_problems(LogLine.model_validate(json.loads(line)))
    assert main(["logs", str(run), "--json"]) == 0
    assert capsys.readouterr().out == written
    assert main(["logs", str(run)]) == 0
    shown = capsys.readouterr().out.splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(shown) == len(entries)
    for out, entry in zip(shown, entries, strict=True):
        assert entry["level"] in out
        assert out.endswith(entry["message"])
    # The relaxation physics' W0717 is the run's only line above info.
    assert main(["logs", str(run), "--json", "--level", "warning"]) == 0
    kept = capsys.readouterr().out.splitlines()
    assert kept == [line for line, e in zip(lines, entries, strict=True) if e["level"] != "info"]
    assert any(json.loads(line)["code"] == "W0717" for line in kept)


def test_a_broken_run_directory_is_e0701(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["logs", str(tmp_path / "none"), "--json"]) == 1
    assert _diagnostic_codes(capsys.readouterr().out) == ["E0701"]
    run = _finished_run(tmp_path)
    copy = tmp_path / "copy" / run.name
    # The run's last line carries data: with a NaN in it, it is in shape but for JSON itself.
    last = json.loads((run / "log.jsonl").read_text().splitlines()[-1])
    nan = json.dumps({**last, "data": {**last["data"], "x": float("nan")}}).encode() + b"\n"
    for appended in (b'{"time": "not a time"}\n', b"\xff\n", nan):
        shutil.rmtree(copy.parent, ignore_errors=True)
        shutil.copytree(run, copy)
        assert main(["logs", str(copy), "--json"]) == 0
        capsys.readouterr()
        with (copy / "log.jsonl").open("ab") as log:
            log.write(appended)
        assert main(["logs", str(copy), "--json"]) == 1
        assert _diagnostic_codes(capsys.readouterr().out) == ["E0701"]


def _running_copy(tmp_path: Path) -> Path:
    """A copy of a run directory taken while the runtime writes it, cut to its complete lines;
    the run is stopped once copied."""
    case, meshinfo = _inputs(tmp_path)
    runs = tmp_path / "runs"
    writer = subprocess.Popen(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RELAXATION), str(runs)],
        env=_env(AUTAN_TEST_PAUSE_MS=PAUSE_MS, AUTAN_TEST_FACTOR="1.0"),
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            found = list(runs.glob("run-*/run.json"))
            if found and (found[0].parent / "log.jsonl").read_bytes().count(b"\n") >= 3:
                break
            time.sleep(0.02)
        [run] = runs.glob("run-*")
        copy = tmp_path / "copy" / run.name
        # The writer's temporaries and checkpoint may be renamed while they are copied.
        shutil.copytree(run, copy, ignore=shutil.ignore_patterns(".*", "*.tmp", "checkpoint.cgns"))
        writer.send_signal(signal.SIGTERM)
        writer.wait(timeout=60)
    finally:
        writer.kill()
        writer.wait()
    assert json.loads((copy / "run.json").read_text())["status"] == "running"
    log = (copy / "log.jsonl").read_bytes()
    (copy / "log.jsonl").write_bytes(log[: log.rindex(b"\n") + 1])
    return copy


def test_follow_holds_back_a_partial_line_and_takes_its_rewrite(tmp_path: Path) -> None:
    # Written to as results.md lets a writer: a partial last line, cut and written again
    # (runtime/src/run_directory.cpp), then the final status.
    copy = _running_copy(tmp_path)
    head = (copy / "log.jsonl").read_bytes()
    last = head.splitlines(keepends=True)[-1]
    follower = Follower(copy)
    try:
        follower.wait_for(head, 60)
        with (copy / "log.jsonl").open("ab") as log:
            log.write(b'{"time": "2026-11-11T00:00:00.000Z", "iteration": null, "lev')
        time.sleep(LIVE_SECONDS)
        assert follower.output() == head
        with (copy / "log.jsonl").open("r+b") as log:
            log.truncate(len(head))
        with (copy / "log.jsonl").open("ab") as log:
            log.write(last)
        follower.wait_for(head + last, LIVE_SECONDS)
        info = json.loads((copy / "run.json").read_text())
        replaced = copy / "run.json.tmp"
        replaced.write_text(json.dumps({**info, "status": "stopped"}))
        replaced.replace(copy / "run.json")
        assert follower.finish() == 0
    finally:
        follower.process.kill()
    assert follower.output() == head + last


@pytest.mark.parametrize("ending", list(ENDINGS))
def test_follow_prints_every_line_live_until_the_final_status(tmp_path: Path, ending: str) -> None:
    case, meshinfo = _inputs(tmp_path)
    runs = tmp_path / "runs"
    writer = subprocess.Popen(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RELAXATION), str(runs)],
        env=_env(AUTAN_TEST_PAUSE_MS=PAUSE_MS, **ENDINGS[ending]),
    )
    follower = None
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not list(runs.glob("run-*/run.json")):
            time.sleep(0.02)
        [run] = runs.glob("run-*")
        follower = Follower(run)
        # When each complete line appears in the file, and whether the run was then running.
        appeared: list[tuple[float, bool]] = []
        while follower.process.poll() is None and time.monotonic() < deadline:
            seen = (run / "log.jsonl").read_bytes().count(b"\n")
            running = json.loads((run / "run.json").read_text())["status"] == "running"
            now = time.monotonic()
            appeared.extend((now, running) for _ in range(seen - len(appeared)))
            ready = follower.printed[0][0] if follower.printed else None
            judged = [t for t, _ in appeared if ready is not None and t > ready]
            stop = ending == "stopped" and judged and now - judged[0] > SPAN_SECONDS + 1
            if stop and writer.poll() is None:
                writer.send_signal(signal.SIGTERM)
            time.sleep(0.02)
        assert follower.finish() == 0
        seen = (run / "log.jsonl").read_bytes().count(b"\n")
        appeared.extend((time.monotonic(), False) for _ in range(seen - len(appeared)))
    finally:
        writer.kill()
        writer.wait()
        if follower is not None:
            follower.process.kill()
    assert json.loads((run / "run.json").read_text())["status"] == ending
    assert follower.output() == (run / "log.jsonl").read_bytes()
    # When each line was printed: the arrival of the chunk that completed it.
    printed_at: list[float] = []
    for arrival, chunk in follower.printed:
        printed_at.extend(arrival for _ in range(chunk.count(b"\n")))
    # Once the follower prints (it is ready), every line appended while the run runs is printed
    # within LIVE_SECONDS, over a span longer than LIVE_SECONDS.
    ready = follower.printed[0][0]
    timed = [
        (printed, appeared_at)
        for printed, (appeared_at, running) in zip(printed_at, appeared, strict=True)
        if running and appeared_at > ready
    ]
    assert timed, "no line was appended while the run ran"
    assert timed[-1][1] - timed[0][1] > SPAN_SECONDS
    assert all(printed - appeared_at <= LIVE_SECONDS for printed, appeared_at in timed)
