import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parents[1] / "lock" / "autan-gpu-lock"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the wrapper reads Linux's /proc")


def wrapper(lock: Path, *args: str, **kwargs) -> subprocess.Popen:
    env = {**os.environ, "AUTAN_GPU_LOCK": str(lock), "AUTAN_AGENT": "tester"}
    return subprocess.Popen([str(WRAPPER), *args], env=env, stderr=subprocess.PIPE, **kwargs)


def events(lock: Path) -> list[list[str]]:
    return [line.split("\t") for line in (lock.parent / "holders.log").read_text().splitlines()]


def wait_for(lock: Path, event: str) -> None:
    log = lock.parent / "holders.log"
    for _ in range(200):
        if log.exists() and any(r[1] == event for r in events(lock)):
            return
        time.sleep(0.05)
    pytest.fail(f"no {event} record")


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] not in "ZX"
    except FileNotFoundError:
        return False


def pid_from(path: Path) -> int:
    for _ in range(200):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    pytest.fail(f"{path} never written")


def free(lock: Path) -> bool:
    with lock.open() as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    path = tmp_path / "gpu.lock"
    path.touch()
    (tmp_path / "holders.log").touch()
    return path


def test_waits_for_the_holder_runs_the_command_and_records_the_hold(lock):
    with lock.open() as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = wrapper(lock, "-m", "unit test", "--", "sh", "-c", "exit 3")
        wait_for(lock, "wait")
        time.sleep(0.5)
        assert proc.poll() is None
    assert proc.wait(timeout=10) == 3
    rows = events(lock)
    assert [r[1] for r in rows] == ["wait", "take", "release"]
    assert all(r[2] == "tester" and r[6] == "unit test" for r in rows)
    assert rows[2][5] == "3"


def test_gives_up_after_the_wait_limit(lock):
    with lock.open() as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = wrapper(lock, "-m", "unit test", "-w", "1", "--", "true")
        assert proc.wait(timeout=10) == 75
    assert [r[1] for r in events(lock)] == ["wait", "timeout"]


def test_the_hold_limit_stops_every_worker_before_releasing(lock, tmp_path):
    # A shell whose worker ignores TERM: only KILL, at the limit, stops it.
    pid_file = tmp_path / "worker"
    worker = f'trap "" TERM; sleep 60 & echo $! > {pid_file}; wait'
    proc = wrapper(lock, "-m", "unit test", "-t", "2", "--", "sh", "-c", worker)
    wait_for(lock, "take")
    start = time.monotonic()
    assert proc.wait(timeout=20) == 124
    assert time.monotonic() - start < 3.5
    assert events(lock)[-1][1] == "overrun"
    assert not alive(pid_from(pid_file))
    assert free(lock)


@pytest.mark.parametrize("hold", ["0", "601", "01100", "1e3"])
def test_refuses_a_hold_outside_one_to_six_hundred_seconds(lock, hold):
    assert wrapper(lock, "-m", "unit test", "-t", hold, "--", "true").wait(timeout=10) == 2
    assert events(lock) == []


def test_no_process_left_behind_outlives_the_hold(lock, tmp_path):
    # Also one a job-control shell put in another process group of the session.
    for shell_options in ("", "set -m;"):
        pid_file = tmp_path / f"worker{len(shell_options)}"
        left = f"{shell_options} sleep 30 >/dev/null 2>&1 & echo $! > {pid_file}"
        assert wrapper(lock, "-m", "unit test", "--", "bash", "-c", left).wait(timeout=20) == 0
        assert not alive(pid_from(pid_file))
        assert free(lock)


def test_a_command_that_cannot_start_is_recorded(lock):
    assert wrapper(lock, "-m", "unit test", "--", "/nonexistent/cmd").wait(timeout=10) == 127
    assert [r[1] for r in events(lock)] == ["wait", "take", "failed"]
    assert free(lock)


def test_a_log_that_is_not_a_regular_file_never_blocks_the_hold(lock):
    log = lock.parent / "holders.log"
    log.unlink()
    os.mkfifo(log)
    proc = wrapper(lock, "-m", "unit test", "--", "true")
    _, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert err
    assert free(lock)


def test_a_signal_stops_the_command_and_is_recorded(lock):
    proc = wrapper(lock, "-m", "unit test", "--", "sleep", "60")
    wait_for(lock, "take")
    time.sleep(0.3)
    proc.terminate()
    assert proc.wait(timeout=15) == 143
    assert events(lock)[-1][1] == "killed"
    assert free(lock)


def test_fields_cannot_forge_records(lock):
    env = {**os.environ, "AUTAN_GPU_LOCK": str(lock), "AUTAN_AGENT": "a\tb\nc\\nd"}
    why = "job\n2026-01-01T00:00:00Z\ttake\tforged\u0085x\u2028y".encode() + b"\xff"
    cmd = [str(WRAPPER).encode(), b"-m", why, b"--", b"echo", b"x\ny"]
    assert subprocess.run(cmd, env=env, capture_output=True, check=False).returncode == 0
    rows = events(lock)
    assert [r[1] for r in rows] == ["wait", "take", "release"]
    assert all(len(r) == 8 for r in rows)


def test_concurrent_records_stay_whole(lock):
    why = "x" * 3000
    procs = [wrapper(lock, "-m", why, "--", "true") for _ in range(8)]
    assert [p.wait(timeout=30) for p in procs] == [0] * 8
    rows = events(lock)
    assert len(rows) == 24
    assert all(len(r) == 8 for r in rows)


def test_a_signal_to_the_callers_process_group_still_cleans_up(lock, tmp_path):
    pid_file = tmp_path / "worker"
    proc = wrapper(lock, "-m", "unit test", "--", "sh", "-c", f"sleep 60 & echo $! > {pid_file}; wait",
                   start_new_session=True)  # fmt: skip
    worker = pid_from(pid_file)
    os.killpg(proc.pid, signal.SIGINT)
    assert proc.wait(timeout=15) == 143
    assert events(lock)[-1][1] == "killed"
    assert not alive(worker)
    assert free(lock)


def test_an_unwritable_log_is_reported_not_fatal(lock):
    (lock.parent / "holders.log").unlink()
    (lock.parent / "holders.log").mkdir()
    proc = wrapper(lock, "-m", "unit test", "--", "true")
    _, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert err
