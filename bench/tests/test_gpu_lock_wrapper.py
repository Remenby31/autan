import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).parents[1] / "lock" / "autan-gpu-lock"
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("flock") is None, reason="flock(1) is util-linux's"
)


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
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().rpartition(")")[2].split()[0] != "Z"


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


def test_the_hold_limit_stops_every_worker_before_releasing(lock):
    # A shell whose worker ignores TERM: only KILL, within the limit, stops it.
    worker = 'trap "" TERM; sleep 60 & echo $!; wait'
    start = time.monotonic()
    proc = wrapper(lock, "-m", "unit test", "-t", "2", "--", "sh", "-c", worker,
                   stdout=subprocess.PIPE)  # fmt: skip
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 124
    assert time.monotonic() - start < 4
    assert events(lock)[-1][1] == "overrun"
    assert not alive(int(out))
    assert free(lock)


@pytest.mark.parametrize("hold", ["0", "601", "01100", "1e3"])
def test_refuses_a_hold_outside_one_to_six_hundred_seconds(lock, hold):
    assert wrapper(lock, "-m", "unit test", "-t", hold, "--", "true").wait(timeout=10) == 2
    assert events(lock) == []


def test_no_process_left_behind_outlives_the_hold(lock):
    left = "sleep 30 >/dev/null 2>&1 & echo $!"
    proc = wrapper(lock, "-m", "unit test", "--", "sh", "-c", left, stdout=subprocess.PIPE)
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert not alive(int(out))
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
    env = {**os.environ, "AUTAN_GPU_LOCK": str(lock), "AUTAN_AGENT": "a\tb\nc"}
    why = "job\n2026-01-01T00:00:00Z\ttake\tforged"
    cmd = [str(WRAPPER), "-m", why, "--", "echo", "x\ny"]
    assert subprocess.run(cmd, env=env, capture_output=True, check=False).returncode == 0
    rows = events(lock)
    assert [r[1] for r in rows] == ["wait", "take", "release"]
    assert all(len(r) == 8 for r in rows)


def test_an_unwritable_log_is_reported_not_fatal(lock):
    (lock.parent / "holders.log").unlink()
    (lock.parent / "holders.log").mkdir()
    proc = wrapper(lock, "-m", "unit test", "--", "true")
    _, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert err
