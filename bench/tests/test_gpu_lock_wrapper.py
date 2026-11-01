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
    return path


def test_waits_for_the_holder_runs_the_command_and_records_the_hold(lock):
    with lock.open() as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = wrapper(lock, "-m", "unit test", "--", "sh", "-c", "exit 3")
        time.sleep(1.5)
        assert proc.poll() is None
    assert proc.wait(timeout=10) == 3
    rows = events(lock)
    assert [r[1] for r in rows] == ["wait", "take", "release"]
    assert all(r[2] == "tester" and r[6] == "unit test" for r in rows)
    assert int(rows[1][4]) >= 1
    assert rows[2][5] == "3"


def test_gives_up_after_the_wait_limit(lock):
    with lock.open() as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = wrapper(lock, "-m", "unit test", "-w", "1", "--", "true")
        assert proc.wait(timeout=10) == 75
    assert [r[1] for r in events(lock)] == ["wait", "timeout"]


def test_stops_the_command_at_the_hold_limit_and_frees_the_lock(lock):
    start = time.monotonic()
    proc = wrapper(lock, "-m", "unit test", "-t", "1", "--", "sleep", "60")
    assert proc.wait(timeout=20) == 124
    assert time.monotonic() - start < 10
    assert events(lock)[-1][1] == "overrun"
    assert free(lock)


def test_refuses_a_hold_beyond_ten_minutes(lock):
    assert wrapper(lock, "-m", "unit test", "-t", "601", "--", "true").wait(timeout=10) == 2
    assert not (lock.parent / "holders.log").exists()


def test_a_process_left_behind_does_not_keep_the_lock(lock):
    # The command starts a background process that outlives it.
    left = "sleep 30 >/dev/null 2>&1 & echo $!"
    proc = wrapper(lock, "-m", "unit test", "--", "sh", "-c", left, stdout=subprocess.PIPE)
    out, _ = proc.communicate(timeout=10)
    pid = int(out)
    try:
        assert proc.returncode == 0
        assert Path(f"/proc/{pid}").exists()
        assert free(lock)
    finally:
        os.kill(pid, 9)


def test_a_signal_stops_the_command_and_is_recorded(lock):
    proc = wrapper(lock, "-m", "unit test", "--", "sleep", "60")
    time.sleep(1)
    proc.terminate()
    assert proc.wait(timeout=10) == 143
    assert events(lock)[-1][1] == "killed"
    assert free(lock)


def test_an_unwritable_log_is_reported_not_fatal(lock):
    (lock.parent / "holders.log").mkdir()
    proc = wrapper(lock, "-m", "unit test", "--", "true")
    _, err = proc.communicate(timeout=10)
    assert proc.returncode == 0
    assert err
