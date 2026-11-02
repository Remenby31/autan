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
    # The default floor (24 GiB) is a Spark's; a test host may have less memory in all.
    floor = [] if "-f" in args else ["-f", "0G"]
    return subprocess.Popen([str(WRAPPER), *floor, *args], env=env, stderr=subprocess.PIPE,
                            **kwargs)  # fmt: skip


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
    # Also one a job-control shell put in another process group, and one in a session of its
    # own whose parent exits at once (orphaned before any scan could see its ancestry).
    for k, left in enumerate((
        "sleep 30 >/dev/null 2>&1 & echo $! > {pid}",
        "set -m; sleep 30 >/dev/null 2>&1 & echo $! > {pid}",
        "setsid -f sh -c 'echo $$ > {pid}; exec sleep 30' >/dev/null 2>&1",
    )):  # fmt: skip
        pid_file = tmp_path / f"worker{k}"
        script = left.format(pid=pid_file)
        assert wrapper(lock, "-m", "unit test", "--", "bash", "-c", script).wait(timeout=20) == 0
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
    cmd = [str(WRAPPER).encode(), b"-f", b"0G", b"-m", why, b"--", b"echo", b"x\ny"]
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


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    pytest.fail("no MemAvailable")


# A command that ignores TERM and leaves a worker in a session of its own (as autan-bench's
# gated binary does): it writes the worker's pid to $1, then "started" lines to $2.
SLEEPER = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[-1] != "worker":
    sys.exit(subprocess.Popen([sys.executable, *sys.argv, "worker"], start_new_session=True).wait())
open(sys.argv[1], "w").write(str(os.getpid()))
with open(sys.argv[2], "a") as log:
    log.write("started\\n")
time.sleep(120)
"""


def sleeper(tmp_path: Path) -> list[str]:
    script = tmp_path / "sleeper.py"
    script.write_text(SLEEPER)
    return [sys.executable, str(script), str(tmp_path / "worker.pid"), str(tmp_path / "log")]


def test_the_floor_kills_a_command_as_soon_as_memory_falls_below_it(lock, tmp_path, meminfo):
    proc = wrapper(lock, "-m", "test", "-t", "60", "-f", "10G", "--", *sleeper(tmp_path))
    worker = pid_from(tmp_path / "worker.pid")
    time.sleep(1)
    assert proc.poll() is None
    start = time.monotonic()
    meminfo(5)
    assert proc.wait(timeout=30) == 71
    # At once, without the TERM grace the command would sit out.
    assert time.monotonic() - start < 2
    assert not alive(worker)
    assert [r[1] for r in events(lock)][-1] == "memory"
    assert free(lock)
    # Above the floor, a command runs to its end.
    meminfo(20)
    assert wrapper(lock, "-m", "test", "-f", "10G", "--", "true").wait(timeout=30) == 0
    assert [r[1] for r in events(lock)][-1] == "release"


def test_the_floor_holds_during_the_grace_of_another_stop(lock, tmp_path, meminfo):
    # Hold 6 s: TERM at 3 s (ignored), KILL at 6 s. Memory falls at 4 s, inside the grace.
    start = time.monotonic()
    proc = wrapper(lock, "-m", "test", "-t", "6", "-f", "10G", "--", *sleeper(tmp_path))
    pid_from(tmp_path / "worker.pid")
    time.sleep(max(0.0, 4 - (time.monotonic() - start)))
    meminfo(5)
    assert proc.wait(timeout=30) == 71
    assert time.monotonic() - start < 5.5
    assert [r[1] for r in events(lock)][-1] == "memory"


def test_waiters_start_one_at_a_time_once_memory_is_back(lock, tmp_path, meminfo):
    # Below the floor, as while the kernel reclaims a holder the floor killed: two commands
    # queue; memory comes back 3 s later; each re-checks the floor once it holds the lock and
    # they run one after the other, never together.
    meminfo(5)
    log = tmp_path / "log"
    step = f"echo start >> {log}; sleep 1; echo end >> {log}"
    waiters = [wrapper(lock, "-m", "test", "-f", "10G", "--", "sh", "-c", step)
               for _ in range(2)]  # fmt: skip
    time.sleep(3)
    assert not log.exists()
    assert all(w.poll() is None for w in waiters)
    meminfo(20)
    assert [w.wait(timeout=60) for w in waiters] == [0, 0]
    assert log.read_text().split() == ["start", "end", "start", "end"]


def test_the_memory_wait_is_bounded_by_the_hold_and_by_r(lock, meminfo):
    meminfo(5)
    start = time.monotonic()
    # -t 4: TERM time at 2 s, before -r 30 runs out.
    held = wrapper(lock, "-m", "test", "-t", "4", "-r", "30", "-f", "10G", "--", "true")
    assert held.wait(timeout=30) == 71
    assert time.monotonic() - start < 4
    start = time.monotonic()
    refused = wrapper(lock, "-m", "test", "-r", "1", "-f", "10G", "--", "true")
    assert refused.wait(timeout=30) == 71
    assert time.monotonic() - start < 3
    assert [r[1] for r in events(lock)][-2:] == ["take", "memory"]


def test_a_self_locking_command_s_time_counts_from_its_start(lock, meminfo):
    # With -n the command takes the lock itself, so the 3 s of waiting for memory are not
    # taken from the 3 s it is given.
    meminfo(5)
    own = wrapper(lock, "-n", "-m", "test", "-t", "3", "-r", "30", "-f", "10G", "--",
                  "flock", str(lock), "sleep", "1")  # fmt: skip
    time.sleep(3)
    meminfo(20)
    assert own.wait(timeout=30) == 0


# The one test on the host's real memory: a command really allocating past the floor is
# killed. Its margin comes from a measurement: on spark-5512 at load 17-29 (2026-11-02, 5968
# samples over 10 min) MemAvailable moved by at most 3.10 GiB within 5 s (p99 2.98, median
# 0.45). The floor is set 1 GiB below what is available and the command takes 8 GiB, so the
# test fails only if the host frees more than 7 GiB meanwhile, over twice that maximum.
REAL_ALLOCATION_GIB = 8


def test_a_command_really_allocating_past_the_floor_is_killed(lock, tmp_path):
    out = tmp_path / "done"
    floor = f"{(mem_available() - (1 << 30)) >> 20}M"
    grow = (f"chunks = [b'\\1' * (128 << 20) for _ in range({REAL_ALLOCATION_GIB * 8})]\n"
            f"open({str(out)!r}, 'w').write('x')\nimport time\ntime.sleep(60)\n")  # fmt: skip
    proc = wrapper(lock, "-m", "test", "-t", "60", "-f", floor, "--", sys.executable, "-c", grow)
    assert proc.wait(timeout=60) == 71
    assert not out.exists()
    assert [r[1] for r in events(lock)][-1] == "memory"


@pytest.mark.parametrize("floor", ["24", "1.5G", "-1G", "24T", "0x10G"])
def test_refuses_a_floor_that_is_not_a_size(lock, floor):
    proc = wrapper(lock, "-m", "test", "-f", floor, "--", "true")
    assert proc.wait(timeout=10) == 2


# Test-only image, pinned: CPU allocation in a real container (not shipped).
PYTHON_IMAGE = (
    "python:3.12-alpine@sha256:e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8"
)


def docker_works() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=30, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def containers(label: str) -> list[str]:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={label}"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return out.split()


@pytest.mark.skipif(not docker_works(), reason="needs a Docker daemon")
def test_a_docker_run_command_has_its_container_killed_with_it(lock, tmp_path, meminfo):
    # A container lives outside the command's processes: killing its docker client alone would
    # leave it running, past the memory floor and past the hold limit. PID 1 of each container
    # is `sleep`, which ignores TERM.
    subprocess.run(["docker", "pull", "-q", PYTHON_IMAGE], capture_output=True, check=True)
    label = f"autan-gpu-lock-test={os.getpid()}-{time.monotonic_ns()}"
    run = ["docker", "run", "--rm", "--label", label, PYTHON_IMAGE, "sleep", "120"]
    try:
        floored = wrapper(lock, "-m", "test", "-t", "60", "-f", "10G", "--", *run)
        for _ in range(300):
            if containers(label):
                break
            time.sleep(0.1)
        meminfo(5)
        assert floored.wait(timeout=60) == 71
        assert [r[1] for r in events(lock)][-1] == "memory"
        meminfo(100)
        stuck = wrapper(lock, "-m", "test", "-t", "4", "--", *run)
        assert stuck.wait(timeout=60) == 124
        # The same with the caller's own --cidfile, which takes precedence over one injected.
        own = wrapper(lock, "-m", "test", "-t", "4", "--", "docker", "run", "--rm",
                      "--cidfile", str(tmp_path / "cid"), *run[2:])  # fmt: skip
        assert own.wait(timeout=60) == 124
        for _ in range(100):
            if not containers(label):
                break
            time.sleep(0.2)
        assert containers(label) == []
    finally:
        for cid in containers(label):
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
