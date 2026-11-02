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


# Real host memory, touched page by page: what a GB10 device allocation also consumes. The
# allocator and its parent ignore TERM; the allocator runs in a session of its own (as
# autan-bench's gated binary does), and writes its pid, then "done" once it has allocated.
ALLOCATE = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sys.argv[-1] != "child":
    sys.exit(subprocess.Popen([sys.executable, *sys.argv, "child"], start_new_session=True).wait())
steps, out, linger, delay = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
open(out + ".pid", "w").write(str(os.getpid()))
time.sleep(delay)
chunks = []
for _ in range(steps):
    chunks.append(b"\\1" * (128 << 20))
    time.sleep(0.05)
open(out, "w").write("done")
time.sleep(linger)
"""


def allocate(lock, floor, steps, out, linger, *, hold="60", delay="0"):
    script = out.parent / "allocate.py"
    script.write_text(ALLOCATE)
    return wrapper(lock, "-m", "test", "-t", hold, "-f", floor, "--", sys.executable,
                   str(script), str(steps), str(out), str(linger), delay)  # fmt: skip


def test_the_host_memory_floor_kills_a_command_that_crosses_it_and_only_that(lock, tmp_path):
    # Each floor is taken from what is available just before (a shared host's memory moves):
    # 1 GiB below it, 4 GiB of allocation crosses it; 2 GiB below it, 256 MiB does not.
    def below_now(gib: int) -> str:
        return f"{(mem_available() - (gib << 30)) >> 20}M"

    done = tmp_path / "done"
    start = time.monotonic()
    greedy = allocate(lock, below_now(1), 32, done, 60)
    assert greedy.wait(timeout=60) == 71
    # At once: no TERM grace, which the allocator would sit out.
    assert time.monotonic() - start < 8
    assert not done.exists()
    assert not alive(pid_from(Path(f"{done}.pid")))
    assert [r[1] for r in events(lock)][-1] == "memory"
    assert free(lock)

    # The floor still holds during the grace of another stop (here the hold limit's).
    # (hold 20 s: TERM at 10 s, KILL at 20 s; the allocation crosses the floor at about 11.5 s)
    start = time.monotonic()
    late = allocate(lock, below_now(1), 32, done, 60, hold="20", delay="11")
    assert late.wait(timeout=60) == 71
    assert time.monotonic() - start < 17
    assert [r[1] for r in events(lock)][-1] == "memory"

    modest = allocate(lock, below_now(2), 2, done, 0)
    assert modest.wait(timeout=60) == 0
    assert done.exists()
    assert [r[1] for r in events(lock)][-1] == "release"

    # Below the floor already: the command never starts.
    done.unlink()
    above = f"{(mem_available() >> 20) + 4096}M"
    refused = wrapper(lock, "-m", "test", "-f", above, "-r", "1", "--", "touch", str(done))
    assert refused.wait(timeout=30) == 71
    assert not done.exists()
    assert [r[1] for r in events(lock)][-2:] == ["take", "memory"]


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
def test_a_docker_run_command_has_its_container_killed_with_it(lock, tmp_path):
    # A container lives outside the command's session: killing its docker client alone would
    # leave it running, past the memory floor and past the hold limit.
    subprocess.run(["docker", "pull", "-q", PYTHON_IMAGE], capture_output=True, check=True)
    label = f"autan-gpu-lock-test={os.getpid()}-{time.monotonic_ns()}"
    try:
        floor = f"{(mem_available() - (1 << 30)) >> 20}M"
        greedy = wrapper(lock, "-m", "test", "-t", "60", "-f", floor, "--", "docker", "run",
                         "--rm", "--label", label, PYTHON_IMAGE, "python", "-c", ALLOCATE, "32",
                         "/tmp/done", "60", "0", "child")  # fmt: skip
        assert greedy.wait(timeout=90) == 71
        assert [r[1] for r in events(lock)][-1] == "memory"
        # PID 1 of the container ignores TERM: only `docker kill` stops it at the limit.
        stuck = wrapper(lock, "-m", "test", "-t", "4", "--", "docker", "run", "--rm", "--label",
                        label, PYTHON_IMAGE, "sleep", "120")  # fmt: skip
        assert stuck.wait(timeout=60) == 124
        # The same with the caller's own --cidfile, which takes precedence over one injected.
        own = wrapper(lock, "-m", "test", "-t", "4", "--", "docker", "run", "--rm",
                      "--cidfile", str(tmp_path / "cid"), "--label", label, PYTHON_IMAGE,
                      "sleep", "120")  # fmt: skip
        assert own.wait(timeout=60) == 124
        for _ in range(100):
            if not containers(label):
                break
            time.sleep(0.2)
        assert containers(label) == []
    finally:
        for cid in containers(label):
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)


def test_waiters_on_memory_being_returned_start_one_at_a_time_once_it_is_back(lock, tmp_path):
    # A process holds 3 GiB of real memory below the floor, as one the floor killed does while
    # the kernel reclaims it, and frees it only when the test says so, 4 s after two commands
    # queued (a wait shorter than that would refuse them). Each command then re-checks the
    # floor once it holds the lock: they run one after the other, never together.
    ready, release, log = tmp_path / "ready", tmp_path / "release", tmp_path / "log"
    hog = subprocess.Popen([sys.executable, "-c", (
        "import os, sys, time\nchunks = [b'\\1' * (128 << 20) for _ in range(24)]\n"
        f"open({str(ready)!r}, 'w').write('x')\n"
        f"while not os.path.exists({str(release)!r}):\n    time.sleep(0.05)\n")])  # fmt: skip
    try:
        for _ in range(400):
            if ready.exists():
                break
            time.sleep(0.05)
        assert hog.poll() is None
        # 1 GiB above what is left while it holds its 3 GiB: below the floor until it exits.
        floor = f"{(mem_available() + (1 << 30)) >> 20}M"
        step = f"echo start $(date +%s.%N) >> {log}; sleep 1; echo end $(date +%s.%N) >> {log}"
        waiters = [wrapper(lock, "-m", "test", "-f", floor, "--", "sh", "-c", step)
                   for _ in range(2)]  # fmt: skip
        time.sleep(4)
        assert not log.exists()
        assert all(w.poll() is None for w in waiters)
        release.touch()
        assert [w.wait(timeout=60) for w in waiters] == [0, 0]
    finally:
        release.touch()
        hog.wait(timeout=30)
    times = [line.split() for line in log.read_text().splitlines()]
    assert [kind for kind, _ in times] == ["start", "end", "start", "end"]
    assert float(times[2][1]) >= float(times[1][1])


def test_the_memory_wait_counts_in_the_hold_and_not_in_a_self_locking_command_s(lock, tmp_path):
    # Below the floor for good: the wait ends with the hold (-t 4: TERM time at 2 s), not after
    # -r 30, and refuses.
    above = f"{(mem_available() >> 20) + 4096}M"
    start = time.monotonic()
    held = wrapper(lock, "-m", "test", "-t", "4", "-r", "30", "-f", above, "--", "true")
    assert held.wait(timeout=30) == 71
    assert time.monotonic() - start < 6
    # With -n the command takes the lock itself and its -t counts from its start: 3 s of
    # waiting for memory do not eat into the 3 s it is given.
    ready, release = tmp_path / "ready", tmp_path / "release"
    hog = subprocess.Popen([sys.executable, "-c", (
        "import os, time\nchunks = [b'\\1' * (128 << 20) for _ in range(24)]\n"
        f"open({str(ready)!r}, 'w').write('x')\n"
        f"while not os.path.exists({str(release)!r}):\n    time.sleep(0.05)\n")])  # fmt: skip
    try:
        for _ in range(400):
            if ready.exists():
                break
            time.sleep(0.05)
        floor = f"{(mem_available() + (1 << 30)) >> 20}M"
        own = wrapper(lock, "-n", "-m", "test", "-t", "3", "-r", "30", "-f", floor, "--",
                      "flock", str(lock), "sleep", "1")  # fmt: skip
        time.sleep(3)
        release.touch()
        assert own.wait(timeout=30) == 0
    finally:
        release.touch()
        hog.wait(timeout=30)
