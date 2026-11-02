import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).parents[1] / "ci" / "ctest-watchdog.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's /proc")


def fake_tools(tmp_path: Path, launcher_body: str) -> dict[str, str]:
    """An MPI launcher and a debugger at the external boundary."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "mpiexec"
    launcher.write_text(f"#!/bin/sh\n{launcher_body}\n")
    gdb = bin_dir / "fake-gdb"
    # It prints, then ignores TERM and never exits: the watchdog must still finish.
    gdb.write_text(
        '#!/bin/sh\necho "fake backtrace of $3"\ntrap "" TERM\nwhile :; do sleep 1; done\n'
    )
    for tool in (launcher, gdb):
        tool.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "AUTAN_WATCHDOG_GDB": str(gdb), "AUTAN_WATCHDOG_GDB_TIMEOUT": "2",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")}  # fmt: skip


def running(pid: str) -> bool:
    """Alive and not a zombie: a container's PID 1 may never reap the killed rank."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_a_hung_launcher_is_dumped_then_killed_and_the_command_fails(tmp_path):
    rank = tmp_path / "rank.pid"
    env = fake_tools(tmp_path, f"sleep 300 & echo $! > {rank}; wait")
    stacks = tmp_path / "stacks"
    start = time.monotonic()
    proc = subprocess.run([str(WATCHDOG), "2", str(stacks), "--", "sh", "-c", "mpiexec"],
                          check=False, env=env, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert proc.returncode != 0
    assert time.monotonic() - start < 30
    pid = rank.read_text().strip()
    (dump,) = stacks.glob(f"*-{pid}.txt")
    assert f"fake backtrace of {pid}" in dump.read_text()
    assert f"fake backtrace of {pid}" in proc.stdout
    assert "MPI hang caught by the watchdog" in (tmp_path / "summary").read_text()
    assert not running(pid)


def test_a_launcher_that_finishes_in_time_is_left_alone(tmp_path):
    env = fake_tools(tmp_path, "sleep 1")
    stacks = tmp_path / "stacks"
    proc = subprocess.run([str(WATCHDOG), "5", str(stacks), "--", "sh", "-c", "mpiexec; exit 3"],
                          check=False, env=env, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert proc.returncode == 3
    assert list(stacks.iterdir()) == []
    assert not (tmp_path / "summary").exists()


# The launcher blocks without any process below it: a read on a FIFO it holds open itself.
HANG_ALONE = 'mkfifo "$0.fifo" 2>/dev/null; exec 3<>"$0.fifo"; read -r _ <&3'
NET = {"AUTAN_WATCHDOG_NET_S": "2"}


def watchdog(env, limit, stacks, command):
    return subprocess.run([str(WATCHDOG), str(limit), str(stacks), "--", "sh", "-c", command],
                          check=False, env=env, capture_output=True, text=True, timeout=90)  # fmt: skip


def test_a_launcher_that_outlives_every_rank_is_killed_without_failing_the_test(tmp_path):
    # #304: the ranks have ended (one ran 3 s, and exited), the launcher stays: past the net's
    # delay it is killed with a warning naming the test, and the test's own status stands.
    env = {**fake_tools(tmp_path, f"sleep 3; {HANG_ALONE}"), **NET}
    stacks = tmp_path / "stacks"
    start = time.monotonic()
    proc = watchdog(env, 60, stacks, "mpiexec; exit 0")
    assert proc.returncode == 0
    assert time.monotonic() - start < 30
    assert "outlived every rank" in proc.stdout
    assert "#304" in proc.stdout
    assert "test: sh -c mpiexec; exit 0" in proc.stdout
    assert "killed after every rank ended" in (tmp_path / "summary").read_text()
    (dump,) = stacks.iterdir()  # the launcher's stack, for the upstream report
    assert dump.name.startswith("net-")
    assert "fake backtrace of" in dump.read_text()


def test_a_launcher_whose_ranks_left_their_exit_markers_is_killed_too(tmp_path):
    # Ranks so quick that the watchdog never saw them: their markers (#380) say they ended.
    runs = tmp_path / "case" / "runs"
    (tmp_path / "case" / "exits").mkdir(parents=True)
    for rank in (0, 1):
        (tmp_path / "case" / "exits" / f"exit-{rank}").write_text(f"2 {4000 + rank}\n")
    env = {**fake_tools(tmp_path, HANG_ALONE), **NET}
    proc = watchdog(env, 60, tmp_path / "stacks", f"mpiexec -n 2 prog --runs {runs}; exit 2")
    assert proc.returncode == 2
    assert "rank exit markers: 2/2" in proc.stdout


def test_a_launcher_with_no_rank_seen_and_markers_missing_is_a_hang(tmp_path):
    # One marker of two: a rank may not have ended, so the net does not apply; the hang
    # watchdog fails the test at its limit.
    runs = tmp_path / "case" / "runs"
    (tmp_path / "case" / "exits").mkdir(parents=True)
    (tmp_path / "case" / "exits" / "exit-0").write_text("0 4000\n")
    env = {**fake_tools(tmp_path, HANG_ALONE), **NET}
    proc = watchdog(env, 6, tmp_path / "stacks", f"mpiexec -n 2 prog --runs {runs}; exit 0")
    assert proc.returncode != 0
    assert "outlived every rank" not in proc.stdout
    assert "MPI hang caught by the watchdog" in (tmp_path / "summary").read_text()


def test_a_launcher_is_never_killed_while_a_rank_lives(tmp_path):
    # A rank that works longer than the net's delay: the launcher waits for it and ends normally.
    env = {**fake_tools(tmp_path, "sleep 6"), **NET}
    proc = watchdog(env, 60, tmp_path / "stacks", "mpiexec; exit 0")
    assert proc.returncode == 0
    assert "outlived every rank" not in proc.stdout
    assert not (tmp_path / "summary").exists()
