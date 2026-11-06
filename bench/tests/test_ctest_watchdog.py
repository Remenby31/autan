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


def test_a_hang_s_dump_tells_a_spinning_rank_from_a_blocked_one(tmp_path):
    """Without a debugger (the runners have none, and no ptrace): /proc says whether each rank
    burns CPU, as ranks busy-polling on an oversubscribed host do, or sleeps, as in a deadlock."""
    spin, block = tmp_path / "spin.pid", tmp_path / "block.pid"
    body = (f"sh -c 'echo $$ > {spin}; while :; do :; done' & "
            f"sh -c 'echo $$ > {block}; exec sleep 300' & wait")  # fmt: skip
    env = {**fake_tools(tmp_path, body), "AUTAN_WATCHDOG_GDB": "/nonexistent/gdb"}
    stacks = tmp_path / "stacks"
    proc = subprocess.run([str(WATCHDOG), "2", str(stacks), "--", "sh", "-c", "mpiexec"],
                          check=False, env=env, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert proc.returncode != 0
    (report,) = stacks.glob("*-activity.txt")
    lines = {line.split()[1]: line for line in report.read_text().splitlines()}
    spinning, blocked = lines[spin.read_text().strip()], lines[block.read_text().strip()]
    # However loaded the host, a spinning shell is runnable and accrues CPU; a sleeping one never.
    assert "state R" in spinning
    assert int(spinning.split("CPU ")[1].split()[0]) > 0
    assert "state S" in blocked
    assert "CPU 0 %" in blocked
    assert report.read_text() in proc.stdout  # in the job's log as well
    assert "no debugger" in proc.stdout


def test_the_watchdog_ends_as_soon_as_its_command_does(tmp_path):
    # Not at the end of a poll: under a lock with a hard limit, a command that passed just before
    # it must not be reported as stopped because the watchdog was asleep.
    env = fake_tools(tmp_path, "true")
    start = time.monotonic()
    proc = subprocess.run([str(WATCHDOG), "120", str(tmp_path / "stacks"), "--", "sh", "-c",
                           "sleep 0.3; exit 7"], check=False, env=env, timeout=60)  # fmt: skip
    assert proc.returncode == 7
    assert time.monotonic() - start < 1.5


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


def rank(r: int, status: int = 0, then: str = "") -> str:
    """A rank as autan-run-relaxation is (#380): its marker, "<status> <pid>", once it ended."""
    marker = f'echo "{status} $$" > "$RANK_DIR/exit-{r}"'
    return f"sh -c '{marker}{then}' &"


def launcher(*ranks: str) -> str:
    """mpiexec that starts its ranks, waits for them, then hangs as PRRTE 3.0.x does (#304)."""
    return " ".join(ranks) + f" wait; {HANG_ALONE}"


def watchdog(env, limit, stacks, command):
    return subprocess.run([str(WATCHDOG), str(limit), str(stacks), "--", "sh", "-c", command],
                          check=False, env=env, capture_output=True, text=True, timeout=90)  # fmt: skip


def exits(tmp_path: Path) -> Path:
    d = tmp_path / "exits"
    d.mkdir()
    return d


def test_a_launcher_that_outlives_every_rank_is_killed_without_failing_the_test(tmp_path):
    # Both ranks ended with their markers; the launcher stays: it is killed, its stack is in
    # the log, and the test's own status stands.
    d = exits(tmp_path)
    env = {**fake_tools(tmp_path, launcher(rank(0), rank(1))), **NET}
    stacks = tmp_path / "stacks"
    start = time.monotonic()
    proc = watchdog(
        env, 60, stacks, f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog; exit 0"
    )
    assert proc.returncode == 0, proc.stdout
    assert time.monotonic() - start < 30
    assert "outlived every rank" in proc.stdout
    assert "#304" in proc.stdout
    assert "ranks proven ended by their exit markers: 2/2" in proc.stdout
    assert "test: sh -c AUTAN_TEST_EXIT_DIR=" in proc.stdout
    assert "fake backtrace of" in proc.stdout  # the stack in the log, not only on disk
    assert "killed after every rank ended" in (tmp_path / "summary").read_text()
    (dump,) = stacks.iterdir()
    assert dump.name.startswith("net-")


def test_markers_are_found_next_to_the_runs_and_with_np_equals(tmp_path):
    # Without AUTAN_TEST_EXIT_DIR, <parent of --runs>/exits; the count as --np=2.
    d = exits(tmp_path)
    env = {**fake_tools(tmp_path, launcher(rank(0), rank(1))), **NET}
    command = f"RANK_DIR={d} mpiexec --np=2 prog --runs={tmp_path / 'runs'}; exit 2"
    proc = watchdog(env, 60, tmp_path / "stacks", command)
    assert proc.returncode == 2, proc.stdout
    assert "ranks proven ended by their exit markers: 2/2" in proc.stdout


@pytest.mark.parametrize(
    ("ranks", "why"),
    [
        ((rank(0),), "one rank of two never wrote its marker"),
        ((rank(0), rank(1, then="; exec sleep 30")), "a rank wrote its marker but still runs"),
    ],
)
def test_without_proof_that_every_rank_ended_it_is_a_hang(tmp_path, ranks, why):
    d = exits(tmp_path)
    env = {**fake_tools(tmp_path, launcher(*ranks)), **NET}
    proc = watchdog(
        env, 8, tmp_path / "stacks", f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog"
    )
    assert proc.returncode != 0, why
    assert "outlived every rank" not in proc.stdout, why
    assert "MPI hang caught by the watchdog" in (tmp_path / "summary").read_text()


def test_stale_or_foreign_markers_prove_nothing(tmp_path):
    # Markers older than the launcher (an earlier run), and files that are not rank markers.
    d = exits(tmp_path)
    gone = subprocess.run(["sh", "-c", "echo $$"], capture_output=True, text=True, check=True)
    for name in ("exit-0", "exit-1"):  # of a process that has ended: only their age is wrong
        (d / name).write_text(f"0 {gone.stdout.strip()}\n")
        os.utime(d / name, (time.time() - 3600, time.time() - 3600))
    (d / "exit-17").write_text("")
    (d / "exit-garbage").write_text("")
    env = {**fake_tools(tmp_path, HANG_ALONE), **NET}
    proc = watchdog(
        env, 8, tmp_path / "stacks", f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog"
    )
    assert proc.returncode != 0
    assert "outlived every rank" not in proc.stdout


def test_ranks_ending_near_the_limit_are_not_a_hang(tmp_path):
    # The ranks end at 4 s, the net's delay ends past the 6 s limit: proof wins over the age.
    d = exits(tmp_path)
    body = launcher(rank(0, then="; sleep 4"), rank(1, then="; sleep 4"))
    env = {**fake_tools(tmp_path, body), "AUTAN_WATCHDOG_NET_S": "4"}
    proc = watchdog(
        env,
        6,
        tmp_path / "stacks",
        f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog; exit 0",
    )
    assert proc.returncode == 0, proc.stdout
    assert "outlived every rank" in proc.stdout
    assert "MPI hang caught" not in (tmp_path / "summary").read_text()


def test_a_launcher_is_never_killed_while_a_rank_lives(tmp_path):
    # A rank that works longer than the net's delay: the launcher waits for it and ends normally.
    d = exits(tmp_path)
    env = {**fake_tools(tmp_path, f"{rank(0, then='; sleep 6')} wait"), **NET}
    proc = watchdog(
        env, 60, tmp_path / "stacks", f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 1 prog"
    )
    assert proc.returncode == 0
    assert "outlived every rank" not in proc.stdout
    assert not (tmp_path / "summary").exists()


def test_a_rank_that_left_the_launcher_s_tree_must_have_ended(tmp_path):
    # Rank 1 reparented away from the launcher (its parent exited): its marker names a process
    # that still runs, so the ranks have not all ended, whatever the launcher's tree says.
    d = exits(tmp_path)
    away = "sh -c 'sh -c \"echo 0 \\$\\$ > $RANK_DIR/exit-1; exec sleep 30\" &' "
    env = {**fake_tools(tmp_path, f"{rank(0)} {away}; wait; {HANG_ALONE}"), **NET}
    proc = watchdog(
        env, 8, tmp_path / "stacks", f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog"
    )
    assert proc.returncode != 0
    assert "outlived every rank" not in proc.stdout


def test_a_process_still_living_below_the_launcher_holds_the_net_back(tmp_path):
    # Every rank ended with its marker, but a helper of the launcher runs 8 s more: the launcher
    # is killed (its stack dumped just before) only once nothing below it lives.
    d = exits(tmp_path)
    done = tmp_path / "helper-done"
    helper = f"sh -c 'sleep 8; touch {done}' >/dev/null 2>&1 &"
    env = {**fake_tools(tmp_path, f"{rank(0)} {rank(1)} {helper} {HANG_ALONE}"), **NET}
    stacks = tmp_path / "stacks"
    command = f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog; exit 0"
    proc = watchdog(env, 60, stacks, command)
    assert proc.returncode == 0, proc.stdout
    (dump,) = stacks.iterdir()
    assert done.exists()
    assert dump.stat().st_mtime >= done.stat().st_mtime


def test_a_launcher_that_handles_term_still_ends_by_a_signal(tmp_path):
    # The test judges the markers only when its launcher was killed by a signal: one that would
    # exit cleanly on TERM is killed outright.
    d = exits(tmp_path)
    body = "trap 'exit 1' TERM; " + launcher(rank(0), rank(1))
    env = {**fake_tools(tmp_path, body), **NET}
    command = f"AUTAN_TEST_EXIT_DIR={d} RANK_DIR={d} mpiexec -n 2 prog; echo launcher=$?"
    proc = watchdog(env, 60, tmp_path / "stacks", command)
    assert "outlived every rank" in proc.stdout
    assert "launcher=137" in proc.stdout


@pytest.mark.parametrize("count", ["-c=2", "-np=2", "-n=2"])
def test_markers_relative_to_the_launcher_s_directory_are_found(tmp_path, count):
    # The launcher runs in case/ with AUTAN_TEST_EXIT_DIR=exits: case/exits, not the watchdog's.
    case = tmp_path / "case"
    (case / "exits").mkdir(parents=True)
    env = {**fake_tools(tmp_path, launcher(rank(0), rank(1))), **NET}
    command = (
        f"cd {case} && AUTAN_TEST_EXIT_DIR=exits RANK_DIR={case}/exits mpiexec {count} prog; exit 0"
    )
    proc = watchdog(env, 60, tmp_path / "stacks", command)
    assert proc.returncode == 0, proc.stdout
    assert "ranks proven ended by their exit markers: 2/2" in proc.stdout


@pytest.mark.parametrize("given", [True, False])
def test_only_the_watched_command_gets_the_ptracer_library(tmp_path, given):
    # The self-hosted runner names it; the command under the watchdog, and its children, preload
    # it. Without it (a hosted runner), nothing is preloaded.
    library = tmp_path / "libautan_ptracer.so"
    library.write_bytes(b"")  # the loader ignores an unloadable preload, with a warning
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    if given:
        env["AUTAN_WATCHDOG_PTRACER"] = str(library)
    seen = tmp_path / "seen"
    proc = subprocess.run(
        [str(WATCHDOG), "30", str(tmp_path / "stacks"), "--", "sh", "-c",
         f'sh -c \'printf "%s" "${{LD_PRELOAD:-}}"\' > {seen}'],
        check=False, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert proc.returncode == 0
    assert seen.read_text() == (str(library) if given else "")
