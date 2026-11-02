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
