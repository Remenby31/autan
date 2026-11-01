import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "lock" / "autan-gpu-lock"
PRIORITY = ROOT / "ci" / "priority.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="flock(1) and /proc are Linux's")


@pytest.fixture
def spark(tmp_path: Path) -> dict[str, str]:
    """A lock directory as deploy.sh prepares it, and the environment of an agent outside CI."""
    for name in ("gpu.lock", "ci.lock", "holders.log"):
        (tmp_path / name).touch()
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    env |= {"AUTAN_GPU_LOCK": str(tmp_path / "gpu.lock"), "RUNNER_TEMP": str(tmp_path)}
    return env


def agent(env: dict[str, str], line: str, order: Path, *args: str) -> subprocess.Popen:
    cmd = [str(WRAPPER), "-m", "agent", *args, "--", "sh", "-c", f"echo {line} >> {order}"]
    return subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE)


def ci_step(env: dict[str, str], order: Path) -> subprocess.Popen:
    """A CI GPU step: it takes the GPU lock with flock and never yields."""
    lock = env["AUTAN_GPU_LOCK"]
    return subprocess.Popen(["flock", lock, "sh", "-c", f"echo ci >> {order}; sleep 0.3"])


def priority(env: dict[str, str], *args: str) -> None:
    subprocess.run([str(PRIORITY), *args], env=env, check=True, timeout=30)


def events(env: dict[str, str]) -> list[str]:
    log = Path(env["AUTAN_GPU_LOCK"]).parent / "holders.log"
    return [line.split("\t")[1] for line in log.read_text().splitlines()]


def lines(order: Path) -> list[str]:
    return order.read_text().split() if order.exists() else []


def test_an_agent_waiting_with_ci_goes_after_ci(spark, tmp_path):
    order = tmp_path / "order"
    # Someone holds the GPU; CI announces itself and queues; an agent queues after it.
    hold = ["--", "sh", "-c", f"echo first >> {order}; sleep 1"]
    first = subprocess.Popen([str(WRAPPER), "-m", "first", *hold], env=spark)
    time.sleep(0.3)
    priority(spark, "start", "5")
    ci = ci_step(spark, order)
    second = agent(spark, "second", order)
    assert ci.wait(timeout=20) == 0
    time.sleep(0.5)
    assert lines(order) == ["first", "ci"]
    assert second.poll() is None
    priority(spark, "stop")
    assert second.wait(timeout=20) == 0
    assert first.wait(timeout=20) == 0
    assert lines(order) == ["first", "ci", "second"]
    assert "yield" in events(spark)


def test_an_agent_never_waits_for_ci_beyond_its_limit(spark, tmp_path):
    priority(spark, "start", "5")
    try:
        start = time.monotonic()
        proc = agent(spark, "late", tmp_path / "order", "-w", "1")
        assert proc.wait(timeout=20) == 75
        assert time.monotonic() - start < 5
        assert events(spark)[-2:] == ["yield", "timeout"]
    finally:
        priority(spark, "stop")


def test_ci_itself_never_yields(spark, tmp_path):
    priority(spark, "start", "5")
    try:
        order = tmp_path / "order"
        proc = agent({**spark, "GITHUB_ACTIONS": "true"}, "ci", order, "-w", "5")
        assert proc.wait(timeout=20) == 0
        assert lines(order) == ["ci"]
    finally:
        priority(spark, "stop")


def test_a_self_locking_command_waits_for_ci_then_takes_the_lock_itself(spark, tmp_path):
    # -n: for autan-bench, which takes the lock itself; nesting it under a held lock would
    # deadlock. The wrapper only waits for CI and for a free lock.
    order = tmp_path / "order"
    lock = spark["AUTAN_GPU_LOCK"]
    priority(spark, "start", "5")
    take = f"flock -n {lock} sh -c 'echo agent >> {order}'"
    proc = subprocess.Popen([str(WRAPPER), "-n", "-m", "agent", "--", "sh", "-c", take], env=spark)
    time.sleep(1)
    assert proc.poll() is None
    ci = ci_step(spark, order)
    assert ci.wait(timeout=20) == 0
    priority(spark, "stop")
    assert proc.wait(timeout=20) == 0
    assert lines(order) == ["ci", "agent"]
    assert "start" in events(spark)


GUARD = ROOT / "lock" / "autan-gpu-guard"


def test_the_guard_records_every_holder_and_never_announces_ci_for_others(spark):
    lock = spark["AUTAN_GPU_LOCK"]
    guard = subprocess.Popen([sys.executable, str(GUARD)], env=spark, stderr=subprocess.PIPE)
    try:
        time.sleep(1)
        subprocess.run(["flock", lock, "sleep", "1.5"], check=True)
        time.sleep(1.5)
        held = events(spark)
        assert held[:2] == ["held", "released"]
        # Not the runner container: ci.lock stays free.
        assert (
            subprocess.run(
                ["flock", "-n", "-x", str(Path(lock).parent / "ci.lock"), "true"], check=False
            ).returncode
            == 0
        )
    finally:
        guard.terminate()
        guard.wait(timeout=10)


def test_the_guard_reads_holders_and_waiters_from_proc_locks():
    loader = importlib.machinery.SourceFileLoader("guard", str(GUARD))
    spec = importlib.util.spec_from_loader("guard", loader)
    guard = importlib.util.module_from_spec(spec)
    loader.exec_module(guard)
    key = "103:02:6985512"
    text = (
        f"22: FLOCK  ADVISORY  WRITE 4242 {key} 0 EOF\n"
        f"22: -> FLOCK  ADVISORY  WRITE 4343 {key} 0 EOF\n"
        "23: POSIX  ADVISORY  WRITE 99 103:02:1 0 EOF\n"
    )
    assert guard.lock_users(key, text) == ({4242}, {4343})
