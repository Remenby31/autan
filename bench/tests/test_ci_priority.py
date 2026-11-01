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
