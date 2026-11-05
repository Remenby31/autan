import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "autan-mem-guard"
GIB = 1 << 30


def load():
    loader = importlib.machinery.SourceFileLoader("autan_mem_guard", str(SCRIPT))
    spec = importlib.util.spec_from_loader("autan_mem_guard", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["autan_mem_guard"] = module
    loader.exec_module(module)
    return module


guard_module = load()

# docker at the boundary: the containers and whether they run in a JSON file, every stop and
# start recorded; a container's id is its name.
FAKE_DOCKER = """#!/usr/bin/env python3
import json, sys, pathlib
state = pathlib.Path(sys.argv[0]).parent / "containers.json"
calls = pathlib.Path(sys.argv[0]).parent / "calls"
c = json.loads(state.read_text())
a = sys.argv[1:]
if a[0] == "ps":
    print("\\n".join(n for n, up in c.items() if up))
elif a[0] == "inspect":
    print(a[-1] if a[-1] in c else "")
elif a[0] in ("stop", "start"):
    with calls.open("a") as h:
        h.write(f"{a[0]} {a[-1]}\\n")
    c[a[-1]] = a[0] == "start"
    state.write_text(json.dumps(c))
"""

CONTAINERS = {
    "autan-noyau-mut1": True, "autan-noyau-mut2": True, "autan-noyau-release": True,
    "autan-sdk-build": True, "autan-ns-nssa": True, "autan-m2-rae2822": True,
    "autan-m2-h1-2048": True, "autan-runner": True, "autan-runner-proxy": True,
    "robostral-api": True, "excalibur-mcp": True,
}  # fmt: skip
LEVEL = {"autan-noyau-mut1": 1, "autan-noyau-mut2": 1, "autan-noyau-release": 2,
         "autan-sdk-build": 2, "autan-ns-nssa": 3, "autan-m2-rae2822": 4, "autan-runner": 5}  # fmt: skip


@pytest.fixture
def host(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "containers.json").write_text(json.dumps(CONTAINERS))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    cgroups = tmp_path / "cgroup"
    for k, name in enumerate(CONTAINERS):
        scope = cgroups / "system.slice" / f"docker-{name}.scope"
        scope.mkdir(parents=True)
        (scope / "memory.peak").write_text(f"{(k + 1) * GIB}\n")
    state = tmp_path / "guard"
    state.mkdir()

    class Host:
        meminfo = tmp_path / "meminfo"
        log = state / "guard.log"

        def free(self, gib_):
            self.meminfo.write_text(
                f"MemTotal: {128 << 20} kB\nMemAvailable: {int(gib_ * (1 << 20))} kB\n"
            )

        def calls(self):
            path = bin_dir / "calls"
            return path.read_text().splitlines() if path.exists() else []

        def guard(self, dry_run=False):
            templates = tuple(
                cgroups / t.relative_to("/sys/fs/cgroup") for t in guard_module.CGROUPS
            )
            return guard_module.Guard(
                state, dry_run=dry_run, meminfo=self.meminfo, cgroups=templates
            )

    return Host()


def test_levels_and_what_is_never_touched():
    for name, level in LEVEL.items():
        assert guard_module.level_of(name) == level - 1
    for name in (
        "autan-m2-h1-2048",
        "autan-m2-h1",
        "autan-runner-proxy",
        "robostral-api",
        "excalibur-mcp",
    ):
        assert guard_module.level_of(name) is None


def test_below_the_floor_one_level_at_a_time_then_back_in_reverse(host):
    guard = host.guard()
    host.free(25)
    for t in range(0, 600, 30):
        guard.tick(float(t))
    # One level per 60 s, in order, each level whole; H1, the proxy and foreign containers never.
    stops = [c.split()[1] for c in host.calls()]
    assert [LEVEL[n] for n in stops] == sorted(LEVEL[n] for n in stops)
    assert set(stops) == set(LEVEL)
    times = {}
    for line in host.log.read_text().splitlines():
        if " STOP " in line:
            times.setdefault(line.split()[3], line)
    assert len(times) == 5
    # The peak read before the stop is logged: autan-ns-nssa's is 5 GiB (the fixture's order).
    assert "5.0 GiB" in next(
        line for line in host.log.read_text().splitlines() if "autan-ns-nssa" in line
    )
    # Between the two thresholds: nothing. Above 40 GiB for 120 s: the last level comes back,
    # then the next 120 s later, the same containers.
    host.free(35)
    before = len(host.calls())
    for t in range(600, 900, 30):
        guard.tick(float(t))
    assert len(host.calls()) == before
    host.free(45)
    started = []
    for t in range(900, 900 + 5 * 120 + 60, 30):
        guard.tick(float(t))
        started = [c.split()[1] for c in host.calls() if c.startswith("start")]
    assert [LEVEL[n] for n in started] == sorted((LEVEL[n] for n in started), reverse=True)
    assert set(started) == set(LEVEL)


def test_a_dip_restarts_the_hysteresis(host):
    guard = host.guard()
    host.free(25)
    guard.tick(0.0)
    host.free(45)
    guard.tick(60.0)
    guard.tick(150.0)
    host.free(38)
    guard.tick(170.0)
    host.free(45)
    guard.tick(200.0)
    guard.tick(300.0)
    assert not any(c.startswith("start") for c in host.calls())
    guard.tick(320.0)
    assert any(c.startswith("start") for c in host.calls())


def test_a_restarted_guard_still_starts_what_an_earlier_one_stopped(host):
    host.free(25)
    host.guard().tick(0.0)
    host.free(45)
    again = host.guard()
    again.tick(1000.0)
    again.tick(1120.0)
    assert sorted(c for c in host.calls() if c.startswith("start")) == [
        "start autan-noyau-mut1", "start autan-noyau-mut2"]  # fmt: skip


def test_dry_run_decides_and_logs_but_never_acts(host):
    guard = host.guard(dry_run=True)
    host.free(25)
    for t in range(0, 300, 30):
        guard.tick(float(t))
    assert host.calls() == []
    log = host.log.read_text()
    assert "DRY-RUN" in log
    assert "autan-noyau-mut1" in log


def test_unknown_memory_counts_as_below_the_floor(host):
    host.meminfo.write_text("garbage\n")
    host.guard().tick(0.0)
    assert "stop autan-noyau-mut1" in host.calls()
