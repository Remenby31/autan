import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "autan-mem-guard"
GIB = 1 << 30

# docker at the boundary: each container's ID, state and stop time in a JSON file; every stop
# and start recorded with its time. Names in fail-start never start.
FAKE_DOCKER = r"""#!/usr/bin/env python3
import fcntl, json, sys, time, pathlib
here = pathlib.Path(sys.argv[0]).parent
state, calls = here / "containers.json", here / "calls"
held = (here / "lock").open("a")
fcntl.flock(held, fcntl.LOCK_EX)  # docker serialises its own state; concurrent calls here too
c = json.loads(state.read_text())
by_id = {v["id"]: k for k, v in c.items()}
a = sys.argv[1:]
if a[0] == "ps":
    if (here / "deny").exists():
        sys.stderr.write("permission denied while trying to connect to the docker API\n")
        sys.exit(1)
    print("\n".join(f"{v['id']} {k}" for k, v in c.items() if v["status"] == "running"))
elif a[0] == "inspect":
    flaky = here / "fail-inspect"
    if flaky.exists() and int(flaky.read_text()) > 0:
        flaky.write_text(str(int(flaky.read_text()) - 1))
        sys.stderr.write("Error response from daemon: context deadline exceeded\n")
        sys.exit(1)
    name = by_id.get(a[-1])
    if name is None:
        sys.stderr.write(f"Error: No such object: {a[-1]}\n")
        sys.exit(1)
    print(c[name]["status"], c[name]["finished"])
elif a[0] in ("stop", "start"):
    name = by_id[a[-1]]
    with calls.open("a") as h:
        h.write(f"{time.monotonic()} {a[0]} {name}\n")
    if a[0] == "start" and name in json.loads((here / "fail-start.json").read_text()):
        sys.exit(1)
    if a[0] == "stop" and name in json.loads((here / "fail-stop.json").read_text()):
        sys.exit(1)
    after = json.loads((here / "after-start.json").read_text())
    if a[0] == "start" and name in after:
        pathlib.Path(after[name][0]).write_text(after[name][1])
    c[name]["status"] = "running" if a[0] == "start" else "exited"
    if a[0] == "stop":
        c[name]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{time.time_ns() % 10**9:09d}Z"
    state.write_text(json.dumps(c))
"""

NAMES = ["autan-noyau-mut1", "autan-noyau-mut2", "autan-noyau-release", "autan-sdk-build",
         "autan-su2-901598", "autan-mg-vvC", "autan-ns-nssa", "autan-m2-rae2822", "autan-runner", "autan-m2-h1-2048", "autan-m2-h1",
         "autan-runner-proxy", "autan-campaign-c440f", "robostral-api", "excalibur-mcp"]  # fmt: skip
LEVEL = {"autan-noyau-mut1": 1, "autan-noyau-mut2": 1, "autan-noyau-release": 2,
         "autan-sdk-build": 2, "autan-su2-901598": 3, "autan-mg-vvC": 4, "autan-m2-rae2822": 5,
         "autan-ns-nssa": 6, "autan-runner": 7}  # fmt: skip
WAIT = 0.6  # the tests' --level-wait and --hysteresis, in seconds


class Host:
    def __init__(self, tmp_path):
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(
            FAKE_DOCKER.replace("#!/usr/bin/env python3", f"#!{sys.executable}")
        )
        (self.bin / "docker").chmod(0o755)
        self.containers = self.bin / "containers.json"
        self.containers.write_text(json.dumps(
            {n: {"id": f"id{k:02d}" + "0" * 60, "status": "running", "finished": "-"}
             for k, n in enumerate(NAMES)}))  # fmt: skip
        for name in ("fail-start.json", "fail-stop.json"):
            (self.bin / name).write_text("[]")
        (self.bin / "after-start.json").write_text("{}")
        self.cgroup = tmp_path / "cgroup"
        for k in range(len(NAMES)):
            scope = self.cgroup / "system.slice" / f"docker-id{k:02d}{'0' * 60}.scope"
            scope.mkdir(parents=True)
            (scope / "memory.peak").write_text(f"{(k + 1) * GIB}\n")
        self.state = tmp_path / "guard"
        self.meminfo = tmp_path / "meminfo"
        self.free(100)

    def free(self, gib_):
        tmp = self.meminfo.with_suffix(".tmp")
        tmp.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {int(gib_ * (1 << 20))} kB\n")
        tmp.replace(self.meminfo)

    def calls(self, verb=None):
        path = self.bin / "calls"
        rows = [line.split() for line in path.read_text().splitlines()] if path.exists() else []
        return [(float(t), v, n) for t, v, n in rows if verb in (None, v)]

    def guard(self, *extra, ticks=0, path=None):
        env = {
            **os.environ,
            "PATH": path if path is not None else f"{self.bin}:{os.environ['PATH']}",
        }
        args = [sys.executable, str(SCRIPT), "--state-dir", str(self.state), "--meminfo", str(self.meminfo),
                "--cgroup-root", str(self.cgroup), "--interval", "0.05", "--level-wait", str(WAIT),
                "--hysteresis", str(WAIT), "--stop-timeout", "1", "--ticks", str(ticks), *extra]  # fmt: skip
        return subprocess.Popen(
            args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def log(self):
        path = self.state / "guard.log"
        return path.read_text() if path.exists() else ""


def until(predicate, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


def levels_in_order(calls, reverse=False):
    levels = [LEVEL[n] for _, _, n in calls]
    return levels == sorted(levels, reverse=reverse)


def first_of_each_level(calls, reverse=False):
    seen = {}
    for t, _, n in calls:
        seen.setdefault(LEVEL[n], t)
    return [seen[k] for k in sorted(seen, reverse=reverse)]


def test_below_the_floor_level_by_level_then_back_in_reverse(host):
    proc = host.guard()
    try:
        host.free(25)
        assert until(lambda: {n for _, _, n in host.calls("stop")} == set(LEVEL))
        stops = host.calls("stop")
        # In order, each level whole, at least --level-wait apart; H1, the proxy, campaigns and
        # foreign containers never.
        assert levels_in_order(stops)
        starts_of_levels = first_of_each_level(stops)
        assert all(b - a >= WAIT * 0.9 for a, b in itertools.pairwise(starts_of_levels))
        # The peak read before the stop is the container's own: autan-ns-nssa's is 7 GiB.
        assert (
            "autan-ns-nssa: MemAvailable 25.0 GiB, below 30.0 GiB; its memory peak 7.0 GiB"
            in host.log()
        )
        # Between the thresholds, nothing comes back.
        host.free(35)
        time.sleep(3 * WAIT)
        assert host.calls("start") == []
        # Above 40 GiB: last level first, each level a full hysteresis after the previous one.
        host.free(45)
        assert until(lambda: {n for _, _, n in host.calls("start")} == set(LEVEL))
        starts = host.calls("start")
        assert levels_in_order(starts, reverse=True)
        firsts = first_of_each_level(starts, reverse=True)
        assert all(b - a >= WAIT * 0.9 for a, b in itertools.pairwise(firsts))
    finally:
        proc.kill()
        proc.wait()


def test_what_someone_else_removed_or_cycled_is_left_alone_and_failed_starts_are_retried(host):
    (host.bin / "fail-start.json").write_text(json.dumps(["autan-noyau-mut2"]))
    proc = host.guard()
    try:
        host.free(25)
        assert until(lambda: len(host.calls("stop")) >= 2)
        host.free(35)
        time.sleep(4 * WAIT)
        # Someone removes mut1 meanwhile; mut2's start fails until the guard gives up.
        c = json.loads(host.containers.read_text())
        del c["autan-noyau-mut1"]
        host.containers.write_text(json.dumps(c))
        host.free(45)
        assert until(lambda: "given up" in host.log())
        mut = [n for _, _, n in host.calls("start") if n.startswith("autan-noyau-mut")]
        assert mut == ["autan-noyau-mut2"] * 3
        assert "LEFT autan-noyau-mut1" in host.log()
    finally:
        proc.kill()
        proc.wait()


def test_a_restarted_guard_starts_what_an_earlier_one_stopped(host):
    host.free(25)
    first = host.guard(ticks=3)
    assert first.wait(timeout=30) == 0
    stopped = {n for _, _, n in host.calls("stop")}
    assert stopped == {"autan-noyau-mut1", "autan-noyau-mut2"}
    # Cycled by someone else meanwhile: started and stopped again, so no longer the guard's.
    c = json.loads(host.containers.read_text())
    c["autan-noyau-mut2"]["finished"] = "later"
    host.containers.write_text(json.dumps(c))
    host.free(45)
    second = host.guard()
    try:
        assert until(lambda: "LEFT autan-noyau-mut2" in host.log())
        assert until(lambda: [n for _, _, n in host.calls("start")] == ["autan-noyau-mut1"])
    finally:
        second.kill()
        second.wait()


def test_dry_run_decides_and_logs_but_never_acts(host):
    host.free(25)
    assert host.guard("--dry-run", ticks=20).wait(timeout=30) == 0
    assert host.calls() == []
    assert "DRY-RUN STOP level 1 autan-noyau-mut1" in host.log()


def test_unknown_memory_counts_as_below_the_floor(host):
    host.meminfo.write_text("garbage\n")
    assert host.guard(ticks=2).wait(timeout=30) == 0
    assert {n for _, _, n in host.calls("stop")} == {"autan-noyau-mut1", "autan-noyau-mut2"}


def test_docker_unreachable_is_an_error_not_nothing_left(host):
    # A missing CLI, and a socket the guard may not use (a user service without the docker group).
    host.free(25)
    out = host.guard(ticks=3, path="/nonexistent")
    assert out.wait(timeout=30) == 0, out.stderr.read()
    (host.bin / "deny").touch()
    assert host.guard(ticks=3).wait(timeout=30) == 0
    log = host.log()
    assert "nothing left to stop" not in log
    assert log.count("ERROR docker cannot list") == 2  # at the start of each guard
    assert "permission denied" in log


def test_a_container_that_will_not_stop_does_not_hold_back_the_next_levels(host):
    (host.bin / "fail-stop.json").write_text(json.dumps(["autan-noyau-mut1"]))
    proc = host.guard()
    try:
        host.free(25)
        assert until(
            lambda: (
                {"autan-noyau-release", "autan-sdk-build"} <= {n for _, _, n in host.calls("stop")}
            )
        )
        assert "FAILED to stop autan-noyau-mut1" in host.log()
    finally:
        proc.kill()
        proc.wait()


def test_docker_not_answering_at_the_restart_is_retried_not_forgotten(host):
    host.free(25)
    assert host.guard(ticks=3).wait(timeout=30) == 0
    (host.bin / "fail-inspect").write_text("1")
    host.free(45)
    proc = host.guard()
    try:
        assert until(
            lambda: (
                {n for _, _, n in host.calls("start")} == {"autan-noyau-mut1", "autan-noyau-mut2"}
            )
        )
        assert "did not answer" in host.log()
        assert "LEFT" not in host.log()
    finally:
        proc.kill()
        proc.wait()


def test_memory_falling_during_the_restarts_holds_the_rest_of_the_level(host):
    host.free(25)
    assert host.guard(ticks=3).wait(timeout=30) == 0
    # Whichever of the two starts first takes the host back down to 32 GiB.
    low = f"MemTotal: {128 << 20} kB\nMemAvailable: {32 << 20} kB\n"
    (host.bin / "after-start.json").write_text(json.dumps(
        {n: [str(host.meminfo), low] for n in ("autan-noyau-mut1", "autan-noyau-mut2")}))  # fmt: skip
    host.free(45)
    proc = host.guard()
    try:
        assert until(lambda: "the rest of its level wait" in host.log())
        assert len(host.calls("start")) == 1
    finally:
        proc.kill()
        proc.wait()


def disk_args(path, max_bytes, max_pct=100.0):
    return ("--disk-path", str(path), "--disk-max-gb", str(max_bytes / 1e9),
            "--disk-max-pct", str(max_pct), "--disk-every", "0.2")  # fmt: skip


def test_a_directory_over_its_size_limit_stops_the_levels_and_holds_the_restarts(host, tmp_path):
    data = tmp_path / "cfd"
    data.mkdir()
    (data / "results").write_bytes(b"x" * 200_000)
    proc = host.guard(*disk_args(data, 100_000))  # memory plentiful: the disk alone
    try:
        assert until(
            lambda: (
                {"autan-noyau-mut1", "autan-noyau-mut2"} <= {n for _, _, n in host.calls("stop")}
            )
        )
        assert (host.state / "ALERT").exists()
        assert "ALERT disk" in host.log()
        # Nothing restarts while the directory is over 90 % of its limit.
        time.sleep(3 * WAIT)
        assert host.calls("start") == []
        (data / "results").unlink()
        assert until(lambda: not (host.state / "ALERT").exists())
        assert until(lambda: len(host.calls("start")) >= 1)
    finally:
        proc.kill()
        proc.wait()


def test_90_percent_of_a_limit_alerts_without_stopping(host, tmp_path):
    data = tmp_path / "cfd"
    data.mkdir()
    (data / "results").write_bytes(b"x" * 95_000)
    assert host.guard(*disk_args(data, 100_000), ticks=20).wait(timeout=30) == 0
    assert (host.state / "ALERT").exists()
    assert host.calls("stop") == []


def test_a_full_filesystem_stops_the_levels(host, tmp_path):
    assert host.guard(*disk_args(tmp_path, 10**15, max_pct=1e-9), ticks=3).wait(timeout=30) == 0
    assert {n for _, _, n in host.calls("stop")} == {"autan-noyau-mut1", "autan-noyau-mut2"}


def test_a_stop_whose_time_was_unread_is_restarted_only_if_nobody_stopped_it_since(host):
    # Docker does not answer the three inspections after each stop of level 1: the stop times
    # stay unread. Then mut2 is started and stopped again by someone else, later.
    (host.bin / "fail-inspect").write_text("6")
    host.free(25)
    assert host.guard(ticks=3).wait(timeout=60) == 0
    assert {n for _, _, n in host.calls("stop")} == {"autan-noyau-mut1", "autan-noyau-mut2"}
    c = json.loads(host.containers.read_text())
    later = time.gmtime(time.time() + 100)
    c["autan-noyau-mut2"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S", later) + ".5Z"
    host.containers.write_text(json.dumps(c))
    host.free(45)
    proc = host.guard()
    try:
        assert until(lambda: "LEFT autan-noyau-mut2" in host.log())
        assert until(lambda: [n for _, _, n in host.calls("start")] == ["autan-noyau-mut1"])
    finally:
        proc.kill()
        proc.wait()


def test_a_failed_du_keeps_the_last_size(host, tmp_path):
    # du's first measure says 600 kB; then it fails on an unreadable entry with a partial total.
    data = tmp_path / "cfd"
    data.mkdir()
    fake_du = host.bin / "du"
    answer = host.bin / "du-answer"
    answer.write_text("600000 0")
    fake_du.write_text(f'#!/bin/sh\nread size code < {answer}\necho "$size\t$3"\nexit $code\n')
    fake_du.chmod(0o755)
    proc = host.guard(*disk_args(data, 100_000))
    try:
        assert until(lambda: len(host.calls("stop")) >= 2)
        answer.write_text("100 1")
        time.sleep(4 * WAIT)
        assert (host.state / "ALERT").exists()
        assert host.calls("start") == []
    finally:
        proc.kill()
        proc.wait()


def test_an_unwritable_alert_does_not_stop_the_memory_protection(host, tmp_path):
    data = tmp_path / "cfd"
    data.mkdir()
    (data / "results").write_bytes(b"x" * 95_000)  # a disk warning, not over
    host.state.mkdir()
    (host.state / "ALERT").mkdir()  # cannot be written as a file
    host.free(25)
    assert host.guard(*disk_args(data, 100_000), ticks=3).wait(timeout=30) == 0
    assert {n for _, _, n in host.calls("stop")} == {"autan-noyau-mut1", "autan-noyau-mut2"}
