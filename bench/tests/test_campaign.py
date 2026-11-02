import fcntl
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from autan_bench.errors import BenchError, Code
from autan_bench.gpu import require_campaigns_frozen

SCRIPT = Path(__file__).parents[1] / "campaign" / "autan-campaign"


def load_campaign():
    loader = importlib.machinery.SourceFileLoader("autan_campaign", str(SCRIPT))
    spec = importlib.util.spec_from_loader("autan_campaign", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


campaign = load_campaign()


def test_lock_use_is_read_from_holders_and_waiters(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    st = lock.stat()
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    other_dev = f"{os.major(st.st_dev) + 1:02x}:{os.minor(st.st_dev):02x}"
    # Lines in the format of proc(5).
    held = f"1: FLOCK  ADVISORY  WRITE 4242 {dev}:{st.st_ino} 0 EOF"
    waiting = f"1: FLOCK  ADVISORY  WRITE 4242 {dev}:1 0 EOF\n1: -> FLOCK  ADVISORY  WRITE 4343 {dev}:{st.st_ino} 0 EOF"
    unrelated = (
        f"2: POSIX  ADVISORY  WRITE 99 {dev}:{st.st_ino + 1} 0 EOF\n"
        f"3: FLOCK  ADVISORY  WRITE 98 {other_dev}:{st.st_ino} 0 EOF"
    )
    assert campaign.lock_in_use(lock, held)
    assert campaign.lock_in_use(lock, f"{unrelated}\n{waiting}")
    assert not campaign.lock_in_use(lock, unrelated)
    assert not campaign.lock_in_use(lock, "")


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/locks is Linux's")
def test_lock_use_follows_a_real_flock(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    proc_locks = Path("/proc/locks")
    assert not campaign.lock_in_use(lock, proc_locks.read_text())
    with lock.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert campaign.lock_in_use(lock, proc_locks.read_text())
    assert not campaign.lock_in_use(lock, proc_locks.read_text())


def test_start_keeps_its_options_out_of_the_command(tmp_path, monkeypatch):
    # docker is the external boundary: a fake records its arguments and reports nothing running.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    # One record per call: arguments separated by US, records by RS (arguments hold newlines).
    fake.write_text(
        f"#!/bin/sh\nprintf '%s\\037' \"$@\" >> {calls}; printf '\\036' >> {calls}\n"
        'case "$1" in inspect) echo "exited 0" ;; esac\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    lock = tmp_path / "gpu.lock"
    lock.touch()
    monkeypatch.setenv("AUTAN_GPU_LOCK", str(lock))
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
    work = tmp_path / "work"
    work.mkdir()
    argv = ["start", "c1", "--cpus", "3", "--dir", str(work), "--", "make", "-j", "--cpus"]
    assert campaign.main(argv) == 0
    records = [r.split("\x1f")[:-1] for r in calls.read_text().split("\x1e")[:-1]]
    create = next(r for r in records if r[0] == "create")
    assert create[create.index("--cpus") + 1] == "3.0"
    assert create[create.index("autan-dev:local") + 1 : create.index("autan-dev:local") + 4] == [
        "nice", "-n", "19"
    ]  # fmt: skip
    assert create[-3:] == ["make", "-j", "--cpus"]


def test_a_measurement_waits_for_every_live_campaign_to_freeze(tmp_path):
    lock = tmp_path / "gpu.lock"
    states = tmp_path / "campaigns"
    states.mkdir()
    require_campaigns_frozen(lock, timeout_s=0.2)
    (states / "a.state").write_text("running\n")
    (states / "a.heartbeat").touch()
    with pytest.raises(BenchError) as err:
        require_campaigns_frozen(lock, timeout_s=0.3)
    assert err.value.code == Code.HOST_BUSY
    (states / "a.state").write_text("frozen\n")
    require_campaigns_frozen(lock, timeout_s=0.2)
    # A campaign whose watcher died stops by itself once its heartbeat is stale.
    (states / "a.state").write_text("running\n")
    old = time.time() - 600
    os.utime(states / "a.heartbeat", (old, old))
    require_campaigns_frozen(lock, timeout_s=0.2)


@pytest.mark.skipif(sys.platform != "linux", reason="GNU stat")
def test_the_supervisor_stops_the_campaign_when_the_heartbeat_is_stale(tmp_path):
    heartbeat = tmp_path / "heartbeat"
    heartbeat.touch()
    left = tmp_path / "pid"
    command = ["sh", "-c", f"sleep 60 & echo $! > {left}; wait"]
    cmd = ["sh", "-c", campaign.SUPERVISOR, "supervisor", str(heartbeat), "3", *command]
    proc = subprocess.Popen(cmd, start_new_session=True)
    time.sleep(1)
    assert proc.poll() is None
    old = time.time() - 600
    os.utime(heartbeat, (old, old))
    assert proc.wait(timeout=10) == 125
    os.killpg(proc.pid, 9)


def test_stop_never_signals_a_process_that_is_not_its_watcher(tmp_path, monkeypatch):
    # A PID file left by a dead watcher, whose PID now belongs to an unrelated process.
    other = subprocess.Popen(["sleep", "30"])
    try:
        fake = tmp_path / "docker"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
        monkeypatch.setattr(campaign, "STATE", tmp_path)
        monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
        (tmp_path / "c2.pid").write_text(f"{other.pid} 12345\n")
        assert campaign.main(["stop", "c2"]) == 0
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="/proc is Linux's")
def test_every_live_campaign_is_excluded_from_the_outside_load(tmp_path, monkeypatch):
    # Two campaigns, each a cgroup whose thread spins, published by their watchers. A third
    # publication with a stale heartbeat and a malformed one exclude nothing.
    spin = [sys.executable, "-c", "while True: pass"]
    procs = [subprocess.Popen(spin) for _ in range(3)]
    try:
        root, states = tmp_path / "cgroups", tmp_path / "campaigns"
        states.mkdir()
        for k, proc in enumerate(procs):
            cgroup = root / f"cg{k}"
            cgroup.mkdir(parents=True)
            (cgroup / "cgroup.threads").write_text(f"{proc.pid}\n")
            (states / f"c{k}.cgroup").write_text(f"{cgroup}\n")
            (states / f"c{k}.heartbeat").touch()
        old = time.time() - 600
        os.utime(states / "c2.heartbeat", (old, old))
        (states / "bad.cgroup").write_bytes(b"\xff/\n")
        (states / "bad.heartbeat").touch()
        monkeypatch.setattr(campaign, "STATES", states)
        monkeypatch.setattr(campaign, "CGROUPS", root)
        time.sleep(0.2)
        # What the outside load excludes (other_runnable = runnable threads - these - the
        # sampler): the spinning threads of the 2 live campaigns, not the stale one nor the
        # malformed one. Measured on spark-5512 at load 17-25: 20 000 of 20 000 samples gave 2,
        # so a sample errs with p <= 1.5e-4 (rule of three, 95 %). Majority of 5: false failure
        # <= C(5,3) p^3 = 3e-11. A defect (stale counted: 3; one or both live ones missed: 1, 0)
        # gives a wrong value on every sample, so it is caught unless 3 of 5 samples err:
        # missed with probability <= 3e-11. Differencing two scans of the whole host instead
        # (the previous form) erred on 91 % of samples there.
        counts = [campaign.campaigns_running() for _ in range(5)]
        assert counts.count(2) >= 3, counts
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()


def until(predicate, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/meminfo and cgroups are Linux's")
def test_a_campaign_gives_way_when_host_memory_falls_below_its_floor(
    tmp_path, monkeypatch, meminfo
):
    # Through the CLI, whose detached watcher enforces the floor. docker is faked at the
    # boundary: it records its calls and reports the container running as this test's own
    # process, so that the container's cgroup is a real one.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    # A stalled daemon: `docker pause` takes 20 s.
    fake.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n'
                    f'case "$1" in inspect) echo "running {os.getpid()}" ;; pause) sleep 20 ;; esac\n')  # fmt: skip
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    lock = tmp_path / "gpu.lock"
    lock.touch()
    monkeypatch.setenv("AUTAN_GPU_LOCK", str(lock))
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
    logs = tmp_path / "autan-campaigns"

    def calls_of(name: str) -> list[str]:
        return [c for c in calls.read_text().splitlines() if c.endswith(f"autan-campaign-{name}")]

    # Below the floor before it starts: never started, removed.
    meminfo(5)
    assert campaign.main(["start", "c1", "--free", "10G", "--dir", str(tmp_path), "--", "x"]) == 0
    assert until(lambda: "not started" in (logs / "c1.log").read_text())
    assert until(lambda: "rm -f autan-campaign-c1" in calls_of("c1"))
    assert "start autan-campaign-c1" not in calls_of("c1")

    # Running, then frozen behind a stalled `docker pause` (a measurement took the GPU lock), when
    # memory falls below the floor: the container is killed at once, not once docker answers.
    meminfo(20)
    assert campaign.main(["start", "c2", "--free", "10G", "--dir", str(tmp_path), "--", "x"]) == 0
    assert until(lambda: "start autan-campaign-c2" in calls_of("c2"))
    measurement = lock.open()
    fcntl.flock(measurement, fcntl.LOCK_EX)
    assert until(lambda: "pause autan-campaign-c2" in calls_of("c2"))
    meminfo(5)
    try:
        assert until(lambda: "kill autan-campaign-c2" in calls_of("c2"), timeout=8)
        fcntl.flock(measurement, fcntl.LOCK_UN)
        assert until(lambda: "below --free" in (logs / "c2.log").read_text(), timeout=40)
        assert until(lambda: "rm -f autan-campaign-c2" in calls_of("c2"))
    finally:
        measurement.close()
