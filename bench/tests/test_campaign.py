import dataclasses
import fcntl
import importlib.machinery
import importlib.util
import json
import os
import signal
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
    # Registered first: its dataclasses resolve their annotations through sys.modules.
    sys.modules["autan_campaign"] = module
    loader.exec_module(module)
    return module


campaign = load_campaign()


@pytest.fixture(autouse=True)
def host_memory(tmp_path, monkeypatch):
    """The host's memory account for `start`'s cap check, never this machine's: 128 GiB in all,
    100 GiB available (autan_memfloor read MEMINFO when the module was loaded)."""
    path = tmp_path / "host-meminfo"
    path.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {100 << 20} kB\n")
    monkeypatch.setattr(campaign.memfloor, "MEMINFO", path)
    return path


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
                    f'case "$1" in inspect) echo "running {os.getpid()}" ;; pause) sleep 20 ;;\n'
                    f'  start) if [ -e {tmp_path}/slow-start ]; then sleep 10; fi ;; esac\n')  # fmt: skip
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
    # memory falls below half the floor: the breach's leader removes it at once (it declares no
    # clean stop, and the host is critical), not once docker answers the pause.
    meminfo(20)
    assert campaign.main(["start", "c2", "--free", "10G", "--dir", str(tmp_path), "--", "x"]) == 0
    assert until(lambda: "start autan-campaign-c2" in calls_of("c2"))
    measurement = lock.open()
    fcntl.flock(measurement, fcntl.LOCK_EX)
    assert until(lambda: "pause autan-campaign-c2" in calls_of("c2"))
    meminfo(4)
    try:
        assert until(lambda: "kill autan-campaign-c2" in calls_of("c2"), timeout=8)
        fcntl.flock(measurement, fcntl.LOCK_UN)
        assert until(lambda: "rm -f autan-campaign-c2" in calls_of("c2"))
    finally:
        measurement.close()

    # Its `docker start` stalls (10 s) when the host falls below half the floor: killed during
    # the start, not once docker answers, then removed.
    (tmp_path / "slow-start").touch()
    meminfo(20)
    assert campaign.main(["start", "c3", "--free", "10G", "--dir", str(tmp_path), "--", "x"]) == 0
    assert until(lambda: "start autan-campaign-c3" in calls_of("c3"))
    meminfo(4)
    assert until(lambda: "kill autan-campaign-c3" in calls_of("c3"), timeout=3)
    assert until(lambda: "rm -f autan-campaign-c3" in calls_of("c3"), timeout=20)


@pytest.mark.parametrize(("given", "expected"), [([], 48 << 30), (["--memory", "20G"], 20 << 30)])
def test_a_campaign_is_capped_at_its_declared_memory(tmp_path, monkeypatch, given, expected):
    # docker is the external boundary: the cap reaches `docker create`, swap included.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
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
    argv = ["start", "c4", *given, "--dir", str(tmp_path), "--", "true"]
    assert campaign.main(argv) == 0
    records = [r.split("\x1f")[:-1] for r in calls.read_text().split("\x1e")[:-1]]
    create = next(r for r in records if r[0] == "create")
    assert create[create.index("--memory") + 1] == f"{expected}b"
    assert create[create.index("--memory-swap") + 1] == f"{expected}b"


def test_a_cap_that_cannot_fit_beside_the_floor_is_refused(tmp_path, monkeypatch):
    # 128 GiB in all, a 24 GiB floor: a 110 GiB cap could only end in the floor's breach, which
    # removes every campaign. Refused before docker is asked anything.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    lock = tmp_path / "gpu.lock"
    lock.touch()
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
    for memory in ("110G", "lots"):
        with pytest.raises(SystemExit) as stop:
            campaign.main(["start", "c5", "--memory", memory, "--dir", str(tmp_path), "--", "x"])
        assert stop.value.code not in (0, None)
    assert not calls.exists()


def test_a_campaign_s_out_of_memory_kills_are_counted_from_its_cgroup(tmp_path):
    cgroup = tmp_path / "cg"
    cgroup.mkdir()
    assert campaign.oom_kills_in(cgroup) == 0  # unreadable: none counted
    (cgroup / "memory.events").write_text("low 0\nhigh 0\nmax 12\noom 2\noom_kill 2\n")
    assert campaign.oom_kills_in(cgroup) == 2


@pytest.mark.parametrize(("state", "code", "said"), [
    ("0 false", 0, "campaign ended;"),
    ("137 true", 1, "killed for exceeding its --memory cap (exit 137)"),
    ("2 false", 1, "campaign ended with exit 2"),
])  # fmt: skip
def test_how_a_campaign_ended_is_logged_and_returned(tmp_path, monkeypatch, state, code, said):
    fake = tmp_path / "docker"
    fake.write_text(f"#!/bin/sh\necho '{state}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    assert campaign.ended_status("c7", "autan-campaign-c7") == code
    assert said in (tmp_path / "autan-campaigns" / "c7.log").read_text()


# A campaign's command as autan-run behaves (#455): at each "iteration" it looks for its stop file;
# found, it writes its restart state (solution.cgns, run.json "stopped") and exits 130. With
# IGNORE, it never looks: a hung program.
SOLVER = """import json, pathlib, sys, time
stop, out, ignore = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3] == "IGNORE"
while True:
    if not ignore and stop.exists():
        (out / "solution.cgns").write_text("state")
        (out / "run.json").write_text(json.dumps({"status": "stopped"}))
        sys.exit(130)
    time.sleep(0.05)
"""
GIB = 1 << 30


class FakeHost:
    """The leader's boundaries. Each campaign is a real process honouring (or not) its stop file;
    docker pause and unpause are SIGSTOP and SIGCONT to it, so a frozen campaign really cannot
    stop, and a pause can be made to land late; memory is a function of what has stopped."""

    def __init__(self, tmp_path, free_before, free_after, critical=False, pause_delay=0.0,
                 pause_fails=False):  # fmt: skip
        self.tmp_path = tmp_path
        self.paused, self.killed, self.events = set(), [], []
        self.killed_at = {}
        self.published, self.procs, self.logs = {}, {}, []
        self.free_before, self.free_after, self.critical = free_before, free_after, critical
        self.pause_delay, self.pause_fails = pause_delay, pause_fails
        self.history = []  # every state published, in order
        self.newcomers = []  # campaigns started after the election
        self.exclusive_from = 0.0  # transitions.lock is the leader's from this time.monotonic()
        self.gpu_until = 0.0  # the GPU lock is held until this time.monotonic()
        self.script = tmp_path / "solver.py"
        self.script.write_text(SOLVER)

    def spawn(self, name, clean_stop, ignore=False):
        out = self.tmp_path / name
        out.mkdir()
        stop = out / "STOP"
        self.procs[name] = subprocess.Popen(
            [sys.executable, str(self.script), str(stop), str(out), "IGNORE" if ignore else "-"]
        )
        return (f"stop-file:{stop}" if clean_stop == "stop-file" else clean_stop), out

    def available(self):
        if self.critical:
            return 10 * GIB
        gone = any(p.poll() is not None or n in self.killed for n, p in self.procs.items())
        return self.free_after if gone else self.free_before

    def host(self, _transitions=None):
        def pause(name):
            def run():
                time.sleep(self.pause_delay)
                if self.pause_fails:
                    return False
                self.procs[name].send_signal(signal.SIGSTOP)
                self.paused.add(name)
                self.published[name] = "frozen"
                self.events.append(("paused", name, time.monotonic()))
                return True

            return campaign.Op(run)

        def unpause(name):
            def run():
                self.procs[name].send_signal(signal.SIGCONT)
                self.paused.discard(name)
                self.events.append(("thawed", name, time.monotonic()))
                return True

            return campaign.Op(run)

        def kill(name):
            self.killed.append(name)
            self.killed_at[name] = time.monotonic()
            self.procs[name].kill()
            self.procs[name].wait()

        return campaign.Host(
            available=self.available,
            exclusive=lambda: time.monotonic() >= self.exclusive_from,
            members=lambda: self.newcomers,
            gpu_busy=lambda: time.monotonic() < self.gpu_until, pause=pause, unpause=unpause,
            publish=lambda name, state: (self.published.__setitem__(name, state),
                                         self.history.append((name, state))), request_stop=campaign.request_stop,
            ended=lambda name: self.procs[name].poll() is not None, kill=kill,
            log=lambda name, message: self.logs.append((name, message)),
        )  # fmt: skip

    def close(self):
        for proc in self.procs.values():
            proc.kill()
            proc.wait()


@pytest.fixture
def breach(tmp_path):
    hosts = []

    def make(**kwargs):
        host = FakeHost(tmp_path, **kwargs)
        hosts.append(host)
        return host

    yield make
    for host in hosts:
        host.close()


def stopped_cleanly(out: Path) -> bool:
    return (out / "solution.cgns").exists() and json.loads(
        (out / "run.json").read_text())["status"] == "stopped"  # fmt: skip


def test_a_breach_freezes_all_then_stops_the_culprit_cleanly(breach):
    # Pauses land late (a slow daemon): the victim is thawed only after its own, else the pause
    # would freeze it mid-stop and its wait would end in removal.
    host = breach(free_before=20 * GIB, free_after=40 * GIB, pause_delay=0.5)
    mode_a, _ = host.spawn("a", "none")
    mode_b, out_b = host.spawn("b", "stop-file")
    mode_c, _ = host.spawn("c", "none")
    members = [campaign.Member("a", 100.0, mode_a, 60, growth=0),
               campaign.Member("b", 50.0, mode_b, 60, growth=10 * GIB)]  # fmt: skip
    # c started after the election, before the leader held transitions.lock: frozen too.
    host.newcomers = [campaign.Member("c", 150.0, mode_c, 60)]
    host.exclusive_from = time.monotonic() + 1.0
    campaign.lead(members, 24 * GIB, host.host())
    assert host.procs["b"].wait(timeout=10) == 130
    assert stopped_cleanly(out_b)
    assert host.killed == []
    # The others are never removed, and left frozen for their own watchers to thaw; nothing is
    # paused before the leader holds transitions.lock.
    assert host.paused == {"a", "c"}
    assert host.published["a"] == "frozen"
    assert host.procs["a"].poll() is None
    assert all(e[2] >= host.exclusive_from for e in host.events if e[0] == "paused")
    order = [e[:2] for e in host.events if e[1] == "b"]
    assert order == [("paused", "b"), ("thawed", "b")]


def test_a_victim_is_not_thawed_while_a_measurement_holds_the_gpu(breach):
    host = breach(free_before=20 * GIB, free_after=40 * GIB)
    host.gpu_until = time.monotonic() + 1.5
    mode, out = host.spawn("b", "stop-file")
    campaign.lead([campaign.Member("b", 1.0, mode, 30)], 24 * GIB, host.host())
    assert host.procs["b"].wait(timeout=10) == 130
    assert stopped_cleanly(out)
    thawed = [e[2] for e in host.events if e[0] == "thawed"]
    assert len(thawed) == 1
    assert thawed[0] >= host.gpu_until


def test_a_campaign_whose_pause_failed_never_reads_frozen(breach):
    # Its pause failed: it runs. A measurement takes the GPU lock between "running" and the thaw.
    host = breach(free_before=20 * GIB, free_after=40 * GIB, pause_fails=True)
    mode, out = host.spawn("b", "stop-file")
    real = host.host()
    reads = iter([False, True, True, True])  # free, then held right after "running"
    gpu = dataclasses.replace(real, gpu_busy=lambda: next(reads, False))
    campaign.lead([campaign.Member("b", 1.0, mode, 30)], 24 * GIB, gpu)
    assert stopped_cleanly(out)
    assert ("b", "frozen") not in host.history


def test_memory_is_watched_while_the_breach_s_docker_calls_settle(breach):
    # Back above the floor at once, but a pause is still under way (3 s) when the host falls
    # below half the floor: the newest is removed then, not once docker has answered.
    host = breach(free_before=0, free_after=0, pause_delay=3.0)
    host.spawn("a", "none")
    host.spawn("b", "none")
    start = time.monotonic()
    real = host.host()
    falling = dataclasses.replace(
        real, available=lambda: 40 * GIB if time.monotonic() - start < 0.5 else 5 * GIB
    )
    members = [campaign.Member("a", 1.0, "none", 30), campaign.Member("b", 2.0, "none", 30)]
    campaign.lead(members, 24 * GIB, falling)
    assert host.killed[0] == "b"
    assert host.killed_at["b"] - start < 2


def test_a_victim_that_does_not_stop_within_its_wait_is_removed(breach):
    host = breach(free_before=20 * GIB, free_after=40 * GIB)
    mode_a, _ = host.spawn("a", "none")
    mode_b, _ = host.spawn("b", "stop-file", ignore=True)
    members = [campaign.Member("a", 100.0, mode_a, 60),
               campaign.Member("b", 200.0, mode_b, 2)]  # fmt: skip
    start = time.monotonic()
    campaign.lead(members, 24 * GIB, host.host())
    assert host.killed == ["b"]
    assert 2 <= host.killed_at["b"] - start < 5
    assert host.procs["a"].poll() is None


@pytest.mark.parametrize("cause", ["no clean stop", "below half the floor", "stop unrequestable"])
def test_the_last_resort_is_at_once(breach, cause):
    # No clean stop declared; the host below half the floor; a stop file that cannot be created.
    host = breach(
        free_before=20 * GIB, free_after=40 * GIB, critical=cause == "below half the floor"
    )
    mode, out = host.spawn("b", "none" if cause == "no clean stop" else "stop-file")
    if cause == "stop unrequestable":
        mode = f"stop-file:{out / 'missing' / 'STOP'}"
    start = time.monotonic()
    campaign.lead([campaign.Member("b", 1.0, mode, 60)], 24 * GIB, host.host())
    assert host.killed == ["b"]
    assert host.killed_at["b"] - start < 1


@pytest.mark.parametrize(("members", "victim"), [
    # The culprit, whatever its stop.
    ([("a", 3.0, "signal", 0), ("b", 1.0, "none", 5 * GIB)], "b"),
    # No culprit: the newest with a clean stop, before a newer one without.
    ([("a", 1.0, "signal", 0), ("b", 2.0, "stop-file:/x", GIB), ("c", 3.0, "none", 0)], "b"),
    # None declares one: the newest.
    ([("a", 1.0, "none", 0), ("b", 2.0, "none", 0)], "b"),
])  # fmt: skip
def test_the_victim_is_the_culprit_else_the_newest_with_a_clean_stop(members, victim):
    chosen, _ = campaign.victim_of(
        [campaign.Member(n, t, mode, 60, growth=g) for n, t, mode, g in members]
    )
    assert chosen.name == victim


@pytest.fixture
def states(tmp_path, monkeypatch):
    states = tmp_path / "campaigns"
    states.mkdir()
    monkeypatch.setattr(campaign, "STATES", states)
    monkeypatch.setattr(campaign, "BREACH_LOCK", states / "breach.lock")
    monkeypatch.setattr(campaign, "TRANSITIONS", states / "transitions.lock")
    monkeypatch.setattr(campaign, "BREACH_GENERATION", states / "breach.generation")
    return states


def published(states, name, *, clean_stop, started, floor, growth=None):
    (states / f"{name}.meta").write_text(json.dumps(
        {"started": started, "clean_stop": clean_stop, "stop_wait_s": 30, "floor": floor}))  # fmt: skip
    (states / f"{name}.heartbeat").touch()
    (states / f"{name}.state").write_text("running\n")
    if growth is not None:
        (states / f"{name}.growth").write_text(f"{growth[0]} {growth[1]}\n")


def test_the_host_s_floor_is_the_highest_declared_and_stale_growth_names_no_culprit(
    breach, states, host_memory
):
    # 30 GiB available: above the leader's own 24G floor, below the other campaign's 48G. The
    # newer campaign published a 10 GiB growth a minute ago, from a stalled watcher: no evidence.
    host_memory.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {30 << 20} kB\n")
    host = breach(free_before=30 * GIB, free_after=60 * GIB)
    mode_a, out_a = host.spawn("a", "stop-file")
    mode_b, _ = host.spawn("b", "none")
    published(states, "a", clean_stop=mode_a, started=1.0, floor=48 * GIB)
    published(states, "b", clean_stop=mode_b, started=2.0, floor=24 * GIB,
              growth=(10 * GIB, time.time() - 60))  # fmt: skip
    assert campaign.handle_breach(24 * GIB, host=host.host) is True
    assert stopped_cleanly(out_a)
    assert host.killed == []


def test_one_watcher_leads_a_breach_the_others_follow(breach, states, host_memory):
    host_memory.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {10 << 20} kB\n")
    host = breach(free_before=10 * GIB, free_after=60 * GIB)
    mode, _ = host.spawn("a", "none")
    published(states, "a", clean_stop=mode, started=1.0, floor=24 * GIB)
    with (states / "breach.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX)  # another watcher leads
        assert campaign.handle_breach(24 * GIB, host=host.host) is False
    assert host.killed == []
    assert campaign.handle_breach(24 * GIB, host=host.host) is True
    assert host.killed == ["a"]


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/locks is Linux's")
def test_a_watcher_s_transition_and_a_breach_never_cross(states):
    with states.joinpath("transitions.lock").open("a") as leader:
        exclusive = campaign.real_host(leader).exclusive
        with campaign.transition_allowed() as allowed:
            assert allowed
            assert not exclusive()  # the leader waits for the watcher's pause or thaw
        assert exclusive()
        with campaign.transition_allowed() as allowed:
            assert not allowed  # and no watcher acts while the leader holds it
    with states.joinpath("breach.lock").open("a") as election:
        fcntl.flock(election, fcntl.LOCK_EX)  # elected, not yet exclusive
        with campaign.transition_allowed() as allowed:
            assert not allowed


def test_a_stop_file_outside_an_existing_directory_is_refused(tmp_path, monkeypatch):
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(
        f'#!/bin/sh\necho "$*" >> {calls}\ncase "$1" in inspect) echo "exited 0" ;; esac\n'
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
    argv = ["start", "c1", "--dir", str(tmp_path), "--clean-stop", "stop-file:control/STOP",
            "--", "x"]  # fmt: skip
    with pytest.raises(SystemExit):
        campaign.main(argv)
    assert not calls.exists() or "create" not in calls.read_text()
    (tmp_path / "control").mkdir()
    assert campaign.main(argv) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/meminfo and cgroups are Linux's")
def test_a_leader_whose_own_campaign_is_the_victim_finishes_the_breach(
    tmp_path, monkeypatch, meminfo
):
    # Through the CLI and the detached watchers, docker faked at the boundary: it records its
    # calls, keeps each container's paused state, and reports every container running as this
    # test's process (a real cgroup). c1's floor (30G) is crossed, c2's and c3's (10G) are not:
    # c1's watcher leads, at the host's floor, 30G. c1 alone declares a clean stop: it goes
    # first, is asked through its stop file (created in its --dir), and, its fake container never
    # ending, is removed after its 10 s wait. Memory does not come back: the leader goes on, after
    # its own campaign has ended, to the newest, c2. Memory then comes back: c3 survives, and its
    # own watcher thaws it.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(
        f'#!/bin/sh\necho "$*" >> {calls}\n'
        'case "$1" in\n'
        f'  inspect) case "$3" in *Paused*) [ -e "{tmp_path}/paused-$4" ] && echo true || echo false ;;\n'
        f'           *) echo "running {os.getpid()}" ;; esac ;;\n'
        f'  pause) touch "{tmp_path}/paused-$2" ;;\n'
        f'  unpause) rm -f "{tmp_path}/paused-$2" ;;\n'
        "esac\n"
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

    def ran(command: str) -> bool:
        return calls.exists() and command in calls.read_text().splitlines()

    for name, free, stop in (("c3", "10G", "none"), ("c2", "10G", "none"),
                             ("c1", "30G", "stop-file:STOP")):  # fmt: skip
        argv = ["start", name, "--free", free, "--dir", str(tmp_path), "--clean-stop", stop,
                "--stop-wait", "10", "--", "x"]  # fmt: skip
        assert campaign.main(argv) == 0
        assert until(lambda n=name: ran(f"start autan-campaign-{n}"))
    meminfo(25)
    assert until((tmp_path / "paused-autan-campaign-c3").exists)
    assert until((tmp_path / "STOP").exists)
    assert until(lambda: ran("kill autan-campaign-c1"), timeout=20)
    assert until(lambda: ran("kill autan-campaign-c2"), timeout=20)
    meminfo(100)  # within the leader's 2 s wait for c2's memory
    assert until(lambda: not (tmp_path / "paused-autan-campaign-c3").exists(), timeout=20)
    assert until(lambda: ran("rm -f autan-campaign-c1") and ran("rm -f autan-campaign-c2"))
    assert not ran("kill autan-campaign-c3")


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/locks is Linux's")
def test_a_start_delayed_by_a_breach_waits_for_a_measurement_begun_meanwhile(tmp_path, monkeypatch):
    # Through the CLI and the detached watcher, docker faked at the boundary. A breach is being
    # led (breach.lock held) when the campaign would start: it waits. A measurement takes the GPU
    # lock, then the breach ends: it still waits, and starts once the measurement is over.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n'
                    f'case "$1" in inspect) echo "running {os.getpid()}" ;; esac\n')  # fmt: skip
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    lock = tmp_path / "gpu.lock"
    lock.touch()
    monkeypatch.setenv("AUTAN_GPU_LOCK", str(lock))
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
    (tmp_path / "campaigns").mkdir()

    def started() -> bool:
        return calls.exists() and "start autan-campaign-c1" in calls.read_text().splitlines()

    with (tmp_path / "campaigns" / "breach.lock").open("a") as breach, lock.open() as measurement:
        fcntl.flock(breach, fcntl.LOCK_EX)
        assert (
            campaign.main(["start", "c1", "--free", "0G", "--dir", str(tmp_path), "--", "x"]) == 0
        )
        assert not until(started, timeout=2)
        fcntl.flock(measurement, fcntl.LOCK_EX)
        fcntl.flock(breach, fcntl.LOCK_UN)
        assert not until(started, timeout=2)
        assert not (tmp_path / "campaigns" / "c1.state").exists()
        fcntl.flock(measurement, fcntl.LOCK_UN)
        assert until(started, timeout=10)
