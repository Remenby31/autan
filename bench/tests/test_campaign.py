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


HOLDER = """import fcntl, pathlib, sys, time
handles = []
for path, mode in zip(sys.argv[2::2], sys.argv[3::2]):
    handle = open(path)
    fcntl.flock(handle, fcntl.LOCK_EX if mode == "ex" else fcntl.LOCK_SH)
    handles.append(handle)
pathlib.Path(sys.argv[1]).touch()
time.sleep(60)
"""


@pytest.fixture
def holders(tmp_path):
    """Processes that hold flocks, as a build or a measurement holds the GPU lock."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER)
    started = []

    def hold(*locks: tuple[Path, str]) -> subprocess.Popen:
        ready = tmp_path / f"ready-{len(started)}"
        args = [str(x) for lock in locks for x in lock]
        proc = subprocess.Popen([sys.executable, str(script), str(ready), *args])
        started.append(proc)
        assert until(ready.exists, timeout=10)
        return proc

    yield hold
    for proc in started:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/locks is Linux's")
def test_only_a_hold_that_says_it_is_untimed_leaves_campaigns_running(
    tmp_path, monkeypatch, holders
):
    lock, untimed = tmp_path / "gpu.lock", tmp_path / "untimed.lock"
    lock.touch()
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "UNTIMED", untimed)

    def wanted() -> bool:
        return campaign.freeze_wanted(Path("/proc/locks").read_text())

    assert not wanted()
    # Before untimed.lock exists (a Spark not redeployed), every hold freezes.
    proc = holders((lock, "ex"))
    assert wanted()
    proc.kill()
    proc.wait()
    untimed.touch()
    # A build: the GPU lock and untimed.lock, by the same process.
    build = holders((lock, "ex"), (untimed, "sh"))
    assert not wanted()
    build.kill()
    build.wait()
    # A measurement: the GPU lock alone; untimed.lock held by another process does not cover it.
    holders((untimed, "sh"))
    holders((lock, "ex"))
    assert wanted()


WRAPPER = Path(__file__).parents[1] / "lock" / "autan-gpu-lock"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/locks and cgroups are Linux's")
def test_the_watcher_freezes_for_a_measurement_not_for_a_build(tmp_path, monkeypatch, holders):
    # Through the CLI and its detached watcher; docker faked at the boundary (it records its
    # calls and reports the container running as this test's own process, a real cgroup). The
    # build is the real wrapper with --untimed.
    calls = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n'
                    f'case "$1" in inspect) echo "running {os.getpid()}" ;; esac\n')  # fmt: skip
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    lock = tmp_path / "gpu.lock"
    for name in ("gpu.lock", "untimed.lock", "holders.log"):
        (tmp_path / name).touch()
    monkeypatch.setenv("AUTAN_GPU_LOCK", str(lock))
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "UNTIMED", tmp_path / "untimed.lock")
    monkeypatch.setattr(campaign, "STATE", tmp_path / "autan-campaigns")
    monkeypatch.setattr(campaign, "STATES", tmp_path / "campaigns")
    state = tmp_path / "campaigns" / "c3.state"

    def calls_of() -> list[str]:
        if not calls.exists():
            return []
        return [c.split()[0] for c in calls.read_text().splitlines()
                if c.endswith("autan-campaign-c3")]  # fmt: skip

    assert campaign.main(["start", "c3", "--free", "0G", "--dir", str(tmp_path), "--", "x"]) == 0
    try:
        assert until(lambda: "start" in calls_of())
        build = subprocess.run(
            [str(WRAPPER), "--untimed", "-f", "0G", "-m", "a build", "--", "sleep", "3"],
            env={**os.environ, "AUTAN_AGENT": "tester"}, check=False, timeout=60,
        )  # fmt: skip
        assert build.returncode == 0
        assert "pause" not in calls_of()  # thirty polls of the watcher: never paused
        measurement = holders((lock, "ex"))
        assert until(lambda: state.read_text().strip() == "frozen")
        assert "pause" in calls_of()
        measurement.kill()
        measurement.wait()
        assert until(lambda: "unpause" in calls_of())
        assert until(lambda: state.read_text().strip() == "running")
    finally:
        campaign.main(["stop", "c3"])


def test_an_owner_without_a_pid_never_says_untimed(tmp_path, monkeypatch):
    """Codex on #452: OFD locks read PID -1; one on untimed.lock must not cover another on the
    GPU lock."""
    lock, untimed = tmp_path / "gpu.lock", tmp_path / "untimed.lock"
    lock.touch()
    untimed.touch()
    monkeypatch.setattr(campaign, "LOCK", lock)
    monkeypatch.setattr(campaign, "UNTIMED", untimed)

    def key(path: Path) -> str:
        st = path.stat()
        return f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"

    ofd = (f"1: OFDLCK ADVISORY WRITE -1 {key(lock)} 0 EOF\n"
           f"2: OFDLCK ADVISORY READ -1 {key(untimed)} 0 EOF\n")  # fmt: skip
    assert campaign.freeze_wanted(ofd)
    same = (f"1: FLOCK ADVISORY WRITE 4242 {key(lock)} 0 EOF\n"
            f"2: FLOCK ADVISORY READ 4242 {key(untimed)} 0 EOF\n")  # fmt: skip
    assert not campaign.freeze_wanted(same)


@pytest.mark.parametrize("paused", [True, False])
def test_an_untimed_hold_s_load_never_ends_a_campaign(paused):
    """Codex on #452: a build's compilers, once it no longer freezes the campaign, counted as a
    foreign load and removed it after a minute. Not while an untimed hold runs."""
    load = campaign.ForeignLoad()
    ended = None
    now = 0.0
    while now < 300 and ended is None:
        ended = load.update(paused, now, lambda: campaign.MAX_OTHER_LOAD + 10)
        now += campaign.POLL_S
    if paused:
        assert ended is None
    else:
        # Foreign: the campaign gives way after its grace.
        assert ended == campaign.MAX_OTHER_LOAD + 10
        assert campaign.LOAD_GRACE_S <= now <= campaign.LOAD_GRACE_S + 30


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
