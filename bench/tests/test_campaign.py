import fcntl
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import pytest

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
    fake.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> {calls}\n'
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
    work = tmp_path / "work"
    work.mkdir()
    argv = ["start", "c1", "--cpus", "3", "--dir", str(work), "--", "make", "-j", "--cpus"]
    assert campaign.main(argv) == 0
    create = next(line for line in calls.read_text().splitlines() if line.startswith("create"))
    assert "--cpus 3.0" in create
    assert create.endswith("autan-dev:local nice -n 19 make -j --cpus")
