import fcntl
import threading
import time

import pytest
from autan_bench.errors import BenchError, Code
from autan_bench.gpu import gpu_lock


def test_gpu_lock_serialises_holders(tmp_path):
    lock = tmp_path / "gpu.lock"
    held = threading.Event()

    def hold() -> None:
        with gpu_lock(lock):
            held.set()
            time.sleep(0.3)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    with gpu_lock(lock) as waited:
        assert waited >= 0.2
    holder.join()
    with gpu_lock(lock) as waited:
        assert waited < 0.1


def test_gpu_lock_gives_up_after_its_timeout(tmp_path):
    lock = tmp_path / "gpu.lock"
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with gpu_lock(lock):
            held.set()
            release.wait()

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    with pytest.raises(BenchError) as err, gpu_lock(lock, timeout_s=0.5):
        pass
    assert err.value.code == Code.LOCK_TIMEOUT
    release.set()
    holder.join()
    with gpu_lock(lock, timeout_s=0.5) as waited:
        assert waited < 0.1


def test_gpu_lock_waits_behind_an_announced_ci_job(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    ci = tmp_path / "ci.lock"
    ci.touch()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with ci.open() as announced:
        fcntl.flock(announced, fcntl.LOCK_SH)
        with pytest.raises(BenchError) as err, gpu_lock(lock, timeout_s=0.5):
            pass
        assert err.value.code == Code.LOCK_TIMEOUT
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        with gpu_lock(lock, timeout_s=0.5) as waited:
            assert waited < 0.1


def test_a_pull_request_s_ci_job_waits_behind_main(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    main = tmp_path / "main.lock"
    main.touch()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with main.open() as announced:
        fcntl.flock(announced, fcntl.LOCK_SH)
        monkeypatch.setenv("GITHUB_REF", "refs/pull/7/merge")
        with pytest.raises(BenchError) as err, gpu_lock(lock, timeout_s=0.5):
            pass
        assert err.value.code == Code.LOCK_TIMEOUT
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        with gpu_lock(lock, timeout_s=0.5) as waited:
            assert waited < 0.1
