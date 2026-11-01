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
