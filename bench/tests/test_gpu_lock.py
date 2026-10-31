import threading
import time

from cfdbench.gpu import gpu_lock


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
