import os
import stat

import pytest
from autan_bench.errors import BenchError, Code
from autan_bench.gpu import require_idle


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch):
    """Puts on PATH an nvidia-smi that reports the given utilization and compute processes."""

    def install(utilization: str, apps: int) -> None:
        script = tmp_path / "nvidia-smi"
        pids = " ".join(str(1000 + i) for i in range(apps))
        script.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            f"  *query-compute-apps*) for p in {pids}; do echo $p; done ;;\n"
            f'  *) echo "{utilization}, 2400, 3003, 20.0, 45, 0x0000000000000000" ;;\n'
            "esac\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    return install


def test_idle_check_needs_a_measured_quiet_gpu(fake_nvidia_smi):
    fake_nvidia_smi("0", 0)
    assert require_idle(interval_s=0.0)["utilization_pct_max"] == 0.0

    for utilization, apps, code in [
        ("30", 0, Code.GPU_BUSY),
        ("0", 1, Code.GPU_BUSY),
        ("[N/A]", 0, Code.NVIDIA_SMI_FAILED),
    ]:
        fake_nvidia_smi(utilization, apps)
        with pytest.raises(BenchError) as err:
            require_idle(interval_s=0.0, timeout_s=0.0)
        assert err.value.code == code
