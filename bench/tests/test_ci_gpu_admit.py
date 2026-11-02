import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci" / "gpu-admit.sh"
# autan-gpu-guard's list, as it wrote it on spark-5512 with Excalibur's vLLM on the GPU.
VLLM = "# 2026-11-02T20:00:31Z 1 foreign\n4132364\tmcpvar-vllm\t23221 MiB\tVLLM::EngineCore\n"
NONE = "# 2026-11-02T20:00:31Z 0 foreign\n"


@pytest.fixture
def spark(tmp_path: Path):
    """A lock directory with the guard's list, and the kernel's memory account as a file."""

    def admit(host: str, tenants: str | None, gib: float, *args: str) -> tuple[int, str, str]:
        if tenants is not None:
            (tmp_path / "foreign-gpu").write_text(tenants)
        write(gib)
        env = {**os.environ, "AUTAN_GPU_LOCK": str(tmp_path / "gpu.lock"),
               "AUTAN_MEMINFO": str(tmp_path / "meminfo"), "AUTAN_BENCH_HOST": host,
               "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"), "AUTAN_ADMIT_POLL_S": "0.1"}  # fmt: skip
        out = subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True,
                             timeout=60, check=False)  # fmt: skip
        summary = (tmp_path / "summary").read_text() if (tmp_path / "summary").exists() else ""
        return out.returncode, out.stdout, summary

    def write(gib: float) -> None:
        tmp = tmp_path / "meminfo.tmp"
        tmp.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {int(gib * (1 << 20))} kB\n")
        tmp.replace(tmp_path / "meminfo")

    admit.write = write
    return admit


def test_a_foreign_tenant_is_not_a_refusal_off_the_timed_hosts(spark):
    code, out, summary = spark("spark-5512", VLLM, 40, "4")
    assert code == 0
    # No annotation: a process name may hold "::", but no line starts with a workflow command.
    assert not [line for line in out.splitlines() if line.startswith("::")]
    # Named in the job's summary all the same.
    assert (
        "GPU tenants outside Autan on spark-5512: VLLM::EngineCore (mcpvar-vllm, 23221 MiB)"
        in summary
    )


def test_a_foreign_tenant_refuses_a_gpu_step_on_a_timed_host(spark):
    code, out, summary = spark("spark-5359", VLLM, 100, "4")
    assert code == 1
    assert (
        "::error::GPU taken by a foreign tenant on spark-5359, which runs timed benchmarks" in out
    )
    assert "mcpvar-vllm" in summary


def test_a_timed_host_without_tenants_admits(spark):
    code, _, summary = spark("spark-5359", NONE, 100, "4")
    assert code == 0
    assert "GPU tenants outside Autan on spark-5359: none" in summary


@pytest.mark.parametrize(("gib", "code"), [(28, 0), (27.9, 1)])
def test_the_step_needs_the_floor_plus_its_declared_need(spark, gib, code):
    # 24 GiB floor + 4 GiB declared: 28 GiB passes, just under waits, then fails.
    got, out, _ = spark("spark-5512", NONE, gib, "4", "1")
    assert got == code
    if code:
        assert "::error::WAITING FOR MEMORY timed out after 1 s on spark-5512" in out
        assert "Not this change's fault" in out


def test_the_step_waits_for_memory_and_starts_once_it_is_back(spark):
    spark.write(10)
    timer = threading.Timer(1.0, spark.write, args=(30,))
    timer.start()
    start = time.monotonic()
    try:
        code, out, _ = spark("spark-5512", NONE, 10, "4", "30")
    finally:
        timer.cancel()
    assert code == 0
    assert "WAITING FOR MEMORY on spark-5512: 10 GiB available, needs 28 GiB" in out
    assert time.monotonic() - start < 10


def test_an_unknown_tenant_list_is_said_and_does_not_refuse(spark):
    code, _, summary = spark("spark-5359", None, 100, "4")
    assert code == 0
    assert "unknown (no list from autan-gpu-guard)" in summary


@pytest.mark.parametrize("args", [["x"], ["4", "-1"], []])
def test_malformed_arguments_are_refused(spark, args):
    code, _, _ = spark("spark-5512", NONE, 100, *args)
    assert code == 2
