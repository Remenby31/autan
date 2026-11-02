import importlib.machinery
import importlib.util
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "ci" / "gpu-admit.sh"
GUARD = ROOT / "lock" / "autan-gpu-guard"


def stamp(age_s: float = 0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_s))


def guard_list(rows: str, age_s: float = 0) -> str:
    """autan-gpu-guard's list, as it wrote it on spark-5512 with Excalibur's vLLM on the GPU."""
    return f"# {stamp(age_s)} {rows.count(chr(10))} foreign\n{rows}"


VLLM = "4132364\tmcpvar-vllm\t23221 MiB\tVLLM::EngineCore\n"


@pytest.fixture
def spark(tmp_path: Path):
    """A lock directory with the guard's list, and the kernel's memory account as a file."""

    def admit(host: str, tenants: str | None, gib: float | None, *args: str,
              poll: str | None = "0.1") -> tuple[int, str, str]:  # fmt: skip
        if tenants is not None:
            (tmp_path / "foreign-gpu").write_text(tenants)
        if gib is not None:
            write(gib)
        env = {**os.environ, "AUTAN_GPU_LOCK": str(tmp_path / "gpu.lock"),
               "AUTAN_MEMINFO": str(tmp_path / "meminfo"), "AUTAN_BENCH_HOST": host,
               "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")}  # fmt: skip
        env.pop("AUTAN_ADMIT_POLL_S", None)
        if poll is not None:
            env["AUTAN_ADMIT_POLL_S"] = poll
        out = subprocess.run([str(SCRIPT), *args], env=env, capture_output=True, text=True,
                             timeout=60, check=False)  # fmt: skip
        summary = (tmp_path / "summary").read_text() if (tmp_path / "summary").exists() else ""
        return out.returncode, out.stdout + out.stderr, summary

    def write(gib: float) -> None:
        tmp = tmp_path / "meminfo.tmp"
        tmp.write_text(f"MemTotal: {128 << 20} kB\nMemAvailable: {int(gib * (1 << 20))} kB\n")
        tmp.replace(tmp_path / "meminfo")

    admit.write = write
    admit.dir = tmp_path
    return admit


def commands(out: str) -> list[str]:
    """The lines the job's log would obey: "::" at a line's start, "##[" anywhere."""
    return [line for line in out.splitlines() if line.startswith("::") or "##[" in line]


def test_a_foreign_tenant_is_not_a_refusal_off_the_timed_hosts(spark):
    code, out, summary = spark("spark-5512", guard_list(VLLM), 40, "4")
    assert code == 0
    assert not commands(out)
    # Named in the job's summary all the same.
    assert "on spark-5512: VLLM::EngineCore (mcpvar-vllm, 23221 MiB)" in summary


@pytest.mark.parametrize("host", ["spark-5359", "spark-5359-main"])
def test_a_foreign_tenant_refuses_a_gpu_step_on_a_timed_host(spark, host):
    code, out, summary = spark(host, guard_list(VLLM), 100, "4")
    assert code == 1
    assert f"::error::GPU taken by a foreign tenant on {host}, which runs timed benchmarks" in out
    assert "mcpvar-vllm" in summary


def test_a_timed_host_without_tenants_admits(spark):
    code, _, summary = spark("spark-5359", guard_list(""), 100, "4")
    assert code == 0
    assert "on spark-5359: none" in summary


@pytest.mark.parametrize(
    ("listing", "said"),
    [
        (guard_list(VLLM, age_s=300), "s old"),
        (f"# {stamp()} nvidia-smi failed\n", "the guard's nvidia-smi failed"),
    ],
)
def test_a_stale_or_failed_list_is_unknown_and_refuses_nothing(spark, listing, said):
    # A tenant the guard saw 5 minutes ago may be long gone: no refusal on its word.
    code, _, summary = spark("spark-5359", listing, 100, "4")
    assert code == 0
    assert said in summary
    assert ": none" not in summary


def test_the_guard_s_own_list_is_read_as_it_writes_it(spark, monkeypatch):
    # The producer, not a copy of its format: autan-gpu-guard's publish().
    monkeypatch.syspath_prepend(str(ROOT / "lock"))
    loader = importlib.machinery.SourceFileLoader("guard", str(GUARD))
    spec = importlib.util.spec_from_loader("guard", loader)
    guard = importlib.util.module_from_spec(spec)
    loader.exec_module(guard)
    monkeypatch.setattr(guard, "FOREIGN", spark.dir / "foreign-gpu")
    guard.publish({4132364: {"where": "mcpvar-vllm", "mib": "23221", "name": "VLLM::EngineCore"}})
    code, _, summary = spark("spark-5359", None, 100, "4")
    assert code == 1
    assert "VLLM::EngineCore (mcpvar-vllm, 23221 MiB)" in summary
    guard.publish(None)
    code, _, summary = spark("spark-5359", None, 100, "4")
    assert code == 0
    assert "unknown (the guard's nvidia-smi failed at " in summary
    guard.publish({})
    code, _, summary = spark("spark-5359", None, 100, "4")
    assert code == 0
    assert summary.splitlines()[-2].endswith(": none")


@pytest.mark.parametrize("host", ["spark-5512##[error]injected", "spark-5512\n::error::injected"])
def test_a_host_name_from_the_job_never_becomes_a_workflow_command(spark, host):
    code, out, summary = spark(host, guard_list(""), 100, "4")
    assert code == 0
    assert not commands(out)
    assert "unnamed-host" in summary


@pytest.mark.parametrize(("gib", "code"), [(28, 0), (27.999, 1)])
def test_the_step_needs_the_floor_plus_its_declared_need(spark, gib, code):
    # 24 GiB floor + 4 GiB declared: exactly 28 GiB passes, a KiB less waits, then fails.
    got, out, _ = spark("spark-5512", guard_list(""), gib, "4", "1")
    assert got == code
    if code:
        assert "::error::WAITING FOR MEMORY timed out after 1 s on spark-5512" in out
        assert "Not this change's fault" in out


@pytest.mark.parametrize(("need", "gib", "code"), [("010", 34, 0), ("010", 33.9, 1), ("08", 32, 0)])
def test_a_leading_zero_is_decimal(spark, need, gib, code):
    assert spark("spark-5512", guard_list(""), gib, need, "1")[0] == code


def test_the_wait_never_exceeds_its_bound(spark):
    # The default poll is 10 s; a 1 s bound still ends within about 1 s.
    start = time.monotonic()
    code, _, _ = spark("spark-5512", guard_list(""), 10, "4", "1", poll=None)
    assert code == 1
    assert time.monotonic() - start < 4


def test_the_step_waits_for_memory_and_starts_once_it_is_back(spark):
    spark.write(10)
    timer = threading.Timer(1.0, spark.write, args=(30,))
    timer.start()
    start = time.monotonic()
    try:
        code, out, _ = spark("spark-5512", guard_list(""), None, "4", "30")
    finally:
        timer.cancel()
    assert code == 0
    assert "WAITING FOR MEMORY on spark-5512: 10.0 GiB available, needs 28 GiB" in out
    assert time.monotonic() - start < 10


def test_an_unknown_tenant_list_is_said_and_does_not_refuse(spark):
    code, _, summary = spark("spark-5359", None, 100, "4")
    assert code == 0
    assert "unknown (no list from autan-gpu-guard)" in summary


@pytest.mark.parametrize("args", [["x"], ["4", "-1"], [], ["99999"]])
def test_malformed_arguments_are_refused(spark, args):
    code, _, _ = spark("spark-5512", guard_list(""), 100, *args)
    assert code == 2
