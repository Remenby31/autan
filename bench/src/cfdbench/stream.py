import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

from cfdbench import gpu, provenance, result
from cfdbench.errors import BenchError
from cfdbench.stats import summarize


def _run_binary(binary: Path, args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run([str(binary), *args], capture_output=True, text=True)
    except OSError as exc:
        raise BenchError("BINARY_FAILED", f"{binary}: {exc}") from exc
    if proc.returncode != 0:
        raise BenchError("BINARY_FAILED", f"{binary} exited {proc.returncode}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BenchError("BINARY_FAILED", f"{binary} printed invalid JSON: {exc}") from exc


def run(
    binary: Path,
    repo: Path,
    *,
    elements: int,
    warmup: int,
    reps: int,
    block: int,
    lock: Path = gpu.DEFAULT_LOCK,
) -> dict[str, Any]:
    """Run the STREAM binary under the GPU lock and return a validated result document."""
    args = ["--elements", str(elements), "--warmup", str(warmup), "--reps", str(reps), "--block", str(block)]
    prov = provenance.collect(repo)
    with gpu.gpu_lock(lock) as waited:
        idle = gpu.require_idle()
        with gpu.Sampler() as sampler:
            raw = _run_binary(binary, args)
        during = sampler.summary()
    if during["compute_apps_max"] > 1:
        raise BenchError(
            "GPU_CONTENDED", f"{during['compute_apps_max']} compute processes shared the GPU during the run"
        )
    if raw.get("verified") is not True:
        raise BenchError("VERIFY_FAILED", "the binary did not verify its arrays")

    benchmarks = []
    for kernel in raw["kernels"]:
        bandwidth = [kernel["bytes_per_rep"] / t / 1e9 for t in kernel["seconds"]]
        benchmarks.append(
            {
                "name": f"stream.{kernel['name']}",
                "metric": "bandwidth",
                "unit": "GB/s",
                "better": "higher",
                "bytes_per_rep": kernel["bytes_per_rep"],
                "samples": bandwidth,
                "stats": summarize(bandwidth),
            }
        )
    doc = {
        "schema": result.SCHEMA_ID,
        "suite": "stream",
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "provenance": {
            **prov,
            "precision": "fp64",
            "cuda_runtime": raw["cuda_runtime"],
            "cuda_driver": raw["cuda_driver"],
            "gpu": raw["device"],
        },
        "conditions": {"lock": str(lock), "lock_wait_s": waited, "idle_check": idle, "during_run": during},
        "config": raw["config"],
        "benchmarks": benchmarks,
    }
    result.validate(doc)
    return doc
