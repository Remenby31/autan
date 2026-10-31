import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from autan_bench import gpu, provenance, result
from autan_bench.errors import BenchError, Code
from autan_bench.stats import summarize


def _run_binary(binary: Path, args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run([str(binary), *args], check=False, capture_output=True, text=True)
    except OSError as exc:
        raise BenchError(Code.BINARY_FAILED, f"{binary}: {exc}") from exc
    if proc.returncode != 0:
        raise BenchError(
            Code.BINARY_FAILED, f"{binary} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BenchError(Code.BINARY_FAILED, f"{binary} printed invalid JSON: {exc}") from exc


def _benchmarks(suite: str, kernel: dict[str, Any]) -> list[dict[str, Any]]:
    name = f"{suite}.{kernel['name']}"
    extra = {"params": kernel["params"]} if "params" in kernel else {}
    bandwidth = [kernel["bytes_per_rep"] / t / 1e9 for t in kernel["seconds"]]
    out = [
        {
            "name": name,
            "metric": "bandwidth",
            "unit": "GB/s",
            "better": "higher",
            "bytes_per_rep": kernel["bytes_per_rep"],
            **extra,
            "samples": bandwidth,
            "stats": summarize(bandwidth),
        }
    ]
    if "cells" in kernel:
        per_cell = [t / kernel["cells"] * 1e9 for t in kernel["seconds"]]
        out.append(
            {
                "name": f"{name}.time_per_cell",
                "metric": "time_per_cell_iteration",
                "unit": "ns",
                "better": "lower",
                "cells_per_rep": kernel["cells"],
                **extra,
                "samples": per_cell,
                "stats": summarize(per_cell),
            }
        )
    return out


def run(
    binary: Path,
    args: list[str],
    *,
    suite: str,
    precision: str,
    lock: Path = gpu.DEFAULT_LOCK,
) -> dict[str, Any]:
    """Run a benchmark binary under the GPU lock and return a validated result document.

    The binary prints one JSON object: ``revision`` {commit, dirty}, ``backend``,
    ``runtime_version``, ``driver_version``, ``device``, ``config``, ``verified`` and
    ``kernels``, each {name, bytes_per_rep, seconds[], optional cells, optional params}.
    """
    with gpu.gpu_lock(lock) as waited:
        idle = gpu.require_idle()
        with gpu.Sampler() as sampler:
            raw = _run_binary(binary, args)
        during = sampler.summary()
    if during["compute_apps_max"] > 1:
        raise BenchError(
            Code.GPU_CONTENDED,
            f"{during['compute_apps_max']} compute processes shared the GPU during the run",
        )
    if raw.get("verified") is not True:
        raise BenchError(Code.VERIFY_FAILED, f"{binary} did not verify its results")
    doc = {
        "schema": result.SCHEMA_ID,
        "suite": suite,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "provenance": {
            **provenance.collect(binary, raw["revision"]),
            "backend": raw["backend"],
            "precision": precision,
            "devices": {"kind": raw["backend"], "count": 1},
            "runtime_version": raw["runtime_version"],
            "driver_version": raw["driver_version"],
            "gpu": raw["device"],
        },
        "conditions": {
            "lock": str(lock),
            "lock_wait_s": waited,
            "idle_check": idle,
            "during_run": during,
        },
        "config": raw["config"],
        "benchmarks": [b for kernel in raw["kernels"] for b in _benchmarks(suite, kernel)],
    }
    result.validate(doc)
    return doc
