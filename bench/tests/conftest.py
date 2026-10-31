import json
from pathlib import Path

import pytest
from autan_bench import result
from autan_bench.stats import summarize


def pytest_configure(config: pytest.Config) -> None:
    # Declared here, not in bench/pyproject.toml, so the marker exists whichever configuration
    # file pytest picks (CI runs these tests from the repository root).
    config.addinivalue_line(
        "markers", "gpu: needs an NVIDIA GPU and the built stream binary (AUTAN_BENCH_STREAM)"
    )


def bench(name: str, samples: list[float], *, unit: str = "GB/s", better: str = "higher") -> dict:
    metric = "bandwidth" if unit == "GB/s" else "time_per_cell_iteration"
    return {
        "name": name,
        "metric": metric,
        "unit": unit,
        "better": better,
        "samples": samples,
        "stats": summarize(samples),
    }


def make_result(
    benchmarks: list[dict],
    *,
    gpu: str = "NVIDIA GB10",
    dirty: bool = False,
    config: dict | None = None,
    image: str | None = "autan-dev@sha256:" + "0" * 64,
) -> dict:
    doc = {
        "schema": result.SCHEMA_ID,
        "suite": "stream",
        "created": "2026-10-31T12:00:00+00:00",
        "provenance": {
            "solver": {"commit": "0" * 40, "dirty": dirty},
            "binary_sha256": "0" * 64,
            "backend": "cuda",
            "precision": "fp64",
            "devices": {"kind": "cuda", "count": 1},
            "host": "test",
            "image": image,
            "image_revision": None,
            "runtime_version": 13000,
            "driver_version": 13000,
            "gpu": {
                "name": gpu,
                "uuid": "GPU-00000000-0000-0000-0000-000000000000",
                "compute_capability": "12.1",
                "sm_count": 48,
                "l2_bytes": 25165824,
                "memory_bytes": 128 * 2**30,
            },
        },
        "conditions": {
            "lock": "/tmp/cfd-gpu.lock",
            "lock_wait_s": 0.0,
            "idle_check": {"samples": 5, "compute_apps_max": 0, "utilization_pct_max": 0.0},
            "during_run": {
                "samples": 3,
                "busy_samples": 2,
                "sm_mhz": {"min": 2400.0, "median": 2400.0, "max": 2400.0},
                "sm_max_mhz": 3003.0,
                "power_w": None,
                "temperature_c": None,
                "clocks_event_reasons": "0x0000000000000000",
                "compute_apps_max": 1,
            },
        },
        "config": config or {"elements": 1 << 28, "element_bytes": 8},
        "benchmarks": benchmarks,
    }
    result.validate(doc)
    return doc


@pytest.fixture
def write_json(tmp_path: Path):
    def write(name: str, doc: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return path

    return write
