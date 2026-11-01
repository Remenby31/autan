import json
import os
import stat
import sys
from pathlib import Path

import pytest
from autan_bench.cli import main

# The binary and the GPU telemetry tools are external processes; all are faked here. The fake
# GPU reads busy while the fake binary runs (Linux: 90 % and one compute process; macOS: 12 %, what
# ioreg shows under a bandwidth-bound Metal kernel), idle otherwise.
FAKE_IOREG = """#!/bin/sh
if [ -e "{busy}" ]; then util=12; else util=3; fi
echo '"PerformanceStatistics" = {{"Device Utilization %"='$util',"Tiler Utilization %"=2}}'
"""
FAKE_SMI = """#!/bin/sh
if [ -e "{busy}" ]; then util=90; apps=1; else util=0; apps=0; fi
case "$*" in
  *query-compute-apps*) [ "$apps" = 1 ] && echo 4242; exit 0 ;;
  *) echo "$util, 2400, 3003, 20.0, 45, 0x0000000000000000" ;;
esac
"""

FAKE_BINARY = """#!{python}
import pathlib, time
busy = pathlib.Path({busy!r})
busy.touch()
time.sleep(0.7)
busy.unlink()
print(pathlib.Path({raw!r}).read_text())
"""


DEVICES = {
    "cuda": {
        "name": "NVIDIA GB10",
        "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "compute_capability": "12.1",
        "sm_count": 48,
        "l2_bytes": 25165824,
        "memory_bytes": 128 * 2**30,
    },
    "metal": {
        "name": "Apple M4 Pro",
        "registry_id": "4294968922",
        "gpu_cores": 20,
        "recommended_max_working_set_bytes": 36 * 2**30,
        "memory_bytes": 48 * 2**30,
    },
}


def raw_output(verified: bool, backend: str) -> dict:
    return {
        "revision": {"commit": "a" * 40, "dirty": False},
        "backend": backend,
        "runtime_version": 13000,
        "driver_version": 13000,
        "device": DEVICES[backend],
        "config": {"mesh_sha256": "b" * 64, "n_cells": 1000},
        "verified": verified,
        "kernels": [
            {
                "name": "flux",
                "bytes_per_rep": 2_000_000_000,
                "cells": 1000,
                "params": {"strategy": "atomics"},
                "seconds": [0.010, 0.020, 0.040],
            }
        ],
    }


@pytest.fixture(params=[("linux", "cuda"), ("darwin", "metal")], ids=["cuda", "metal"])
def fake_gpu(request, tmp_path, monkeypatch):
    system, backend = request.param
    monkeypatch.setattr(sys, "platform", system)
    busy = tmp_path / "busy"
    for tool, text in (("nvidia-smi", FAKE_SMI), ("ioreg", FAKE_IOREG)):
        script = tmp_path / tool
        script.write_text(text.format(busy=busy))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    def binary(verified: bool = True) -> Path:
        path = tmp_path / "kernel-bench"
        raw = tmp_path / "raw.json"
        raw.write_text(json.dumps(raw_output(verified, backend)))
        path.write_text(FAKE_BINARY.format(python=sys.executable, busy=str(busy), raw=str(raw)))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    return binary


def test_run_derives_bandwidth_and_time_per_cell_from_kernel_times(fake_gpu, tmp_path):
    out = tmp_path / "r.json"
    argv = ["run", "--suite", "codegen", "--binary", str(fake_gpu()), "--precision", "fp32"]
    assert main([*argv, "--out", str(out), "--lock", str(tmp_path / "lock"), "--", "--x", "1"]) == 0
    doc = json.loads(out.read_text())
    bw, per_cell = doc["benchmarks"]
    assert bw["name"] == "codegen.flux"
    # 2e9 bytes in 10, 20, 40 ms: 200, 100, 50 GB/s.
    assert bw["samples"] == pytest.approx([200.0, 100.0, 50.0], rel=1e-12)
    assert per_cell["name"] == "codegen.flux.time_per_cell"
    # 1000 cells in 10, 20, 40 ms: 1e4, 2e4, 4e4 ns per cell.
    assert per_cell["samples"] == pytest.approx([1e4, 2e4, 4e4], rel=1e-12)
    assert (per_cell["better"], per_cell["unit"]) == ("lower", "ns")
    assert bw["params"] == per_cell["params"] == {"strategy": "atomics"}
    assert doc["provenance"]["precision"] == "fp32"
    assert doc["provenance"]["solver"] == {"commit": "a" * 40, "dirty": False}
    assert doc["config"]["n_cells"] == 1000
    during = doc["conditions"]["during_run"]
    assert during["busy_samples"] >= 1
    backend = doc["provenance"]["backend"]
    assert doc["provenance"]["devices"]["kind"] == backend
    # macOS lists no GPU processes: contention is recorded as unknown, not as zero.
    assert during["compute_apps_max"] == (1 if backend == "cuda" else None)
    assert (during["clocks_event_reasons"] is None) == (backend == "metal")


def test_run_refuses_an_unverified_result(fake_gpu, tmp_path, capsys):
    out = tmp_path / "r.json"
    argv = ["run", "--suite", "codegen", "--binary", str(fake_gpu(verified=False))]
    assert (
        main([*argv, "--precision", "fp32", "--out", str(out), "--lock", str(tmp_path / "l")]) == 2
    )
    assert "error[VERIFY_FAILED]" in capsys.readouterr().err
    assert not out.exists()
