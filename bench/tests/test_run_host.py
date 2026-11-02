"""autan-bench run --suite host: a host loop benchmark (backend cpu), measured under the GPU lock
with the CPU campaigns frozen and the GPU left idle."""

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest
from autan_bench import gpu
from autan_bench.cli import main

# The binary and nvidia-smi are faked. The fake GPU reads idle unless `gpu-busy` exists: a host
# benchmark must find it idle throughout.
FAKE_SMI = """#!/bin/sh
if [ -e "{busy}" ]; then util=90; apps=1; else util=0; apps=0; fi
case "$*" in
  *query-compute-apps*) [ "$apps" = 1 ] && echo 4242; exit 0 ;;
  *) echo "$util, 210, 3003, 5.0, 45, 0x0000000000000000" ;;
esac
"""
# With HOST_BENCH_TOUCHES_GPU set, something works on the GPU while the host loops run.
FAKE_HOST_BINARY = """#!{python}
import os, pathlib, time
busy = pathlib.Path({busy!r})
if os.environ.get("HOST_BENCH_TOUCHES_GPU"):
    busy.touch()
time.sleep(0.7)
busy.unlink(missing_ok=True)
print(pathlib.Path({raw!r}).read_text())
"""
CORE = {"name": "Cortex-X925", "cpus": [9], "midr": "0x00000000410fd851", "max_khz": 3900000,
        "online_cpus": 20}  # fmt: skip


def raw_output() -> dict:
    return {
        "revision": {"commit": "a" * 40, "dirty": False},
        "backend": "cpu",
        "runtime_version": 130300,
        "driver_version": 0,
        "device": CORE,
        "config": {"mesh_sha256": "b" * 64, "n_cells": 1000, "cpus": "9"},
        "verified": True,
        "kernels": [
            {
                "name": "euler.roe_faces",
                "bytes_per_rep": 2_000_000,
                "cells": 1000,
                "params": {"strategy": "scatter"},
                "seconds": [0.010, 0.011, 0.012],
            },
        ],
    }


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    stat_file = tmp_path / "proc-stat"
    stat_file.write_text("cpu  1 2 3 4\nprocs_running 2\nprocs_blocked 0\n")
    monkeypatch.setattr(gpu, "PROC_STAT", stat_file)
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(busy=tmp_path / "gpu-busy"))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(raw_output()))
    binary = tmp_path / "host-bench"
    binary.write_text(
        FAKE_HOST_BINARY.format(
            python=sys.executable, raw=str(raw), busy=str(tmp_path / "gpu-busy")
        )
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    lock = tmp_path / "locks" / "gpu.lock"
    lock.parent.mkdir()
    lock.touch()
    return {"binary": binary, "lock": lock, "tmp": tmp_path}


def run(host, out: Path) -> int:
    argv = ["run", "--suite", "host", "--binary", str(host["binary"]), "--precision",
            "fp32+acc64", "--lock", str(host["lock"]), "--out", str(out), "--", "--cpu", "9"]  # fmt: skip
    return main(argv)


def test_a_host_benchmark_records_its_core_and_an_idle_gpu(host, tmp_path):
    out = tmp_path / "r.json"
    assert run(host, out) == 0
    doc = json.loads(out.read_text())
    assert doc["provenance"]["backend"] == "cpu"
    assert doc["provenance"]["gpu"] == CORE  # the pinned core, as the result schema's cpu_device
    during = doc["conditions"]["during_run"]
    assert during["busy_samples"] == 0
    assert during["compute_apps_max"] == 0
    names = [b["name"] for b in doc["benchmarks"]]
    assert names == ["host.euler.roe_faces", "host.euler.roe_faces.time_per_cell"]


def test_a_gpu_busy_before_a_host_benchmark_holds_it_back(host, tmp_path, capsys):
    (host["tmp"] / "gpu-busy").touch()
    assert run(host, tmp_path / "r.json") != 0
    assert "GPU_BUSY" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_a_gpu_busy_during_a_host_benchmark_fails_it(host, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOST_BENCH_TOUCHES_GPU", "1")
    assert run(host, tmp_path / "r.json") != 0
    assert "GPU_CONTENDED" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_a_host_benchmark_waits_for_the_cpu_campaigns_to_freeze(
    host, tmp_path, capsys, monkeypatch
):
    # A live campaign that never freezes: the host loops are never timed beside it (HOST_BUSY).
    campaigns = host["lock"].parent / "campaigns"
    campaigns.mkdir()
    (campaigns / "c356.state").write_text("running\n")
    (campaigns / "c356.heartbeat").touch()
    real = gpu.require_campaigns_frozen
    monkeypatch.setattr(gpu, "require_campaigns_frozen", lambda lock: real(lock, timeout_s=1.0))
    started = time.monotonic()
    assert run(host, tmp_path / "r.json") != 0
    assert time.monotonic() - started < 10
    assert "HOST_BUSY" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()
    # Frozen, the same run goes ahead.
    (campaigns / "c356.state").write_text("frozen\n")
    assert run(host, tmp_path / "r.json") == 0
