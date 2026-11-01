import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from autan_bench import gpu
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

# A gated binary (--ready): it records whether the GPU lock is held while it sets up and after
# the go line, then measures as FAKE_BINARY does.
GATE = """import fcntl, sys
def held():
    with open({lock!r}) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
seen = [held()]
print("reading the mesh", file=sys.stderr)
print("autan-bench: ready", file=sys.stderr, flush=True)
seen += [sys.stdin.readline().strip(), held()]
pathlib.Path({seen!r}).write_text(json.dumps(seen))
"""

FAKE_BINARY = """#!{python}
import json, pathlib, time
{gate}
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
        "setup_seconds": {"read": 1.5, "renumber": 0.25},
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


def fake_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """The kernel's count of runnable threads, observer included: /proc/stat, and on macOS
    `ps -A -M -o stat=`, which appends each thread's state after the truncated command (here
    with a sleeping process whose arguments look like a state, and a localised %CPU)."""
    stat_file = tmp_path / "proc-stat"
    stat_file.write_text(f"cpu  1 2 3 4\nprocs_running {count}\nprocs_blocked 0\n")
    monkeypatch.setattr(gpu, "PROC_STAT", stat_file)
    ps = tmp_path / "ps"
    running = "admin  4242   ??    9,0 R    31T   0:00.01   0:00.02 spin       R"
    sleeping = "admin  4243   ??    0,0 S    31T   0:00.00   0:00.00 x 1 R 2    S"
    ps.write_text(
        "#!/bin/sh\necho 'USER PID TT %CPU STAT PRI STIME UTIME COMMAND'\n"
        f"i=0; while [ $i -lt {count} ]; do echo '{running}'; i=$((i+1)); done\n"
        f"echo '{sleeping}'\n"
    )
    ps.chmod(ps.stat().st_mode | stat.S_IEXEC)


@pytest.fixture(params=[("linux", "cuda"), ("darwin", "metal")], ids=["cuda", "metal"])
def fake_gpu(request, tmp_path, monkeypatch):
    system, backend = request.param
    monkeypatch.setattr(sys, "platform", system)
    fake_runnable(tmp_path, monkeypatch, 3)
    busy = tmp_path / "busy"
    for tool, text in (("nvidia-smi", FAKE_SMI), ("ioreg", FAKE_IOREG)):
        script = tmp_path / tool
        script.write_text(text.format(busy=busy))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    def binary(verified: bool = True, gated: bool = False) -> Path:
        path = tmp_path / "kernel-bench"
        raw = tmp_path / "raw.json"
        raw.write_text(json.dumps(raw_output(verified, backend)))
        gate = (
            GATE.format(lock=str(tmp_path / "lock"), seen=str(tmp_path / "seen")) if gated else ""
        )
        path.write_text(FAKE_BINARY.format(python=sys.executable, busy=str(busy), raw=str(raw),
                                           gate=gate))  # fmt: skip
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
    assert doc["conditions"]["setup_seconds"] == {"read": 1.5, "renumber": 0.25}
    load = doc["conditions"]["host_load"]
    assert load["cpus"] >= 1
    assert min(load["before_1m"], load["after_1m"]) >= 0
    assert load["before_runnable"] == load["after_runnable"] == 2
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


def test_run_refuses_a_busy_host_when_asked(fake_gpu, tmp_path, monkeypatch, capsys):
    # The gate reads runnable threads now, not the load average: a high average left by a
    # campaign frozen a moment ago does not refuse the run.
    binary = fake_gpu()
    monkeypatch.setattr(os, "getloadavg", lambda: (100.0, 100.0, 100.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 20)
    out = tmp_path / "r.json"
    argv = ["run", "--suite", "codegen", "--binary", str(binary), "--precision", "fp32"]
    argv += ["--out", str(out), "--lock", str(tmp_path / "l"), "--max-load", "0.15"]
    assert main(argv) == 0
    assert json.loads(out.read_text())["conditions"]["host_load"]["before_1m"] == 100.0
    out.unlink()
    fake_runnable(tmp_path, monkeypatch, 5)
    assert main(argv) == 2
    assert "error[HOST_BUSY]" in capsys.readouterr().err
    assert not out.exists()


def test_runnable_threads_see_busy_processes():
    spin = [sys.executable, "-c", "while True: pass"]
    procs = [subprocess.Popen(spin) for _ in range(3)]
    try:
        time.sleep(0.2)
        assert gpu.runnable_threads() >= 3
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()


def test_a_gated_binary_sets_up_without_the_lock_and_measures_under_it(fake_gpu, tmp_path):
    out = tmp_path / "r.json"
    (tmp_path / "lock").touch()
    argv = ["run", "--suite", "codegen", "--binary", str(fake_gpu(gated=True)), "--precision",
            "fp32", "--out", str(out), "--lock", str(tmp_path / "lock"), "--ready", "30"]  # fmt: skip
    assert main(argv) == 0
    assert json.loads((tmp_path / "seen").read_text()) == [False, "go", True]
    conditions = json.loads(out.read_text())["conditions"]
    assert conditions["setup_outside_lock_s"] >= 0


def test_a_gated_binary_that_never_gets_ready_fails_without_the_lock(tmp_path, capsys):
    lock = tmp_path / "lock"
    for body, timeout in (("sys.exit(3)", "30"), ("time.sleep(60)", "1")):
        binary = tmp_path / "setup"
        binary.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n")
        binary.chmod(0o755)
        argv = ["run", "--suite", "s", "--binary", str(binary), "--precision", "fp32",
                "--out", str(tmp_path / "r.json"), "--lock", str(lock), "--ready", timeout]  # fmt: skip
        start = time.monotonic()
        assert main(argv) == 2
        assert time.monotonic() - start < 10
        assert "error[BINARY_FAILED]" in capsys.readouterr().err
        # The lock was never created, hence never taken.
        assert not lock.exists()
