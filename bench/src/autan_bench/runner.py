import datetime as dt
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from autan_bench import gpu, provenance, result
from autan_bench.errors import BenchError, Code
from autan_bench.stats import summarize

_log = logging.getLogger(__name__)


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


READY = "autan-bench: ready"


class _Gated:
    """A benchmark binary run with --gate: it sets up (CPU work only, no GPU context), prints
    READY on stderr and waits for one line on stdin before its timed work."""

    def __init__(self, binary: Path, args: list[str]) -> None:
        self.binary = binary
        try:
            self.proc = subprocess.Popen(
                [str(binary), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )  # fmt: skip
        except OSError as exc:
            raise BenchError(Code.BINARY_FAILED, f"{binary}: {exc}") from exc
        self.ready = threading.Event()
        self.stderr: list[str] = []
        self.stdout = ""
        self.readers = [
            threading.Thread(target=self._read_stderr, daemon=True),
            threading.Thread(target=self._read_stdout, daemon=True),
        ]
        for reader in self.readers:
            reader.start()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            if line.rstrip("\n") == READY:
                self.ready.set()
            else:
                self.stderr.append(line)

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        self.stdout = self.proc.stdout.read()

    def wait_ready(self, timeout_s: float) -> float:
        start = time.monotonic()
        while not self.ready.wait(0.1):
            if self.proc.poll() is not None:
                self._join()
                raise BenchError(
                    Code.BINARY_FAILED,
                    f"{self.binary} exited {self.proc.returncode} before it was ready: "
                    f"{''.join(self.stderr).strip()}",
                )
            if time.monotonic() - start > timeout_s:
                self.kill()
                raise BenchError(
                    Code.BINARY_FAILED, f"{self.binary} not ready after {timeout_s:.0f} s"
                )
        return time.monotonic() - start

    def go(self) -> dict[str, Any]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write("go\n")
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait()
        self._join()
        if self.proc.returncode != 0:
            raise BenchError(
                Code.BINARY_FAILED,
                f"{self.binary} exited {self.proc.returncode}: {''.join(self.stderr).strip()}",
            )
        try:
            return json.loads(self.stdout)
        except json.JSONDecodeError as exc:
            raise BenchError(
                Code.BINARY_FAILED, f"{self.binary} printed invalid JSON: {exc}"
            ) from exc

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._join()

    def _join(self) -> None:
        for reader in self.readers:
            reader.join(timeout=10)


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
    lock_timeout_s: float | None = None,
    max_load_per_cpu: float | None = None,
    ready_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run a benchmark binary under the GPU lock and return a validated result document.

    The binary prints one JSON object: ``revision`` {commit, dirty}, ``backend``,
    ``runtime_version``, ``driver_version``, ``device``, ``config``, ``verified`` and
    ``kernels``, each {name, bytes_per_rep, seconds[], optional cells, optional params}.
    With ``ready_timeout_s``, the binary is gated: started without the lock, it sets up (CPU
    work only: no GPU context before ``go``), prints ``autan-bench: ready`` on stderr and waits
    for a line on stdin; the lock covers only what follows.
    """
    gated = _Gated(binary, args) if ready_timeout_s is not None else None
    try:
        setup = gated.wait_ready(ready_timeout_s) if gated else None
        return _locked_run(binary, args, gated, setup, suite, precision, lock, lock_timeout_s,
                           max_load_per_cpu)  # fmt: skip
    finally:
        if gated:
            gated.kill()


def _locked_run(
    binary: Path, args: list[str], gated: _Gated | None, setup: float | None, suite: str,
    precision: str, lock: Path, lock_timeout_s: float | None, max_load_per_cpu: float | None,
) -> dict[str, Any]:  # fmt: skip
    with gpu.gpu_lock(lock, lock_timeout_s) as waited:
        gpu.require_campaigns_frozen(lock)
        idle = gpu.require_idle()
        before = gpu.host_load()
        per_cpu = before["runnable"] / before["cpus"]
        if max_load_per_cpu is not None and per_cpu > max_load_per_cpu:
            raise BenchError(
                Code.HOST_BUSY,
                f"{before['runnable']:g} runnable threads on {before['cpus']} CPUs exceed "
                f"{max_load_per_cpu} per CPU",
            )
        if per_cpu > gpu.LOAD_WARN_PER_CPU:
            _log.warning(
                "%g runnable threads on %d CPUs: other processes share the memory bandwidth",
                before["runnable"],
                before["cpus"],
            )
        with gpu.Sampler() as sampler:
            raw = gated.go() if gated else _run_binary(binary, args)
        during = sampler.summary()
        after = gpu.host_load()
    if (during["compute_apps_max"] or 0) > 1:
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
            **({"os_version": raw["os_version"]} if "os_version" in raw else {}),
        },
        "conditions": {
            "lock": provenance.path_label(lock),
            "lock_wait_s": waited,
            **({"setup_outside_lock_s": setup} if setup is not None else {}),
            "idle_check": idle,
            "host_load": {
                "cpus": before["cpus"],
                "before_runnable": before["runnable"],
                "after_runnable": after["runnable"],
                "before_1m": before["load_1m"],
                "after_1m": after["load_1m"],
            },
            "during_run": during,
            **({"setup_seconds": raw["setup_seconds"]} if "setup_seconds" in raw else {}),
        },
        "config": raw["config"],
        "benchmarks": [b for kernel in raw["kernels"] for b in _benchmarks(suite, kernel)],
    }
    result.validate(doc)
    return doc
