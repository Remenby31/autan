import contextlib
import dataclasses
import datetime as dt
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
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
        self.ready = threading.Event()
        # Setup is timed from the launch: processes started together are waited for in turn.
        self.launched = time.monotonic()
        self.ready_at: float | None = None
        self.stderr: list[str] = []
        self.stdout = ""
        self.readers: list[threading.Thread] = []
        try:
            # Its own session: the whole tree it starts is killed with it. Diagnostic bytes that
            # are not UTF-8 must not stop the readers, which would leave a pipe undrained.
            self.proc = subprocess.Popen(
                [str(binary), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True,
            )  # fmt: skip
        except OSError as exc:
            raise BenchError(Code.BINARY_FAILED, f"{binary}: {exc}") from exc
        try:
            self.readers = [
                threading.Thread(target=self._read_stderr, daemon=True),
                threading.Thread(target=self._read_stdout, daemon=True),
            ]
            for reader in self.readers:
                reader.start()
        except BaseException:
            self.kill()
            raise

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            if line.rstrip("\n") == READY:
                self.ready_at = time.monotonic()
                self.ready.set()
            else:
                self.stderr.append(line)

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        self.stdout = self.proc.stdout.read()

    def wait_ready(self, timeout_s: float) -> float:
        """Wait for the ready line, at most ``timeout_s`` from the launch; returns the setup's
        duration, from the launch to the ready line."""
        start = self.launched
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
        assert self.ready_at is not None
        if self.ready_at - start > timeout_s:
            self.kill()
            raise BenchError(Code.BINARY_FAILED, f"{self.binary} not ready after {timeout_s:.0f} s")
        return self.ready_at - start

    def go(self) -> dict[str, Any]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write("go\n")
            self.proc.stdin.close()
        except BrokenPipeError as exc:
            raise BenchError(
                Code.BINARY_FAILED, f"{self.binary} closed its stdin before 'go'"
            ) from exc
        self.proc.wait()
        # Helpers it left behind would keep the pipes open: the output is complete only once
        # the whole session is gone.
        self._kill_group()
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
        self._kill_group()
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self._join()

    def _kill_group(self) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)

    def _join(self) -> None:
        for reader in self.readers:
            reader.join()


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
    gated = None
    try:
        gated = _Gated(binary, args) if ready_timeout_s is not None else None
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
        try:
            return _measure(binary, args, gated, setup, suite, precision, lock, waited,
                            max_load_per_cpu)  # fmt: skip
        except BaseException:
            # Stopped before the lock is released: the binary must never run without it.
            if gated:
                gated.kill()
            raise


def _measure(
    binary: Path, args: list[str], gated: _Gated | None, setup: float | None, suite: str,
    precision: str, lock: Path, waited: float, max_load_per_cpu: float | None,
) -> dict[str, Any]:  # fmt: skip
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
    # The host's activity brackets the run itself: read right before it starts and right after it
    # returns, before the summaries and the after-load sampling (#400).
    activity_before = gpu.host_activity()
    with gpu.Sampler() as sampler:
        raw = gated.go() if gated else _run_binary(binary, args)
        activity_after = gpu.host_activity()
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
            **(
                {"host_activity": {"before": activity_before, "after": activity_after}}
                if activity_before is not None and activity_after is not None
                else {}
            ),
            **({"setup_seconds": raw["setup_seconds"]} if "setup_seconds" in raw else {}),
        },
        "config": raw["config"],
        "benchmarks": [b for kernel in raw["kernels"] for b in _benchmarks(suite, kernel)],
    }
    result.validate(doc)
    return doc


@dataclasses.dataclass(frozen=True)
class Series:
    """One paired series: the side of each run, its result, the lock's wait and hold."""

    sides: list[str]
    docs: list[dict[str, Any]]
    lock_wait_s: float
    hold_s: float


def pair_series(
    change: Path,
    base: Path,
    args: list[str],
    sides: list[str],
    *,
    suite: str,
    precision: str,
    lock: Path = gpu.DEFAULT_LOCK,
    lock_timeout_s: float | None = None,
    max_load_per_cpu: float | None = None,
    on_run: Callable[[int, str, dict[str, Any]], None] | None = None,
    ready_timeout_s: float | None = None,
) -> Series:
    """Run the change's and the base's binaries with the same arguments, in the order of
    ``sides``. Each run is the measurement :func:`run` makes, with its own campaign, idle and
    load checks. ``on_run(index, side, result)`` is called after each run.

    Ungated, the whole series is one hold of the GPU lock (#403), so no other job runs between
    the two sides. Gated (``ready_timeout_s``, binaries run with their gate), each pair is one
    hold: both of its processes set up outside the lock (at most these two alive), then the lock
    is taken, both runs go, one after the other, and it is released; the statistic is per pair,
    so other jobs between pairs widen the interval without biasing the ratio (#403)."""
    binaries = {"change": change, "base": base}
    if ready_timeout_s is None:
        with gpu.gpu_lock(lock, lock_timeout_s) as waited:
            start = time.monotonic()
            docs = []
            for i, side in enumerate(sides):
                # Only the first run waited for the lock; the others ran within the same hold.
                doc = _measure(binaries[side], args, None, None, suite, precision, lock,
                               waited if i == 0 else 0.0, max_load_per_cpu)  # fmt: skip
                docs.append(doc)
                # Kept as soon as measured: a later run that fails loses only itself.
                if on_run:
                    on_run(i, side, doc)
            return Series(sides, docs, waited, time.monotonic() - start)
    docs, waited_total, held_total = [], 0.0, 0.0
    for first in range(0, len(sides), 2):
        pair = sides[first : first + 2]
        gated: list[_Gated] = []
        try:
            # One at a time: if the second cannot start, the first is killed by the finally.
            gated.extend(_Gated(binaries[side], args) for side in pair)
            setups = [g.wait_ready(ready_timeout_s) for g in gated]
            with gpu.gpu_lock(lock, lock_timeout_s) as waited:
                start = time.monotonic()
                try:
                    for k, side in enumerate(pair):
                        doc = _measure(binaries[side], args, gated[k], setups[k], suite,
                                       precision, lock, waited if k == 0 else 0.0,
                                       max_load_per_cpu)  # fmt: skip
                        docs.append(doc)
                        if on_run:
                            on_run(first + k, side, doc)
                finally:
                    # Stopped before the lock is released: a binary never runs without it.
                    for g in gated:
                        g.kill()
                    held_total += time.monotonic() - start
            waited_total += waited
        finally:
            for g in gated:
                g.kill()
    return Series(sides, docs, waited_total, held_total)
