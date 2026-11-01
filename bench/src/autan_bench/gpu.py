import contextlib
import fcntl
import logging
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from autan_bench.errors import BenchError, Code

_log = logging.getLogger(__name__)

# One lock per Spark, outside sticky /tmp so that every uid can open it (docs/infra/bench.md).
# A Mac has a single user: the lock sits in the home directory.
DEFAULT_LOCK = (
    Path.home() / ".autan-gpu.lock"
    if sys.platform == "darwin"
    else Path("/home/admin/autan-gpu/gpu.lock")
)
IDLE_UTILIZATION_PCT = 5.0
BUSY_UTILIZATION_PCT = 50.0
# macOS "Device Utilization %" stays at 8-18 % while a bandwidth-bound Metal kernel saturates
# memory (M4 Pro, STREAM): there a sample is busy as soon as it is above the idle threshold.
BUSY_UTILIZATION_PCT_MACOS = IDLE_UTILIZATION_PCT

_GPU_FIELDS = (
    "utilization.gpu",
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
    "temperature.gpu",
    "clocks_event_reasons.active",
)


def _nvidia_smi(*args: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args], check=True, capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchError(Code.NVIDIA_SMI_FAILED, f"nvidia-smi {' '.join(args)}: {exc}") from exc
    return [line.strip() for line in out.splitlines() if line.strip()]


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class GpuSample:
    utilization_pct: float | None
    sm_mhz: float | None
    sm_max_mhz: float | None
    power_w: float | None
    temperature_c: float | None
    event_reasons: int | None
    # None where the platform does not list GPU processes (macOS): contention is then unchecked.
    compute_apps: int | None


def sample(gpu_index: int = 0) -> GpuSample:
    """One telemetry sample: nvidia-smi on Linux, the IOAccelerator registry on macOS."""
    return _sample_ioreg() if sys.platform == "darwin" else _sample_nvidia(gpu_index)


def _sample_ioreg() -> GpuSample:
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchError(Code.TELEMETRY_FAILED, f"ioreg: {exc}") from exc
    found = re.search(r'"Device Utilization %"=(\d+)', out)
    if found is None:
        raise BenchError(Code.TELEMETRY_FAILED, "ioreg reports no GPU device utilization")
    return GpuSample(
        utilization_pct=float(found.group(1)),
        sm_mhz=None,
        sm_max_mhz=None,
        power_w=None,
        temperature_c=None,
        event_reasons=None,
        compute_apps=None,
    )


def _sample_nvidia(gpu_index: int) -> GpuSample:
    rows = _nvidia_smi(
        f"--id={gpu_index}", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"
    )
    if len(rows) != 1:
        raise BenchError(Code.NVIDIA_SMI_FAILED, f"expected one GPU row, got {rows!r}")
    util, sm, sm_max, power, temp, reasons = (v.strip() for v in rows[0].split(","))
    apps = _nvidia_smi(f"--id={gpu_index}", "--query-compute-apps=pid", "--format=csv,noheader")
    return GpuSample(
        utilization_pct=_number(util),
        sm_mhz=_number(sm),
        sm_max_mhz=_number(sm_max),
        power_w=_number(power),
        temperature_c=_number(temp),
        event_reasons=int(reasons, 16) if reasons.startswith("0x") else None,
        compute_apps=len(apps),
    )


@contextlib.contextmanager
def gpu_lock(path: Path = DEFAULT_LOCK) -> Iterator[float]:
    """Hold an exclusive flock on ``path``; yields the seconds spent waiting for it."""
    # flock needs only a read-only descriptor, so the lock works on a file owned by another user.
    # O_CREAT is used only when the file is missing: in a sticky /tmp, fs.protected_regular
    # refuses O_CREAT on another user's existing file even for reading.
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    with os.fdopen(fd) as handle:
        start = time.monotonic()
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log.warning("waiting for %s", path)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield time.monotonic() - start
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_idle(
    gpu_index: int = 0, window: int = 5, interval_s: float = 0.2, timeout_s: float = 5.0
) -> dict[str, float | int]:
    """Wait until ``window`` consecutive samples show no compute process and low utilization.

    ``utilization.gpu`` is a trailing average, so a GPU released just before still reads busy
    for up to a second; a GPU that is not idle within ``timeout_s`` fails with ``GPU_BUSY``.
    """
    seen: list[GpuSample] = []
    deadline = time.monotonic() + timeout_s
    while True:
        current = sample(gpu_index)
        if current.utilization_pct is None:
            raise BenchError(
                Code.NVIDIA_SMI_FAILED,
                f"GPU {gpu_index} reports no utilization; cannot prove it idle",
            )
        seen.append(current)
        recent = seen[-window:]
        listed = [s.compute_apps for s in recent if s.compute_apps is not None]
        apps = max(listed) if listed else None
        util = max(s.utilization_pct or 0.0 for s in recent)
        if len(recent) == window and not apps and util <= IDLE_UTILIZATION_PCT:
            return {"samples": window, "compute_apps_max": apps, "utilization_pct_max": util}
        if time.monotonic() > deadline:
            raise BenchError(
                Code.GPU_BUSY,
                f"GPU {gpu_index} not idle after {timeout_s:.0f} s: {apps} compute process(es),"
                f" utilization up to {util:.0f} %",
            )
        time.sleep(interval_s)


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


@dataclass
class Sampler:
    """Samples clocks, power and temperature in the background while a benchmark runs."""

    gpu_index: int = 0
    interval_s: float = 0.2
    samples: list[GpuSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _error: BenchError | None = None

    def __enter__(self) -> Self:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(sample(self.gpu_index))
            except BenchError as err:
                self._error = err
                return
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        if not self.samples:
            raise BenchError(Code.NO_SAMPLES, "the run ended before any clock sample was taken")

        # Clocks, power and temperature describe the GPU while it works: setup and host-side
        # verification leave it idle at base clock for part of the run.
        threshold = BUSY_UTILIZATION_PCT_MACOS if sys.platform == "darwin" else BUSY_UTILIZATION_PCT
        busy = [s for s in self.samples if (s.utilization_pct or 0.0) > threshold]
        if not busy:
            raise BenchError(Code.NO_SAMPLES, "no clock sample was taken while the GPU was busy")

        def values(name: str) -> list[float]:
            return [v for s in busy if (v := getattr(s, name)) is not None]

        listed = [s.compute_apps for s in self.samples if s.compute_apps is not None]
        masks = [s.event_reasons for s in busy if s.event_reasons is not None]
        reasons = None
        for mask in masks:
            reasons = (reasons or 0) | mask
        return {
            "samples": len(self.samples),
            "busy_samples": len(busy),
            "busy_threshold_pct": threshold,
            "utilization_pct": _spread(
                [u for s in self.samples if (u := s.utilization_pct) is not None]
            ),
            "sm_mhz": _spread(values("sm_mhz")),
            "sm_max_mhz": max(values("sm_max_mhz"), default=None),
            "power_w": _spread(values("power_w")),
            "temperature_c": _spread(values("temperature_c")),
            "clocks_event_reasons": None if reasons is None else f"0x{reasons:016x}",
            "compute_apps_max": max(listed) if listed else None,
        }
