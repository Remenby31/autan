import contextlib
import fcntl
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from cfdbench.errors import BenchError

DEFAULT_LOCK = Path("/tmp/cfd-gpu.lock")
IDLE_UTILIZATION_PCT = 5.0

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
        raise BenchError("NVIDIA_SMI_FAILED", f"nvidia-smi {' '.join(args)}: {exc}") from exc
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
    compute_apps: int


def sample(gpu_index: int = 0) -> GpuSample:
    rows = _nvidia_smi(
        f"--id={gpu_index}", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"
    )
    if len(rows) != 1:
        raise BenchError("NVIDIA_SMI_FAILED", f"expected one GPU row, got {rows!r}")
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
    with open(path, "a") as handle:
        start = time.monotonic()
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"cfdbench: waiting for {path}", file=sys.stderr, flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield time.monotonic() - start
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_idle(gpu_index: int = 0, samples: int = 5, interval_s: float = 0.2) -> dict[str, float | int]:
    """Fail with ``GPU_BUSY`` unless no compute process runs and utilization stays low."""
    seen = []
    for i in range(samples):
        if i:
            time.sleep(interval_s)
        seen.append(sample(gpu_index))
    apps = max(s.compute_apps for s in seen)
    util = max((s.utilization_pct or 0.0) for s in seen)
    if apps or util > IDLE_UTILIZATION_PCT:
        raise BenchError(
            "GPU_BUSY",
            f"GPU {gpu_index} is not idle: {apps} compute process(es), utilization up to {util:.0f} %",
        )
    return {"samples": samples, "compute_apps_max": apps, "utilization_pct_max": util}


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

    def __enter__(self) -> "Sampler":
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
            raise BenchError("NO_SAMPLES", "the run ended before any clock sample was taken")

        def values(name: str) -> list[float]:
            return [v for s in self.samples if (v := getattr(s, name)) is not None]

        reasons = 0
        for s in self.samples:
            reasons |= s.event_reasons or 0
        return {
            "samples": len(self.samples),
            "sm_mhz": _spread(values("sm_mhz")),
            "sm_max_mhz": max(values("sm_max_mhz"), default=None),
            "power_w": _spread(values("power_w")),
            "temperature_c": _spread(values("temperature_c")),
            "clocks_event_reasons": f"0x{reasons:016x}",
            "compute_apps_max": max(s.compute_apps for s in self.samples),
        }
