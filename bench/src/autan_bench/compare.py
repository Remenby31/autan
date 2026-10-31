from dataclasses import dataclass
from typing import Any

from autan_bench.errors import BenchError, Code

DEFAULT_THRESHOLD = 0.05


@dataclass(frozen=True)
class Row:
    name: str
    unit: str
    baseline: float
    current: float
    change: float
    regression: bool


def compare(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> list[Row]:
    """Compare medians of every baseline benchmark; a change worse than ``threshold`` is a regression."""
    gpu_now = current["provenance"]["gpu"]["name"]
    gpu_base = baseline["provenance"]["gpu"]["name"]
    if gpu_now != gpu_base:
        raise BenchError(
            Code.GPU_MISMATCH, f"result is from {gpu_now!r}, baseline from {gpu_base!r}"
        )
    if current["config"] != baseline["config"]:
        raise BenchError(
            Code.INCOMPATIBLE, f"configs differ: {current['config']!r} != {baseline['config']!r}"
        )
    by_name = {b["name"]: b for b in current["benchmarks"]}
    rows = []
    for base in baseline["benchmarks"]:
        now = by_name.get(base["name"])
        if now is None:
            raise BenchError(
                Code.MISSING_BENCHMARK, f"{base['name']} is in the baseline but not in the result"
            )
        for key in ("metric", "unit", "better"):
            if now[key] != base[key]:
                raise BenchError(
                    Code.INCOMPATIBLE, f"{base['name']}: {key} {now[key]!r} != {base[key]!r}"
                )
        change = now["stats"]["median"] / base["stats"]["median"] - 1.0
        worse = -change if base["better"] == "higher" else change
        rows.append(
            Row(
                name=base["name"],
                unit=base["unit"],
                baseline=base["stats"]["median"],
                current=now["stats"]["median"],
                change=change,
                regression=worse > threshold,
            )
        )
    return rows
