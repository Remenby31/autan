from dataclasses import dataclass
from typing import Any

from autan_bench.errors import BenchError, Code
from autan_bench.stats import median_ci

DEFAULT_THRESHOLD = 0.05
# A change of exactly the threshold is tolerated; the ratio carries a few ulps of round-off.
_ROUND_OFF = 1e-12


# Verdicts: a median worse than the threshold is a regression only when the 95 % bootstrap
# intervals of the two medians are disjoint; with overlapping intervals it is inconclusive and
# the measurement is repeated (docs/infra/bench.md).
OK = "ok"
REGRESSION = "regression"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Row:
    name: str
    unit: str
    baseline: float
    current: float
    change: float
    baseline_ci: tuple[float, float]
    current_ci: tuple[float, float]
    verdict: str


def compare(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> list[Row]:
    """Judge every baseline benchmark: ok, regression, or inconclusive (see the verdicts above)."""
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
        base_ci, now_ci = median_ci(base["samples"]), median_ci(now["samples"])
        higher = base["better"] == "higher"
        separated = now_ci[1] < base_ci[0] if higher else now_ci[0] > base_ci[1]
        if worse - threshold <= _ROUND_OFF:
            verdict = OK
        else:
            verdict = REGRESSION if separated else INCONCLUSIVE
        rows.append(
            Row(
                name=base["name"],
                unit=base["unit"],
                baseline=base["stats"]["median"],
                current=now["stats"]["median"],
                change=change,
                baseline_ci=base_ci,
                current_ci=now_ci,
                verdict=verdict,
            )
        )
    return rows
