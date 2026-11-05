from dataclasses import dataclass
from typing import Any

from autan_bench.errors import BenchError, Code
from autan_bench.stats import median_ci

DEFAULT_THRESHOLD = 0.05
# A machine is fit to measure when every STREAM benchmark reaches this share of its baseline
# (the criterion of #404): a regression gate's 5 % passes a machine a few % slow.
HEALTH_MINIMUM = 0.97
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


def _matched(current: dict[str, Any], baseline: dict[str, Any]) -> list[tuple[dict, dict]]:
    """Each baseline benchmark with the result's, once both are shown comparable: same GPU,
    configuration, and each benchmark's metric, unit and direction."""
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
    pairs = []
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
        pairs.append((now, base))
    return pairs


def health(
    current: dict[str, Any], baseline: dict[str, Any], minimum: float = HEALTH_MINIMUM
) -> list[tuple[str, float, bool]]:
    """Each baseline benchmark's share of its baseline median in the result, in the metric's
    direction (1 = as fast), and whether it reaches ``minimum``: the machine's health, not a
    regression verdict (docs/infra/bench.md)."""
    rows = []
    for now, base in _matched(current, baseline):
        value, reference = now["stats"]["median"], base["stats"]["median"]
        share = value / reference if base["better"] == "higher" else reference / value
        rows.append((base["name"], share, share >= minimum - _ROUND_OFF))
    return rows


def compare(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> list[Row]:
    """Judge every baseline benchmark: ok, regression, or inconclusive (see the verdicts above)."""
    rows = []
    for now, base in _matched(current, baseline):
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
