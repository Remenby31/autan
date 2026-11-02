"""The paired gate (#403): a change against its base, measured alternately in one hold of the lock.

Robust to the machine's state, which an absolute baseline is not (#400): both sides are measured
in the same conditions, minutes apart, so whatever slows the machine slows both. The runs come
in pairs, one of each side, each side first in turn. For each benchmark and pair k, r_k is the
change's median over the base's, inverted for a "lower is better" metric so that r < 1 always
means worse; R is the geometric mean of the r_k. A slowdown of the machine shared by both runs
of a pair cancels in r_k, and a linear drift, which favours the first run of each pair, cancels
between pairs up to 1/(number of pairs) of one run's drift. The 95 % interval is a percentile
bootstrap over both levels: each resample draws the pairs with replacement, then the samples
within each run of each drawn pair. Verdicts, as the absolute gate's transposed: a regression
when the change is more than ``threshold`` worse in the metric's own terms (R < 1 - threshold
for "higher is better", R < 1 / (1 + threshold) for "lower is better", as ``compare`` judges)
and the interval's upper bound < 1; inconclusive when R is that low but the interval contains
1; ok otherwise.
"""

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

from autan_bench.errors import BenchError, Code

SCHEMA_ID = "autan.bench-pair/1"
DEFAULT_THRESHOLD = 0.05
DEFAULT_PAIRS = 5
# The declared number (#403): fewer pairs are not a verdict, a non-passing INSUFFICIENT_PAIRS.
MIN_PAIRS = DEFAULT_PAIRS
RESAMPLES = 2000
SEED = 0
# A ratio of exactly the threshold is tolerated; the division carries a few ulps of round-off.
_ROUND_OFF = 1e-12

OK = "ok"
REGRESSION = "regression"
INCONCLUSIVE = "inconclusive"


def order(pairs: int) -> list[str]:
    """The side of each run, change then base, then base then change, and so on for `pairs` pairs, so that a linear drift of the
    machine weighs equally on both sides."""
    out: list[str] = []
    for i in range(pairs):
        out += ["change", "base"] if i % 2 == 0 else ["base", "change"]
    return out


@dataclass(frozen=True)
class Row:
    name: str
    unit: str
    better: str
    ratio: float
    ci: tuple[float, float]
    verdict: str
    change_median: float
    base_median: float
    change_run_cv: float
    base_run_cv: float


def _pair_ratio(change: list[float], base: list[float], better: str) -> float:
    c, b = statistics.median(change), statistics.median(base)
    if not (math.isfinite(c) and math.isfinite(b) and c > 0 and b > 0):
        raise BenchError(Code.NON_POSITIVE_MEDIAN, f"medians {c!r} and {b!r}: no ratio")
    return c / b if better == "higher" else b / c


def _geomean(ratios: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(r) for r in ratios))


def limit(better: str, threshold: float) -> float:
    """The lowest R that passes: the change at most ``threshold`` worse in the metric's terms
    (a time per cell 5 % longer is R = 1/1.05, not 0.95)."""
    return 1.0 - threshold if better == "higher" else 1.0 / (1.0 + threshold)


def _run_cv(runs: list[list[float]]) -> float:
    """Coefficient of variation of the run medians: the noise between runs."""
    medians = [statistics.median(r) for r in runs]
    mean = statistics.fmean(medians)
    return statistics.stdev(medians) / mean if len(medians) > 1 and mean else 0.0


def judge(name: str, unit: str, better: str, change_runs: list[list[float]],
          base_runs: list[list[float]], threshold: float = DEFAULT_THRESHOLD,
          resamples: int = RESAMPLES, seed: int = SEED,
          min_pairs: int = MIN_PAIRS) -> Row:  # fmt: skip
    """The verdict on one benchmark from each side's runs (lists of samples), pair k being
    ``change_runs[k]`` and ``base_runs[k]``."""
    if better not in ("higher", "lower"):
        raise BenchError(Code.INCOMPATIBLE, f"{name}: better is higher or lower, not {better!r}")
    if not change_runs or not all(change_runs) or not all(base_runs):
        raise BenchError(Code.TOO_FEW_RUNS, f"{name}: each side needs runs with samples")
    if len(change_runs) < min_pairs:
        raise BenchError(
            Code.INSUFFICIENT_PAIRS,
            f"{name}: {len(change_runs)} pair(s), the gate needs {min_pairs}",
        )
    if len(change_runs) != len(base_runs):
        raise BenchError(
            Code.INCOMPATIBLE,
            f"{name}: {len(change_runs)} change runs and {len(base_runs)} base runs do not pair",
        )
    pairs = list(zip(change_runs, base_runs, strict=True))
    ratio = _geomean([_pair_ratio(c, b, better) for c, b in pairs])
    rng = random.Random(seed)

    def resample() -> float:
        drawn = rng.choices(pairs, k=len(pairs))
        return _geomean([_pair_ratio(rng.choices(c, k=len(c)), rng.choices(b, k=len(b)), better)
                         for c, b in drawn])  # fmt: skip

    boot = sorted(resample() for _ in range(resamples))
    lo = boot[int(0.025 * (resamples - 1))]
    hi = boot[int(0.975 * (resamples - 1))]
    if ratio >= limit(better, threshold) - _ROUND_OFF:
        verdict = OK
    elif hi < 1.0:
        verdict = REGRESSION
    else:
        verdict = INCONCLUSIVE
    pooled_c = [x for r in change_runs for x in r]
    pooled_b = [x for r in base_runs for x in r]
    return Row(name, unit, better, ratio, (lo, hi), verdict, statistics.median(pooled_c),
               statistics.median(pooled_b), _run_cv(change_runs), _run_cv(base_runs))  # fmt: skip


def judge_all(
    change_docs: list[dict[str, Any]],
    base_docs: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Row]:
    """Every benchmark of the base's results, judged against the change's (result documents,
    one per run, in run order: the k-th of each side form pair k). All runs must come from one GPU and one config, and agree on each
    benchmark's metric, unit and direction; a benchmark missing from any run is an error."""
    if not change_docs or not base_docs:
        raise BenchError(Code.TOO_FEW_RUNS, "each side needs at least one run")
    docs = change_docs + base_docs
    first = base_docs[0]
    for doc in docs:
        gpu, gpu0 = doc["provenance"]["gpu"]["name"], first["provenance"]["gpu"]["name"]
        if gpu != gpu0:
            raise BenchError(Code.GPU_MISMATCH, f"runs on {gpu!r} and {gpu0!r}")
        if doc["config"] != first["config"]:
            raise BenchError(
                Code.INCOMPATIBLE, f"configs differ: {doc['config']!r} != {first['config']!r}"
            )

    def runs(side: list[dict[str, Any]], ref: dict[str, Any]) -> list[list[float]]:
        out = []
        for doc in side:
            match = [b for b in doc["benchmarks"] if b["name"] == ref["name"]]
            if not match:
                raise BenchError(Code.MISSING_BENCHMARK, f"{ref['name']} is missing from a run")
            for key in ("metric", "unit", "better"):
                if match[0][key] != ref[key]:
                    raise BenchError(
                        Code.INCOMPATIBLE,
                        f"{ref['name']}: {key} {match[0][key]!r} != {ref[key]!r}",
                    )
            out.append(match[0]["samples"])
        return out

    # Every benchmark's runs gathered before any is judged: a missing or changed benchmark is
    # reported as such, not as whatever the first judged benchmark would raise.
    gathered = [(ref, runs(change_docs, ref), runs(base_docs, ref)) for ref in first["benchmarks"]]
    return [
        judge(ref["name"], ref["unit"], ref["better"], change, base, threshold)
        for ref, change, base in gathered
    ]


def outcome(rows: list[Row]) -> str:
    """The worst verdict: regression, then inconclusive, then ok."""
    verdicts = {r.verdict for r in rows}
    return (
        REGRESSION if REGRESSION in verdicts else INCONCLUSIVE if INCONCLUSIVE in verdicts else OK
    )


def series_record(
    sides: list[str], lock_wait_s: float, hold_s: float, rows: list[Row], verdict: str
) -> dict[str, Any]:
    """A series as pair.json records it: its order, the lock's cost, and each benchmark's row."""
    return {
        "order": sides,
        "lock_wait_s": lock_wait_s,
        "hold_s": hold_s,
        "outcome": verdict,
        "rows": [
            {
                "name": r.name,
                "unit": r.unit,
                "better": r.better,
                "ratio": r.ratio,
                "ci": list(r.ci),
                "verdict": r.verdict,
                "change_median": r.change_median,
                "base_median": r.base_median,
                "change_run_cv": r.change_run_cv,
                "base_run_cv": r.base_run_cv,
            }
            for r in rows
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    """pair.json as a job summary: per series, the lock's cost and each benchmark's ratio."""
    verdict = {OK: "ok", REGRESSION: "**REGRESSION**", INCONCLUSIVE: "**INCONCLUSIVE**"}
    final = summary["outcome"]
    if final == INCONCLUSIVE and len(summary["series"]) > 1:
        final_text = "**INCONCLUSIVE: noise too high**"
    else:
        final_text = verdict[final]
    lines = [f"### Paired gate, {summary['suite']}: {final_text}", ""]
    for number, series in enumerate(summary["series"], 1):
        order = " ".join(side[0].upper() for side in series["order"])
        lines += [
            (
                f"Series {number}: {len(series['order'])} runs ({order}), lock waited "
                f"{series['lock_wait_s']:.0f} s, held {series['hold_s']:.0f} s."
            ),
            "",
            "| benchmark | change | base | ratio | 95 % CI | run CV change / base | verdict |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
        lines += [
            f"| {r['name']} | {r['change_median']:.4g} {r['unit']} | {r['base_median']:.4g} "
            f"{r['unit']} | {r['ratio']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | "
            f"{r['change_run_cv']:.2%} / {r['base_run_cv']:.2%} | {verdict[r['verdict']]} |"
            for r in series["rows"]
        ]
        lines.append("")
    return "\n".join(lines) + "\n"
