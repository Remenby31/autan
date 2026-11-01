import random
import statistics


def summarize(samples: list[float]) -> dict[str, float | int]:
    # Quartiles by linear interpolation between order statistics (Hyndman & Fan 1996, type 7).
    if len(samples) < 2:
        msg = f"at least two samples are needed, got {len(samples)}"
        raise ValueError(msg)
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    return {
        "n": len(samples),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean,
        "min": min(samples),
        "max": max(samples),
    }


def median_ci(
    samples: list[float], level: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the median (Efron & Tibshirani 1993, ch. 13).

    The generator is seeded, so the same samples always give the same interval.
    """
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples))) for _ in range(resamples)
    )
    tail = (1.0 - level) / 2.0
    return medians[int(tail * resamples)], medians[int((1.0 - tail) * resamples) - 1]
