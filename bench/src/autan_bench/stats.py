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
