import pytest
from autan_bench.stats import median_ci, summarize


def test_summary_statistics_follow_their_definitions():
    # Hyndman & Fan type 7 quartiles of 1..4: 1 + 0.75 (2 - 1), 2.5, 3 + 0.25 (4 - 3).
    s = summarize([4.0, 1.0, 3.0, 2.0])
    assert (s["q1"], s["median"], s["q3"], s["iqr"]) == (1.75, 2.5, 3.25, 1.5)
    assert (s["min"], s["max"], s["n"]) == (1.0, 4.0, 4)
    # Sample standard deviation of 9, 10, 11 is 1, so CV = 1 / 10.
    s = summarize([9.0, 10.0, 11.0])
    assert s["mean"] == 10.0
    assert s["stdev"] == pytest.approx(1.0, rel=1e-15)
    assert s["cv"] == pytest.approx(0.1, rel=1e-15)
    with pytest.raises(ValueError, match="two samples"):
        summarize([1.0])


def test_median_interval_is_reproducible_and_brackets_the_median():
    samples = [float(x) for x in (5, 1, 9, 3, 7, 2, 8, 4, 6, 10, 11)]
    low, high = median_ci(samples)
    # Every bootstrap median is a sample value or the mean of two, so the interval lies within
    # the sample range; it contains the sample median 6; the seeded generator repeats it.
    assert min(samples) <= low <= 6.0 <= high <= max(samples)
    assert median_ci(samples) == (low, high)
