import json

import pytest
from autan_bench.cli import main
from conftest import bench, make_result

# gb10.json's STREAM medians on 2026-11-02 (GB/s): pinned here, so that refreshing the baseline
# never changes what these tests check.
BASE = {"copy": 234.88, "scale": 236.24, "add": 235.53, "triad": 235.66}


def stream(shares: dict[str, float] | None = None, **kwargs) -> dict:
    """A STREAM result whose kernels read `share` of BASE (1 if not given)."""
    shares = shares or {}
    return make_result(
        [bench(f"stream.{k}", [v * shares.get(k, 1.0)] * 3) for k, v in BASE.items()], **kwargs
    )


@pytest.fixture
def baseline(write_json):
    return write_json("gb10.json", stream())


@pytest.mark.parametrize(("share", "code"), [(1.0, 0), (0.97, 0), (0.9699, 4)])
def test_a_machine_is_fit_from_97_percent_of_its_baseline(write_json, baseline, share, code):
    now = write_json("now.json", stream({"triad": share}))
    assert main(["health", str(now), str(baseline)]) == code


def test_spark_5359_on_2026_11_02_was_not_fit_though_the_5_percent_gate_passed_it(
    write_json, baseline
):
    # The STREAM medians of the cancelled bisection's first probe (#400): 3.9-4.7 % slow.
    measured = {"copy": 225.26, "scale": 225.12, "add": 226.31, "triad": 225.86}
    now = write_json("now.json", stream({k: v / BASE[k] for k, v in measured.items()}))
    assert main(["compare", str(now), str(baseline)]) == 0
    assert main(["health", str(now), str(baseline)]) == 4


def test_a_lower_is_better_metric_is_judged_in_its_direction(write_json):
    base = write_json(
        "base.json", make_result([bench("k.t", [100.0] * 3, unit="ns", better="lower")])
    )
    for time, code in ((103.0, 0), (103.2, 4)):  # 100/103 = 97.1 %, 100/103.2 = 96.9 %
        now = write_json(
            "now.json", make_result([bench("k.t", [time] * 3, unit="ns", better="lower")])
        )
        assert main(["health", str(now), str(base)]) == code


def test_a_custom_minimum_is_honoured(write_json, baseline):
    now = write_json("now.json", stream({"add": 0.96}))
    assert main(["health", str(now), str(baseline), "--min", "0.95"]) == 0


@pytest.mark.parametrize(
    "now",
    [
        stream(gpu="Apple M4 Pro"),
        stream(config={"elements": 1}),
        make_result([bench("stream.copy", [BASE["copy"]] * 3)]),
        make_result(
            [bench(f"stream.{k}", [v] * 3, unit="ns", better="lower") for k, v in BASE.items()]
        ),
    ],
    ids=["another GPU", "another config", "a missing benchmark", "another direction"],
)
def test_results_that_do_not_match_the_baseline_are_errors_not_verdicts(write_json, baseline, now):
    assert main(["health", str(write_json("now.json", now)), str(baseline)]) == 2


def test_unreadable_or_non_finite_files_are_errors_not_verdicts(tmp_path, write_json, baseline):
    text = json.dumps(stream()).replace(f"{BASE['triad'] * 1.0}", "Infinity", 1)
    (tmp_path / "inf.json").write_text(text)
    (tmp_path / "broken.json").write_text("{")
    empty = json.loads(json.dumps(stream()))
    empty["benchmarks"] = []
    for path in (tmp_path / "inf.json", tmp_path / "broken.json", tmp_path / "absent.json"):
        assert main(["health", str(path), str(baseline)]) == 2
    assert (
        main(["health", str(write_json("now.json", stream())), str(write_json("e.json", empty))])
        == 2
    )
