import pytest
from autan_bench.cli import main
from conftest import bench, make_result


def scaled(samples: list[float], factor: float) -> list[float]:
    return [x * factor for x in samples]


BASE = [100.0, 101.0, 99.0, 100.5, 99.5]


@pytest.mark.parametrize(
    ("factor", "unit", "better", "exit_code"),
    [
        (1.0, "GB/s", "higher", 0),
        (0.951, "GB/s", "higher", 0),
        (0.949, "GB/s", "higher", 1),
        (1.30, "GB/s", "higher", 0),
        (1.049, "ns", "lower", 0),
        (1.051, "ns", "lower", 1),
        (0.70, "ns", "lower", 0),
    ],
)
def test_compare_fails_only_beyond_five_percent_worse(
    write_json, capsys, factor, unit, better, exit_code
):
    base = write_json(
        "base.json", make_result([bench("stream.triad", BASE, unit=unit, better=better)])
    )
    now = make_result([bench("stream.triad", scaled(BASE, factor), unit=unit, better=better)])
    assert main(["compare", str(write_json("now.json", now)), str(base)]) == exit_code
    assert ("error[REGRESSION]" in capsys.readouterr().err) == (exit_code == 1)


@pytest.mark.parametrize(
    ("current", "code"),
    [
        (make_result([bench("stream.copy", BASE)]), "MISSING_BENCHMARK"),
        (make_result([bench("stream.triad", BASE)], gpu="NVIDIA RTX PRO 6000"), "GPU_MISMATCH"),
        (make_result([bench("stream.triad", BASE, unit="ns", better="lower")]), "INCOMPATIBLE"),
        (make_result([bench("stream.triad", BASE)], config={"elements": 1 << 25}), "INCOMPATIBLE"),
    ],
)
def test_compare_refuses_results_it_cannot_judge(write_json, capsys, current, code):
    base = write_json("base.json", make_result([bench("stream.triad", BASE)]))
    assert main(["compare", str(write_json("now.json", current)), str(base)]) == 2
    assert f"error[{code}]" in capsys.readouterr().err
