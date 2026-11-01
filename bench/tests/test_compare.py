import pytest
from autan_bench.cli import main
from conftest import bench, make_result


def scaled(samples: list[float], factor: float) -> list[float]:
    return [x * factor for x in samples]


BASE = [100.0, 101.0, 99.0, 100.5, 99.5]


# Samples spread over ±40 %: a 6 % shift of the median stays inside the noise.
WIDE = [60.0, 70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0]


def test_compare_calls_a_shift_within_the_noise_inconclusive(write_json, capsys):
    base = write_json("base.json", make_result([bench("stream.triad", WIDE)]))
    now = make_result([bench("stream.triad", scaled(WIDE, 0.94))])
    assert main(["compare", str(write_json("now.json", now)), str(base)]) == 3
    assert "error[INCONCLUSIVE]" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("factor", "unit", "better", "exit_code"),
    [
        (1.0, "GB/s", "higher", 0),
        (0.95, "GB/s", "higher", 0),
        (0.949, "GB/s", "higher", 1),
        (1.30, "GB/s", "higher", 0),
        (1.05, "ns", "lower", 0),
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
        (
            make_result(
                [bench("stream.triad", BASE)], config={"elements": 1 << 25, "element_bytes": 8}
            ),
            "INCOMPATIBLE",
        ),
    ],
)
def test_compare_refuses_results_it_cannot_judge(write_json, capsys, current, code):
    base = write_json("base.json", make_result([bench("stream.triad", BASE)]))
    assert main(["compare", str(write_json("now.json", current)), str(base)]) == 2
    assert f"error[{code}]" in capsys.readouterr().err
