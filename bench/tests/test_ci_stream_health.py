import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import bench, make_result

SCRIPT = Path(__file__).parents[1] / "ci" / "stream-health.py"
BASELINE = Path(__file__).parents[1] / "baselines" / "gb10.json"
KERNELS = ("copy", "scale", "add", "triad")


def base_median(kernel: str) -> float:
    doc = json.loads(BASELINE.read_text())
    return next(b["stats"]["median"] for b in doc["benchmarks"] if b["name"] == f"stream.{kernel}")


def stream(write_json, shares: dict[str, float]) -> Path:
    """A STREAM result whose every kernel reads `share` of gb10.json's median (one sample)."""
    benchmarks = [bench(f"stream.{k}", [base_median(k) * shares.get(k, 1.0)] * 3) for k in KERNELS]
    return write_json("now.json", make_result(benchmarks))


def health(*args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          text=True, check=False)  # fmt: skip


@pytest.mark.parametrize(("share", "code"), [(1.0, 0), (0.97, 0), (0.9699, 4)])
def test_the_machine_is_fit_from_97_percent_of_the_baseline(write_json, share, code):
    out = health(stream(write_json, {"triad": share}), BASELINE)
    assert out.returncode == code
    assert ("LOW" in out.stdout) == bool(code)


def test_spark_5359_on_2026_11_02_was_not_fit_though_the_5_percent_gate_passed_it(write_json):
    # The STREAM medians of the cancelled bisection's first probe (#400): 3.9-4.7 % slow, which
    # the regression gate's 5 % let through, so every probe "confirmed" the loops' regression.
    measured = {"copy": 225.26, "scale": 225.12, "add": 226.31, "triad": 225.86}
    shares = {k: v / base_median(k) for k, v in measured.items()}
    assert all(0.95 < s < 0.97 for s in shares.values())
    assert health(stream(write_json, shares), BASELINE).returncode == 4


def test_a_custom_minimum_is_honoured(write_json):
    assert health(stream(write_json, {"add": 0.96}), BASELINE, "--min", "0.95").returncode == 0


def test_a_missing_benchmark_or_unreadable_file_is_an_error_not_a_verdict(write_json, tmp_path):
    partial = make_result([bench("stream.copy", [base_median("copy")] * 3)])
    assert health(write_json("partial.json", partial), BASELINE).returncode == 2
    (tmp_path / "broken.json").write_text("{")
    assert health(tmp_path / "broken.json", BASELINE).returncode == 2
    assert health(tmp_path / "absent.json", BASELINE).returncode == 2
