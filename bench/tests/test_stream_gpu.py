import json
import os
from pathlib import Path

import pytest
from autan_bench.cli import main

# DGX Spark datasheet: 128 GB LPDDR5x, 256-bit, 273 GB/s.
GB10_THEORETICAL_GBPS = 273.0

pytestmark = pytest.mark.gpu


@pytest.fixture
def stream_binary() -> Path:
    path = os.environ.get("AUTAN_BENCH_STREAM")
    if not path:
        if os.environ.get("AUTAN_BENCH_REQUIRE_GPU"):
            pytest.fail("AUTAN_BENCH_REQUIRE_GPU is set but AUTAN_BENCH_STREAM is not")
        pytest.skip("set AUTAN_BENCH_STREAM to the built stream binary to run GPU tests")
    return Path(path)


def test_stream_run_produces_a_valid_physical_result(stream_binary, tmp_path):
    out = tmp_path / "stream.json"
    repo = Path(__file__).resolve().parents[2]
    argv = ["stream", "--binary", str(stream_binary), "--out", str(out), "--repo", str(repo)]
    assert main([*argv, "--elements", str(1 << 25), "--warmup", "3", "--reps", "20"]) == 0
    doc = json.loads(out.read_text())
    assert [b["name"] for b in doc["benchmarks"]] == [
        "stream.copy",
        "stream.scale",
        "stream.add",
        "stream.triad",
    ]
    for b in doc["benchmarks"]:
        assert b["stats"]["n"] == 20
        assert b["stats"]["median"] > 0
        if doc["provenance"]["gpu"]["name"] == "NVIDIA GB10":
            assert b["stats"]["max"] <= GB10_THEORETICAL_GBPS
    assert main(["validate", str(out)]) == 0
    assert main(["compare", str(out), str(out)]) == 0


def test_stream_binary_rejects_bad_arguments(stream_binary, tmp_path, capsys):
    out = tmp_path / "stream.json"
    assert (
        main(["stream", "--binary", str(stream_binary), "--out", str(out), "--elements", "3"]) == 2
    )
    assert "error[BINARY_FAILED]" in capsys.readouterr().err
    assert not out.exists()
