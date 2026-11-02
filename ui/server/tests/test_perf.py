import json
import subprocess
from pathlib import Path

from autan_ui import create_app
from fastapi.testclient import TestClient

BENCH = Path(__file__).parents[3] / "bench"
RECORDED = BENCH / "tests" / "data" / "history"
LOCAL = "http://127.0.0.1"


def produced_history(tmp_path: Path) -> Path:
    """history.json as gpu-bench's history job writes it, from two recorded bench artifacts."""
    out = tmp_path / "perf-history"
    subprocess.run([Path(__file__).parents[2] / "scripts" / "sample-perf.sh", out], check=True)
    return out


def test_the_perf_history_is_main_s_record_per_commit_as_a_percent_of_the_stream_peak(
    tmp_path: Path,
) -> None:
    perf = produced_history(tmp_path)
    client = TestClient(create_app(tmp_path / "runs", static=None, perf_root=perf), base_url=LOCAL)

    doc = client.get("/api/perf").json()

    stream = json.loads((BENCH / "baselines" / "gb10.json").read_text())
    peak = max(b["stats"]["median"] for b in stream["benchmarks"])
    assert doc["peak_GBps"] == peak
    newest = max(RECORDED.iterdir())
    measured = {
        b["name"]: b["stats"]["median"]
        for path in newest.glob("*.json")
        for b in json.loads(path.read_text())["benchmarks"]
    }
    last = doc["records"][-1]["benchmarks"]
    assert {name: b["median"] for name, b in last.items()} == measured
    for b in last.values():
        if b["unit"] == "GB/s":
            assert b["percent_of_peak"] == 100 * b["median"] / peak
    assert "codegen.advection_atomic" in doc["tracked"]

    history = perf / "history.json"
    history.write_text(json.dumps(json.loads(history.read_text()) | {"records": {}}))
    assert client.get("/api/perf").status_code == 422
    unset = TestClient(create_app(tmp_path / "runs", static=None), base_url=LOCAL)
    assert unset.get("/api/perf").status_code == 404
