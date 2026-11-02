import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from autan_ui import create_app
from fastapi.testclient import TestClient

BENCH = Path(__file__).parents[3] / "bench"
RECORDED = BENCH / "tests" / "data" / "history"
LOCAL = "http://127.0.0.1"


def produced_history(tmp_path: Path) -> Path:
    """history.json as gpu-bench's history job writes it (scripts/sample-perf.sh): three commits of
    main, the middle one measured STREAM only, the newest blamed by a bisection."""
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
    assert "codegen.advection_atomic" not in doc["records"][1]["benchmarks"]
    assert doc["records"][-1]["regression"] == {
        "status": "culprit",
        "good": doc["records"][1]["commit"],
        "bad": doc["records"][-1]["commit"],
    }

    history = perf / "history.json"
    produced = json.loads(history.read_text())

    def refused(change: Callable[[dict[str, Any]], None]) -> bool:
        doc = json.loads(json.dumps(produced))
        change(doc)
        history.write_text(json.dumps(doc))
        return bool(client.get("/api/perf").status_code == 422)

    def copy(doc: dict[str, Any]) -> dict[str, Any]:
        return cast("dict[str, Any]", doc["records"][0]["benchmarks"]["stream.copy"])

    def timing(doc: dict[str, Any]) -> dict[str, Any]:
        return cast(
            "dict[str, Any]", doc["records"][0]["benchmarks"]["codegen.roe_atomic.time_per_cell"]
        )

    assert refused(lambda doc: doc.update(records={}))
    assert refused(lambda doc: copy(doc).pop("percent_of_peak"))
    assert refused(lambda doc: timing(doc).update(percent_of_peak=1.0))
    assert refused(lambda doc: doc["records"][0].update(regression={}))
    assert refused(lambda doc: doc["records"][0].update(regression=None))
    unset = TestClient(create_app(tmp_path / "runs", static=None), base_url=LOCAL)
    assert unset.get("/api/perf").status_code == 404
