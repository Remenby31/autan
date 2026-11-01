import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "history"))
import history

RECORDED = HERE / "data" / "history"  # two gpu-bench artifacts as `gh run download` writes them
BASELINES = HERE.parent / "baselines"


def artifacts(tmp_path: Path, culprit: bool = False) -> Path:
    out = tmp_path / "artifacts"
    shutil.copytree(RECORDED, out)
    if culprit:
        newest = out / "41335764169"
        commit = json.loads((newest / "stream.json").read_text())["provenance"]["solver"]["commit"]
        (newest / "bisect").mkdir()
        summary = {"status": "culprit", "culprit": commit, "good": "a" * 40, "bad": commit}
        (newest / "bisect" / "summary.json").write_text(json.dumps(summary))
    return out


def test_each_measured_commit_is_one_record_with_its_tracked_metrics(tmp_path):
    h = history.build(artifacts(tmp_path), BASELINES)
    assert [r["commit"][:7] for r in h["records"]] == ["2399886", "cdb9344"]
    assert h["peak_GBps"] == max(
        b["stats"]["median"]
        for b in json.loads((BASELINES / "gb10.json").read_text())["benchmarks"]
    )
    rec = h["records"][1]
    stream = json.loads((RECORDED / "41335764169" / "stream.json").read_text())
    copy = next(b for b in stream["benchmarks"] if b["name"] == "stream.copy")
    assert rec["benchmarks"]["stream.copy"]["median"] == copy["stats"]["median"]
    assert rec["benchmarks"]["stream.copy"]["percent_of_peak"] == (
        100 * copy["stats"]["median"] / h["peak_GBps"]
    )
    assert "codegen.lsq_gradient_bounds.time_per_cell" in h["tracked"]
    assert rec["benchmarks"]["codegen.lsq_gradient_bounds.time_per_cell"]["unit"] == "ns"
    assert rec["host"] == "spark-5359"
    assert "regression" not in rec


def test_a_bisected_culprit_is_marked_in_the_data_and_the_page(tmp_path):
    out = tmp_path / "out"
    assert history.main([str(artifacts(tmp_path, culprit=True)), "--baselines", str(BASELINES),
                         "--out", str(out)]) == 0  # fmt: skip
    records = json.loads((out / "history.json").read_text())["records"]
    assert [("regression" in r) for r in records] == [False, True]
    rows = list(csv.DictReader((out / "history.csv").open()))
    assert {r["culprit"] for r in rows if r["commit"].startswith("cdb9344")} == {"True"}
    page = (out / "history.html").read_text()
    tracked = json.loads((out / "history.json").read_text())["tracked"]
    assert page.count("<figure>") == len(tracked)
    assert "culprit of a regression" in page


def test_files_that_are_not_results_are_ignored(tmp_path):
    out = artifacts(tmp_path)
    (out / "41330656690" / "notes.json").write_text("[]")
    (out / "empty").mkdir()
    assert len(history.build(out, BASELINES)["records"]) == 2
