import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "history"))
import history

RECORDED = HERE / "data" / "history"  # two gpu-bench artifacts as `gh run download` writes them
BASELINES = HERE.parent / "baselines"
OLD, NEW = "41330656690", "41335764169"


def commit_of(run_dir: Path) -> str:
    return json.loads((run_dir / "stream.json").read_text())["provenance"]["solver"]["commit"]


def remeasure(src: Path, dst: Path, commit: str | None = None, created: str | None = None,
              host: str | None = None) -> None:  # fmt: skip
    """The results of src as if the bench had measured them again (another commit, time, host)."""
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        doc = json.loads(path.read_text())
        if commit:
            doc["provenance"]["solver"]["commit"] = commit
        if created:
            doc["created"] = created
        if host:
            doc["provenance"]["host"] = host
        (dst / path.name).write_text(json.dumps(doc))


def artifacts(tmp_path: Path) -> tuple[Path, list[str]]:
    out = tmp_path / "artifacts"
    shutil.copytree(RECORDED, out)
    return out, [commit_of(RECORDED / OLD), commit_of(RECORDED / NEW)]


def build(out: Path, order: list[str]) -> dict:
    return history.build(out, order, BASELINES)


def test_each_measured_commit_is_one_record_with_its_tracked_metrics(tmp_path):
    out, order = artifacts(tmp_path)
    h = build(out, order)
    assert [r["commit"] for r in h["records"]] == order
    assert h["peak_GBps"] == max(
        b["stats"]["median"]
        for b in json.loads((BASELINES / "gb10.json").read_text())["benchmarks"]
    )
    rec = h["records"][1]
    stream = json.loads((RECORDED / NEW / "stream.json").read_text())
    copy = next(b for b in stream["benchmarks"] if b["name"] == "stream.copy")
    assert rec["benchmarks"]["stream.copy"]["median"] == copy["stats"]["median"]
    assert rec["benchmarks"]["stream.copy"]["percent_of_peak"] == (
        100 * copy["stats"]["median"] / h["peak_GBps"]
    )
    assert "codegen.lsq_gradient_bounds.time_per_cell" in h["tracked"]
    assert rec["benchmarks"]["codegen.lsq_gradient_bounds.time_per_cell"]["unit"] == "ns"
    assert rec["host"] == "spark-5359"
    assert "regression" not in rec


def test_commits_follow_main_and_a_remeasured_commit_keeps_its_latest_measurement(tmp_path):
    out, (old, new) = artifacts(tmp_path)
    # A rerun of the older commit's workflow, after the newer commit was measured.
    remeasure(
        RECORDED / OLD, out / "41400000000", created="2026-11-02T09:00:00+00:00", host="rerun"
    )
    remeasure(RECORDED / NEW, out / "41400000001", commit="f" * 40)
    h = build(out, [old, new])
    assert [(r["commit"], r["host"]) for r in h["records"]] == [(old, "rerun"), (new, "spark-5359")]


FAKE_MEASURE = """#!{python}
import json, pathlib, sys
sys.path.insert(0, {tests!r})
from test_history import remeasure
sha, out = sys.argv[1], pathlib.Path(sys.argv[2])
verdicts = json.loads(pathlib.Path({verdicts!r}).read_text())
remeasure(pathlib.Path({src!r}), out, commit=sha)
sys.exit(verdicts[sha])
"""


def test_a_bisection_adds_its_probes_and_marks_its_culprit(tmp_path):
    """Real producer: bench/ci/bisect.py over a main of 5 commits, the third one regressing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    shas = []
    for k in range(5):
        cmd = ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
               "-m", f"c{k}"]  # fmt: skip
        if k == 0:
            subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), *cmd], check=True)
        shas.append(subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip())  # fmt: skip
    order = subprocess.run(["git", "-C", str(repo), "rev-list", "--first-parent", "--reverse",
                            "HEAD"], capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert order == shas
    # The probes are the external boundary (a GPU measurement): a fake writes real result files.
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(json.dumps({s: int(k >= 2) for k, s in enumerate(shas)}))
    fake = tmp_path / "measure"
    fake.write_text(FAKE_MEASURE.format(python=sys.executable, tests=str(HERE), src=str(RECORDED / NEW),
                                        verdicts=str(verdicts)))  # fmt: skip
    fake.chmod(0o755)
    out = tmp_path / "artifacts"
    remeasure(RECORDED / OLD, out / "1", commit=shas[0])
    run = out / "2"
    remeasure(RECORDED / NEW, run, commit=shas[4])
    remeasure(run, run / "bisect" / shas[4])  # as gpu-bench seeds the bisection
    subprocess.run([sys.executable, str(HERE.parent / "ci" / "bisect.py"), "--good", shas[0],
                    "--bad", shas[4], "--out", str(run / "bisect"), "--compare", "false", "--",
                    str(fake)], cwd=repo, check=True)  # fmt: skip

    dest = tmp_path / "out"
    order_file = tmp_path / "order"
    order_file.write_text("\n".join(order) + "\n")
    assert history.main([str(out), "--order", str(order_file), "--baselines", str(BASELINES),
                         "--out", str(dest)]) == 0  # fmt: skip
    records = json.loads((dest / "history.json").read_text())["records"]
    assert [r["commit"] for r in records] == [shas[0], shas[1], shas[2], shas[4]]
    assert [r["commit"] for r in records if "regression" in r] == [shas[2]]
    rows = list(csv.DictReader((dest / "history.csv").open()))
    assert {r["culprit"] for r in rows if r["commit"] == shas[2]} == {"True"}
    page = (dest / "history.html").read_text()
    tracked = json.loads((dest / "history.json").read_text())["tracked"]
    assert page.count("<figure>") == len(tracked)
    assert page.count("culprit of a regression") == len(
        [n for n in tracked if n in records[2]["benchmarks"]]
    )


def test_artifact_strings_cannot_become_spreadsheet_formulas(tmp_path):
    out, order = artifacts(tmp_path)
    shutil.rmtree(out / NEW)
    remeasure(RECORDED / NEW, out / NEW, host="=HYPERLINK(1)")
    dest = tmp_path / "out"
    order_file = tmp_path / "order"
    order_file.write_text("\n".join(order))
    history.main([str(out), "--order", str(order_file), "--baselines", str(BASELINES),
                  "--out", str(dest)])  # fmt: skip
    hosts = {r["host"] for r in csv.DictReader((dest / "history.csv").open())}
    assert hosts == {"spark-5359", "'=HYPERLINK(1)"}


def test_files_that_are_not_results_are_ignored(tmp_path):
    out, order = artifacts(tmp_path)
    (out / OLD / "notes.json").write_text("[]")
    (out / "empty").mkdir()
    assert len(build(out, order)["records"]) == 2
