import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CI = Path(__file__).parents[1] / "ci"
TRACKED = "codegen.flux.time_per_cell"


def git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, list[str]]:
    """A main branch: the last commit that passed the gate, then 8 commits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    shas = []
    for k in range(9):
        git(repo, "commit", "-q", "--allow-empty", "-m", f"c{k}")
        shas.append(git(repo, "rev-parse", "HEAD"))
    return repo, shas


# The probes are the external boundary (a GPU measurement on the Spark): a fake answers from a
# table of verdicts and writes the result file a real probe would.
FAKE_MEASURE = """#!{python}
import json, pathlib, sys
verdicts = json.loads(pathlib.Path({verdicts!r}).read_text())
sha, out = sys.argv[1], pathlib.Path(sys.argv[2])
with open({log!r}, "a") as log:
    log.write(sha + "\\n")
code, median = verdicts[sha]
out.mkdir(parents=True, exist_ok=True)
doc = {{"benchmarks": [{{"name": {tracked!r}, "stats": {{"median": median}}}}]}}
(out / "codegen-hex-n128-fp32.json").write_text(json.dumps(doc))
sys.exit(code)
"""


def bisect(tmp_path, repo, good, bad, verdicts, compare_exit):
    fake = tmp_path / "measure"
    log = tmp_path / "probes.log"
    (tmp_path / "verdicts.json").write_text(json.dumps(verdicts))
    fake.write_text(FAKE_MEASURE.format(python=sys.executable, verdicts=str(tmp_path / "verdicts.json"),
                                        log=str(log), tracked=TRACKED))  # fmt: skip
    fake.chmod(0o755)
    compare = tmp_path / "compare"
    compare.write_text(f"#!/bin/sh\necho 'codegen.flux  +8.0%'\nexit {compare_exit}\n")
    compare.chmod(0o755)
    out = tmp_path / "bisect"
    cmd = [sys.executable, str(CI / "bisect.py"), "--good", good, "--bad", bad, "--out", str(out),
           "--compare", str(compare), "--", str(fake)]  # fmt: skip
    subprocess.run(cmd, cwd=repo, check=False)
    probed = log.read_text().split() if log.exists() else []
    return json.loads((out / "summary.json").read_text()), probed


def verdicts_from(shas: list[str], first_bad: int, **override: int) -> dict[str, list]:
    table = {s: [1 if k >= first_bad else 0, 2.0 if k >= first_bad else 1.0]
             for k, s in enumerate(shas)}  # fmt: skip
    for index, code in override.items():
        table[shas[int(index[1:])]][0] = code
    return table


def test_a_culprit_in_the_middle_of_eight_commits_takes_three_probes(tmp_path, history):
    repo, shas = history
    summary, probed = bisect(tmp_path, repo, shas[0], shas[8], verdicts_from(shas, 5), 1)
    assert summary["status"] == "culprit"
    assert summary["culprit"] == shas[5]
    assert summary["parent"] == shas[4]
    assert len(probed) == 3
    assert shas[8] not in probed


def test_a_shift_spread_over_commits_blames_none(tmp_path, history):
    repo, shas = history
    summary, _ = bisect(tmp_path, repo, shas[0], shas[8], verdicts_from(shas, 5), 0)
    assert summary["status"] == "spread"
    assert "culprit" not in summary
    assert (summary["last_passing"], summary["first_failing"]) == (shas[4], shas[5])


@pytest.mark.parametrize(
    ("code", "status"), [(4, "machine-unhealthy"), (2, "error")], ids=["stream", "lock-lost"]
)
def test_a_probe_that_cannot_measure_stops_the_bisection(tmp_path, history, code, status):
    # 4: STREAM failed three times; 2: any other failure, such as LOCK_TIMEOUT when the GPU lock
    # could not be taken again. Nothing after that point is measured or interpreted.
    repo, shas = history
    verdicts = verdicts_from(shas, 5, k4=code)
    summary, probed = bisect(tmp_path, repo, shas[0], shas[8], verdicts, 1)
    assert summary["status"] == status
    assert summary["at"] == shas[4]
    assert "culprit" not in summary
    assert "first_failing" not in summary
    assert probed[-1] == shas[4]


def test_the_gate_measures_again_and_lets_the_second_run_decide(tmp_path):
    # uv is faked at the boundary: compare answers from a queue, a measurement is counted.
    answers = tmp_path / "answers"
    runs = tmp_path / "runs"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f'case "$*" in *compare*) code=$(head -1 {answers}); sed -i.bak 1d {answers}; '
        "exit $code ;;\n"
        f"  *) echo run >> {runs} ;; esac\n"
    )
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    gate = [str(CI / "gate.sh"), "r.json", "b.json", "--", "uv", "run", "measure"]
    for first, second, expected in ((1, 0, 0), (1, 1, 1), (0, None, 0)):
        answers.write_text("".join(f"{c}\n" for c in (first, second) if c is not None))
        runs.unlink(missing_ok=True)
        assert subprocess.run(gate, env=env, check=False).returncode == expected
        assert runs.exists() == (first != 0)


def fake_gh(tmp_path: Path, prs: str) -> tuple[dict[str, str], Path]:
    calls = tmp_path / "gh-calls"
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir()
    gh.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\037" "$@" >> {calls}; printf "\\036" >> {calls}\n'
        'case "$1 $2" in\n'
        f'  "api repos/o/r/commits/"*) echo "{prs}" ;;\n'
        '  "issue create") cat > /dev/null; echo https://github.com/o/r/issues/9 ;;\n'
        '  "pr comment") cat > /dev/null ;;\n'
        "esac\n"
    )
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{gh.parent}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r"}
    return env, calls


def report(tmp_path, history, compare_exit, prs):
    repo, shas = history
    summary, _ = bisect(tmp_path, repo, shas[0], shas[8], verdicts_from(shas, 5), compare_exit)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"benchmarks": [{"name": TRACKED, "unit": "ns"}]}))
    env, calls = fake_gh(tmp_path, prs)
    cmd = [sys.executable, str(CI / "report_regression.py"), str(tmp_path / "bisect"),
           str(baseline), "https://ci/run/1"]  # fmt: skip
    assert subprocess.run(cmd, env=env, check=False).returncode == 0
    records = [r.split("\x1f")[:-1] for r in calls.read_text().split("\x1e")[:-1]]
    return summary, records


def test_a_culprit_gets_an_issue_and_a_comment_on_its_pull_request(tmp_path, history):
    _, records = report(tmp_path, history, 1, "42")
    issue = next(r for r in records if r[:2] == ["issue", "create"])
    assert issue[issue.index("--label") + 1] == "perf-regression"
    assert "#42" in issue[issue.index("--title") + 1]
    assert next(r for r in records if r[:2] == ["pr", "comment"])[:3] == ["pr", "comment", "42"]


def test_a_spread_shift_gets_an_issue_and_blames_no_pull_request(tmp_path, history):
    _, records = report(tmp_path, history, 0, "42")
    issue = next(r for r in records if r[:2] == ["issue", "create"])
    assert "no single culprit" in issue[issue.index("--title") + 1]
    assert not any(r[:2] == ["pr", "comment"] for r in records)
