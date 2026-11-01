import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci" / "changes.sh"
LAST = Path(__file__).parents[1] / "ci" / "last-measured.sh"


def git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "HOME": str(repo),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    out = subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    return out.stdout.decode().strip()


def commit(repo: Path, path: str) -> str:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"{path} {len(git(repo, 'log', '--oneline', '--all').splitlines())}\n")
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", path)
    return git(repo, "rev-parse", "HEAD")


def decide(repo: Path, event: str, draft: str = "false", before: str = "") -> dict[str, str]:
    out = subprocess.run(
        [str(SCRIPT), event, draft, before], cwd=repo, check=True, capture_output=True, text=True
    )
    return dict(line.split("=", 1) for line in out.stdout.splitlines())


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    commit(tmp_path, "README.md")
    return tmp_path


def pull_request_merge(repo: Path, branch: str) -> None:
    """Check out what GitHub tests for a pull request: its branch merged into the current base."""
    git(repo, "checkout", "-q", "--detach", "main")
    git(repo, "merge", "-q", "--no-ff", "--no-edit", branch)


def test_a_pull_request_that_merged_main_is_judged_on_its_own_files(repo):
    git(repo, "checkout", "-q", "-b", "ui-change")
    commit(repo, "ui/src/app.ts")
    git(repo, "checkout", "-q", "main")
    commit(repo, "core/src/mesh.cpp")
    commit(repo, "bench/src/autan_bench/gpu.py")
    git(repo, "checkout", "-q", "ui-change")
    git(repo, "merge", "-q", "--no-edit", "main")
    pull_request_merge(repo, "ui-change")
    assert decide(repo, "pull_request") == {"bench": "false", "harness": "false"}


def test_a_pull_request_is_measured_when_it_changes_measured_code(repo):
    git(repo, "checkout", "-q", "-b", "kernel")
    commit(repo, "HEAD")
    commit(repo, "kernels/flux.cu")
    pull_request_merge(repo, "kernel")
    assert decide(repo, "pull_request") == {"bench": "true", "harness": "false"}
    assert decide(repo, "pull_request", draft="true") == {"bench": "false", "harness": "false"}


def test_a_push_is_judged_against_the_last_measured_commit(repo):
    # A kernel change (A) whose run was cancelled while pending, then a documentation change (B):
    # B's run must still measure A.
    measured = git(repo, "rev-parse", "HEAD")
    commit(repo, "kernels/flux.cu")
    commit(repo, "docs/index.md")
    assert decide(repo, "push", before=measured) == {"bench": "true", "harness": "false"}
    measured = git(repo, "rev-parse", "HEAD")
    commit(repo, "docs/other.md")
    assert decide(repo, "push", before=measured) == {"bench": "false", "harness": "false"}


def fake_gh(tmp_path: Path, runs: list[dict], jobs: dict[int, str], fail: bool = False) -> dict:
    """gh is the external boundary: it answers with API-shaped JSON and applies the caller's
    --jq filter with jq, as `gh api --jq` does."""
    data = tmp_path / "api"
    data.mkdir()
    (data / "runs.json").write_text(json.dumps({"workflow_runs": runs}))
    for run_id, conclusion in jobs.items():
        body = {"jobs": [{"name": "changes", "conclusion": "success"},
                         {"name": "bench", "conclusion": conclusion}]}  # fmt: skip
        (data / f"jobs-{run_id}.json").write_text(json.dumps(body))
    gh = tmp_path / "bin" / "gh"
    gh.parent.mkdir()
    gh.write_text(
        "#!/bin/sh\n"
        + ("exit 1\n" if fail else "")
        + '[ "$1" = api ] && [ "$3" = --jq ] || { echo "unexpected: $*" >&2; exit 2; }\n'
        + f'case "$2" in\n  *workflows/gpu-bench.yml/runs*) f={data}/runs.json ;;\n'
        + f"  */runs/*/jobs) id=${{2%/jobs}}; f={data}/jobs-${{id##*/}}.json ;;\nesac\n"
        + 'jq -r "$4" "$f"\n'
    )
    gh.chmod(0o755)
    return {**os.environ, "PATH": f"{gh.parent}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r"}


def last_measured(env: dict[str, str]) -> str:
    out = subprocess.run([str(LAST)], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def run(run_id: int, sha: str, event: str = "push", branch: str = "main") -> dict:
    return {"id": run_id, "head_sha": sha * 40, "event": event, "head_branch": branch}


def test_the_last_measured_commit_is_the_newest_run_whose_bench_job_passed(tmp_path):
    # Newest first: a docs-only push (bench skipped), a pull request from main (it measures a
    # merge tree, not its head), then a measured push.
    runs = [run(4, "d"), run(3, "c", event="pull_request"), run(2, "b"), run(1, "a")]
    env = fake_gh(tmp_path, runs, {4: "skipped", 3: "success", 2: "success", 1: "success"})
    assert last_measured(env) == "b" * 40


def test_without_an_answer_from_the_api_the_push_is_measured(tmp_path):
    assert last_measured(fake_gh(tmp_path, [], {}, fail=True)) == ""


def test_without_a_readable_base_everything_runs(repo):
    commit(repo, "docs/index.md")
    assert decide(repo, "workflow_dispatch") == {"bench": "true", "harness": "true"}
    assert decide(repo, "push", before="0" * 40) == {"bench": "true", "harness": "true"}


@pytest.mark.parametrize(
    ("path", "bench", "harness"),
    [
        ("kernels/flux.cu", "true", "false"),
        ("io/gmsh.cpp", "true", "false"),
        ("docker/dev/Dockerfile", "true", "true"),
        ("bench/src/autan_bench/stats.py", "true", "true"),
        ("bench/stream/stream.cu", "true", "true"),
        ("bench/cmake/revision.cmake", "true", "true"),
        ("bench/baselines/gb10.json", "true", "true"),
        ("bench/CMakeLists.txt", "true", "true"),
        ("vv/src/autan_vv/meshes/families.py", "true", "false"),
        (".github/workflows/gpu-bench.yml", "true", "false"),
        ("bench/tests/test_run.py", "false", "true"),
        ("bench/ci/gate.sh", "false", "true"),
        ("bench/lock/autan-gpu-lock", "false", "true"),
        ("bench/campaign/autan-campaign", "false", "true"),
        ("docker/runner/entrypoint.sh", "false", "true"),
        ("docs/infra/bench.md", "false", "false"),
        ("ui/src/app.ts", "false", "false"),
    ],
)
def test_each_path_is_measured_tested_or_neither(repo, path, bench, harness):
    # Measured on the reference Spark: measured code, the image and the measurement method.
    # The harness's GPU tests alone (tests runner): its tests, CI scripts, lock and campaign tools.
    git(repo, "checkout", "-q", "-b", "change")
    commit(repo, path)
    pull_request_merge(repo, "change")
    assert decide(repo, "pull_request") == {"bench": bench, "harness": harness}
