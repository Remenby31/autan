import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci" / "changes.sh"


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
    commit(repo, "kernels/flux.cu")
    pull_request_merge(repo, "kernel")
    assert decide(repo, "pull_request") == {"bench": "true", "harness": "false"}
    assert decide(repo, "pull_request", draft="true") == {"bench": "false", "harness": "false"}


def test_a_push_compares_with_the_commit_it_replaced(repo):
    before = git(repo, "rev-parse", "HEAD")
    commit(repo, "docker/dev/Dockerfile")
    assert decide(repo, "push", before=before) == {"bench": "true", "harness": "true"}
    before = git(repo, "rev-parse", "HEAD")
    commit(repo, "docs/index.md")
    assert decide(repo, "push", before=before) == {"bench": "false", "harness": "false"}


def test_without_a_readable_base_everything_runs(repo):
    commit(repo, "docs/index.md")
    assert decide(repo, "workflow_dispatch") == {"bench": "true", "harness": "true"}
    assert decide(repo, "push", before="0" * 40) == {"bench": "true", "harness": "true"}
