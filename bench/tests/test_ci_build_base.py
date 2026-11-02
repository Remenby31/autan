import os
import stat
import subprocess
from pathlib import Path

import pytest
from test_ci_changes import git

SCRIPT = Path(__file__).parents[1] / "ci" / "build-base.sh"

# The lock wrapper runs the command after -- (or exits with $LOCK_STATUS, as when the lock is not
# had); cmake fails in a source tree holding a file named BROKEN, and records each call.
FAKE_LOCK = """#!/usr/bin/env bash
[[ -n ${LOCK_STATUS:-} ]] && exit "$LOCK_STATUS"
while [[ $1 != -- ]]; do shift; done
shift
exec "$@"
"""
FAKE_CMAKE = """#!/usr/bin/env bash
echo "$*" >> "$CMAKE_LOG"
if [[ $1 == -S && ( -e $2/BROKEN || -e $2/../BROKEN ) ]]; then exit 1; fi
"""


def commit(repo: Path, path: str, text: str = "x\n") -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", path)


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    executable(repo / "bench" / "lock" / "autan-gpu-lock", FAKE_LOCK)
    git(repo, "add", "bench/lock/autan-gpu-lock")
    git(repo, "commit", "-q", "-m", "lock")
    executable(tmp_path / "bin" / "cmake", FAKE_CMAKE)
    return repo


def build(repo: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    tmp = repo.parent
    summary = tmp / "summary.md"
    full = {**os.environ, "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
            "CMAKE_LOG": str(tmp / "cmake.log"), "GITHUB_STEP_SUMMARY": str(summary), **env}  # fmt: skip
    return subprocess.run([str(SCRIPT), *args], cwd=repo, env=full, capture_output=True,
                          text=True, check=False)  # fmt: skip


def test_the_base_is_built_from_its_own_tree(repo):
    commit(repo, "bench/CMakeLists.txt", "base\n")
    base = git(repo, "rev-parse", "HEAD")
    commit(repo, "bench/CMakeLists.txt", "change\n")
    out = build(repo, str(repo.parent / "base"))
    assert out.returncode == 0, out.stderr
    worktree = repo.parent / "base"
    assert git(worktree, "rev-parse", "HEAD") == base
    assert (worktree / "bench" / "CMakeLists.txt").read_text() == "base\n"
    calls = (repo.parent / "cmake.log").read_text().splitlines()
    assert calls == [
        f"-S {worktree}/bench -B {worktree}/build/bench",
        f"--build {worktree}/build/bench",
    ]
    # Removed on request, and pruned from git's list.
    assert build(repo, "--remove", str(worktree)).returncode == 0
    assert not worktree.exists()
    assert str(worktree) not in git(repo, "worktree", "list")


def test_a_base_that_does_not_build_is_base_build_failed(repo):
    commit(repo, "BROKEN")
    commit(repo, "fix.txt")
    out = build(repo, str(repo.parent / "base"))
    assert out.returncode == 1
    assert "::error title=autan-bench::BASE BUILD FAILED: build (exit 1) at HEAD^1" in out.stdout
    assert "BASE BUILD FAILED" in (repo.parent / "summary.md").read_text()


def test_a_missing_base_is_base_build_failed(repo):
    out = build(repo, str(repo.parent / "base"), "0" * 40)
    assert out.returncode == 1
    assert "BASE BUILD FAILED: no such commit" in out.stdout


@pytest.mark.parametrize("status", ["1", "75", "124"])
def test_a_lock_failure_is_not_blamed_on_the_base(repo, status):
    # 1: the real wrapper's exit on a missing lock file (codex on #428), before any build.
    commit(repo, "fix.txt")
    out = build(repo, str(repo.parent / "base"), LOCK_STATUS=status)
    assert out.returncode == int(status)
    assert "BASE BUILD FAILED" not in out.stdout
    assert "did not start" in out.stdout


def test_the_lock_is_waited_for_as_long_as_the_job_allows(repo):
    commit(repo, "fix.txt")
    log = repo.parent / "lock-args"
    lock = repo / "bench" / "lock" / "autan-gpu-lock"
    lock.write_text(
        FAKE_LOCK.replace("#!/usr/bin/env bash\n", f'#!/usr/bin/env bash\necho "$*" > {log}\n')
    )
    git(repo, "commit", "-qam", "log the lock's arguments")
    assert build(repo, str(repo.parent / "base"), AUTAN_BENCH_LOCK_TIMEOUT="1500").returncode == 0
    assert " -w 1500 " in f" {log.read_text()} "


def test_a_worktree_left_by_a_cancelled_job_is_replaced(repo):
    commit(repo, "fix.txt")
    dir_ = repo.parent / "base"
    assert build(repo, str(dir_)).returncode == 0
    (dir_ / "stale").write_text("left over")
    assert build(repo, str(dir_)).returncode == 0
    assert not (dir_ / "stale").exists()
