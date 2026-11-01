"""The revision benchmark binaries record (codegen/bench/revision.cmake)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or ""
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def revision(tmp: Path, source: Path, commit: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmake", f"-DGIT={GIT}", f"-DSOURCE_DIR={source}", f"-DAUTAN_COMMIT={commit}",
         f"-DOUTPUT_FILE={tmp / 'revision.h'}", "-P",
         str(ROOT / "codegen" / "bench" / "revision.cmake")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip


def recorded(tmp: Path, source: Path, commit: str = "") -> tuple[str, str]:
    assert revision(tmp, source, commit).returncode == 0
    text = (tmp / "revision.h").read_text()
    c = text.split('AUTAN_BENCH_COMMIT "')[1].split('"')[0]
    return c, text.split('AUTAN_BENCH_DIRTY "')[1].split('"')[0]


def git(repo: Path, *args: str) -> str:
    cmd = [GIT, "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def test_revision_of_a_checkout_an_export_and_an_export_inside_another_repository(
    tmp_path: Path,
) -> None:
    # A source export (sdist, git archive) has no .git: the build goes on, recording AUTAN_COMMIT
    # when given, else "unknown" marked dirty; even unpacked inside another repository.
    outer = tmp_path / "outer"
    export = outer / "autan"
    export.mkdir(parents=True)
    git(outer, "init", "-q")
    git(outer, "commit", "-q", "--allow-empty", "-m", "outer")
    assert recorded(tmp_path, export) == ("unknown", "true")
    assert recorded(tmp_path, export, COMMIT) == (COMMIT, "false")
    # A checkout records its own HEAD, dirty when tracked files changed, whatever AUTAN_COMMIT.
    (export / "f").write_text("a")
    git(export, "init", "-q")
    git(export, "add", "f")
    git(export, "commit", "-q", "-m", "c")
    head = git(export, "rev-parse", "HEAD")
    assert recorded(tmp_path, export, COMMIT) == (head, "false")
    (export / "f").write_text("b")
    assert recorded(tmp_path, export, COMMIT) == (head, "true")
    # AUTAN_COMMIT follows the runtime's rule: 40 lowercase hexadecimal characters.
    for bad in ("deadbeef", 'x" "', COMMIT.upper()):
        assert revision(tmp_path, outer / "missing", bad).returncode != 0
