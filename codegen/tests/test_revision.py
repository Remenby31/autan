"""The revision benchmark binaries record (codegen/bench/revision.cmake), outside a git checkout."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def revision(tmp: Path, source: Path, commit: str = "") -> str:
    out = tmp / "revision.h"
    subprocess.run(
        ["cmake", f"-DGIT={shutil.which('git') or ''}", f"-DSOURCE_DIR={source}",
         f"-DAUTAN_COMMIT={commit}", f"-DOUTPUT_FILE={out}", "-P",
         str(ROOT / "codegen" / "bench" / "revision.cmake")],
        check=True, capture_output=True,
    )  # fmt: skip
    return out.read_text()


def test_an_export_without_git_builds_with_the_given_commit_or_unknown(tmp_path: Path) -> None:
    # A source export (sdist, git archive) has no .git: the build goes on, recording AUTAN_COMMIT
    # when given, else "unknown" marked dirty.
    export = tmp_path / "export"
    export.mkdir()
    unknown = revision(tmp_path, export)
    assert '#define AUTAN_BENCH_COMMIT "unknown"' in unknown
    assert '#define AUTAN_BENCH_DIRTY "true"' in unknown
    commit = "0123456789abcdef0123456789abcdef01234567"
    given = revision(tmp_path, export, commit)
    assert f'#define AUTAN_BENCH_COMMIT "{commit}"' in given
    assert '#define AUTAN_BENCH_DIRTY "false"' in given
