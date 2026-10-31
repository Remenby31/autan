"""The commands of the user guides run as written, from the repository root."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT

PAGES = sorted(
    page
    for folder in ("getting-started", "guides")
    for page in (ROOT / "docs" / folder).glob("*.md")
    if not page.name.endswith(".fr.md")
)
BLOCK = re.compile(r"^```(sh|python|text)\n(.*?)^```", re.MULTILINE | re.DOTALL)
CODE = re.compile(r"\b[EWD]\d{4}\b")


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_page_commands_succeed_with_the_codes_shown(page: Path, tmp_path: Path) -> None:
    blocks = BLOCK.findall(page.read_text())
    # The test environment is already synchronised; `uv sync` would rebuild the package under test.
    runnable = [
        i
        for i, (language, code) in enumerate(blocks)
        if language != "text" and not code.startswith("uv sync")
    ]
    assert runnable, f"{page.name} has no sh or python block"
    env = {**os.environ, "TMPDIR": str(tmp_path), "UV_LOCKED": "1"}
    for i in runnable:
        language, code = blocks[i]
        argv = (
            ["bash", "-euo", "pipefail", "-c", code]
            if language == "sh"
            else [sys.executable, "-c", code]
        )
        done = subprocess.run(
            argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
        )
        assert done.returncode == 0, f"{page.name}: {code}\n{done.stdout}\n{done.stderr}"
        if i + 1 < len(blocks) and blocks[i + 1][0] == "text":
            shown = set(CODE.findall(blocks[i + 1][1]))
            assert shown <= set(CODE.findall(done.stdout + done.stderr)), f"{page.name}: {code}"
