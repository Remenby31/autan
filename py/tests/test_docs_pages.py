"""The commands of the user guides run as written, from the repository root."""

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
BLOCK = re.compile(r"^```(sh|python)\n(.*?)^```", re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_page_commands_succeed(page: Path) -> None:
    blocks = BLOCK.findall(page.read_text())
    assert blocks, f"{page.name} has no sh or python block"
    for language, code in blocks:
        argv = (
            ["bash", "-euo", "pipefail", "-c", code]
            if language == "sh"
            else [sys.executable, "-c", code]
        )
        done = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
        )
        assert done.returncode == 0, f"{page.name}: {code}\n{done.stdout}\n{done.stderr}"
