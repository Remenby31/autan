"""The diagnostic registry: every code used is registered and documented."""

import re
from pathlib import Path

from cfd.errors import REGISTRY, reference_page

ROOT = Path(__file__).resolve().parents[2]
CODE = re.compile(r"\b[EWD]\d{4}\b")


def test_reference_page_is_generated_from_the_registry() -> None:
    assert (ROOT / "docs" / "sdk" / "errors.md").read_text() == reference_page()


def test_every_code_used_or_documented_is_registered() -> None:
    sources = [*(ROOT / "py" / "cfd").glob("*.py"), *(ROOT / "docs" / "contracts").glob("*.md")]
    used = {code for path in sources for code in CODE.findall(path.read_text())}
    core = {code for code in used if 101 <= int(code[1:]) <= 199 and code[0] == "E"}
    assert used - core <= set(REGISTRY)
