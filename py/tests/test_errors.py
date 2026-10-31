"""The diagnostic registry: every code used is registered and documented."""

import json
import re
from pathlib import Path

from autan.errors import REGISTRY, reference_page, registry_document

ROOT = Path(__file__).resolve().parents[2]
CODE = re.compile(r"\b[EWD]\d{4}\b")
RANGE = re.compile(r"\b[EWD]\d{4}`?\s*(?:[\u2013-]|to)\s*`?[EWD]\d{4}\b")
MESH_CODE_ROW = re.compile(r"^\| (E01\d\d) \| `[a-z_]+` \| (.+?) \|$", re.MULTILINE)


def test_reference_page_is_generated_from_the_registry() -> None:
    assert (ROOT / "docs" / "sdk" / "errors.md").read_text() == reference_page()


def test_embedded_registry_export_is_generated_from_the_registry() -> None:
    # docs/contracts/diagnostics.json is what the C++ runtime embeds; `autan explain --all --json`.
    exported = json.loads((ROOT / "docs" / "contracts" / "diagnostics.json").read_text())
    assert exported == registry_document()
    assert set(exported["codes"]) == set(REGISTRY)


def test_every_code_used_or_documented_is_registered() -> None:
    sources = [*(ROOT / "py" / "autan").glob("*.py"), *(ROOT / "docs" / "contracts").glob("*.md")]
    used = {code for path in sources for code in CODE.findall(RANGE.sub("", path.read_text()))}
    assert used <= set(REGISTRY)


def test_mesh_codes_mirror_the_mesh_contract() -> None:
    # docs/contracts/mesh.md owns the mesh core codes and when they are raised.
    table = dict(MESH_CODE_ROW.findall((ROOT / "docs" / "contracts" / "mesh.md").read_text()))
    registered = {code: entry for code, entry in REGISTRY.items() if code.startswith("E01")}
    assert set(registered) == set(table)
    for code, trigger in table.items():
        assert registered[code].explanation == trigger
