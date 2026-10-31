import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cfd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = ROOT / "py" / "examples"


def mesh_document(name: str) -> dict[str, Any]:
    document: dict[str, Any] = json.loads((FIXTURES / f"{name}.meshinfo.json").read_text())
    return document


@pytest.fixture
def crm_mesh() -> cfd.MeshInfo:
    return cfd.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json")


@pytest.fixture
def mesh_variant() -> Callable[..., cfd.MeshInfo]:
    """A fixture MeshInfo after ``edit`` mutates its JSON document."""

    def build(name: str, edit: Callable[[dict[str, Any]], None]) -> cfd.MeshInfo:
        document = copy.deepcopy(mesh_document(name))
        edit(document)
        return cfd.MeshInfo.from_document(document, cfd.Location(file=name))

    return build


@pytest.fixture
def write_case(tmp_path: Path) -> Callable[[str], Path]:
    def write(body: str, name: str = "case.py") -> Path:
        path = tmp_path / name
        path.write_text(body)
        return path

    return write


CRM_HL = (EXAMPLES / "crm_hl.py").read_text()
