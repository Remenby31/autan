"""MeshInfo documents through the public reader."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cfd
import jsonschema
import pytest
from cfd.cli import json_schema

from tests.conftest import FIXTURES, mesh_document


@pytest.mark.parametrize("name", ["crm-hl", "heat-exchanger"])
def test_fixtures_are_valid_for_the_reader_and_the_published_schema(name: str) -> None:
    mesh = cfd.MeshInfo.load(FIXTURES / f"{name}.meshinfo.json")
    assert mesh.document() == mesh_document(name)
    jsonschema.Draft202012Validator(json_schema("mesh-info")).validate(mesh_document(name))


def _zones(d: dict[str, Any]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = d["boundary_zones"]
    return zones


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(schema="cfd.mesh-info/2"),
        lambda d: d.update(extra=1),
        lambda d: d["source"].update(sha256="ABC"),
        lambda d: d["counts"]["cells_by_type"].update(hexa=1),
        lambda d: d["cell_zones"][0].update(cells=1),
        lambda d: _zones(d).append(dict(_zones(d)[0])),
        lambda d: _zones(d)[0].update(type="periodic"),
        lambda d: d["interfaces"].append({"zones": ["fluid", "nowhere"], "faces": 1}),
        lambda d: d["interfaces"].append({"zones": ["fluid", "fluid"], "faces": 1}),
        lambda d: d["quality"].update(max_non_orthogonality_deg=181.0),
        lambda d: d.pop("quality"),
    ],
)
def test_invalid_documents_are_rejected_with_e0202(
    edit: Callable[[dict[str, Any]], None], mesh_variant: Callable[..., cfd.MeshInfo]
) -> None:
    with pytest.raises(cfd.CaseError) as error:
        mesh_variant("crm-hl", edit)
    assert set(error.value.codes) == {"E0202"}


def test_unreadable_file_is_e0202(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{")
    with pytest.raises(cfd.CaseError) as error:
        cfd.MeshInfo.load(path)
    assert error.value.codes == ["E0202"]
