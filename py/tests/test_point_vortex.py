"""The point-vortex correction of a farfield (case-schema.md#boundaries)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import jsonschema
import pydantic
import pytest
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, FIXTURES

VORTEX = (
    "from autan.units import m\n"
    "case.boundary('farfield').farfield(point_vortex=autan.PointVortex("
    "surfaces={surfaces!r}, center=(0.25 * m, 0 * m, 0 * m)))\n"
)


def prepared(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str], surfaces: str
) -> dict[str, Any]:
    case = write_case(CRM_HL + VORTEX.format(surfaces=surfaces))
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document: dict[str, Any] = json.loads(capsys.readouterr().out)
    return document


def test_the_vortex_reaches_the_solver_input(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    document = prepared(write_case, capsys, "wing.*")
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    boundaries = {b["zone"]: b for b in document["boundaries"]}
    assert boundaries["farfield"]["point_vortex"] == {
        "zones": ["wing.slat", "wing.main", "wing.flap"],
        "center": [0.25, 0.0, 0.0],
    }
    assert all(b["point_vortex"] is None for z, b in boundaries.items() if z != "farfield")


@pytest.mark.parametrize("surfaces", ["nothing*", "symmetry", "*"])
def test_a_vortex_needs_walls(surfaces: str, write_case: Callable[..., Path]) -> None:
    case = autan.load(write_case(CRM_HL + VORTEX.format(surfaces=surfaces)))
    diagnostics = case.plan(autan.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json")).diagnostics
    assert [d.code for d in diagnostics if d.severity == "error"] == ["E0425"]


def test_the_vortex_round_trips_and_is_hashed(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    plain = autan.load(write_case(CRM_HL + "case.boundary('farfield').farfield()\n", name="a.py"))
    vortex = autan.load(write_case(CRM_HL + VORTEX.format(surfaces="wing.*"), name="b.py"))
    assert "point_vortex" not in plain.spec.document()["boundaries"][-1]["condition"]
    assert plain.spec.hash() != vortex.spec.hash()
    copy = tmp_path / "copy.toml"
    copy.write_text(vortex.to_toml())
    assert autan.load(copy).spec == vortex.spec


@pytest.mark.parametrize("zone", ["wing.main", "symmetry"])
def test_both_validators_keep_the_vortex_on_a_farfield_over_walls(
    zone: str, write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    document = prepared(write_case, capsys, "wing.*")
    boundaries = {b["zone"]: b for b in document["boundaries"]}
    vortex = boundaries["farfield"]["point_vortex"]
    if zone == "wing.main":  # a vortex on a wall
        boundaries[zone]["point_vortex"] = vortex
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    else:  # a vortex over a symmetry plane
        vortex["zones"] = [zone]
    with pytest.raises(pydantic.ValidationError):
        autan.solver_input.SolverInput.model_validate(document)
