"""Periodic pairs of boundary zones: a mesh fact (mesh-info.md), never a case condition
(case-schema.md#boundary-resolution)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import jsonschema
import pytest
from autan import cgns
from autan.cli import json_schema, main

from tests.conftest import CRM_HL

PAIR = {
    "zones": ["fuselage", "empennage"],
    "transform": {"kind": "translation", "vector": [0.0, 2.5, 0.0]},
    "faces": 95000,
}


def periodic(document: dict[str, Any], *pairs: dict[str, Any]) -> None:
    """Make fuselage and empennage periodic zones listed with ``pairs`` (PAIR by default);
    one ``{}`` drops the ``periodic`` key."""
    for zone in document["boundary_zones"]:
        if zone["name"] in ("fuselage", "empennage"):
            zone.update(type="periodic", faces=95000)
    document["periodic"] = list(pairs or [PAIR])
    if pairs == ({},):
        del document["periodic"]


@pytest.fixture
def mesh(mesh_variant: Callable[..., autan.MeshInfo], tmp_path: Path) -> Path:
    path = tmp_path / "periodic.meshinfo.json"
    path.write_text(json.dumps(mesh_variant("crm-hl", periodic).document()))
    return path


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: periodic(d, PAIR | {"zones": ["fuselage", "fuselage"]}),
        lambda d: periodic(d, PAIR | {"zones": ["fuselage", "farfield"]}),
        lambda d: periodic(d, PAIR | {"faces": 94000}),
        lambda d: periodic(d, PAIR | {"transform": {"kind": "translation", "vector": [0, 0, 0]}}),
        lambda d: periodic(d, PAIR | {"transform": {"kind": "rotation", "vector": [0, 1, 0]}}),
        lambda d: periodic(d, PAIR, PAIR),
        lambda d: periodic(d, {}),
    ],
)
def test_periodic_pairs_must_match_their_zones(
    edit: Callable[[dict[str, Any]], None], mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        mesh_variant("crm-hl", edit)
    assert error.value.codes == ["E0202"]


def test_a_periodic_pair_resolves_from_the_mesh(
    write_case: Callable[..., Path], mesh: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = write_case(CRM_HL)
    plan = autan.load(case).plan(autan.MeshInfo.load(mesh))
    assert plan.ok, plan.diagnostics
    resolved = {b.zone: (b.condition, b.provenance) for b in plan.boundaries}
    assert resolved["fuselage"] == ({"kind": "periodic", "partner": "empennage"}, "mesh")
    assert resolved["empennage"] == ({"kind": "periodic", "partner": "fuselage"}, "mesh")

    assert main(["prepare", str(case), "--mesh-info", str(mesh)]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["periodic"] == [{k: PAIR[k] for k in ("zones", "transform")}]
    assert {"fuselage", "empennage"}.isdisjoint(b["zone"] for b in document["boundaries"])

    document["boundaries"].append({**document["boundaries"][0], "zone": "fuselage"})
    with pytest.raises(ValueError, match="periodic"):
        autan.solver_input.SolverInput.model_validate(document)


def test_a_wildcard_skips_periodic_zones_and_a_name_is_e0424(
    write_case: Callable[..., Path], mesh: Path
) -> None:
    info = autan.MeshInfo.load(mesh)
    every = autan.load(write_case(CRM_HL + "case.boundary('*').slip()\n", name="every.py"))
    resolved = {b.zone: b.condition for b in every.plan(info).boundaries}
    assert resolved["fuselage"] == {"kind": "periodic", "partner": "empennage"}

    named = autan.load(write_case(CRM_HL + "case.boundary('fuselage').slip()\n"))
    assert [d.code for d in named.plan(info).diagnostics if d.severity == "error"] == ["E0424"]


def test_a_periodic_zone_is_written_without_condition(
    write_case: Callable[..., Path], mesh: Path, tmp_path: Path
) -> None:
    case = autan.load(write_case(CRM_HL))
    result = case.run(
        autan.MeshInfo.load(mesh), backend=autan.FakeBackend(iterations=2), runs=tmp_path
    )
    zone_bc = cgns.read(result.path / "solution.cgns").child("Base").child("Zone").child("ZoneBC")
    assert zone_bc.child("fuselage").text() == "UserDefined"
    assert [p.zones for p in result.mesh_info.periodic] == [("fuselage", "empennage")]
