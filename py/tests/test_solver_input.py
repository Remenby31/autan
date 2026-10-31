"""The solver input through `autan prepare`; expected values recomputed from the contracts."""

import json
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import jsonschema
import pytest
from autan.cli import json_schema, main

from tests.conftest import FIXTURES

R_UNIVERSAL = 8.314462618


def prepare(
    capsys: pytest.CaptureFixture[str], case: Path, mesh: str, *extra: str
) -> tuple[int, Any]:
    status = main(["prepare", str(case), "--mesh-info", str(FIXTURES / mesh), *extra])
    out = capsys.readouterr().out
    return status, json.loads(out) if status == 0 else out


def test_crm_hl_input_is_resolved_and_closed_in_si(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    status, document = prepare(
        capsys,
        example("crm_hl.py"),
        "crm-hl.meshinfo.json",
        "--device-kind",
        "cuda",
        "--device-count",
        "8",
        "--deterministic",
    )
    assert status == 0
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["devices"] == {"kind": "cuda", "count": 8}
    assert document["deterministic"] is True

    case = tomllib.loads(document["case"]["toml"])
    assert autan.CaseSpec.from_document(case, autan.Location()).hash() == document["case"]["sha256"]

    temperature, mach, re_number, length = 288.15, 0.2, 5.49e6, 7.0
    r = R_UNIVERSAL / 0.0289647
    mu = 1.716e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 110.4) / (temperature + 110.4)
    speed = mach * math.sqrt(1.4 * r * temperature)
    rho = re_number * mu / (speed * length)
    alpha = math.radians(19.57)
    fs = document["freestream"]
    assert fs["density"] == pytest.approx(rho, rel=1e-12)
    assert fs["pressure"] == pytest.approx(rho * r * temperature, rel=1e-12)
    assert fs["velocity"] == pytest.approx(
        [speed * math.cos(alpha), 0.0, speed * math.sin(alpha)], rel=1e-12
    )
    assert fs["aoa"] == pytest.approx(alpha, rel=1e-12)
    assert fs["nu_tilde"] == pytest.approx(3.0 * mu / rho, rel=1e-12)
    assert (fs["k"], fs["omega"]) == (None, None)

    kinds = {b["zone"]: b["kind"] for b in document["boundaries"]}
    assert kinds == {
        "wing.slat": "wall",
        "wing.main": "wall",
        "wing.flap": "wall",
        "fuselage": "wall",
        "empennage": "wall",
        "symmetry": "symmetry",
        "farfield": "farfield",
    }
    [forces] = document["monitors"]
    assert forces["zones"] == ["wing.slat", "wing.main", "wing.flap"]
    assert forces["ref_area"] == pytest.approx(191.84)
    assert forces["ref_length"] == pytest.approx(7.0)
    assert document["time"] == {"kind": "steady", "max_iterations": 10000, "residual_orders": 8.0}


def test_heat_exchanger_media_and_inlets_in_si(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    status, document = prepare(capsys, example("heat_exchanger.py"), "heat-exchanger.meshinfo.json")
    assert status == 0
    media = {z["zone"]: z["medium"] for z in document["cell_zones"]}
    assert media["co2"]["gas_constant"] == pytest.approx(R_UNIVERSAL / 0.0440095, rel=1e-12)
    assert media["co2"]["viscosity"] == {
        "law": "sutherland",
        "mu_ref": 1.370e-5,
        "t_ref": 273.15,
        "s": 222.0,
    }
    assert media["plate"] == {
        "kind": "solid",
        "name": "copper",
        "density": 8933.0,
        "specific_heat": 385.0,
        "conductivity": 401.0,
    }
    inlet = next(b for b in document["boundaries"] if b["zone"] == "air-inlet")
    assert (inlet["kind"], inlet["total_pressure"], inlet["total_temperature"]) == (
        "inlet",
        pytest.approx(103000.0),
        pytest.approx(400.0),
    )
    outlet = next(b for b in document["boundaries"] if b["zone"] == "co2-outlet")
    assert outlet["static_pressure"] == pytest.approx(101325.0)
    turbulence = document["physics"]["turbulence"]
    assert (turbulence["intensity"], turbulence["viscosity_ratio"]) == (0.001, 0.009)


def test_plan_errors_block_the_input(
    capsys: pytest.CaptureFixture[str], write_case: Callable[[str], Path], tmp_path: Path
) -> None:
    document = json.loads((FIXTURES / "crm-hl.meshinfo.json").read_text())
    document["boundary_zones"][4]["type"] = None
    info = tmp_path / "untyped.meshinfo.json"
    info.write_text(json.dumps(document))
    case = write_case((FIXTURES.parent.parent / "examples" / "crm_hl.py").read_text())
    assert main(["prepare", str(case), "--mesh-info", str(info), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert [d["code"] for d in out["diagnostics"]] == ["E0412"]


def test_input_rejects_unknown_keys() -> None:
    schema = json_schema("solver-input")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(
            {"schema": "autan.solver-input/1", "extra": 1}
        )


@pytest.mark.parametrize("name", ["case", "mesh-info", "results", "solver-input"])
def test_each_published_schema_has_its_identifier(name: str) -> None:
    schema = json_schema(name)
    assert schema["$id"] == autan.version.SCHEMAS[name]
    assert schema["properties"]["schema"]["const"] == autan.version.SCHEMAS[name]
