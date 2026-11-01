"""Cell zones starting from their own state (case-schema.md#cell-zones), e.g. a shock tube.

Expected densities come from the ideal-gas law p = rho R T with R = R_u / M.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import jsonschema
import numpy as np
import pytest
from autan.cli import json_schema, main

R_AIR = 8.314462618 / 0.0289647
SOD = (
    "import autan\nfrom autan.units import K, Pa, m, s\n"
    "case = autan.Case('sod', mesh='crm-hl.cgns')\n"
    "case.physics = autan.Compressible(turbulence=autan.Inviscid())\n"
    "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
    "still = (0 * m / s, 0 * m / s, 0 * m / s)\n"
    "case.cell_zone('left').fluid(autan.gases.AIR, initial=autan.State("
    "pressure=1e5 * Pa, temperature=348.4 * K, velocity=still))\n"
    "case.cell_zone('right').fluid(autan.gases.AIR, initial=autan.State("
    "pressure=1e4 * Pa, temperature=278.7 * K, velocity=still))\n"
)


def split(document: dict[str, Any]) -> None:
    cells = document["counts"]["cells"]
    document["cell_zones"] = [
        {"name": "left", "type": None, "native_type": None, "cells": cells // 2},
        {"name": "right", "type": None, "native_type": None, "cells": cells - cells // 2},
    ]
    document["interfaces"] = [{"zones": ["left", "right"], "faces": 1000}]


@pytest.fixture
def tube(mesh_variant: Callable[..., autan.MeshInfo], tmp_path: Path) -> Path:
    mesh = mesh_variant("crm-hl", split)
    path = tmp_path / "tube.meshinfo.json"
    path.write_text(json.dumps(mesh.document()))
    return path


def test_each_zone_starts_from_its_state(
    write_case: Callable[..., Path], tube: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = write_case(
        SOD + "for zone in ('wing.*', 'fuselage', 'empennage'):\n    case.boundary(zone).slip()\n"
    )
    assert main(["prepare", str(case), "--mesh-info", str(tube)]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    initial = {z["zone"]: z["initial"] for z in document["cell_zones"]}
    assert initial["left"]["density"] == pytest.approx(1e5 / (R_AIR * 348.4), rel=1e-12)
    assert initial["right"]["density"] == pytest.approx(1e4 / (R_AIR * 278.7), rel=1e-12)
    assert initial["left"]["velocity"] == [0.0, 0.0, 0.0]
    assert (initial["right"]["pressure"], initial["right"]["temperature"]) == (1e4, 278.7)


def test_the_fake_backend_writes_the_zone_states(
    write_case: Callable[..., Path], tube: Path, tmp_path: Path
) -> None:
    case = autan.load(
        write_case(
            SOD + "case.boundary('wing.*').slip()\n"
            "case.boundary('fuselage').slip()\n"
            "case.boundary('empennage').slip()\n"
        )
    )
    result = case.run(
        autan.MeshInfo.load(tube), backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs"
    )
    density = result.field("Density")
    np.testing.assert_allclose(density[result.cell_zone("left")], 1e5 / (R_AIR * 348.4), rtol=1e-12)
    np.testing.assert_allclose(
        density[result.cell_zone("right")], 1e4 / (R_AIR * 278.7), rtol=1e-12
    )


@pytest.mark.parametrize(
    "state",
    [
        "autan.State(pressure=1e5 * Pa, velocity=still)",
        "autan.State(pressure=-1e5 * Pa, temperature=300 * K, velocity=still)",
    ],
)
def test_a_compressible_state_needs_temperature_and_positive_pressure(
    state: str, write_case: Callable[..., Path]
) -> None:
    body = (
        SOD.split("case.cell_zone", maxsplit=1)[0]
        + f"case.cell_zone('left').fluid(autan.gases.AIR, initial={state})\n"
    )
    with pytest.raises(autan.CaseError) as error:
        _ = autan.load(write_case(body)).spec
    assert error.value.codes == ["E0507"]


def test_an_incompressible_state_takes_no_temperature(write_case: Callable[..., Path]) -> None:
    body = (
        "import autan\nfrom autan.units import K, Pa, kg, m, s\n"
        "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
        " viscosity=1e-3 * Pa * s)\n"
        "case = autan.Case('c', mesh='crm-hl.cgns')\n"
        "case.physics = autan.Incompressible(turbulence=autan.Laminar(), fluid=water)\n"
        "case.freestream(velocity=2 * m / s)\n"
        "case.cell_zone('fluid').fluid(water, initial=autan.State(pressure=0 * Pa,"
        " temperature=300 * K, velocity=(1 * m / s, 0 * m / s, 0 * m / s)))\n"
    )
    with pytest.raises(autan.CaseError) as error:
        _ = autan.load(write_case(body)).spec
    assert error.value.codes == ["E0507"]


def test_zone_states_round_trip(write_case: Callable[..., Path], tmp_path: Path) -> None:
    case = autan.load(write_case(SOD))
    document = case.spec.document()
    left = document["cell_zones"][0]["medium"]["initial"]
    assert left == {
        "pressure": "100000.0 Pa",
        "temperature": "348.4 K",
        "velocity": ["0.0 m / s", "0.0 m / s", "0.0 m / s"],
    }
    jsonschema.Draft202012Validator(json_schema("case")).validate(document)
    toml = tmp_path / "copy.toml"
    toml.write_text(case.to_toml())
    assert autan.load(toml).spec == case.spec


def test_wall_pressure_is_that_of_the_wall_cells(
    write_case: Callable[..., Path], tube: Path, tmp_path: Path
) -> None:
    case = autan.load(
        write_case(
            SOD + "case.boundary('wing.*').slip()\n"
            "case.boundary('fuselage').slip()\n"
            "case.boundary('empennage').slip()\n"
        )
    )
    result = case.run(
        autan.MeshInfo.load(tube), backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs"
    )
    _, pressure = result.wall("Pressure")
    assert set(np.unique(pressure)) <= {1e5, 1e4}  # the zone states, not the freestream


@pytest.mark.parametrize(
    ("body", "edit"),
    [
        (SOD, lambda doc: doc["cell_zones"][0]["medium"]["initial"].pop("temperature")),
        (
            (
                "import autan\nfrom autan.units import Pa, kg, m, s\n"
                "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
                " viscosity=1e-3 * Pa * s)\n"
                "case = autan.Case('c', mesh='crm-hl.cgns')\n"
                "case.physics = autan.Incompressible(turbulence=autan.Laminar(), fluid=water)\n"
                "case.freestream(velocity=2 * m / s)\n"
                "case.cell_zone('fluid').fluid(water, initial=autan.State(pressure=0 * Pa,"
                " velocity=(1 * m / s, 0 * m / s, 0 * m / s)))\n"
            ),
            lambda doc: doc["cell_zones"][0]["medium"]["initial"].update(temperature="300 K"),
        ),
    ],
)
def test_schema_and_reader_reject_the_same_zone_states(
    body: str,
    edit: Callable[[dict[str, Any]], None],
    write_case: Callable[..., Path],
    tmp_path: Path,
) -> None:
    document = autan.load(write_case(body)).spec.document()
    validator = jsonschema.Draft202012Validator(json_schema("case"))
    validator.validate(document)
    edit(document)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(document))
    with pytest.raises(autan.CaseError) as error:
        autan.load(path)
    assert error.value.codes == ["E0507"]
