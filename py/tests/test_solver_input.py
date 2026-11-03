"""The solver input through `autan prepare`; expected values recomputed from the contracts."""

import hashlib
import json
import math
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import jsonschema
import pytest
import rfc8785
from autan.cli import json_schema, main
from pydantic import ValidationError

from tests.conftest import CRM_HL, EXAMPLES, FIXTURES, mesh_document, place_meshes, prepared

R_UNIVERSAL = 8.314462618


def prepare(
    capsys: pytest.CaptureFixture[str], case: Path, mesh: str | Path, *extra: str
) -> tuple[int, Any]:
    status = main(["prepare", str(case), "--mesh-info", str(FIXTURES / mesh), *extra])
    out, err = capsys.readouterr()
    return status, json.loads(out) if status == 0 else out + err


def test_crm_hl_input_is_resolved_and_closed_in_si(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path], tmp_path: Path
) -> None:
    mesh_info = tmp_path / "crm-hl.meshinfo.json"
    mesh_info.write_text(json.dumps(prepared(mesh_document("crm-hl"), ranks=8)))
    status, document = prepare(
        capsys,
        example("crm_hl.py"),
        mesh_info,
        "--device-kind",
        "cuda",
        "--device-count",
        "8",
    )
    assert status == 0
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["devices"] == {"kind": "cuda", "count": 8}
    planned = autan.load(example("crm_hl.py")).plan(
        autan.MeshInfo.load(mesh_info), gpus=8, device_kind="cuda"
    )
    assert document["memory"] == {
        "bytes_host": planned.memory.bytes_host,
        "bytes_device": planned.memory.bytes_device,
    }
    assert document["reproducibility"] == "run"  # the case's default (case-schema.md#physics)
    # solver-input.md: the MeshInfo the plan used travels with the input, for the same mesh.
    assert document["mesh_info"] == autan.MeshInfo.load(mesh_info).document()
    document["mesh"]["sha256"] = "1" * 64
    with pytest.raises(ValidationError):
        autan.solver_input.SolverInput.model_validate(document)
    document["mesh"]["sha256"] = document["mesh_info"]["source"]["sha256"]

    # The case hash is SHA-256 of the RFC 8785 form of the (fully defaulted) case document.
    canonical = rfc8785.dumps(tomllib.loads(document["case"]["toml"]))
    assert hashlib.sha256(canonical).hexdigest() == document["case"]["sha256"]

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
    assert document["time"] == {
        "kind": "steady",
        "max_iterations": 10000,
        "residual_orders": 8.0,
        "linear_solver": "multigrid",
        "coupled_turbulence": False,
        "matrix_free": False,
        "checkpoint": {"seconds": 1800.0},
    }


def test_the_coupled_turbulence_step_is_an_sa_option(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str], crm_mesh: autan.MeshInfo
) -> None:
    # case-schema.md#time: off by default; on for compressible SA only (E0426), with the
    # multigrid on one rank (E0427), and always reported as validated on the TMR plate only.
    def codes(turbulence: str, time: str, gpus: int = 1) -> list[str]:
        body = CRM_HL.replace("autan.SA()", turbulence) + f"case.time = autan.Steady({time})\n"
        try:
            planned = autan.load(write_case(body)).plan(crm_mesh, gpus=gpus)
        except autan.CaseError as error:
            return error.codes
        return sorted(d.code for d in planned.diagnostics if d.code[1:3] == "04")

    on = "coupled_turbulence=True"
    assert codes("autan.SA()", "") == []
    assert codes("autan.SA()", on) == ["W0426"]
    assert codes("autan.SST()", on) == ["E0426"]
    assert codes("autan.Laminar()", on) == ["E0426"]
    assert codes("autan.SA()", f"{on}, linear_solver='gauss-seidel'") == ["E0427", "W0426"]
    assert codes("autan.SA()", on, gpus=2) == ["E0427", "W0426"]
    case = write_case(CRM_HL + f"case.time = autan.Steady({on})\n")
    jsonschema.Draft202012Validator(json_schema("case")).validate(autan.load(case).spec.document())
    status, document = prepare(capsys, case, "crm-hl.meshinfo.json")
    assert status == 0
    assert document["time"]["coupled_turbulence"] is True
    # The runtime and the published schema take a JSON boolean only.
    with pytest.raises(ValidationError):
        autan.solver_input.SolverInput.model_validate(
            document | {"time": document["time"] | {"coupled_turbulence": "true"}}
        )
    status, refused = prepare(capsys, case, "crm-hl.meshinfo.json", "--device-count", "2")
    assert status != 0
    assert "E0427" in refused


def test_the_matrix_free_step_is_a_laminar_option(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str], crm_mesh: autan.MeshInfo
) -> None:
    # case-schema.md#time: off by default; on for compressible laminar flow only (E0429), with
    # the multigrid on one rank (E0427), and always reported as validated on the TMR laminar
    # flat plate only.
    def codes(turbulence: str, time: str, gpus: int = 1) -> list[str]:
        body = CRM_HL.replace("autan.SA()", turbulence) + f"case.time = autan.Steady({time})\n"
        try:
            planned = autan.load(write_case(body)).plan(crm_mesh, gpus=gpus)
        except autan.CaseError as error:
            return error.codes
        return sorted(d.code for d in planned.diagnostics if d.code[1:3] == "04")

    on = "matrix_free=True"
    assert codes("autan.Laminar()", "") == []
    assert codes("autan.Laminar()", on) == ["W0429"]
    assert codes("autan.SA()", on) == ["E0429"]
    assert codes("autan.Inviscid()", on) == ["E0429"]
    assert codes("autan.Laminar()", f"{on}, linear_solver='lines'") == ["E0427", "W0429"]
    assert codes("autan.Laminar()", on, gpus=2) == ["E0427", "W0429"]
    case = write_case(
        CRM_HL.replace("autan.SA()", "autan.Laminar()") + f"case.time = autan.Steady({on})\n"
    )
    jsonschema.Draft202012Validator(json_schema("case")).validate(autan.load(case).spec.document())
    status, document = prepare(capsys, case, "crm-hl.meshinfo.json")
    assert status == 0
    assert document["time"]["matrix_free"] is True
    with pytest.raises(ValidationError):
        autan.solver_input.SolverInput.model_validate(
            document | {"time": document["time"] | {"matrix_free": 1}}
        )


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


def _valid(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path], name: str, mesh: str
) -> dict[str, Any]:
    status, document = prepare(capsys, example(name), mesh)
    assert status == 0
    result: dict[str, Any] = document
    return result


def _set_boundary(document: dict[str, Any], **values: Any) -> None:
    document["boundaries"][0].update(values)


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(extra=1),
        lambda d: d.pop("schema"),
        lambda d: d["time"].pop("kind"),
        lambda d: d["mesh"].update(scale=True),
        lambda d: d["mesh"].update(scale="1.0"),
        lambda d: d["mesh"].update(scale=0.0),
        lambda d: d["monitors"][0].pop("kind"),
        lambda d: _set_boundary(d, kind="inlet"),
        lambda d: _set_boundary(d, static_pressure=1.0),
        lambda d: d["physics"]["turbulence"].update(nu_tilde_ratio=None),
        lambda d: d["physics"]["turbulence"].update(intensity=0.1),
        lambda d: d.update(
            time={"kind": "unsteady", "steps": 10, "dt": None, "cfl": None, "average_from": None}
        ),
        lambda d: d.update(
            time={"kind": "unsteady", "steps": 10, "dt": 1e-3, "cfl": 1.0, "average_from": None}
        ),
        lambda d: d["physics"]["turbulence"].update(
            model="wmles", wall_model="equilibrium", nu_tilde_ratio=None
        ),
    ],
)
def test_schema_and_reader_reject_the_same_documents(
    edit: Callable[[dict[str, Any]], None],
    capsys: pytest.CaptureFixture[str],
    example: Callable[[str], Path],
) -> None:
    document = _valid(capsys, example, "crm_hl.py", "crm-hl.meshinfo.json")
    validator = jsonschema.Draft202012Validator(json_schema("solver-input"))
    validator.validate(document)
    autan.solver_input.SolverInput.model_validate(document)
    edit(document)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(document)
    with pytest.raises(ValidationError):
        autan.solver_input.SolverInput.model_validate(document)


def test_each_zone_starts_in_its_own_medium(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    document = _valid(capsys, example, "heat_exchanger.py", "heat-exchanger.meshinfo.json")
    initial = {z["zone"]: z["initial"] for z in document["cell_zones"]}
    pressure, temperature = 101325.0, 300.0
    rho = pressure / (R_UNIVERSAL / 0.0440095 * temperature)
    mu = 1.370e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 222.0) / (temperature + 222.0)
    speed = 0.1 * math.sqrt(1.4 * R_UNIVERSAL / 0.0289647 * temperature)
    k = 1.5 * (0.001 * speed) ** 2
    assert initial["co2"]["density"] == pytest.approx(rho, rel=1e-12)
    assert initial["co2"]["pressure"] == pytest.approx(pressure)
    assert initial["co2"]["omega"] == pytest.approx(rho * k / (0.009 * mu), rel=1e-12)
    assert initial["plate"] == {"kind": "solid", "temperature": temperature}


def test_prepare_reports_plan_warnings_and_bad_arguments(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path], tmp_path: Path
) -> None:
    case = example("crm_hl.py")
    case.write_text(case.read_text() + 'case.boundary("fuselage").slip()\n')
    argv = ["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]
    assert main([*argv, "--json"]) == 0
    warnings = json.loads(capsys.readouterr().err)
    assert [w["code"] for w in warnings] == ["W0415"]
    with pytest.raises(SystemExit) as exit_:
        main([*argv, "--device-count", "0"])
    assert exit_.value.code == 2
    assert main([*argv, "--json", "--output", str(tmp_path)]) == 1
    assert [d["code"] for d in json.loads(capsys.readouterr().out)["diagnostics"]] == ["E0806"]


@pytest.mark.parametrize("name", ["case", "mesh-info", "results", "solver-input"])
def test_each_published_schema_has_its_identifier(name: str) -> None:
    schema = json_schema(name)
    assert schema["$id"] == autan.version.SCHEMAS[name]
    assert schema["properties"]["schema"]["const"] == autan.version.SCHEMAS[name]


# Exact by definition: the international inch (1959) and foot.
@pytest.mark.parametrize(("unit", "metres"), [("m", 1.0), ("in", 0.0254), ("ft", 0.3048)])
def test_mesh_unit_scales_the_mesh_alone(
    unit: str,
    metres: float,
    capsys: pytest.CaptureFixture[str],
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
) -> None:
    body = CRM_HL.replace('mesh="crm-hl.cgns"', f'mesh="crm-hl.cgns", mesh_unit="{unit}"')
    path = write_case(body)
    case = autan.load(path)
    assert case.spec.document()["mesh_unit"] == unit
    planned = case.plan(crm_mesh).mesh
    assert planned["unit"] == unit
    # The fixture's bounds are in its file's units; plan gives them in metres.
    assert planned["bounds"]["max"] == [80.0 * metres, 50.0 * metres, 30.0 * metres]
    # mesh-info.md: the lines are counted on the mesh at the case's scale.
    status, refused = prepare(capsys, path, "crm-hl.meshinfo.json")
    assert (status == 0) == (unit == "m")
    if unit != "m":
        assert "E0210" in refused
    mesh_info = tmp_path / "scaled.meshinfo.json"
    mesh_info.write_text(json.dumps(prepared(mesh_document("crm-hl"), scale=metres)))
    status, document = prepare(capsys, path, mesh_info)
    assert status == 0
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["mesh"]["scale"] == metres
    autan.solver_input.SolverInput.model_validate(document)
    with pytest.raises(ValidationError):  # a scale that is not the case's unit
        autan.solver_input.SolverInput.model_validate(
            document | {"mesh": document["mesh"] | {"scale": 2 * metres}}
        )
    # Every other length carries its own unit: the reference length stays 7 m.
    assert document["monitors"][0]["ref_length"] == pytest.approx(7.0)
    in_metres = autan.load(write_case(CRM_HL, "metres.py")).spec
    assert (case.spec.hash() == in_metres.hash()) == (unit == "m")


@pytest.mark.parametrize("mode", ["none", "run", "ranks"])
def test_reproducibility_reaches_the_runtime_and_the_hash(
    mode: str,
    write_case: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    body = CRM_HL.replace("autan.SA())", f"autan.SA(), reproducibility={mode!r})")
    case = write_case(body, name=f"{mode}.py")
    spec = autan.load(case).spec
    copy = tmp_path / "copy.toml"
    copy.write_text(autan.load(case).to_toml())
    assert autan.load(copy).spec == spec
    assert (spec.hash() == autan.load(write_case(CRM_HL, name="default.py")).spec.hash()) == (
        mode == "run"  # the default
    )
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["reproducibility"] == mode


def test_the_runtime_mode_is_the_case_s(
    write_case: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    case = write_case(CRM_HL)
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    document["reproducibility"] = "none"  # the embedded case says "run"
    with pytest.raises(ValidationError, match="reproducibility"):
        autan.solver_input.SolverInput.model_validate(document)


@pytest.mark.parametrize("linear_solver", ["multigrid", "gauss-seidel", "lines"])
def test_the_linear_solver_is_a_steady_choice(
    write_case: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    linear_solver: str,
) -> None:
    body = CRM_HL + f"case.time = autan.Steady(linear_solver='{linear_solver}')\n"
    case = write_case(body)
    copy = tmp_path / "copy.toml"
    copy.write_text(autan.load(case).to_toml())
    assert autan.load(copy).spec == autan.load(case).spec
    assert main(["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]) == 0
    document = json.loads(capsys.readouterr().out)
    validator = jsonschema.Draft202012Validator(json_schema("solver-input"))
    validator.validate(document)
    assert document["time"]["linear_solver"] == linear_solver
    unsteady = {"kind": "unsteady", "steps": 10, "dt": 1e-3, "cfl": None, "average_from": None}
    with pytest.raises(jsonschema.ValidationError):  # an unsteady run has no linear solver choice
        validator.validate(document | {"time": unsteady | {"linear_solver": "multigrid"}})


@pytest.mark.parametrize("repeats", ["monitors", "point vortex"])
def test_the_input_stays_within_the_size_its_plan_counts(
    repeats: str, tmp_path: Path, mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    # plan.md: the solver input as `autan prepare` writes it is at most the plan's "solver
    # input". Every free text at its 4096-byte bound in control characters, which JSON
    # escapes 6 times, on 256 walls and 64 cell zones whose names the monitors or a point
    # vortex repeat.
    def long(tag: str) -> str:
        return "\x01" * (4096 - len(tag)) + tag

    def rename(document: dict[str, Any]) -> None:
        zones = document["boundary_zones"]
        walls = [z for z in zones if z["type"] == "wall"]
        faces = sum(z["faces"] for z in walls)
        split = [
            {"name": long(f"wall-{i}"), "type": "wall", "native_type": None, "faces": faces // 256}
            for i in range(256)
        ]
        split[0]["faces"] += faces % 256
        others = [z | {"name": long(f"{z['type']}-{z['name']}")} for z in zones if z not in walls]
        document["boundary_zones"] = split + others
        cells = document["counts"]["cells"]
        document["cell_zones"] = [
            {"name": long(f"fluid-{i}"), "type": None, "native_type": None, "cells": cells // 64}
            for i in range(64)
        ]
        document["cell_zones"][0]["cells"] += cells % 64

    mesh = mesh_variant("crm-hl", rename)
    place_meshes(tmp_path)
    case = autan.load(EXAMPLES / "crm_hl.py").spec.document()
    case["name"] = long("case")
    case["physics"]["gas"]["name"] = long("gas")
    farfield: dict[str, Any] = {"kind": "farfield"}
    if repeats == "point vortex":
        farfield["point_vortex"] = {"surfaces": "*wall-*", "center": ["0 m", "0 m", "0 m"]}
        case["monitors"] = []
    else:
        case["monitors"] = [
            {**case["monitors"][0], "name": long(f"forces-{i}"), "surfaces": "*"} for i in range(8)
        ]
    case["boundaries"] = [
        {"zones": "*wall-*", "condition": {"kind": "wall"}},
        {"zones": "*farfield-*", "condition": farfield},
    ]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    loaded = autan.load(path)
    devices = autan.Devices(kind="cuda", count=1)

    document = autan.solver_input.build(loaded, mesh, devices=devices).document()
    written = json.dumps(document, indent=2) + "\n"

    bound = {c.name: c for c in loaded.plan(mesh).memory.components}["solver input"]
    assert len(written.encode()) <= bound.bytes_host
