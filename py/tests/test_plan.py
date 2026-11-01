"""Case resolution against MeshInfo: provenance, coded diagnostics, memory model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import pytest

from tests.conftest import CRM_HL, FIXTURES, HEAT_EXCHANGER

HEAT = HEAT_EXCHANGER


def plan_of(body: str, mesh: autan.MeshInfo, write_case: Callable[[str], Path]) -> autan.Plan:
    return autan.load(write_case(body)).plan(mesh)


def diagnostic_codes(result: autan.Plan) -> list[str]:
    return sorted(d.code for d in result.diagnostics)


def zone_type(name: str, value: str | None) -> Callable[[dict[str, Any]], None]:
    def edit(document: dict[str, Any]) -> None:
        for zone in document["boundary_zones"] + document["cell_zones"]:
            if zone["name"] == name:
                zone["type"] = value

    return edit


def test_crm_hl_example_plans_with_provenance(
    crm_mesh: autan.MeshInfo, example: Callable[[str], Path]
) -> None:
    result = autan.load(example("crm_hl.py")).plan(crm_mesh)
    assert result.ok
    assert result.diagnostics == ()
    resolved = {
        b.zone: (b.condition["kind"] if b.condition else None, b.provenance)
        for b in result.boundaries
    }
    assert resolved == {
        "wing.slat": ("wall", "code"),
        "wing.main": ("wall", "code"),
        "wing.flap": ("wall", "code"),
        "fuselage": ("wall", "mesh"),
        "empennage": ("wall", "mesh"),
        "symmetry": ("symmetry", "code"),
        "farfield": ("farfield", "mesh"),
    }
    [fluid] = result.cell_zones
    assert fluid.provenance == "default"
    assert fluid.medium is not None
    assert fluid.medium["material"]["name"] == "air"


def test_heat_exchanger_example_plans_two_gases_and_a_solid(
    example: Callable[[str], Path],
) -> None:
    mesh = autan.MeshInfo.load(FIXTURES / "heat-exchanger.meshinfo.json")
    result = autan.load(example("heat_exchanger.py")).plan(mesh)
    assert result.ok, [d.render() for d in result.diagnostics]
    media = {
        z.zone: (z.medium["kind"], z.medium["material"]["name"], z.provenance)
        for z in result.cell_zones
        if z.medium
    }
    assert media == {
        "air": ("fluid", "air", "mesh"),
        "co2": ("fluid", "co2", "code"),
        "plate": ("solid", "copper", "code"),
    }


@pytest.mark.parametrize(
    ("mesh_name", "edit", "body", "expected"),
    [
        # Untyped zone, no rule.
        ("crm-hl", zone_type("empennage", None), CRM_HL, ["E0412"]),
        # Empty untyped zone needs nothing.
        (
            "crm-hl",
            lambda d: d["boundary_zones"].append(
                {"name": "ghost", "type": None, "native_type": None, "faces": 0}
            ),
            CRM_HL,
            [],
        ),
        ("crm-hl", None, CRM_HL + "case.boundary('pylon').wall()\n", ["E0410"]),
        ("crm-hl", None, CRM_HL + "case.boundary('wing.main').wall()\n", ["E0411"]),
        ("crm-hl", zone_type("farfield", "outlet"), CRM_HL, ["E0413"]),
        ("crm-hl", zone_type("fuselage", "thermal-wall"), CRM_HL, ["E0413"]),
        ("crm-hl", None, CRM_HL + "case.boundary('farfield').symmetry()\n", ["W0415"]),
        ("crm-hl", None, CRM_HL.replace("autan.SA()", "autan.Inviscid()"), ["E0414"]),
        (
            "crm-hl",
            None,
            CRM_HL + "from autan.units import s\n"
            "case.boundary('farfield').velocity_inlet(velocity=3 * m / s)\n",
            ["E0414", "W0415"],
        ),
        ("crm-hl", None, CRM_HL + "case.cell_zone('core').fluid(autan.gases.CO2)\n", ["E0501"]),
        (
            "crm-hl",
            None,
            CRM_HL + "case.cell_zone('fl*').fluid(autan.gases.CO2)\n"
            "case.cell_zone('*').fluid(autan.gases.AIR)\n",
            ["E0502"],
        ),
        ("crm-hl", None, CRM_HL + "case.monitor(autan.forces('pylon*', name='p'))\n", ["E0601"]),
        ("crm-hl", None, CRM_HL + "case.monitor(autan.forces('fuselage'))\n", ["E0602"]),
        ("crm-hl", lambda d: d["source"].update(sha256="0" * 64), CRM_HL, ["E0203"]),
        ("crm-hl", None, CRM_HL.replace('"crm-hl.cgns"', '"absent.cgns"'), ["W0205"]),
        # Mesh slip overridden by a no-slip wall changes the kind.
        (
            "crm-hl",
            zone_type("fuselage", "slip"),
            CRM_HL + "case.boundary('fuselage').wall()\n",
            ["W0415"],
        ),
        (
            "crm-hl",
            lambda d: d["quality"].update(max_non_orthogonality_deg=87.0),
            CRM_HL,
            ["W0204"],
        ),
        ("heat-exchanger", None, HEAT.replace('case.cell_zone("plate")', "#"), ["E0503"]),
        (
            "heat-exchanger",
            lambda d: d["interfaces"].append({"zones": ["air", "co2"], "faces": 10}),
            HEAT,
            ["E0505"],
        ),
        # The same gas written in other units is the same material: no species transport.
        (
            "heat-exchanger",
            lambda d: d["interfaces"].append({"zones": ["air", "co2"], "faces": 10}),
            HEAT.replace(
                "autan.gases.CO2",
                "autan.IdealGas(**{**autan.gases.AIR.model_dump(), 'name': 'a',"
                " 'molar_mass': 28.9647 * autan.units.ureg.gram / autan.units.mol})",
            ),
            [],
        ),
    ],
)
def test_resolution_diagnostics(
    mesh_name: str,
    edit: Callable[[dict[str, Any]], None] | None,
    body: str,
    expected: list[str],
    mesh_variant: Callable[..., autan.MeshInfo],
    write_case: Callable[[str], Path],
) -> None:
    mesh = mesh_variant(mesh_name, edit or (lambda _: None))
    result = plan_of(body, mesh, write_case)
    assert diagnostic_codes(result) == expected
    assert result.ok == (not any(code.startswith("E") for code in expected))


def test_missing_boundary_condition_points_at_the_case_and_suggests_the_fix(
    mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[[str], Path]
) -> None:
    path = write_case(CRM_HL)
    result = autan.load(path).plan(mesh_variant("crm-hl", zone_type("empennage", None)))
    [found] = result.diagnostics
    assert found.code == "E0412"
    assert found.data == {"zone": "empennage"}
    assert found.fix.startswith("case.boundary('empennage').wall()")
    assert found.location is not None
    assert found.location.file == str(path)


def test_incompressible_physics_rejects_solids_and_gases(
    write_case: Callable[[str], Path],
) -> None:
    mesh = autan.MeshInfo.load(FIXTURES / "heat-exchanger.meshinfo.json")
    body = (
        "import autan\nfrom autan.units import Pa, kg, m, s\n"
        "water = autan.ConstantDensityFluid(name='water', density=998 * kg / m**3,"
        " viscosity=1e-3 * Pa * s)\n"
        "case = autan.Case('hx', mesh='heat-exchanger.msh')\n"
        "case.physics = autan.Incompressible(turbulence=autan.SST(), fluid=water)\n"
        "case.freestream(velocity=2 * m / s)\n"
        "case.cell_zone('co2').fluid(autan.gases.CO2)\n"
        "case.cell_zone('plate').solid(autan.materials.COPPER)\n"
        "case.boundary('*-inlet').velocity_inlet(velocity=2 * m / s)\n"
        "case.boundary('*-outlet').outlet(static_pressure=101325 * Pa)\n"
    )
    assert diagnostic_codes(plan_of(body, mesh, write_case)) == ["E0504", "E0506"]


def tiny_mesh(boundary_faces: int) -> autan.MeshInfo:
    return autan.MeshInfo.from_document(
        {
            "schema": "autan.mesh-info/1",
            "source": {"path": "m.cgns", "format": "cgns", "sha256": "0" * 64},
            "counts": {
                "cells": 10,
                "faces": 40,
                "nodes": 30,
                "face_nodes": 160,
                "cells_by_type": {
                    "tetra": 0,
                    "pyramid": 0,
                    "prism": 0,
                    "hexa": 10,
                    "polyhedron": 0,
                },
            },
            "boundary_zones": [
                {
                    "name": "farfield",
                    "type": "farfield",
                    "native_type": None,
                    "faces": boundary_faces,
                }
            ],
            "cell_zones": [{"name": "fluid", "type": None, "native_type": None, "cells": 10}],
            "interfaces": [],
            "periodic": [],
            "quality": {"max_non_orthogonality_deg": 0.0, "max_faces_per_cell": 6},
        },
        autan.Location(),
    )


def test_the_memory_estimate_counts_what_each_run_allocates(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo
) -> None:
    # docs/sdk/plan.md, autan.memory/2: per rank, by component; the values are validated
    # against measured peaks, this checks which run allocates what and where.
    def memory(body: str = CRM_HL, **options: Any) -> autan.Memory:
        return autan.load(write_case(body)).plan(crm_mesh, **options).memory

    def names(estimate: autan.Memory) -> set[str]:
        return {c.name for c in estimate.components}

    cpu = memory(device_kind="cpu")
    assert names(cpu) >= {"mesh", "physics", "jacobian", "multigrid", "solution output"}
    assert cpu.bytes_device == 0
    # The peak: every persistent component and the largest transient one.
    persistent = sum(c.bytes_host for c in cpu.components if not c.transient)
    assert cpu.bytes_host == persistent + max(c.bytes_host for c in cpu.components if c.transient)
    # A device run solves explicitly: no Jacobian or linear solver, the state on the device.
    cuda = memory(device_kind="cuda")
    assert not names(cuda) & {"jacobian", "multigrid"}
    assert cuda.bytes_device > 0
    assert cuda.bytes_per_gpu == cuda.bytes_host + cuda.bytes_device
    gs = memory(
        CRM_HL + "case.time = autan.Steady(linear_solver='gauss-seidel')\n", device_kind="cpu"
    )
    assert gs.bytes_host < cpu.bytes_host
    unsteady = memory(CRM_HL + "case.time = autan.Unsteady(steps=10, cfl=1.0)\n", device_kind="cpu")
    assert not names(unsteady) & {"jacobian", "multigrid"}
    # Every rank keeps the global mesh; its share of the rest shrinks with the ranks.
    four = memory(gpus=4, device_kind="cuda")
    by_name = {c.name: c for c in four.components}
    assert by_name["mesh"] == {c.name: c for c in cuda.components}["mesh"]
    assert four.bytes_device < cuda.bytes_device / 3
    assert four.bytes_total == 4 * four.bytes_per_gpu
    fp64 = memory(
        CRM_HL.replace("autan.SA())", 'autan.SA(), precision="fp64")'), device_kind="cuda"
    )
    assert fp64.bytes_device > cuda.bytes_device


@pytest.mark.parametrize(
    ("turbulence", "expected"),
    [
        # docs/sdk/plan.md: 5 compressible equations, +1 for SA, +2 for SST, base model for DDES.
        ("autan.Inviscid()", 5),
        ("autan.SA()", 6),
        ("autan.SST()", 7),
        ("autan.DDES(base=autan.SA())", 6),
        ("autan.DDES()", 7),
        ("autan.WMLES()", 5),
    ],
)
def test_equation_count_per_turbulence_model(
    turbulence: str, expected: int, write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo
) -> None:
    body = CRM_HL.replace("autan.SA()", turbulence).replace(".wall()", ".slip()")
    body += "from autan.units import s\ncase.time = autan.Unsteady(steps=10, dt=1e-3 * s)\n"
    assert autan.load(write_case(body)).plan(crm_mesh).memory.equations == expected
