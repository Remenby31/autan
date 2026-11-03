"""Case resolution against MeshInfo: provenance, coded diagnostics, memory model."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import autan
import pytest

from tests.conftest import CRM_HL, FIXTURES, HEAT_EXCHANGER, mesh_document, prepared

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


TINY = (
    "import autan\n"
    "from autan.units import K, Pa\n"
    "case = autan.Case('tiny', mesh='m.cgns')\n"
    "case.physics = autan.Compressible(turbulence=autan.Inviscid())\n"
    "case.freestream(mach=0.5, pressure=101325 * Pa, temperature=288.15 * K)\n"
    "case.boundary('farfield').farfield()\n"
    "case.time = autan.Steady(linear_solver='multigrid')\n"
)


def tiny_mesh(boundary_faces: int) -> autan.MeshInfo:
    return autan.MeshInfo.from_document(
        {
            "schema": "autan.mesh-info/1",
            "source": {
                "path": "m.cgns",
                "format": "cgns",
                "sha256": "0" * 64,
                "bytes": 4096,
                "logical_bytes": 4096,
                "descriptor_bytes": 2048,
                "zone_names_bytes": 512,
                "filtered_bytes": 0,
                "cgns_nodes": 40,
                "point_list_ids": 128,
            },
            "counts": {
                "cells": 10,
                "faces": 40,
                "nodes": 30,
                "source_nodes": 30,
                "face_nodes": 160,
                "cells_by_type": {
                    "tetra": 0,
                    "pyramid": 0,
                    "prism": 0,
                    "hexa": 10,
                    "polyhedron": 0,
                },
                "stencil_entries": 240,
                "periodic_ghosts": 0,
            },
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
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
            "quality": {
                "max_non_orthogonality_deg": 0.0,
                "max_faces_per_cell": 6,
                "lines": {"order": "rcm", "scale": 1.0, "cells": 0, "long_cells": 0},
                "multigrid": {"cells_by_level": [10], "edges_by_level": [20]},
            },
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
    # Every rank keeps the global mesh; its share of the rest shrinks with the ranks, from the
    # partition the MeshInfo gives for them (mesh-info.md), without which the plan is refused.
    assert "E0210" in diagnostic_codes(autan.load(write_case(CRM_HL)).plan(crm_mesh, gpus=4))
    on_four = autan.MeshInfo.from_document(
        prepared(mesh_document("crm-hl"), ranks=4), autan.Location()
    )
    four = autan.load(write_case(CRM_HL)).plan(on_four, gpus=4, device_kind="cuda").memory
    # A steady CPU run with the lines solver keeps its lines whole: a partition without them is
    # not the run's.
    lines = autan.load(write_case(CRM_HL + "case.time = autan.Steady(linear_solver='lines')\n"))
    assert "E0210" in diagnostic_codes(lines.plan(on_four, gpus=4, device_kind="cpu"))
    assert "E0210" not in diagnostic_codes(lines.plan(on_four, gpus=4, device_kind="cuda"))
    by_name = {c.name: c for c in four.components}
    assert by_name["mesh"] == {c.name: c for c in cuda.components}["mesh"]
    assert four.bytes_device < cuda.bytes_device / 3
    assert four.bytes_total == 4 * four.bytes_per_gpu
    fp64 = memory(
        CRM_HL.replace("autan.SA())", 'autan.SA(), precision="fp64")'), device_kind="cuda"
    )
    assert fp64.bytes_device > cuda.bytes_device
    # The components plan.md gives in closed form, on 10 hexahedra, 40 faces (20 boundary),
    # 30 nodes and 160 face nodes, 5 equations: the mesh 24N + 16F + 4P + 12F_i + 4F_b + 20C,
    # its geometry 56F + 32C and colourings 12C; the output gathered and copied, 2 x 8C bytes
    # for each of a steady run's 2(n + 1) + n + (n + 2) + (n + 5) = 34 fields.
    tiny = autan.load(write_case(TINY)).plan(tiny_mesh(20), device_kind="cpu").memory
    by_name = {c.name: c for c in tiny.components}
    assert by_name["mesh"].bytes_host == 720 + 640 + 640 + 240 + 80 + 200 + 2240 + 320 + 120
    assert by_name["runtime"].bytes_host == 26 * 2**20  # measured, plan.md
    # The solver input at most: the case's TOML escaped (6 B a byte), the MeshInfo indented
    # one level deeper (4 B a line), each zone's name and resolved condition or medium as JSON
    # with 1 KiB of entry, three paths at PATH_MAX escaped, and 64 KiB; kept once, read and
    # parsed once more (41 times).
    tiny_case = autan.load(write_case(TINY))
    resolved = tiny_case.plan(tiny_mesh(20), device_kind="cpu")
    [farfield], [fluid] = resolved.boundaries, resolved.cell_zones

    def escaped(value: object) -> int:
        return len(json.dumps(value))

    toml = len(tiny_case.to_toml().encode()) - len("m.cgns") + 4096  # the path at PATH_MAX
    bound = (
        6 * toml
        + len(mesh_text := json.dumps(tiny_mesh(20).document(), indent=2))
        + 4 * (mesh_text.count("\n") + 1)
        + escaped("farfield")
        + escaped(farfield.condition)
        + 1024
        + escaped("fluid")
        + escaped(fluid.medium)
        + 1024
        + 3 * 6 * 4096
    )
    assert by_name["solver input"].bytes_host == bound + 2**16
    assert by_name["solver input setup"].bytes_host == 41 * (bound + 2**16)
    # The writer adds the coordinates 24N and 64-bit connectivity 8 max(F, C) + 8 max(P, K).
    assert by_name["solution output"].bytes_host == 2 * 34 * 80 + 24 * 30 + 8 * 41 + 8 * 160
    # RCM order keeps the renumbered mesh (the mesh terms without geometry) and its maps.
    renumbered = 720 + 640 + 640 + 240 + 80 + 200 + 4 * 10 + 5 * 40 + 4 * 30
    assert by_name["renumbered mesh"].bytes_host == renumbered
    # Reading a CGNS file (plan.md, the readers' inventory), 10 hexahedra, 20 boundary faces:
    # elements 144 B a hexahedron, 112 B a boundary face, labels 64 B per boundary face and per
    # cell, then 24N + 4C + 52F + 16P; the file's 2 048 B of descriptors (the library holds
    # them from cg_open), its 512 B of ZoneNames parsed (40 B a byte), and HDF5's default
    # caches: 4 MiB of metadata, 8 MiB of chunks and a 64 KiB sieve.
    standard = 144 * 10 + 112 * 20 + 64 * 20 + 64 * 10 + 24 * 30 + 4 * 10 + 52 * 40 + 16 * 160
    # Sections read in chunks through 1 MiB and 32 KiB of buffers, the edges and points they
    # skip kept as runs, 24 B for each of the 30 elements and one more.
    standard += 2**20 + 32 * 2**10 + 24 * (10 + 20 + 1)
    caches = 4 * 2**20 + 8 * 2**20 + 64 * 2**10
    # cg_open's tree of 40 nodes, 1 MiB and 16 KiB a node, and the 1 MiB of data it reads at once;
    # 8 B for each of its 128 point-list entries, read through a 1 MiB chunk
    tree = 2**20 + 16 * 2**10 * 40 + 2**20 + 8 * 128 + 2**20
    assert by_name["mesh reading"].bytes_host == standard + 2048 + 40 * 512 + caches + tree
    # A Fluent file of 4 096 B: S + 53N_s + 20C + 16P + 45F, and 16 B for each of the 16 384
    # fluid and solid zone records it may hold.
    fluent: dict[str, Any] = tiny_mesh(20).document()
    fluent["source"].update(
        format="fluent", logical_bytes=4096, descriptor_bytes=0, zone_names_bytes=0, cgns_nodes=0
    )
    read = autan.load(write_case(TINY)).plan(
        autan.MeshInfo.from_document(fluent, autan.Location()), device_kind="cpu"
    )
    reading = {c.name: c for c in read.memory.components}["mesh reading"].bytes_host
    assert reading == 4096 + 53 * 30 + 20 * 10 + 16 * 160 + 45 * 40 + 16 * 16_384
    # A Gmsh file of 4 096 B: S + 60N_s - 24N - 8C + 52F + 20P, and its batch of 65 536
    # element-block entities of 8 B.
    fluent["source"]["format"] = "gmsh"
    read = autan.load(write_case(TINY)).plan(
        autan.MeshInfo.from_document(fluent, autan.Location()), device_kind="cpu"
    )
    reading = {c.name: c for c in read.memory.components}["mesh reading"].bytes_host
    assert reading == 4096 + 60 * 30 - 24 * 30 - 8 * 10 + 52 * 40 + 20 * 160 + 65_536 * 8
    filtered: dict[str, Any] = tiny_mesh(20).document()
    filtered["source"]["filtered_bytes"] = 3 * 2**20  # a dataset inflated beside its input
    compressed = autan.load(write_case(TINY)).plan(
        autan.MeshInfo.from_document(filtered, autan.Location()), device_kind="cpu"
    )
    reading = {c.name: c for c in compressed.memory.components}["mesh reading"].bytes_host
    assert reading == by_name["mesh reading"].bytes_host + 3 * 2**20
    # The multigrid's levels from quality.multigrid: one coarse level of 2 nodes and 1 edge adds
    # 1000 per node and 410 per edge, and moves the dense LU from 5 x 10 to 5 x 2 unknowns.
    two_levels: dict[str, Any] = tiny_mesh(20).document()
    two_levels["quality"]["multigrid"] = {"cells_by_level": [10, 2], "edges_by_level": [20, 1]}
    deeper = autan.load(write_case(TINY)).plan(
        autan.MeshInfo.from_document(two_levels, autan.Location()), device_kind="cpu"
    )
    multigrid = {c.name: c for c in deeper.memory.components}["multigrid"].bytes_host
    dense = 8 * 50 * 50 + 8 * 50, 8 * 10 * 10 + 8 * 10
    assert multigrid - by_name["multigrid"].bytes_host == 1000 * 2 + 410 * 1 + dense[1] - dense[0]
    # The physics' setup (setup-memory.md): the packed geometry 8(25F + 4C + 4K), the stencil's
    # twins and FP64 weights 32 a stencil entry, its marks 4C + 4F_b and a vertex stencil's node
    # lists 16N + 8P, on 10 cells, 40 faces (20 boundary), K = 2F_i + F_b = 60 incidences,
    # 240 stencil entries, 30 nodes and 160 face nodes.
    marks = 4 * 10 + 4 * 20 + 16 * 30 + 8 * 160
    expected_setup = 8 * (25 * 40 + 4 * 10 + 4 * 60) + 32 * 240 + marks
    assert by_name["physics setup"].bytes_host == expected_setup
    # Under reproducibility "ranks", the global order: 12K + F kept by the physics, and a first
    # 12K built and freed while packing.
    ranks = autan.load(
        write_case(TINY.replace("Inviscid())", 'Inviscid(), reproducibility="ranks")'))
    ).plan(tiny_mesh(20), device_kind="cpu")
    keyed = {c.name: c for c in ranks.memory.components}
    assert keyed["physics"].bytes_host - by_name["physics"].bytes_host == 12 * 60 + 40
    assert keyed["physics setup"].bytes_host - expected_setup == 12 * 60
    # A device run's face scratch: 5 values of 8 B a face, on the device.
    cuda = autan.load(write_case(TINY)).plan(tiny_mesh(20), device_kind="cuda").memory
    assert {c.name: c for c in cuda.components}["device face scratch"].bytes_device == 40 * 40
    # Building the hierarchy (setup-memory.md): the larger of the isotropic loop,
    # 96 F_i + 284 C + 72 F_b, and the coarse-edge build, 69 F_i + 48 F_1 + 296 C + 156 C_1 + 144 F_b,
    # on 20 interior and 20 boundary faces, and the coarse level's 2 cells and 1 edge.
    assert by_name["multigrid setup"].bytes_host == 69 * 20 + 296 * 10 + 144 * 20
    setup = {c.name: c for c in deeper.memory.components}["multigrid setup"].bytes_host
    assert setup == 69 * 20 + 48 * 1 + 296 * 10 + 156 * 2 + 144 * 20
    # A start reads the previous solution on every rank and takes the rank's share: 8 (C + C_r)
    # bytes per field, the 34 written fields for a continuation, the 6 primitives for a warm one;
    # and its descriptors at their bounds, held from cg_open: run metadata of 64 KiB, also
    # copied and parsed (40 B a byte), and zone names of 25 600 B for each of the 2 zones and
    # the document; HDF5's default caches; and cg_open's tree, 1 MiB and 16 KiB a node, at the
    # 16 384 nodes the runtime accepts in a start file.
    modes: tuple[tuple[Literal["continue", "warm"], int], ...] = (("continue", 34), ("warm", 6))
    for mode, fields in modes:
        started = autan.load(write_case(TINY)).plan(tiny_mesh(20), device_kind="cpu", start=mode)
        assert {c.name: c for c in started.memory.components}["start"].bytes_host == (
            8 * (10 + 10) * fields
            + 42 * 64 * 2**10
            + 25_600 * 3
            + caches
            + 2**20
            + 16 * 2**10 * 16_384
            + 2**20
        )


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
