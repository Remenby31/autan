import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.solver_input
import pytest

from tests.conftest import CRM_HL

# Overlapping groups as a zones.json gives them: families in capitals, components dotted.
GROUPS = {
    "FUSE": ["fuselage"],
    "WING": ["wing.slat", "wing.main", "wing.flap"],
    "wing.high-lift": ["wing.slat", "wing.flap"],
    "wing.main": ["wing.main"],
}
SIDECAR = json.dumps(
    {
        "schema": "autan.zone-groups/1",
        "mesh_sha256": "",
        "provenance": {"note": "test"},
        "groups": GROUPS,
    }
).encode()
SIDECAR_SHA256 = hashlib.sha256(SIDECAR).hexdigest()


def with_groups(document: dict[str, Any]) -> None:
    document["groups"] = [{"name": n, "zones": z} for n, z in sorted(GROUPS.items())]
    document["groups_source"] = {
        "path": "crm-hl.cgns.zones.json",
        "sha256": SIDECAR_SHA256,
        "provenance": {"note": "test"},
    }


@pytest.fixture
def grouped(
    mesh_variant: Callable[..., autan.MeshInfo], write_case: Callable[..., Path]
) -> autan.MeshInfo:
    write_case(SIDECAR.decode(), "crm-hl.cgns.zones.json")
    return mesh_variant("crm-hl", with_groups)


BODY = (
    CRM_HL.replace('case.boundary("wing.*").wall()', 'case.boundary(group="WING").wall()')
    .replace('autan.forces("wing.*"', 'autan.forces(group="wing.*"')
    .replace(
        'case.boundary("symmetry").symmetry()',
        'case.boundary("symmetry").symmetry()\ncase.boundary(group="FUSE").wall()',
    )
)


def test_a_group_selects_the_union_of_its_groups_zones(
    grouped: autan.MeshInfo, write_case: Callable[..., Path]
) -> None:
    case = autan.load(write_case(BODY))
    plan = case.plan(grouped)
    assert plan.ok, [d.render() for d in plan.diagnostics]
    rules = {b.zone: b.rule for b in plan.boundaries}
    assert rules == {
        "wing.slat": 0,
        "wing.main": 0,
        "wing.flap": 0,
        "fuselage": 2,
        "empennage": None,
        "symmetry": 1,
        "farfield": None,
    }
    document = autan.solver_input.build(
        case, grouped, devices=autan.Devices(kind="cpu", count=1), deterministic=False
    )
    # wing.* takes wing.high-lift and wing.main: each zone once, in MeshInfo order.
    assert document.monitors[0].zones == ("wing.slat", "wing.main", "wing.flap")
    assert document.mesh.groups_sha256 == SIDECAR_SHA256
    assert grouped.groups_of("wing.slat") == ("WING", "wing.high-lift")


@pytest.mark.parametrize(
    ("replace", "code"),
    [
        (('group="WING"', 'group="TAIL"'), "E0426"),
        (('autan.forces(group="wing.*"', 'autan.forces(group="tail.*"'), "E0426"),
        (('group="FUSE"', 'group="wing.main"'), "E0411"),  # wing.main is also in WING
    ],
)
def test_group_selection_errors(
    replace: tuple[str, str],
    code: str,
    grouped: autan.MeshInfo,
    write_case: Callable[..., Path],
) -> None:
    plan = autan.load(write_case(BODY.replace(*replace))).plan(grouped)
    assert code in [d.code for d in plan.diagnostics]


def test_a_point_vortex_takes_a_group(
    grouped: autan.MeshInfo, write_case: Callable[..., Path]
) -> None:
    body = BODY.replace(
        'case.boundary("symmetry").symmetry()',
        'case.boundary("symmetry").symmetry()\ncase.boundary("farfield").farfield('
        'point_vortex=autan.PointVortex(group="{}", center=(0 * m, 0 * m, 0 * m)))',
    )
    assert autan.load(write_case(body.format("WING"))).plan(grouped).ok
    plan = autan.load(write_case(body.format("TAIL"))).plan(grouped)
    assert [d.code for d in plan.diagnostics if d.severity == "error"] == ["E0426"]


@pytest.mark.parametrize("selection", ['"wing.*", group="WING"', ""])
def test_a_selection_gives_zones_or_a_group(
    selection: str, write_case: Callable[..., Path]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        autan.load(write_case(CRM_HL + f"case.boundary({selection}).wall()\n")).spec  # noqa: B018
    assert error.value.codes == ["E0003"]


@pytest.mark.parametrize(
    ("sidecar", "groups", "code"),
    [
        (None, True, "W0211"),  # the MeshInfo has groups, the file is gone
        (b"{}", True, "E0210"),  # the file changed since the MeshInfo
        (SIDECAR, False, "E0210"),  # a file the MeshInfo does not know
    ],
)
def test_plan_checks_the_groups_file_like_the_mesh(
    sidecar: bytes | None,
    groups: bool,
    code: str,
    mesh_variant: Callable[..., autan.MeshInfo],
    write_case: Callable[..., Path],
) -> None:
    path = write_case(CRM_HL)
    if sidecar is not None:
        (path.parent / "crm-hl.cgns.zones.json").write_bytes(sidecar)
    mesh = mesh_variant("crm-hl", with_groups if groups else lambda _: None)
    assert code in [d.code for d in autan.load(path).plan(mesh).diagnostics]


def test_a_run_records_the_groups_file_in_its_identity(
    grouped: autan.MeshInfo, write_case: Callable[..., Path], tmp_path: Path
) -> None:
    case = autan.load(write_case(BODY))
    result = case.run(grouped, backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs")
    run = result.run
    assert run.mesh.groups_sha256 == SIDECAR_SHA256
    parts = [run.case.sha256, run.mesh.sha256, SIDECAR_SHA256, run.solver.commit]
    parts += [run.precision, run.solver.backend, run.devices.kind, str(run.devices.count)]
    assert run.identity == hashlib.sha256("\n".join(parts).encode()).hexdigest()
    assert autan.Results.open(result.path).run == run  # mesh-info.json and run.json agree
