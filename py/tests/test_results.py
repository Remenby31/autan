"""Run directories written by the fake backend, read back through ``cfd.Results``.

Expected values come from docs/contracts/results.md and case-schema.md (ideal-gas
law, Sutherland's law, the force projections), recomputed here independently.
"""

import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cfd
import numpy as np
import pytest
from cfd import cgns
from cfd.cli import main

from tests.conftest import CRM_HL, EXAMPLES, FIXTURES

R_UNIVERSAL = 8.314462618
CGNSCHECK = shutil.which("cgnscheck")


@pytest.fixture
def crm_run(tmp_path: Path, crm_mesh: cfd.MeshInfo) -> cfd.Results:
    return cfd.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=cfd.FakeBackend(iterations=60), runs=tmp_path / "runs"
    )


def heat_run(runs: Path, **options: Any) -> cfd.Results:
    mesh = cfd.MeshInfo.load(FIXTURES / "heat-exchanger.meshinfo.json")
    return cfd.load(EXAMPLES / "heat_exchanger.py").run(
        mesh, backend=cfd.FakeBackend(**options), runs=runs
    )


def test_run_directory_and_metadata_follow_the_contract(crm_run: cfd.Results) -> None:
    path = crm_run.path
    assert sorted(p.name for p in path.iterdir()) == [
        "case.toml",
        "history.jsonl",
        "mesh-info.json",
        "run.json",
        "solution.cgns",
    ]
    run = json.loads((path / "run.json").read_text())
    assert run["schema"] == "cfd.results/1"
    assert re.fullmatch(r"run-\d{8}T\d{6}Z-[0-9a-f]{8}", run["run_id"])
    assert path.name == run["run_id"]
    assert run["status"] == "converged"
    assert run["solver"]["backend"] == "fake"
    assert re.fullmatch(r"[0-9a-f]{40}", run["solver"]["commit"])
    assert run["case"]["sha256"] == cfd.load(path / "case.toml").spec.hash()
    assert run["mesh"]["path"] == str((EXAMPLES / "crm-hl.cgns").resolve())
    parts = [
        run["case"]["sha256"],
        run["mesh"]["sha256"],
        run["solver"]["commit"],
        run["precision"],
        "fake",
        run["devices"]["kind"],
        str(run["devices"]["count"]),
    ]
    assert run["identity"] == hashlib.sha256("\n".join(parts).encode()).hexdigest()
    assert run["iterations"] == 60
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", run["finished"])


def test_history_lines_carry_every_equation_and_monitors_at_their_period(
    crm_run: cfd.Results,
) -> None:
    lines = [json.loads(line) for line in (crm_run.path / "history.jsonl").read_text().splitlines()]
    assert [line["iteration"] for line in lines] == list(range(1, 61))
    for line in lines:
        assert list(line["residuals"]) == [
            "density",
            "momentum-x",
            "momentum-y",
            "momentum-z",
            "energy",
            "nu-tilde",
        ]
        assert ("forces" in line["monitors"]) == (line["iteration"] % 10 == 0)
        assert line["time"] is None
    first, last = lines[0]["residuals"]["density"], lines[-1]["residuals"]["density"]
    assert isinstance(crm_run.case.time, cfd.Steady)
    assert math.log10(first / last) >= crm_run.case.time.residual_orders - 1e-9


def test_force_coefficients_are_the_contract_projections(crm_run: cfd.Results) -> None:
    forces = crm_run.forces
    assert list(forces.iteration) == list(range(10, 61, 10))
    alpha = math.radians(19.57)
    e_d = np.array([math.cos(alpha), 0.0, math.sin(alpha)])
    e_l = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
    rho = crm_run.field("Density")[0]
    speed = float(np.linalg.norm([crm_run.field(f"Velocity{a}")[0] for a in "XYZ"]))
    q_area = 0.5 * rho * speed**2 * 191.84
    force = np.stack([forces.fx, forces.fy, forces.fz])
    np.testing.assert_allclose(forces.cd, e_d @ force / q_area, rtol=1e-12)
    np.testing.assert_allclose(forces.cl, e_l @ force / q_area, rtol=1e-12)
    np.testing.assert_allclose(forces.cs, np.cross(e_l, e_d) @ force / q_area, atol=1e-12)
    np.testing.assert_allclose(forces.cmy, forces.my / (q_area * 7.0), rtol=1e-12)


def test_crm_hl_freestream_fields_follow_ideal_gas_and_sutherland(crm_run: cfd.Results) -> None:
    temperature, mach, re_number, length = 288.15, 0.20, 5.49e6, 7.0
    r = R_UNIVERSAL / 0.0289647
    mu = 1.716e-5 * (temperature / 273.15) ** 1.5 * (273.15 + 110.4) / (temperature + 110.4)
    speed = mach * math.sqrt(1.4 * r * temperature)
    rho = re_number * mu / (speed * length)
    np.testing.assert_allclose(crm_run.field("Density"), rho, rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("Pressure"), rho * r * temperature, rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("Temperature"), temperature)
    alpha = math.radians(19.57)
    np.testing.assert_allclose(crm_run.field("VelocityX"), speed * math.cos(alpha), rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("VelocityZ"), speed * math.sin(alpha), rtol=1e-12)
    np.testing.assert_allclose(crm_run.field("TurbulentSANuTilde"), 3.0 * mu / rho, rtol=1e-12)
    faces, pressure = crm_run.wall("Pressure")
    assert faces.size > 0
    np.testing.assert_allclose(pressure, rho * r * temperature, rtol=1e-12)


def test_heat_exchanger_zones_carry_their_own_media(tmp_path: Path) -> None:
    result = heat_run(tmp_path)
    pressure, temperature = 101325.0, 300.0
    for zone, molar_mass in (("air", 0.0289647), ("co2", 0.0440095)):
        cells = result.cell_zone(zone)
        assert cells.size > 0
        expected = pressure / (R_UNIVERSAL / molar_mass * temperature)
        np.testing.assert_allclose(result.field("Density")[cells], expected, rtol=1e-12)
    plate = result.cell_zone("plate")
    np.testing.assert_allclose(result.field("Density")[plate], 8933.0)
    np.testing.assert_allclose(result.field("VelocityX")[plate], 0.0)
    assert set(result.history.equations) >= {"k", "omega"}
    cells = np.concatenate([result.cell_zone(z) for z in ("air", "co2", "plate")])
    assert sorted(cells.tolist()) == list(range(result.field("Density").size))


@pytest.mark.skipif(CGNSCHECK is None, reason="cgnscheck (CGNS library tools) not installed")
@pytest.mark.parametrize("example", ["crm_hl", "heat_exchanger"])
def test_solution_passes_the_cgns_library_checker(example: str, tmp_path: Path) -> None:
    result = (
        heat_run(tmp_path, iterations=5)
        if example == "heat_exchanger"
        else cfd.load(EXAMPLES / "crm_hl.py").run(
            cfd.MeshInfo.load(FIXTURES / "crm-hl.meshinfo.json"),
            backend=cfd.FakeBackend(iterations=5),
            runs=tmp_path,
        )
    )
    assert CGNSCHECK is not None
    done = subprocess.run(
        [CGNSCHECK, str(result.path / "solution.cgns")], capture_output=True, text=True, check=False
    )
    assert "ERROR" not in done.stdout + done.stderr
    assert done.returncode == 0


def test_cgns_tree_layout(crm_run: cfd.Results) -> None:
    tree = cgns.read(crm_run.path / "solution.cgns")
    base = tree.child("Base")
    zone = base.child("Zone")
    _, cells, _ = np.asarray(zone.data).ravel()
    faces = int(zone.child("NGonElements").child("ElementRange").data[1])
    assert list(zone.child("NFaceElements").child("ElementRange").data) == [
        faces + 1,
        faces + cells,
    ]
    names = json.loads(base.child("ZoneNames").text())
    assert set(names["boundary"].values()) == {z.name for z in crm_run.mesh_info.boundary_zones}
    bc_types = {b.name: b.text() for b in zone.child("ZoneBC").with_label("BC_t")}
    assert bc_types["wing.main"] == "BCWallViscousHeatFlux"
    assert bc_types["farfield"] == "BCFarfield"
    assert bc_types["symmetry"] == "BCSymmetryPlane"
    metadata = json.loads(base.child("RunMetadata").text())
    assert metadata["identity"] == crm_run.run.identity


def test_cgns_node_names_avoid_collisions(
    tmp_path: Path, mesh_variant: Callable[..., cfd.MeshInfo], write_case: Callable[[str], Path]
) -> None:
    long_name = "a-boundary-zone-name-longer-than-32"

    def edit(document: dict[str, Any]) -> None:
        document["boundary_zones"][0]["name"] = long_name
        document["boundary_zones"][1]["name"] = "fluid"
        document["boundary_zones"][2]["name"] = "ZoneBC"

    mesh = mesh_variant("crm-hl", edit)
    body = CRM_HL.replace('case.boundary("wing.*").wall()', 'case.boundary("*").wall()')
    body = body.replace('case.boundary("symmetry").symmetry()\n', "")
    case = cfd.load(write_case(body.replace('forces("wing.*"', 'forces("fuselage"')))
    result = case.run(mesh, backend=cfd.FakeBackend(iterations=2), runs=tmp_path)
    names = json.loads(
        cgns.read(result.path / "solution.cgns").child("Base").child("ZoneNames").text()
    )
    assert names["boundary"]["boundary-0000"] == long_name
    assert names["boundary"]["boundary-0001"] == "fluid"
    assert names["cell"]["cell-zone-0000"] == "fluid"
    assert names["boundary"]["ZoneBC"] == "ZoneBC"


def test_divergence_is_reported_with_its_cell(tmp_path: Path, crm_mesh: cfd.MeshInfo) -> None:
    result = cfd.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=cfd.FakeBackend(iterations=100, diverge_at=40), runs=tmp_path
    )
    assert result.run.status == "diverged"
    [found] = result.run.diagnostics
    assert found.code == "D0107"
    assert found.severity == "divergence"
    assert found.data["iteration"] == 40
    assert 0 <= found.data["cell"] < result.field("Density").size
    assert found.data["zone"] == "fluid"
    assert result.history.iterations[-1] == 39


def test_unsteady_run_records_time_and_average(
    tmp_path: Path, write_case: Callable[[str], Path], crm_mesh: cfd.MeshInfo
) -> None:
    body = CRM_HL.replace("cfd.SA()", "cfd.DDES(base=cfd.SA())") + (
        "from cfd.units import s\ncase.time = cfd.Unsteady(steps=40, dt=1e-4 * s, average_from=10)\n"
    )
    result = cfd.load(write_case(body)).run(crm_mesh, backend=cfd.FakeBackend(), runs=tmp_path)
    assert result.run.status == "completed"
    assert result.run.time == pytest.approx(40 * 1e-4)
    assert result.run.averaging is not None
    assert (result.run.averaging.from_step, result.run.averaging.to_step) == (11, 40)
    lines = [json.loads(line) for line in (result.path / "history.jsonl").read_text().splitlines()]
    assert lines[4]["time"] == pytest.approx(5e-4)
    np.testing.assert_array_equal(result.field("Density", mean=True), result.field("Density"))


def test_deterministic_runs_produce_identical_data(tmp_path: Path, crm_mesh: cfd.MeshInfo) -> None:
    case = cfd.load(EXAMPLES / "crm_hl.py")
    first = case.run(crm_mesh, backend=cfd.FakeBackend(iterations=20), runs=tmp_path)
    second = case.run(crm_mesh, backend=cfd.FakeBackend(iterations=20), runs=tmp_path)
    assert first.run.run_id != second.run.run_id
    assert first.run.identity == second.run.identity
    assert (first.path / "history.jsonl").read_bytes() == (
        second.path / "history.jsonl"
    ).read_bytes()
    for name in ("Density", "VelocityX", "Pressure", "TurbulentSANuTilde"):
        assert first.field(name).tobytes() == second.field(name).tobytes()


def test_history_is_appended_while_the_run_is_live(tmp_path: Path, crm_mesh: cfd.MeshInfo) -> None:
    case = cfd.load(EXAMPLES / "crm_hl.py")
    worker = threading.Thread(
        target=case.run,
        args=(crm_mesh,),
        kwargs={"backend": cfd.FakeBackend(iterations=40, pace=0.02), "runs": tmp_path},
    )
    worker.start()
    seen: tuple[str, int] | None = None
    deadline = time.monotonic() + 10
    while seen is None and time.monotonic() < deadline:
        for run_json in tmp_path.glob("*/run.json"):
            history = run_json.parent / "history.jsonl"
            lines = len(history.read_text().splitlines()) if history.exists() else 0
            status = json.loads(run_json.read_text())["status"]
            if status == "running" and 0 < lines < 40:
                seen = (status, lines)
        time.sleep(0.01)
    worker.join()
    assert seen is not None
    [path] = tmp_path.iterdir()
    assert cfd.Results.open(path).run.status == "converged"


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        (lambda p: (p / "run.json").write_text('{"schema": "cfd.results/2"}'), "E0701"),
        (
            lambda p: (p / "case.toml").write_text(
                (p / "case.toml").read_text().replace("mach = 0.2", "mach = 0.3")
            ),
            "E0701",
        ),
        (lambda p: (p / "solution.cgns").unlink(), "E0701"),
        (
            lambda p: (p / "history.jsonl").write_text(
                "\n".join(reversed((p / "history.jsonl").read_text().splitlines()))
            ),
            "E0701",
        ),
    ],
)
def test_reader_rejects_non_conforming_directories(
    damage: Callable[[Path], object], code: str, crm_run: cfd.Results
) -> None:
    damage(crm_run.path)
    with pytest.raises(cfd.CaseError) as error:
        cfd.Results.open(crm_run.path)
    assert error.value.codes == [code]


def test_diverged_run_without_solution_opens_for_its_history(
    tmp_path: Path, crm_mesh: cfd.MeshInfo
) -> None:
    result = cfd.load(EXAMPLES / "crm_hl.py").run(
        crm_mesh, backend=cfd.FakeBackend(iterations=30, diverge_at=10), runs=tmp_path
    )
    (result.path / "solution.cgns").unlink()
    reopened = cfd.Results.open(result.path)
    assert reopened.history.iterations[-1] == 9
    with pytest.raises(cfd.CaseError) as error:
        reopened.field("Density")
    assert error.value.codes == ["E0703"]


def test_run_needs_an_explicit_backend_and_a_valid_plan(
    tmp_path: Path, crm_mesh: cfd.MeshInfo, mesh_variant: Callable[..., cfd.MeshInfo]
) -> None:
    case = cfd.load(EXAMPLES / "crm_hl.py")
    with pytest.raises(cfd.CaseError) as error:
        case.run(crm_mesh, runs=tmp_path)
    assert error.value.codes == ["E0804"]

    def untype(document: dict[str, Any]) -> None:
        document["boundary_zones"][4]["type"] = None

    with pytest.raises(cfd.CaseError) as error:
        case.run(mesh_variant("crm-hl", untype), backend=cfd.FakeBackend(), runs=tmp_path)
    assert error.value.codes == ["E0412"]
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_results_lookup_errors(crm_run: cfd.Results) -> None:
    with pytest.raises(cfd.CaseError) as error:
        crm_run.history.monitor("lift")
    assert error.value.codes == ["E0702"]
    with pytest.raises(cfd.CaseError) as error:
        crm_run.field("Vorticity")
    assert error.value.codes == ["E0703"]


def test_cli_run_reports_divergence_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "run",
        str(EXAMPLES / "crm_hl.py"),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
        "--runs",
        str(tmp_path),
        "--json",
    ]
    assert main([*argv, "--iterations", "20", "--diverge-at", "5"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["status"] == "diverged"
    assert out["run"]["diagnostics"][0]["code"] == "D0107"
    assert cfd.Results.open(out["path"]).run.run_id == out["run"]["run_id"]
    assert main([*argv, "--iterations", "20"]) == 0
