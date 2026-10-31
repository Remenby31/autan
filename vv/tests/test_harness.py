"""The verification harness on run directories from real producers.

- `autan.FakeBackend` runs (the SDK's producer) against the freestream oracle computed independently
  from case-schema.md: the fake writes the freestream state, so errors are rounding.
- Runs written through `autan`'s writers (`autan_vv.runs`) on the V&V mesh families, with fields
  equal to an oracle plus an error of known size C h^p g(x): the harness must recover p, pass the
  gate for p = 2 and fail it for p = 1.
"""

import json
import math
import re
from pathlib import Path

import autan
import numpy as np
import pytest
from autan.units import K, Pa, kg, mol

from autan_vv.cli import main
from autan_vv.errors import VVError
from autan_vv.gates import GATES
from autan_vv.harness import UNIVERSAL_GAS_CONSTANT, read_geometry, verify
from autan_vv.meshes.families import generate_family
from autan_vv.oracles.registry import get_oracle
from autan_vv.runs import mesh_info, write_run

pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
ROOT = Path(__file__).resolve().parents[2]


def _gas(gas_constant: float, gamma: float = 1.4) -> autan.IdealGas:
    return autan.IdealGas(
        name="manufactured",
        molar_mass=UNIVERSAL_GAS_CONSTANT / gas_constant * kg / mol,
        gamma=gamma,
        prandtl=0.72,
        viscosity=autan.ConstantViscosity(mu=10.0 * Pa * autan.units.s),
    )


def _case(path: Path, oracle_name: str, *, turbulence: object = None) -> autan.Case:
    oracle = get_oracle(oracle_name) if oracle_name != "freestream" else None
    case = autan.Case(f"verify-{oracle_name}", mesh=path)
    gas = _gas(oracle.gas_constant if oracle else 287.0)
    case.physics = autan.Compressible(turbulence=turbulence or autan.Inviscid(), precision="fp64", gas=gas)
    case.freestream(mach=0.3, pressure=1e5 * Pa, temperature=300 * K)
    case.boundary("*").farfield()
    return case


def _family(tmp_path: Path, kind: str, levels: tuple[int, ...], oracle_name: str, order: float,
            amplitude: float = 0.05) -> list[Path]:
    """Runs whose fields are the oracle times (1 + amplitude h^order sin(...)): errors ~ h^order."""
    oracle = get_oracle(oracle_name)
    family, meshes = generate_family(kind, levels, tmp_path / kind)
    paths = []
    for level, mesh in zip(family.levels, meshes, strict=True):
        path = tmp_path / kind / level.file
        info = mesh_info(mesh, path, "fluent" if kind == "poly" else "gmsh")
        case = _case(path, oracle_name)
        x = mesh.geometry.cell_centroid
        g = 1.0 + 0.5 * np.sin(2 * x[:, 0] + x[:, 1]) * np.cos(x[:, 2])
        exact = oracle.fields(x)
        h = (1.0 / mesh.n_cells) ** (1 / 3)
        fields = {k: v * (1.0 + amplitude * h**order * g) for k, v in exact.items()}
        paths.append(write_run(tmp_path / "runs", case, mesh, info, fields).path)
    return paths


@pytest.mark.parametrize("kind", ["hex", "poly", "hybrid"])
def test_geometry_rebuilt_from_solution_cgns_matches_the_mesh(tmp_path: Path, kind: str) -> None:
    (path,) = _family(tmp_path, kind, (4,), "mms-euler", 2.0)
    results = autan.Results.open(path)
    rebuilt = read_geometry(results)
    _, meshes = generate_family(kind, (4,), tmp_path / "again")
    original = meshes[0].geometry
    np.testing.assert_allclose(rebuilt.geometry.cell_volume, original.cell_volume, rtol=1e-13)
    np.testing.assert_allclose(rebuilt.geometry.cell_centroid, original.cell_centroid, rtol=0, atol=1e-14)
    assert rebuilt.geometry.cell_volume.sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize(("order", "passes"), [(2.0, True), (1.0, False)])
def test_order_gate_recovers_the_order_of_the_error(tmp_path: Path, order: float, passes: bool) -> None:
    runs = _family(tmp_path, "hex", (4, 8, 16), "mms-euler", order)
    report = verify(runs, "mms-euler", "euler-mms-order")
    for name in ("rho", "rhou", "rhoE", "Pressure"):
        fit = report["fits"][name]["L2"]
        assert fit["order"] == pytest.approx(order, abs=0.02)
        assert fit["pairwise_orders"][-1] == pytest.approx(order, abs=0.02)
    verdict = report["verdict"]
    assert verdict["pass"] is passes
    assert verdict["evidence"] is False  # these runs carry backend "fake"
    assert [r["n_cells"] for r in report["runs"]] == [64, 512, 4096]
    assert all(len(r["mesh_sha256"]) == 64 and len(r["solver_commit"]) == 40 for r in report["runs"])


def test_fake_backend_run_preserves_the_freestream(tmp_path: Path) -> None:
    family, meshes = generate_family("hex", (3,), tmp_path / "mesh")
    path = tmp_path / "mesh" / family.levels[0].file
    case = autan.Case("fake-freestream", mesh=path)
    case.physics = autan.Compressible(turbulence=autan.Laminar(), precision="fp64")
    case.freestream(mach=0.4, re=2.0e6, ref_length=1.0 * autan.units.m, temperature=250 * K, aoa=3.0 * autan.units.deg)
    case.boundary("*").farfield()
    results = case.run(mesh_info(meshes[0], path, "gmsh"), backend=autan.FakeBackend(), runs=tmp_path / "runs")
    report = verify([results.path], "freestream", "freestream-preservation")
    verdict = report["verdict"]
    assert verdict["pass"], verdict["checks"]
    assert not verdict["evidence"]
    assert any("fake" in reason for reason in verdict["not_evidence_because"])
    (run,) = report["runs"]
    assert run["n_cells"] == 512  # the fake's own 8^3 cube, not the case mesh


def test_runs_that_do_not_fit_the_oracle_are_rejected(tmp_path: Path) -> None:
    (path,) = _family(tmp_path, "hex", (4,), "mms-euler", 2.0)
    for against in ("ringleb", "couette"):
        with pytest.raises(VVError) as err:
            verify([path], against)
        assert err.value.code == "V0724"
    with pytest.raises(VVError) as err:
        verify([path, path, path], "mms-euler")
    assert err.value.code == "V0727"
    with pytest.raises(VVError) as err:
        verify([path], "mms-euler", "euler-mms-order")
    assert err.value.code == "V0729"
    with pytest.raises(VVError) as err:
        verify([path], "mms-euler", "ringleb-order")
    assert err.value.code == "V0728"


def test_two_dimensional_oracles_need_one_cell_thick_meshes(tmp_path: Path) -> None:
    (path,) = _family(tmp_path, "hex", (4,), "mms-euler", 2.0)
    # The case is Euler with R = 287, as the vortex is not: compatibility is checked first.
    with pytest.raises(VVError) as err:
        verify([path], "isentropic-vortex")
    assert err.value.code == "V0724"


def test_cli_verify_writes_a_strict_report_and_exits_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = [str(p) for p in _family(tmp_path, "hex", (4, 8, 16), "mms-euler", 1.0)]
    out = tmp_path / "verdict.json"
    with pytest.raises(SystemExit) as exit_info:
        main(["verify", *runs, "--against", "mms-euler", "--gate", "euler-mms-order", "--out", str(out)])
    assert exit_info.value.code == 1
    assert "gate euler-mms-order: FAIL" in capsys.readouterr().out

    def reject(token: str) -> None:
        raise AssertionError(token)

    report = json.loads(out.read_text(), parse_constant=reject)
    assert report["schema"] == "autan.vv-verdict/1"


def test_every_gate_is_stated_in_acceptance_md() -> None:
    text = (ROOT / "docs" / "validation" / "acceptance.md").read_text()
    for gate in GATES.values():
        row = next((line for line in text.splitlines() if f"`{gate.id}`" in line), None)
        assert row is not None, gate.id
        numbers = {float(v) for v in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", row)}
        if gate.measure == "order":
            assert {gate.lower, gate.upper} <= numbers or math.isclose(gate.upper - gate.lower, 0.2), gate.id
        else:
            assert gate.upper in numbers, gate.id
