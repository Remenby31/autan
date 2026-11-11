"""The Couette cases of a couette mesh family (autan-vv meshes couette, autan-vv couette-cases)."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import autan
import numpy as np
import pytest
from autan import face_states
from autan.meshinfo import mesh_sha256
from autan.results import field_names
from autan.schema import ConstantViscosity, OracleState, Steady, Wall
from autan_vv.cli import main
from autan_vv.errors import VVError
from autan_vv.harness import verify
from autan_vv.meshes.families import generate_level
from autan_vv.runs import mesh_info, write_run

if TYPE_CHECKING:
    from pathlib import Path

# Issue #596's definition, from White (2006), section 3.2: u = U y/H and
# T = T_b + (T_t - T_b) y/H + Pr U^2/(2 cp) (y/H)(1 - y/H), uniform p, rho = p/(R T).
H, U, T_B, T_T, P, MU, PR, GAMMA, R = 1.0, 200.0, 300.0, 320.0, 1.0e5, 1.0e-2, 0.72, 1.4, 287.0
CP = GAMMA * R / (GAMMA - 1.0)
# The column's side across x and z, the same at every level: the cell of the finest level, n = 128.
WIDTH = 1.0 / 128.0
ZONES = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
LEVELS = (8, 16, 32, 64)


def _exact(y: np.ndarray) -> dict[str, np.ndarray]:
    eta = y / H
    t = T_B + (T_T - T_B) * eta + PR * U**2 / (2.0 * CP) * eta * (1.0 - eta)
    zero = np.zeros_like(eta)
    return {
        "Density": P / (R * t),
        "VelocityX": U * eta,
        "VelocityY": zero,
        "VelocityZ": zero,
        "Pressure": np.full_like(eta, P),
    }


def test_a_level_is_one_column_of_n_cells_along_y() -> None:
    # A 1D oracle along y: one layer across x and z, the same at every level (harness.md).
    for n in (4, 10):
        mesh, version = generate_level("couette", n)
        assert version == ""
        assert mesh.n_cells == n
        assert {z.name for z in mesh.zones} == ZONES
        np.testing.assert_array_equal(mesh.points.min(axis=0), 0.0)
        np.testing.assert_allclose(mesh.points.max(axis=0), [WIDTH, H, WIDTH], rtol=1e-15)
        np.testing.assert_allclose(mesh.geometry.cell_volume, WIDTH**2 * H / n, rtol=1e-12)
    for n, perturbation in ((0, 0.0), (8, 0.3)):
        with pytest.raises(VVError) as err:
            generate_level("couette", n, perturbation=perturbation)
        assert err.value.code == "V0309"


def test_every_level_imposes_the_exact_states_and_solves_one_case(tmp_path: Path) -> None:
    levels = ",".join(map(str, LEVELS))
    main(["meshes", "couette", "--levels", levels, "--out", str(tmp_path / "mesh")])
    main(["couette-cases", "--family", str(tmp_path / "mesh"), "--out", str(tmp_path / "cases")])
    documents = []
    for n in LEVELS:
        case = autan.load(tmp_path / "cases" / f"couette-n{n:03d}.toml")
        spec = case.spec
        kinds = {r.zones: r.condition.kind for r in spec.boundaries}
        assert kinds == {
            "ymin": "wall",
            "ymax": "oracle-state",
            "xmin": "oracle-state",
            "xmax": "oracle-state",
            "zmin": "symmetry",
            "zmax": "symmetry",
        }
        wall = next(r.condition for r in spec.boundaries if r.zones == "ymin")
        assert isinstance(wall, Wall)
        assert wall.temperature is not None
        assert math.isclose(wall.temperature.to("K").magnitude, T_B)
        gas = spec.physics.gas  # type: ignore[union-attr]
        assert isinstance(gas.viscosity, ConstantViscosity)
        assert math.isclose(gas.viscosity.mu.to("Pa * s").magnitude, MU, rel_tol=1e-15)
        assert math.isclose(gas.prandtl, PR, rel_tol=1e-15)
        assert math.isclose(gas.gamma, GAMMA, rel_tol=1e-15)
        molar = gas.molar_mass.to("kg / mol").magnitude
        assert math.isclose(8.314462618 / molar, R, rel_tol=1e-12)
        assert isinstance(spec.time, Steady)
        assert spec.time.linear_solver == "gauss-seidel"
        assert spec.time.max_iterations == 5000
        fs = spec.freestream
        assert fs.mach is not None
        assert math.isclose(
            fs.mach * math.sqrt(GAMMA * R * 0.5 * (T_B + T_T)), U / 2, rel_tol=1e-12
        )
        # The SDK reads the face states as `autan plan` does: valid, for this mesh, the oracle
        # at every face centroid of the three zones.
        condition = next(r.condition for r in spec.boundaries if r.zones == "ymax")
        assert isinstance(condition, OracleState)
        states = case.resolve(condition.path)
        assert condition.sha256 == hashlib.sha256(states.read_bytes()).hexdigest()
        document = face_states.parse(states.read_bytes())
        assert not isinstance(document, str), document
        assert document.mesh_sha256 == mesh_sha256(case.resolve(spec.mesh))
        assert set(document.zones) == {"ymax", "xmin", "xmax"}
        y = (np.arange(n) + 0.5) * H / n
        mid = np.full(n, WIDTH / 2)
        expected = {
            "ymax": np.array([[WIDTH / 2, H, WIDTH / 2]]),
            "xmin": np.column_stack([np.zeros(n), y, mid]),
            "xmax": np.column_stack([np.full(n, WIDTH), y, mid]),
        }
        for name, zone in document.zones.items():
            centroids = np.array(zone.centroids)
            assert zone.faces == len(expected[name])
            order = np.argsort(centroids[:, 1])
            np.testing.assert_allclose(centroids[order], expected[name], rtol=0, atol=1e-15)
            exact = _exact(centroids[:, 1])
            values = np.array(zone.values)
            for k, field in enumerate(exact):
                np.testing.assert_allclose(values[:, k], exact[field], rtol=1e-14, atol=1e-12)
        dump = spec.model_dump(mode="json")
        dump.pop("name")
        dump.pop("mesh")
        for rule in dump["boundaries"]:
            rule["condition"].pop("path", None)
            rule["condition"].pop("sha256", None)
        documents.append(dump)
    assert all(d == documents[0] for d in documents)


def test_the_cases_meet_both_gates_with_the_exact_velocity(tmp_path: Path) -> None:
    # The real cases through the harness: it accepts their gas and walls as the oracle's (it refuses
    # a mismatch with V0724), the exact velocity passes the max-error gate, and a density error of
    # order 2 gives a temperature order of 2.
    levels = ",".join(map(str, LEVELS))
    main(["meshes", "couette", "--levels", levels, "--out", str(tmp_path / "mesh")])
    main(["couette-cases", "--family", str(tmp_path / "mesh"), "--out", str(tmp_path / "cases")])
    manifest = json.loads((tmp_path / "mesh" / "manifest.json").read_text())
    runs = []
    for level in manifest["levels"]:
        case = autan.load(tmp_path / "cases" / f"couette-n{level['n']:03d}.toml")
        mesh, _ = generate_level("couette", level["n"])
        y = mesh.geometry.cell_centroid[:, 1]
        exact = _exact(y)
        h = H / level["n"]
        exact["Density"] = exact["Density"] * (1.0 + 0.05 * h**2 * (1.0 + 0.5 * np.sin(3.0 * y)))
        exact["Temperature"] = exact["Pressure"] / (R * exact["Density"])
        fields = {k: exact[k] for k in field_names(case.spec)}
        info = mesh_info(mesh, case.resolve(case.spec.mesh), "fluent")
        runs.append(write_run(tmp_path / "runs", case, mesh, info, fields).path)
    for gate in ("couette-velocity-exact", "couette-temperature-order"):
        verdict = verify(runs, "couette", gate)["verdict"]
        assert verdict["pass"], (gate, verdict["checks"])
