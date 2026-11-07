"""The Sod cases of a sod-tube mesh family (autan-vv meshes sod-tube, autan-vv sod-cases)."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import autan
import numpy as np
from autan import initial_state
from autan.meshinfo import mesh_sha256
from autan_vv.cases.shock_tube import END_TIME, STEPS_PER_CELL
from autan_vv.cli import main
from autan_vv.meshes.families import BOX_ZONES, generate_level
from autan_vv.meshes.tube import WIDTH
from autan_vv.oracles.inviscid import GAMMA, shock_tube

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_a_level_is_one_layer_of_n_cells_along_x() -> None:
    # The order study of a 1D oracle measures h along x and needs every cell to span the tube's
    # cross-section (harness.md); the cross-section stays WIDTH x WIDTH at every level.
    for n in (4, 10):
        mesh, version = generate_level("sod-tube", n)
        assert version == ""
        assert mesh.n_cells == n
        assert {z.name for z in mesh.zones} == set(BOX_ZONES)
        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        assert np.allclose(lo, 0.0)
        assert np.allclose(hi, [1.0, WIDTH, WIDTH])
        # Every cell holds the eight corners of one slab [i/n, (i+1)/n] x [0, w]^2.
        assert mesh.points.shape[0] == 4 * (n + 1)


def test_every_level_solves_one_case_from_its_own_initial_state(tmp_path: Path) -> None:
    # t = 0.2 with the finest level's step on every level, every face a symmetry plane, each level
    # started from the exact solution at t = 0 on its own mesh.
    main(["meshes", "sod-tube", "--levels", "8,16", "--out", str(tmp_path / "mesh")])
    manifest = json.loads((tmp_path / "mesh" / "manifest.json").read_text())
    assert manifest["gmsh_version"] == ""
    main(["sod-cases", "--family", str(tmp_path / "mesh"), "--out", str(tmp_path / "cases")])
    documents = []
    for n in (8, 16):
        path = tmp_path / "cases" / f"sod-tube-n{n:03d}.toml"
        case = autan.load(path)
        spec = case.spec
        assert spec.time.kind == "unsteady"
        assert spec.time.dt is not None
        assert spec.time.steps == STEPS_PER_CELL * 16
        assert math.isclose(
            spec.time.steps * spec.time.dt.to("s").magnitude, END_TIME, rel_tol=1e-15
        )
        assert {r.zones for r in spec.boundaries} == set(BOX_ZONES)
        assert {r.condition.kind for r in spec.boundaries} == {"symmetry"}
        start = spec.initial_state
        assert start is not None
        state = case.resolve(start.path)
        assert start.sha256 == hashlib.sha256(state.read_bytes()).hexdigest()
        document = initial_state.parse(state.read_bytes())
        assert not isinstance(document, str), document
        assert document.mesh_sha256 == mesh_sha256(case.resolve(spec.mesh))
        assert document.cells == n
        dump = spec.model_dump(mode="json")
        for key in ("name", "mesh", "initial_state"):
            dump.pop(key)
        documents.append(dump)
    assert documents[0] == documents[1]


def test_the_ends_keep_the_initial_state_until_the_end_time() -> None:
    # The reflecting ends hold only if no wave reaches them by t = 0.2: the exact solution there is
    # still the initial state, from the left end to beyond its first cells and from beyond the
    # shock to the right end.
    oracle = shock_tube("sod")
    x = np.r_[np.linspace(0.0, 0.2, 41), np.linspace(0.9, 1.0, 21)]
    points = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    start, end = oracle.evaluate(points, None), oracle.evaluate(points, END_TIME)
    for field in ("Density", "VelocityX", "Pressure"):
        assert np.array_equal(start[field], end[field]), field


def test_the_finest_level_steps_at_a_cfl_number_below_one_quarter() -> None:
    # dt = END_TIME / (STEPS_PER_CELL n) and dx = 1 / n: the CFL number of the fastest wave of the
    # exact solution, sampled finely over the tube and the run, is dt max(|u| + c) / dx.
    oracle = shock_tube("sod")
    x = np.linspace(0.0, 1.0, 4001)
    points = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    fastest = 0.0
    for t in np.linspace(1e-3, END_TIME, 50):
        w = oracle.evaluate(points, float(t))
        c = np.sqrt(GAMMA * w["Pressure"] / w["Density"])
        fastest = max(fastest, float(np.max(np.abs(w["VelocityX"]) + c)))
    assert END_TIME / STEPS_PER_CELL * fastest < 0.25


def test_a_mesh_file_other_than_its_manifest_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The initial state is computed on the regenerated level: a file the manifest does not record
    # would be paired with centroids that are not its own.
    main(["meshes", "sod-tube", "--levels", "8", "--out", str(tmp_path / "mesh")])
    main(["meshes", "sod-tube", "--levels", "16", "--out", str(tmp_path / "other")])
    (tmp_path / "mesh" / "sod-tube-n008.msh").write_bytes(
        (tmp_path / "other" / "sod-tube-n016.msh").read_bytes()
    )
    family, out = str(tmp_path / "mesh"), str(tmp_path / "c")
    assert main(["sod-cases", "--family", family, "--out", out]) == 2
    assert "V0805" in capsys.readouterr().err
