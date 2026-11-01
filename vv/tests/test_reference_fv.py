"""The reference finite-volume solve: exact on linear fields, second order on every mesh family."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from autan_vv.cli import main
from autan_vv.errors import VVError
from autan_vv.meshes.families import KINDS, generate_level
from autan_vv.mms import advection_diffusion, euler
from autan_vv.mms.solutions import X
from autan_vv.order import fit_order
from autan_vv.reference_fv import solve

pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
x, y, z = X


@pytest.mark.parametrize("kind", KINDS)
def test_linear_solution_is_reproduced_to_rounding(kind: str) -> None:
    # LSQ gradients, reconstruction and the face gradient are exact for linear fields.
    ms = advection_diffusion(solution=1 + 2 * x - 3 * y + sp.Rational(1, 2) * z)
    mesh, _ = generate_level(kind, 5, perturbation=0.3, seed=4)
    sol = solve(mesh, ms, rtol=1e-15)
    # |phi| <= 4 on the cube; a few hundred ulps covers the conditioning of the solve.
    assert sol.error_linf < 256 * np.finfo(np.float64).eps * 4.0


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
@pytest.mark.parametrize("diffusivity", ["1e-24", "1", "1e20"])
def test_solve_is_invariant_to_the_scale_of_the_operator(precision: str, diffusivity: str) -> None:
    # A constant field is linear: exact up to rounding whatever the magnitude of the coefficients.
    ms = advection_diffusion(
        velocity=("0", "0", "0"), diffusivity=diffusivity, solution=sp.Integer(1)
    )
    mesh, _ = generate_level("hex", 3)
    eps = float(np.finfo(np.float32 if precision == "fp32" else np.float64).eps)
    assert solve(mesh, ms, precision).error_linf < 64 * eps


# Levels chosen so that every family is in its asymptotic range while the test stays fast.
LEVELS = {
    "tet": (8, 12, 16),
    "hex": (8, 12, 16),
    "prism": (8, 12, 16),
    "hybrid": (6, 12, 18),
    "poly": (8, 12, 16),
    "agglomerated": (8, 16, 24),
}


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
@pytest.mark.parametrize("perturbation", [0.0, 0.3])
@pytest.mark.parametrize("kind", KINDS)
def test_observed_order_is_two_on_every_family(
    kind: str, perturbation: float, precision: str
) -> None:
    ms = advection_diffusion()
    eps = float(np.finfo(np.float32 if precision == "fp32" else np.float64).eps)
    h, e = [], []
    for n in LEVELS[kind]:
        mesh, _ = generate_level(kind, n, perturbation=perturbation, seed=1)
        sol = solve(mesh, ms, precision)
        assert sol.residual_norm < 1024 * eps
        h.append((mesh.geometry.cell_volume.sum() / mesh.n_cells) ** (1 / 3))
        e.append(sol.error_l2)
    fit = fit_order(h, e)
    assert fit.pairwise_orders[-1] > 1.85, fit
    assert fit.order > 1.85, fit


@pytest.mark.parametrize(
    ("ms", "code"),
    [(euler(), "V0401"), (advection_diffusion(diffusivity="0"), "V0403")],
    ids=["other-equations", "pure-advection"],
)
def test_solver_rejects_problems_it_cannot_solve(ms: object, code: str) -> None:
    mesh, _ = generate_level("hex", 3)
    with pytest.raises(VVError) as err:
        solve(mesh, ms)  # type: ignore[arg-type]
    assert err.value.code == code


def test_cli_verify_writes_report_plot_and_manifest(tmp_path: Path) -> None:
    assert (
        main(["reference", "hex", "--levels", "4,8", "--precision", "fp32", "--out", str(tmp_path)])
        == 0
    )

    def reject(token: str) -> None:
        raise AssertionError(token)

    report = json.loads((tmp_path / "order.json").read_text(), parse_constant=reject)
    assert report["precision"] == "fp32"
    assert len(report["mms_sha256"]) == 64
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert report["commit"] == head.stdout.strip()
    assert isinstance(report["dirty"], bool)
    assert report["fits"]["L2"]["order_ci95"] == [None, None]
    assert report["schema"] == "autan.order-study/1"
    assert report["mesh_family"]["levels"][0]["file"] == "hex-n004.msh"
    assert report["fits"]["L2"]["order"] > 1.8
    assert (tmp_path / "order.png").stat().st_size > 0
    assert (tmp_path / "hex-n008.msh").exists()
