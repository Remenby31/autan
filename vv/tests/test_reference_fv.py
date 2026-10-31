"""The reference finite-volume solve: exact on linear fields, second order on every mesh family."""

import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from cfd_vv.cli import main
from cfd_vv.errors import VVError
from cfd_vv.meshes.families import KINDS, generate_level
from cfd_vv.mms import advection_diffusion, euler
from cfd_vv.mms.solutions import X
from cfd_vv.order import fit_order
from cfd_vv.reference_fv import solve

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
@pytest.mark.parametrize("perturbation", [0.0, 0.3])
@pytest.mark.parametrize("kind", KINDS)
def test_observed_order_is_two_on_every_family(kind: str, perturbation: float) -> None:
    ms = advection_diffusion()
    h, e = [], []
    for n in LEVELS[kind]:
        mesh, _ = generate_level(kind, n, perturbation=perturbation, seed=1)
        sol = solve(mesh, ms)
        assert sol.residual_norm < 1e-10
        h.append((mesh.geometry.cell_volume.sum() / mesh.n_cells) ** (1 / 3))
        e.append(sol.error_l2)
    fit = fit_order(h, e)
    assert fit.pairwise_orders[-1] > 1.85, fit
    assert fit.order > 1.85, fit


def test_solver_rejects_other_equations() -> None:
    mesh, _ = generate_level("hex", 3)
    with pytest.raises(VVError) as err:
        solve(mesh, euler())
    assert err.value.code == "V0401"


def test_cli_verify_writes_report_plot_and_manifest(tmp_path: Path) -> None:
    assert main(["verify", "hex", "--levels", "4,8", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "order.json").read_text())
    assert report["schema"] == "cfd-vv/order-study/1"
    assert report["mesh_family"]["levels"][0]["file"] == "hex-n004.msh"
    assert report["fits"]["L2"]["order"] > 1.8
    assert (tmp_path / "order.png").stat().st_size > 0
    assert (tmp_path / "hex-n008.msh").exists()
