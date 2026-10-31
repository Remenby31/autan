"""Generated tangents and adjoints: dot-product identity, complex step, conservation.

Gap = |<y, J x> - <J^T y, x>| / (|y| |J x| + |J^T y| |x|), criteria of research section 8:
<= 1e-13 in fp64, <= 1e-5 in fp32 and fp32+acc64; tangent vs complex step <= 1e-12 in fp64.
"""

from __future__ import annotations

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS

GAP = {"fp64": 1e-13, "fp32+acc64": 1e-5, "fp32": 1e-5}


def gap(y: np.ndarray, jx: np.ndarray, jty: np.ndarray, x: np.ndarray) -> float:
    a, b = float(np.vdot(y, jx)), float(np.vdot(jty, x))
    return abs(a - b) / (np.linalg.norm(y) * np.linalg.norm(jx) + np.linalg.norm(jty) * np.linalg.norm(x))


@pytest.fixture(scope="module", params=KINDS)
def mesh(request: pytest.FixtureRequest) -> s.Mesh:
    return s.mesh(meshes.build(request.param, 6, 0.15, seed=21))


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("policy", PRECISIONS)
def test_residual_adjoint_passes_the_dot_product_test(mesh: s.Mesh, policy: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    q, q_b, src = rng.standard_normal(mesh.n_cells), rng.standard_normal(mesh.n_faces), rng.standard_normal(mesh.n_cells)
    x, y = rng.standard_normal(mesh.n_cells), rng.standard_normal(mesh.n_cells)
    jx = s.advdiff_derivative(mesh, policy, "tangent", q, q_b, src, mms.VELOCITY, mms.NU, x)
    jty = s.advdiff_derivative(mesh, policy, "adjoint", q, q_b, src, mms.VELOCITY, mms.NU, y)
    assert gap(y, jx, jty, x) <= GAP[policy]


def test_residual_tangent_equals_complex_step(mesh: s.Mesh) -> None:
    rng = np.random.default_rng(5)
    q, q_b, src = rng.standard_normal(mesh.n_cells), rng.standard_normal(mesh.n_faces), rng.standard_normal(mesh.n_cells)
    x = rng.standard_normal(mesh.n_cells)
    jx = s.advdiff_derivative(mesh, "fp64", "tangent", q, q_b, src, mms.VELOCITY, mms.NU, x)
    cs = s.advdiff_complex_step(mesh, q, q_b, src, mms.VELOCITY, mms.NU, x)
    assert np.abs(jx - cs).max() <= 1e-12 * np.abs(cs).max()


def test_adjoint_of_the_residual_sum_vanishes_away_from_the_boundary(mesh: s.Mesh) -> None:
    # Interior fluxes cancel in sum_c R_c, so d(sum R)/dq is zero on cells whose values reach no
    # boundary flux (more than two layers from the boundary).
    m = s.mesh(meshes.build("tet", 8, 0.15, seed=4))
    rng = np.random.default_rng(6)
    q, q_b, src = rng.standard_normal(m.n_cells), rng.standard_normal(m.n_faces), rng.standard_normal(m.n_cells)
    qbar = s.advdiff_derivative(m, "fp64", "adjoint", q, q_b, src, mms.VELOCITY, mms.NU, np.ones(m.n_cells))
    near = set(m.owner[m.n_interior :].tolist())
    for _ in range(2):
        layer = set(near)
        for f in range(m.n_interior):
            o, nb = int(m.owner[f]), int(m.neighbour[f])
            if o in layer or nb in layer:
                near.update((o, nb))
    far = np.array([c for c in range(m.n_cells) if c not in near])
    assert far.size > 0
    assert np.abs(qbar[far]).max() <= 1e-13 * np.abs(qbar).max()


def euler_states(m: s.Mesh, rng: np.random.Generator) -> np.ndarray:
    """Subsonic states with near-zero normal velocity on part of the mesh, so that Harten's fix
    acts on |V| as well as away from it."""
    x = m.cell_centroid
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3)) * (x[:, :1] > 0.5)
    p = 1.0 + 0.2 * rng.random(m.n_cells)
    e = p / 0.4 + 0.5 * rho * np.sum(vel * vel, axis=1)
    return np.column_stack([rho, rho[:, None] * vel, e])


@pytest.mark.parametrize("policy", PRECISIONS)
def test_roe_adjoint_passes_the_dot_product_test(mesh: s.Mesh, policy: str) -> None:
    rng = np.random.default_rng(8)
    u = euler_states(mesh, rng)
    x, y = rng.standard_normal(u.shape), rng.standard_normal(u.shape)
    jx = s.roe(mesh, policy, "tangent", u, x)
    jty = s.roe(mesh, policy, "adjoint", u, y)
    assert gap(y, jx, jty, x) <= GAP[policy]


def test_roe_tangent_equals_complex_step(mesh: s.Mesh) -> None:
    rng = np.random.default_rng(9)
    u = euler_states(mesh, rng)
    x = rng.standard_normal(u.shape)
    jx = s.roe(mesh, "fp64", "tangent", u, x)
    cs = s.roe_complex_step(mesh, u, x)
    assert np.abs(jx - cs).max() <= 1e-12 * np.abs(cs).max()


def test_roe_flux_is_consistent_and_conservative(mesh: s.Mesh) -> None:
    # F(U, U) is the physical flux, so a uniform state leaves every cell closed by interior faces
    # at rest; the face loop moves flux between cells, so its residual and the adjoint of a
    # uniform seed sum to zero.
    u = np.tile([1.2, 0.3, -0.2, 0.1, 2.9], (mesh.n_cells, 1))
    r = s.roe(mesh, "fp64", "residual", u)
    interior = np.setdiff1d(np.arange(mesh.n_cells), mesh.owner[mesh.n_interior :])
    assert np.abs(r[interior]).max() <= 1e-13 * 3.0
    rng = np.random.default_rng(10)
    u = euler_states(mesh, rng)
    r = s.roe(mesh, "fp64", "residual", u)
    assert np.abs(r.sum(axis=0)).max() <= 1e-12 * np.abs(r).sum()
    ubar = s.roe(mesh, "fp64", "adjoint", u, np.ones_like(u))
    assert np.abs(ubar).max() == 0.0


@pytest.mark.parametrize("policy", PRECISIONS)
def test_min_max_reductions_with_ties_pass_the_dot_product_test(mesh: s.Mesh, policy: str) -> None:
    # Values rounded to one decimal: many neighbours tie, and the tangent and the adjoint must
    # select the same incidence (the first in list order).
    rng = np.random.default_rng(12)
    q = np.round(rng.standard_normal(mesh.n_cells), 1)
    q_b = np.round(rng.standard_normal(mesh.n_faces), 1)
    x, y = rng.standard_normal(mesh.n_cells), rng.standard_normal((mesh.n_cells, 5))
    jx = s.lsq_bounds_derivative(mesh, policy, "tangent", q, q_b, x)
    jty = s.lsq_bounds_derivative(mesh, policy, "adjoint", q, q_b, y)
    assert gap(y, jx, jty, x) <= GAP[policy]
    # The bounds' tangents are tangents of selected values: each is dq of the cell or of a neighbour.
    assert np.all(np.isin(jx[:, 3], x) | (jx[:, 3] == 0.0)) if policy == "fp64" else True
