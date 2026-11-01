"""Generated advection-diffusion residual: linear exactness, conservation and MMS order."""

from __future__ import annotations

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps

G = np.array([0.3, -1.2, 0.7])


@pytest.mark.parametrize("perturbation", [0.0, 0.15])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_linear_field_has_zero_residual(kind: str, perturbation: float, policy: str) -> None:
    # div(a q - nu grad q) = a . G for q = 1 + G . x: the scheme is exact for linear fields,
    # including the convective flux through warped faces (face moments).
    m = s.mesh(meshes.build(kind, 6, perturbation, seed=5))
    q = 1.0 + m.cell_centroid @ G
    q_b = np.zeros(m.n_faces)
    q_b[m.n_interior :] = 1.0 + m.face_centroid[m.n_interior :] @ G
    src = np.full(m.n_cells, np.dot(mms.VELOCITY, G))
    res, _ = s.advdiff_residual(m, policy, q, q_b, src, mms.VELOCITY, mms.NU)
    scale = np.abs(m.face_area).sum(axis=1).max() * 3.0 / m.h  # largest single face flux
    assert np.abs(res).max() <= 2000 * eps(policy) * scale


@pytest.mark.parametrize("kind", KINDS)
def test_changing_cells_away_from_the_boundary_keeps_the_residual_sum(kind: str) -> None:
    # Interior fluxes cancel in the sum over cells, so only boundary fluxes (which read cells
    # within two layers of the boundary through their gradients) and sources remain.
    m = s.mesh(meshes.build(kind, 16 if kind == "poly" else 8, 0.15, seed=9))
    rng = np.random.default_rng(1)
    q = rng.standard_normal(m.n_cells)
    q_b = rng.standard_normal(m.n_faces)
    src = rng.standard_normal(m.n_cells)
    near = set(m.owner[m.n_interior :].tolist())
    for _ in range(2):
        layer = set(near)
        for f in range(m.n_interior):
            o, nb = int(m.owner[f]), int(m.neighbour[f])
            if o in layer or nb in layer:
                near.update((o, nb))
    far = np.array([c for c in range(m.n_cells) if c not in near])
    assert far.size > 0
    r1, _ = s.advdiff_residual(m, "fp64", q, q_b, src, mms.VELOCITY, mms.NU)
    q2 = q.copy()
    q2[far] += rng.standard_normal(far.size)
    r2, _ = s.advdiff_residual(m, "fp64", q2, q_b, src, mms.VELOCITY, mms.NU)
    assert abs(r1.sum() - r2.sum()) <= 1e3 * eps("fp64") * np.abs(r1).sum()
    assert np.abs(r1 - r2).max() > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_mms_solution_converges_at_second_order(kind: str, policy: str) -> None:
    h, e = [], []
    for n in (8, 16, 24) if kind == "poly" else (4, 8, 16):
        m = s.mesh(meshes.build(kind, n, 0.15, seed=n))
        sol = mms.solve(m, policy, tol=1e-12 if policy == "fp64" else 1e-6)
        h.append(m.h)
        e.append(sol.error_l2)
    fit, _ = mms.observed_order(h, e)
    assert fit >= 1.9
