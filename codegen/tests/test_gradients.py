"""Generated gradient loops against linear fields, whose gradients are exact (Syrakos et al. 2017)."""

from __future__ import annotations

import autan_spike as s
import meshes
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps

G = np.array([0.3, -1.2, 0.7])


def linear(x: np.ndarray) -> np.ndarray:
    return 1.0 + x @ G


@pytest.fixture(scope="module", params=[(k, p) for k in KINDS for p in (0.0, 0.15)], ids=str)
def mesh(request: pytest.FixtureRequest) -> s.Mesh:
    kind, perturbation = request.param
    return s.mesh(meshes.build(kind, 8, perturbation, seed=3))


def _boundary_values(m: s.Mesh, f: object) -> np.ndarray:
    q_b = np.zeros(m.n_faces)
    q_b[m.n_interior :] = f(m.face_centroid[m.n_interior :])  # type: ignore[operator]
    return q_b


@pytest.mark.parametrize("policy", PRECISIONS)
def test_lsq_gradient_is_exact_for_linear_fields(mesh: s.Mesh, policy: str) -> None:
    g, _, _ = s.gradients(
        mesh, policy, "lsq", linear(mesh.cell_centroid), _boundary_values(mesh, linear)
    )
    # Rounding of q (|q| <= 3) amplified by the LSQ weights (|w| ~ 1/h).
    assert np.abs(g - G).max() <= 400 * eps(policy) * 3.0 * 8


@pytest.mark.parametrize("policy", PRECISIONS)
def test_corrected_green_gauss_has_the_exact_gradient_as_fixed_point(
    mesh: s.Mesh, policy: str
) -> None:
    grad0 = np.tile(G, (mesh.n_cells, 1))
    g, _, _ = s.gradients(
        mesh, policy, "gg", linear(mesh.cell_centroid), _boundary_values(mesh, linear), 1, grad0
    )
    assert np.abs(g - G).max() <= 400 * eps(policy) * 3.0 * 8


def test_lsq_bounds_are_min_and_max_over_cell_and_face_neighbours(mesh: s.Mesh) -> None:
    rng = np.random.default_rng(7)
    q = rng.standard_normal(mesh.n_cells)
    q_b = rng.standard_normal(mesh.n_faces)
    _, lo, hi = s.gradients(mesh, "fp64", "lsq", q, q_b)
    exp_lo, exp_hi = q.copy(), q.copy()
    ni = mesh.n_interior
    for f in range(mesh.n_faces):
        o = mesh.owner[f]
        other = q[mesh.neighbour[f]] if f < ni else q_b[f]
        exp_lo[o], exp_hi[o] = min(exp_lo[o], other), max(exp_hi[o], other)
        if f < ni:
            nb = mesh.neighbour[f]
            exp_lo[nb], exp_hi[nb] = min(exp_lo[nb], q[o]), max(exp_hi[nb], q[o])
    np.testing.assert_array_equal(lo, exp_lo)
    np.testing.assert_array_equal(hi, exp_hi)
