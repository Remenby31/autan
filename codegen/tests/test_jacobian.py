"""Jacobian outputs of loops (dsl.Jacobian): per-face blocks of the fused Roe flux loop.

Oracle: assembling the blocks into dR/dU (R_L += F, R_R -= F) must give the matrix whose columns
are complex-step derivatives of the residual (Martins, Sturdza and Alonso 2003).
"""

from __future__ import annotations

import autan_spike as s
import meshes
import numpy as np
import pytest
from conftest import KINDS, eps


def euler_states(m: s.Mesh, rng: np.random.Generator) -> np.ndarray:
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3))
    p = 1.0 + 0.2 * rng.random(m.n_cells)
    return np.column_stack([rho, rho[:, None] * vel, p / 0.4 + 0.5 * rho * np.sum(vel * vel, axis=1)])


def assembled(m: s.Mesh, jac: np.ndarray) -> np.ndarray:
    n = 5 * m.n_cells
    out = np.zeros((n, n))
    for f in range(m.n_interior):
        sides = (int(m.owner[f]), int(m.neighbour[f]))
        for k, col in enumerate(sides):
            block = jac[f, k]
            out[5 * sides[0] : 5 * sides[0] + 5, 5 * col : 5 * col + 5] += block
            out[5 * sides[1] : 5 * sides[1] + 5, 5 * col : 5 * col + 5] -= block
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_face_jacobians_assemble_into_the_complex_step_jacobian_of_the_residual(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 3, 0.15, seed=5))
    u = euler_states(m, np.random.default_rng(1))
    res, jac = s.roe_jacobian(m, "fp64", u)
    np.testing.assert_array_equal(res, s.roe(m, "fp64", "residual", u))
    n = 5 * m.n_cells
    reference = np.zeros((n, n))
    for c in range(m.n_cells):
        for k in range(5):
            du = np.zeros_like(u)
            du[c, k] = 1.0
            reference[:, 5 * c + k] = s.roe_complex_step(m, u, du).ravel()
    assert np.abs(assembled(m, jac) - reference).max() <= 1e-13 * np.abs(reference).max()


@pytest.mark.gpu
@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_device_face_jacobians_equal_host(strategy: str, policy: str) -> None:
    m = s.mesh(meshes.build("tet", 4, 0.15, seed=6))
    u = euler_states(m, np.random.default_rng(2))
    host_res, host_jac = s.roe_jacobian(m, policy, u)
    dev_res, dev_jac = s.roe_jacobian(m, policy, u, strategy)
    interior = slice(0, m.n_interior)
    assert np.abs(dev_jac[interior] - host_jac[interior]).max() <= 500 * eps(policy) * np.abs(host_jac).max()
    assert np.abs(dev_res - host_res).max() <= 500 * eps(policy) * np.abs(host_res).max()
