"""Generated GPU loops: every race strategy equals the host loop to rounding, and the
deterministic strategies are bitwise reproducible across runs and launch shapes.

Needs AUTAN_SPIKE_LIB pointing to libautan_spike_gpu (built with -DAUTAN_ENABLE_CUDA=ON).
"""

from __future__ import annotations

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", params=KINDS)
def case(request: pytest.FixtureRequest) -> tuple[s.Mesh, dict[str, np.ndarray]]:
    m = s.mesh(meshes.build(request.param, 8, 0.15, seed=11))
    rng = np.random.default_rng(3)
    data = {
        "q": rng.standard_normal(m.n_cells),
        "q_b": rng.standard_normal(m.n_faces),
        "src": rng.standard_normal(m.n_cells),
        "grad": rng.standard_normal((m.n_cells, 3)),
    }
    return m, data


@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_gpu_residual_equals_host_residual(
    case: tuple[s.Mesh, dict[str, np.ndarray]], policy: str, strategy: str
) -> None:
    m, d = case
    host, hg = s.advdiff_residual(m, policy, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU)
    dev, dg = s.gpu_advdiff_residual(
        m, policy, strategy, 128, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU
    )
    scale = np.abs(host).max()
    assert np.abs(dev - host).max() <= 200 * eps(policy) * scale
    assert np.abs(dg - hg).max() <= 200 * eps(policy) * np.abs(hg).max()
    adv_host = s.advection(m, policy, d["q"], d["grad"], mms.VELOCITY)
    adv_dev = s.gpu_advection(m, policy, strategy, 128, d["q"], d["grad"], mms.VELOCITY)
    assert np.abs(adv_dev - adv_host).max() <= 200 * eps(policy) * np.abs(adv_host).max()


@pytest.mark.parametrize("strategy", s.STRATEGIES)
def test_gpu_green_gauss_equals_host(
    case: tuple[s.Mesh, dict[str, np.ndarray]], strategy: str
) -> None:
    m, d = case
    host, _, _ = s.gradients(m, "fp64", "gg", d["q"], d["q_b"], 3, d["grad"])
    dev, _, _ = s.gpu_gradients(m, "fp64", "gg", strategy, 128, d["q"], d["q_b"], 3, d["grad"])
    assert np.abs(dev - host).max() <= 200 * eps("fp64") * np.abs(host).max()
    lsq_h, lo_h, hi_h = s.gradients(m, "fp64", "lsq", d["q"], d["q_b"])
    lsq_d, lo_d, hi_d = s.gpu_gradients(m, "fp64", "lsq", strategy, 128, d["q"], d["q_b"])
    np.testing.assert_allclose(lsq_d, lsq_h, rtol=0, atol=200 * eps("fp64") * np.abs(lsq_h).max())
    np.testing.assert_array_equal(lo_d, lo_h)
    np.testing.assert_array_equal(hi_d, hi_h)


@pytest.mark.parametrize("strategy", s.DETERMINISTIC)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_deterministic_strategies_are_bitwise_reproducible(
    case: tuple[s.Mesh, dict[str, np.ndarray]], policy: str, strategy: str
) -> None:
    m, d = case
    runs = []
    for k in range(10):
        block = 64 if k % 2 else 256
        res, grad = s.gpu_advdiff_residual(
            m, policy, strategy, block, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU
        )
        gg, _, _ = s.gpu_gradients(m, policy, "gg", strategy, block, d["q"], d["q_b"], 2, d["grad"])
        norm = np.sqrt(np.sum(res * res))
        runs.append((res.tobytes(), grad.tobytes(), gg.tobytes(), norm))
    assert len(set(runs)) == 1


@pytest.mark.parametrize("strategy", s.STRATEGIES)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_gpu_tangent_and_adjoint_equal_host(
    case: tuple[s.Mesh, dict[str, np.ndarray]], policy: str, strategy: str
) -> None:
    m, d = case
    rng = np.random.default_rng(4)
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3))
    p = 1.0 + 0.2 * rng.random(m.n_cells)
    u = np.column_stack([rho, rho[:, None] * vel, p / 0.4 + 0.5 * rho * np.sum(vel * vel, axis=1)])
    x = rng.standard_normal(u.shape)
    for mode in ("residual", "tangent", "adjoint"):
        host = s.roe(m, policy, mode, u, x)
        dev = s.gpu_roe(m, policy, strategy, 128, mode, u, x)
        assert np.abs(dev - host).max() <= 500 * eps(policy) * np.abs(host).max()
    y = rng.standard_normal(m.n_cells)
    host = s.advdiff_derivative(
        m, policy, "adjoint", d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU, y
    )
    dev = s.gpu_advdiff_adjoint(
        m, policy, strategy, 128, d["q"], d["q_b"], d["src"], mms.VELOCITY, mms.NU, y
    )
    assert np.abs(dev - host).max() <= 500 * eps(policy) * np.abs(host).max()


@pytest.mark.parametrize("strategy", s.DETERMINISTIC)
def test_face_loop_adjoints_are_bitwise_reproducible(
    case: tuple[s.Mesh, dict[str, np.ndarray]], strategy: str
) -> None:
    # The adjoint of a face loop is a face loop: it runs in the same race strategies. (The adjoint
    # of a cell reduction scatters with atomics in this spike and is not covered.)
    m, _ = case
    rng = np.random.default_rng(5)
    rho = 1.0 + 0.2 * rng.random(m.n_cells)
    vel = 0.3 * rng.standard_normal((m.n_cells, 3))
    u = np.column_stack([rho, rho[:, None] * vel, 2.5 + 0.5 * rho * np.sum(vel * vel, axis=1)])
    y = rng.standard_normal(u.shape)
    runs = {
        s.gpu_roe(m, "fp32+acc64", strategy, 64 if k % 2 else 256, "adjoint", u, y).tobytes()
        for k in range(10)
    }
    assert len(runs) == 1
