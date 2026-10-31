"""Manufactured sources checked against oracles that do not reuse the symbolic derivation.

- hand-derived sources for simple scalar fields;
- exact steady solutions of the Euler and Navier-Stokes equations, whose source must vanish;
- an independent finite-difference evaluation of the RANS-SA residual written from the TMR equations.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import sympy as sp

from cfd_vv.errors import VVError
from cfd_vv.mms import CATALOG, Flow, advection_diffusion, euler, navier_stokes, rans_sa
from cfd_vv.mms.solutions import GAMMA, ManufacturedSolution, Validity, X

x, y, z = X
Vec = npt.NDArray[np.float64]


def test_advection_diffusion_source_matches_hand_derivation() -> None:
    ms = advection_diffusion(
        velocity=("1", "0.7", "0.4"), diffusivity="0.1", solution=sp.sin(sp.pi * x)
    )
    expected = sp.pi * sp.cos(sp.pi * x) + sp.Rational(1, 10) * sp.pi**2 * sp.sin(sp.pi * x)
    assert sp.simplify(ms.sources["phi"] - expected) == 0


def _vanishes(ms: ManufacturedSolution, points: Vec) -> None:
    sources = ms.evaluator("sources")(points)
    conserved = ms.evaluator("conserved")(points)
    for name, value in sources.items():
        scale = np.max(np.abs(conserved[name])) + 1.0
        assert np.max(np.abs(value)) < 1e-9 * scale, name


def _points(n: int = 50, seed: int = 7) -> Vec:
    return np.random.default_rng(seed).uniform(0.05, 0.95, (n, 3))


def test_euler_source_vanishes_for_a_steady_isentropic_vortex() -> None:
    # Shu (1998) isentropic vortex without convection, in units where p = rho T: an exact steady solution.
    eps = sp.Integer(5)
    xb, yb = x - sp.Rational(1, 2), y - sp.Rational(1, 2)
    r2 = (xb**2 + yb**2) * 16
    du = eps / (2 * sp.pi) * sp.exp((1 - r2) / 2) * 4
    t = 1 - (GAMMA - 1) * eps**2 / (8 * GAMMA * sp.pi**2) * sp.exp(1 - r2)
    rho = t ** (1 / (GAMMA - 1))
    vortex = Flow(rho=rho, u=(-yb * du, xb * du, sp.Integer(0)), p=rho * t)
    _vanishes(euler(vortex), _points())


def test_navier_stokes_source_vanishes_for_compressible_couette_flow() -> None:
    # Plane Couette flow with viscous heating: k T'' + mu U^2 = 0, uniform pressure, rho = p / (R T).
    mu, pr, u_wall, p0 = sp.Integer(10), sp.Rational(72, 100), sp.Integer(50), sp.Integer(100000)
    k = GAMMA * 287 / (GAMMA - 1) * mu / pr
    temperature = 300 + 20 * y - mu * u_wall**2 / (2 * k) * y**2
    flow = Flow(rho=p0 / (287 * temperature), u=(u_wall * y, sp.Integer(0), sp.Integer(0)), p=p0)
    _vanishes(navier_stokes(flow, viscosity="10", prandtl="0.72"), _points())


def test_navier_stokes_source_differs_from_euler_by_the_viscous_terms() -> None:
    inviscid, viscous = euler().evaluator("sources"), navier_stokes().evaluator("sources")
    pts = _points(5)
    assert not np.allclose(inviscid(pts)["rhoE"], viscous(pts)["rhoE"])
    np.testing.assert_allclose(inviscid(pts)["rho"], viscous(pts)["rho"], rtol=1e-14)


# --- independent finite-difference residual for RANS-SA ------------------------------------------------------


def _grad(f: Callable[[Vec], Vec], p: Vec, h: float) -> Vec:
    """Fourth-order central differences of a scalar or vector function of a point."""
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append((-f(p + 2 * e) + 8 * f(p + e) - 8 * f(p - e) + f(p - 2 * e)) / (12 * h))
    return np.stack(cols, axis=-1)


def _sa_residual(ms: ManufacturedSolution, p: Vec, h: float) -> dict[str, float]:
    """RANS-SA residual from the TMR equations (SA, compressible conservative form), by finite differences."""
    c = ms.parameters
    gamma, gas, mu, pr, prt = (
        c["gamma"],
        c["gas_constant"],
        c["viscosity"],
        c["prandtl"],
        c["prandtl_turbulent"],
    )
    cp = gamma * gas / (gamma - 1)
    fns = {k: sp.lambdify(X, v, "numpy") for k, v in ms.fields.items()}

    def prim(q: Vec) -> Vec:
        return np.array([float(fns[k](*q)) for k in ("rho", "u", "v", "w", "p", "nu_hat")])

    def mu_t(q: Vec) -> float:
        rho, nu_hat = prim(q)[[0, 5]]
        chi = nu_hat * rho / mu
        return float(rho * nu_hat * chi**3 / (chi**3 + c["cv1"] ** 3))

    def fluxes(q: Vec) -> Vec:
        """Convective minus viscous fluxes, rows = equations, cols = directions."""
        rho, u, v, w, pres, nu_hat = prim(q)
        vel = np.array([u, v, w])
        grad_vel = _grad(lambda s: prim(s)[1:4], q, h)
        grad_t = _grad(lambda s: np.array(prim(s)[4] / (gas * prim(s)[0])), q, h)
        grad_nu = _grad(lambda s: np.array(prim(s)[5]), q, h)
        mut = mu_t(q)
        mu_eff = mu + mut
        tau = mu_eff * (grad_vel + grad_vel.T - 2.0 / 3.0 * np.trace(grad_vel) * np.eye(3))
        energy = pres / (gamma - 1) + 0.5 * rho * vel @ vel
        k_eff = cp * (mu / pr + mut / prt)
        out = np.zeros((6, 3))
        out[0] = rho * vel
        out[1:4] = rho * np.outer(vel, vel) + pres * np.eye(3) - tau
        out[4] = (energy + pres) * vel - tau @ vel - k_eff * grad_t
        out[5] = rho * vel * nu_hat - rho * (mu / rho + nu_hat) * grad_nu / c["sigma"]
        return out

    jac = _grad(fluxes, p, h)  # (6, 3, 3): d flux_ij / d x_k
    div = np.einsum("ijj->i", jac)

    rho, nu_hat = prim(p)[[0, 5]]
    grad_vel = _grad(lambda s: prim(s)[1:4], p, h)
    vort = np.array(
        [
            grad_vel[2, 1] - grad_vel[1, 2],
            grad_vel[0, 2] - grad_vel[2, 0],
            grad_vel[1, 0] - grad_vel[0, 1],
        ]
    )
    omega = np.linalg.norm(vort)
    d = p[1]
    nu = mu / rho
    chi = nu_hat / nu
    fv1 = chi**3 / (chi**3 + c["cv1"] ** 3)
    fv2 = 1 - chi / (1 + chi * fv1)
    s_tilde = omega + nu_hat * fv2 / (c["kappa"] ** 2 * d**2)
    r = min(nu_hat / (s_tilde * c["kappa"] ** 2 * d**2), 10.0)
    g = r + c["cw2"] * (r**6 - r)
    fw = g * ((1 + c["cw3"] ** 6) / (g**6 + c["cw3"] ** 6)) ** (1 / 6)
    cw1 = c["cb1"] / c["kappa"] ** 2 + (1 + c["cb2"]) / c["sigma"]
    ft2 = c["ft2"] * c["ct3"] * np.exp(-c["ct4"] * chi**2)
    grad_nu = _grad(lambda s: np.array(prim(s)[5]), p, h)
    grad_rho = _grad(lambda s: np.array(prim(s)[0]), p, h)
    rhs = (
        rho * c["cb1"] * (1 - ft2) * s_tilde * nu_hat
        - rho * (cw1 * fw - c["cb1"] / c["kappa"] ** 2 * ft2) * (nu_hat / d) ** 2
        + (rho * c["cb2"] * grad_nu @ grad_nu - (nu + nu_hat) * grad_rho @ grad_nu) / c["sigma"]
    )
    names = ["rho", "rhou", "rhov", "rhow", "rhoE", "rhonu"]
    out = {n: float(div[i]) for i, n in enumerate(names)}
    out["rhonu"] -= float(rhs)
    return out


@pytest.mark.parametrize("ft2", [True, False])
@pytest.mark.parametrize("point", [(0.3, 0.4, 0.6), (0.71, 0.12, 0.23), (0.55, 0.03, 0.85)])
def test_rans_sa_source_matches_independent_finite_differences(
    ft2: bool, point: tuple[float, float, float]
) -> None:
    ms = rans_sa(ft2=ft2)
    p = np.array(point)
    numerical = _sa_residual(ms, p, h=min(2e-3, p[1] / 20))
    symbolic = ms.evaluator("sources")(p[None, :])
    for name, value in numerical.items():
        # Relative to the largest flux-divergence term, the scale at which the residual is formed.
        scale = max(abs(value), abs(float(symbolic[name][0])), 1.0)
        assert float(symbolic[name][0]) == pytest.approx(value, rel=2e-6, abs=2e-6 * scale), name


def test_rans_sa_exercises_eddy_viscosity_and_destruction() -> None:
    pts = _points(2000)
    pts[:, 1] = np.geomspace(1e-3, 1.0, pts.shape[0])
    aux = rans_sa().evaluator("auxiliary")(pts)
    assert aux["chi"].max() > 5.0, "mu_t must exceed mu somewhere"
    assert aux["r"].max() > 0.5, "r must reach O(1) so that f_w is nonlinear"


def test_validity_violation_raises_a_coded_error() -> None:
    ms = euler()
    broken = ManufacturedSolution(
        **{**ms.__dict__, "validity": (Validity("negative", -ms.fields["p"], lower=0.0),)}
    )
    with pytest.raises(VVError) as err:
        broken.check_validity()
    assert err.value.code == "V0201"


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_catalog_sources_are_finite_inside_the_domain(name: str) -> None:
    ms = CATALOG[name]()
    values = ms.evaluator("sources")(_points(100))
    assert all(np.all(np.isfinite(v)) for v in values.values())
    assert set(ms.sources) == set(ms.conserved)
