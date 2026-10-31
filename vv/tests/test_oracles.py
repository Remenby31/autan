"""Oracles checked against what defines them: the equations (residuals by finite differences),
published tables (Toro 2009, Ganapol 2010) and jump conditions."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from autan_vv.errors import VVError
from autan_vv.mms import euler
from autan_vv.oracles import Oracle, conserved, from_mms
from autan_vv.oracles.inviscid import isentropic_vortex, ringleb, shock_tube
from autan_vv.oracles.registry import ORACLES, get_oracle
from autan_vv.oracles.riemann import State, sample, star_region
from autan_vv.oracles.shu_osher import shu_osher, solve_1d
from autan_vv.oracles.viscous import blasius, blasius_profile, couette, poiseuille
from autan_vv.order import fit_order


def _euler_residual(oracle: Oracle, point: np.ndarray, time: float | None, h: float) -> np.ndarray:
    """dU/dt + dF/dx + dG/dy + dH/dz of the Euler equations, fourth-order central differences."""

    def flux(p: np.ndarray, t: float | None) -> tuple[np.ndarray, np.ndarray]:
        f = oracle.fields(p[None, :], t)
        u = conserved(f, oracle.gamma)
        state = np.array([u[k][0] for k in ("rho", "rhou", "rhov", "rhow", "rhoE")])
        vel = np.array([f["VelocityX"][0], f["VelocityY"][0], f["VelocityZ"][0]])
        pres = f["Pressure"][0]
        fluxes = np.outer(state, vel)
        fluxes[1:4] += pres * np.eye(3)
        fluxes[4] += pres * vel
        return state, fluxes

    def d(g: Callable[[np.ndarray], np.ndarray], k: int) -> np.ndarray:
        e = np.zeros(3)
        e[k] = h
        return np.asarray(
            (-g(point + 2 * e) + 8 * g(point + e) - 8 * g(point - e) + g(point - 2 * e)) / (12 * h)
        )

    def column(k: int) -> Callable[[np.ndarray], np.ndarray]:
        return lambda p: flux(p, time)[1][:, k]

    total = np.sum([d(column(k), k) for k in range(3)], axis=0)
    if time is not None:
        dt = h
        total = total + (
            -flux(point, time + 2 * dt)[0]
            + 8 * flux(point, time + dt)[0]
            - 8 * flux(point, time - dt)[0]
            + flux(point, time - 2 * dt)[0]
        ) / (12 * dt)
    return np.asarray(total)


def _ringleb_point(speed: float, k: float, gamma: float = 1.4, lower: bool = False) -> np.ndarray:
    c = math.sqrt(1 - 0.5 * (gamma - 1) * speed**2)
    rho = c ** (2 / (gamma - 1))
    j = 1 / c + 1 / (3 * c**3) + 1 / (5 * c**5) - 0.5 * math.log((1 + c) / (1 - c))
    x = (1 / speed**2 - 2 / k**2) / (2 * rho) + j / 2
    y = math.sqrt(1 - speed**2 / k**2) / (k * rho * speed)
    return np.array([x, -y if lower else y, 0.0])


@pytest.mark.parametrize(
    ("speed", "k", "lower"),
    [(0.5, 0.7, False), (0.6, 1.0, True), (0.9, 1.2, False), (1.3, 1.5, True)],
)
def test_ringleb_solves_the_euler_equations_and_recovers_its_hodograph(
    speed: float, k: float, lower: bool
) -> None:
    oracle = ringleb()
    point = _ringleb_point(speed, k, lower=lower)
    f = oracle.fields(point[None, :])
    assert math.hypot(f["VelocityX"][0], f["VelocityY"][0]) == pytest.approx(speed, rel=1e-12)
    assert np.max(np.abs(_euler_residual(oracle, point, None, 1e-4))) < 1e-9
    # Stagnation temperature 1 (isentropic, units of the stagnation state): T + (gamma-1)/2 V^2 = 1.
    assert f["Temperature"][0] + 0.2 * speed**2 == pytest.approx(1.0, rel=1e-12)


def test_ringleb_velocity_is_continuous_across_the_symmetry_axis() -> None:
    oracle = ringleb()
    above = oracle.fields(_ringleb_point(0.7, 0.7000001)[None, :])
    below = oracle.fields(_ringleb_point(0.7, 0.7000001, lower=True)[None, :])
    assert above["VelocityX"][0] == pytest.approx(below["VelocityX"][0], abs=1e-3)
    assert above["VelocityY"][0] == pytest.approx(below["VelocityY"][0], rel=1e-6)


@pytest.mark.parametrize("time", [0.0, 3.7, 40.0])
def test_convected_vortex_solves_the_unsteady_euler_equations(time: float) -> None:
    oracle = isentropic_vortex(mach=0.5, angle=0.3)
    for point in ([0.4, -0.7, 0.0], [1.3, 0.2, 0.0]):
        shift = np.array([0.5 * math.cos(0.3), 0.5 * math.sin(0.3), 0.0]) * time
        p = np.array(point) + shift
        assert np.max(np.abs(_euler_residual(oracle, p, time, 1e-3))) < 1e-9
    with pytest.raises(VVError) as err:
        oracle.fields(np.zeros((1, 3)))
    assert err.value.code == "V0701"


def test_vortex_returns_to_its_initial_state_after_one_period() -> None:
    oracle = isentropic_vortex(mach=0.5)
    pts = np.random.default_rng(0).uniform(-10, 10, (50, 3))
    start, end = oracle.fields(pts, 0.0), oracle.fields(pts, 20.0 / 0.5)
    for name in ("Density", "VelocityX", "Pressure"):
        np.testing.assert_allclose(end[name], start[name], rtol=1e-13, atol=1e-15)


# Toro (2009), Table 4.3: exact star-region solutions of tests 1-5 (gamma = 1.4), as printed.
TORO = {
    1: ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ("0.30313", "0.92745", "0.42632", "0.26557")),
    2: ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), ("0.00189", "0.00000", "0.02185", "0.02185")),
    3: ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), ("460.894", "19.5975", "0.57506", "5.99924")),
    4: ((1.0, 0.0, 0.01), (1.0, 0.0, 100.0), ("46.0950", "-6.19633", "5.99242", "0.57511")),
    5: (
        (5.99924, 19.5975, 460.894),
        (5.99242, -6.19633, 46.0950),
        ("1691.64", "8.68975", "14.2823", "31.0426"),
    ),
}


def _star(left: tuple[float, ...], right: tuple[float, ...]) -> np.ndarray:
    star = star_region(State(*left), State(*right), 1.4)
    return np.array([star.p, star.u, star.rho_left, star.rho_right])


@pytest.mark.parametrize("test", sorted(TORO))
def test_riemann_star_states_match_toro(test: int) -> None:
    left, right, printed = TORO[test]
    got = _star(left, right)
    # Tolerance: half a unit of the printed digit, plus, for test 5 whose inputs are themselves
    # Toro's six-digit outputs, the effect of half a unit of rounding on each input.
    half_units = {5: ((5e-6, 5e-5, 5e-4), (5e-6, 5e-6, 5e-5))}.get(test, ((0, 0, 0), (0, 0, 0)))
    propagated = np.zeros(4)
    for side in range(2):
        for k in range(3):
            if half_units[side][k]:
                states = [list(left), list(right)]
                states[side][k] += half_units[side][k]
                propagated += np.abs(_star(tuple(states[0]), tuple(states[1])) - got)
    for value, text, extra in zip(got, printed, propagated, strict=True):
        unit = 10.0 ** -len(text.split(".")[1])
        assert value == pytest.approx(float(text), abs=0.5 * unit + extra)


def test_sod_shock_satisfies_rankine_hugoniot() -> None:
    left, right = State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1)
    xi = np.linspace(-2, 2, 400001)
    rho, u, p = sample(left, right, 1.4, xi)
    jump = int(np.argmax(np.abs(np.diff(p[xi > 1.0])))) + int(np.sum(xi <= 1.0))
    s = 0.5 * (xi[jump] + xi[jump + 1])
    a, b = jump - 2, jump + 3
    energy = p / 0.4 + 0.5 * rho * u**2
    for q, f in ((rho, rho * u), (rho * u, rho * u**2 + p), (energy, (energy + p) * u)):
        # [F] = s [U] across a shock moving at s.
        assert f[b] - f[a] == pytest.approx(s * (q[b] - q[a]), rel=2e-4)


def test_shock_tube_oracle_samples_the_riemann_solution() -> None:
    oracle = shock_tube("sod")
    x = np.array([[0.0, 0.3, 0.1], [0.5 + 0.92745 * 0.2 - 0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
    f = oracle.fields(x, 0.2)
    assert f["Density"][0] == 1.0
    assert f["Pressure"][1] == pytest.approx(0.30313, rel=5e-5)
    assert f["Density"][2] == 0.125
    assert oracle.fields(x, 0.0)["Density"].tolist() == [1.0, 0.125, 0.125]


def test_blasius_constants_match_ganapol() -> None:
    # Ganapol (2010), arXiv:1006.3888: f''(0) = 0.332057336215196, eta - f -> 1.72078765752246.
    curvature, displacement, _ = blasius_profile()
    assert curvature == pytest.approx(0.332057336215196, abs=1e-13)
    assert displacement == pytest.approx(1.72078765752246, abs=1e-10)


def test_blasius_fields_have_the_classical_wall_shear_and_edge_velocity() -> None:
    oracle = blasius(speed=10.0, viscosity=1.5e-5, density=1.2)
    nu = 1.5e-5 / 1.2
    x = 0.5
    y = np.array([1e-7, 2e-7, 1.0])
    f = oracle.fields(np.column_stack([np.full(3, x), y, np.zeros(3)]))
    # Cf sqrt(Re_x) = 2 f''(0) = 0.664: du/dy at the wall = 0.332 U sqrt(U/(nu x)).
    slope = (f["VelocityX"][1] - f["VelocityX"][0]) / 1e-7
    assert slope == pytest.approx(0.332057336 * 10.0 * math.sqrt(10.0 / (nu * x)), rel=1e-5)
    assert f["VelocityX"][2] == pytest.approx(10.0, rel=1e-12)
    assert f["VelocityY"][2] == pytest.approx(0.5 * 1.72078766 * math.sqrt(nu * 10.0 / x), rel=1e-8)


def test_couette_temperature_balances_viscous_heating() -> None:
    oracle = couette()
    p = oracle.parameters
    h = 1e-3
    y = np.array([0.3 - h, 0.3, 0.3 + h])
    f = oracle.fields(np.column_stack([np.zeros(3), y, np.zeros(3)]))
    cp = 1.4 * 287.0 / 0.4
    conductivity = cp * p["viscosity"] / p["prandtl"]
    t2 = (f["Temperature"][0] - 2 * f["Temperature"][1] + f["Temperature"][2]) / h**2
    heating = p["viscosity"] * (p["wall_speed"] / p["height"]) ** 2
    assert conductivity * t2 + heating == pytest.approx(0.0, abs=1e-6 * heating)
    walls = oracle.fields(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert walls["Temperature"].tolist() == pytest.approx([300.0, 320.0])
    assert walls["VelocityX"].tolist() == pytest.approx([0.0, 50.0])
    assert walls["Pressure"] * 1.0 == pytest.approx(walls["Density"] * 287.0 * walls["Temperature"])


def test_poiseuille_balances_pressure_gradient_and_shear() -> None:
    oracle = poiseuille(gradient=2.0, viscosity=0.01)
    h = 1e-3
    pts = np.array([[0.2, 0.4 - h, 0], [0.2, 0.4, 0], [0.2, 0.4 + h, 0], [0.2 + h, 0.4, 0]])
    f = oracle.fields(pts)
    u2 = (f["VelocityX"][0] - 2 * f["VelocityX"][1] + f["VelocityX"][2]) / h**2
    dpdx = (f["Pressure"][3] - f["Pressure"][1]) / h
    assert -dpdx + 0.01 * u2 == pytest.approx(0.0, abs=1e-7)
    assert "Temperature" not in f


def test_mms_oracle_returns_the_manufactured_fields() -> None:
    ms = euler()
    oracle = from_mms(ms)
    pts = np.random.default_rng(1).uniform(0, 1, (5, 3))
    f = oracle.fields(pts)
    raw = ms.evaluator("fields")(pts)
    np.testing.assert_array_equal(f["Density"], raw["rho"])
    np.testing.assert_array_equal(f["VelocityZ"], raw["w"])
    np.testing.assert_allclose(f["Temperature"], raw["p"] / (raw["rho"] * 287.0), rtol=1e-15)
    assert oracle.physics == "euler"


def test_reference_1d_solver_converges_to_the_exact_sod_solution() -> None:
    # The second source that makes the Shu-Osher reference usable for regression.
    exact = shock_tube("sod")
    errors, sizes = [], []
    for cells in (100, 200, 400, 800):
        dx = 1.0 / cells
        x = (np.arange(cells) + 0.5) * dx
        left = x <= 0.5
        rho, _, _ = solve_1d(
            np.where(left, 1.0, 0.125), np.zeros(cells), np.where(left, 1.0, 0.1), dx, 0.2
        )
        ref = exact.fields(np.column_stack([x, np.zeros((cells, 2))]), 0.2)
        errors.append(float(np.mean(np.abs(rho - ref["Density"]))))
        sizes.append(dx)
    fit = fit_order(sizes, errors)
    # Banks, Aslam and Rider (2008): L1 order of a discontinuous solution between 2/3 and 1.
    assert 0.6 < fit.order < 1.1
    assert errors[-1] < 4e-3


def test_shu_osher_reference_places_the_shock_and_keeps_positivity(tmp_path: Path) -> None:
    oracle = shu_osher(cells=1600, cache_dir=tmp_path)
    x = np.linspace(-5, 5, 2001)
    f = oracle.fields(np.column_stack([x, np.zeros((x.size, 2))]), 1.8)
    assert oracle.reference_only
    assert np.all(f["Density"] > 0)
    assert np.all(f["Pressure"] > 0)
    shock = x[int(np.argmax(np.abs(np.diff(f["Pressure"]))))]
    # Shock near x = 2.4 at t = 1.8 (Shu and Osher 1989, Fig. 14).
    assert 2.2 < shock < 2.6
    assert list(tmp_path.glob("shu-osher-*.npy")), "the reference is cached"


def test_registry_builds_every_oracle_and_rejects_unknown_names() -> None:
    for name in sorted(set(ORACLES) - {"shu-osher"}):  # its reference takes a minute to compute
        oracle = get_oracle(name)
        assert oracle.name == name or name.startswith("mms-")
        f = oracle.fields(
            np.array([[0.3, 0.4, 0.5]])
            if oracle.name != "ringleb"
            else _ringleb_point(0.6, 1.0)[None, :],
            1.0 if oracle.unsteady else None,
        )
        assert set(f) >= {"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"}
    with pytest.raises(VVError) as err:
        get_oracle("nope")
    assert err.value.code == "V0704"
