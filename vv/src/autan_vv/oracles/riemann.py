"""Exact solution of the Riemann problem for the 1D Euler equations of an ideal gas.

E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., Springer, 2009,
chapter 4: pressure function (eqs. 4.6-4.8), Newton iteration from the two-rarefaction guess
(eq. 4.46), sampling of the self-similar solution (section 4.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError

FloatArray = npt.NDArray[np.float64]
EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class State:
    """Primitive state of a 1D ideal gas."""

    rho: float
    u: float
    p: float


@dataclass(frozen=True)
class StarRegion:
    """Pressure and velocity between the two nonlinear waves, and the densities on either side."""

    p: float
    u: float
    rho_left: float
    rho_right: float


def _f(p: float, s: State, gamma: float) -> tuple[float, float]:
    """Toro eqs. 4.6-4.7 and their derivatives: shock if p > p_K, rarefaction otherwise."""
    a = math.sqrt(gamma * s.p / s.rho)
    if p > s.p:
        big_a = 2.0 / ((gamma + 1.0) * s.rho)
        big_b = (gamma - 1.0) / (gamma + 1.0) * s.p
        root = math.sqrt(big_a / (p + big_b))
        return (p - s.p) * root, root * (1.0 - 0.5 * (p - s.p) / (big_b + p))
    ratio = p / s.p
    exponent = (gamma - 1.0) / (2.0 * gamma)
    value = 2.0 * a / (gamma - 1.0) * (ratio**exponent - 1.0)
    return value, ratio ** (-(gamma + 1.0) / (2.0 * gamma)) / (s.rho * a)


def star_region(left: State, right: State, gamma: float) -> StarRegion:
    """Solve for the star region (V0702 if the data generate vacuum)."""
    a_l = math.sqrt(gamma * left.p / left.rho)
    a_r = math.sqrt(gamma * right.p / right.rho)
    if 2.0 * (a_l + a_r) / (gamma - 1.0) <= right.u - left.u:
        msg = "the initial data generate vacuum"
        raise VVError(msg, code="V0702")
    z = (gamma - 1.0) / (2.0 * gamma)
    # Two-rarefaction pressure (Toro eq. 4.46): exact when both waves are rarefactions.
    p = (
        (a_l + a_r - 0.5 * (gamma - 1.0) * (right.u - left.u))
        / (a_l / left.p**z + a_r / right.p**z)
    ) ** (1.0 / z)
    if p > min(left.p, right.p):
        for _ in range(100):
            f_l, d_l = _f(p, left, gamma)
            f_r, d_r = _f(p, right, gamma)
            p_new = max(p - (f_l + f_r + right.u - left.u) / (d_l + d_r), 1e-14 * p)
            converged = abs(p_new - p) <= 64.0 * EPS * p_new
            p = p_new
            if converged:
                break
        else:
            msg = "the star-pressure iteration did not converge"
            raise VVError(msg, code="V0703")
    f_l, _ = _f(p, left, gamma)
    f_r, _ = _f(p, right, gamma)
    u = 0.5 * (left.u + right.u) + 0.5 * (f_r - f_l)
    return StarRegion(p, u, _rho_star(p, left, gamma), _rho_star(p, right, gamma))


def _rho_star(p: float, s: State, gamma: float) -> float:
    if p > s.p:
        g = (gamma - 1.0) / (gamma + 1.0)
        return float(s.rho * (p / s.p + g) / (g * p / s.p + 1.0))
    return float(s.rho * (p / s.p) ** (1.0 / gamma))


def sample(
    left: State, right: State, gamma: float, xi: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Density, velocity and pressure at similarity coordinates xi = (x - x0) / t (Toro section 4.5)."""
    star = star_region(left, right, gamma)
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)
    for side, s, sign, rho_star in (
        (0, left, 1.0, star.rho_left),
        (1, right, -1.0, star.rho_right),
    ):
        mask = xi <= star.u if side == 0 else xi > star.u
        a = math.sqrt(gamma * s.p / s.rho)
        x = xi[mask]
        r, v, q = np.full(x.shape, rho_star), np.full(x.shape, star.u), np.full(x.shape, star.p)
        if star.p > s.p:
            shock = s.u - sign * a * math.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * star.p / s.p + (gamma - 1.0) / (2.0 * gamma)
            )
            outside = sign * (x - shock) < 0.0
        else:
            a_star = a * (star.p / s.p) ** ((gamma - 1.0) / (2.0 * gamma))
            head, tail = s.u - sign * a, star.u - sign * a_star
            outside = sign * (x - head) < 0.0
            fan = ~outside & (sign * (x - tail) < 0.0)
            c = 2.0 / (gamma + 1.0) + sign * (gamma - 1.0) / ((gamma + 1.0) * a) * (s.u - x[fan])
            r[fan] = s.rho * c ** (2.0 / (gamma - 1.0))
            v[fan] = 2.0 / (gamma + 1.0) * (sign * a + 0.5 * (gamma - 1.0) * s.u + x[fan])
            q[fan] = s.p * c ** (2.0 * gamma / (gamma - 1.0))
        r[outside], v[outside], q[outside] = s.rho, s.u, s.p
        rho[mask], u[mask], p[mask] = r, v, q
    return rho, u, p
