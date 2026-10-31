"""Exact inviscid solutions: Ringleb flow, the convected isentropic vortex, shock tubes.

The Ringleb flow and the vortex are written in units where the stagnation (Ringleb) or
free-stream (vortex) density, speed of sound and temperature are 1: gamma = 1.4 and
R = 1/gamma, so that T = p/(rho R) = c^2.
"""

from __future__ import annotations

import math

import numpy as np

from autan_vv.oracles import FloatArray, Oracle
from autan_vv.oracles.riemann import State, sample

GAMMA = 1.4


def _ringleb_state(
    speed: FloatArray, gamma: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    c = np.sqrt(1.0 - 0.5 * (gamma - 1.0) * speed**2)
    rho = c ** (2.0 / (gamma - 1.0))
    p = c ** (2.0 * gamma / (gamma - 1.0)) / gamma
    j = 1.0 / c + 1.0 / (3.0 * c**3) + 1.0 / (5.0 * c**5) - 0.5 * np.log((1.0 + c) / (1.0 - c))
    return c, rho, p, j


def ringleb_fields(points: FloatArray, gamma: float = GAMMA) -> dict[str, FloatArray]:
    """Ringleb flow at points (x, y) of the plane (z ignored).

    The hodograph solution gives x = (1/V^2 - 2/k^2)/(2 rho) + J/2 and y = +-sqrt(1 - V^2/k^2)/(k rho V)
    for speed V and streamline constant k, so that (x - J/2)^2 + y^2 = 1/(4 rho^2 V^4), which is
    solved for V by bisection; then 1/k^2 = (1/V^2 - 2 rho (x - J/2))/2 and, with sin(theta) = V/k,
    u = V cos(theta) sign(y), v = V sin(theta): the flow turns around the origin, crossing y = 0
    vertically. Chiocchia, AGARD-AR-211, 1985; HiOCFD1 problem C1.2.
    """
    x, y = points[:, 0], points[:, 1]
    # Bracket away from V = 0 (1 - c rounds to 0) and from the cavitation speed (c = 0).
    lo = np.full(x.shape, 1e-3)
    hi = np.full(x.shape, math.sqrt(2.0 / (gamma - 1.0)) * (1.0 - 1e-6))

    def residual(speed: FloatArray) -> FloatArray:
        _, rho, _, j = _ringleb_state(speed, gamma)
        return (x - 0.5 * j) ** 2 + y**2 - 1.0 / (4.0 * rho**2 * speed**4)

    r_lo = residual(lo)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)
        same = np.sign(r_mid) == np.sign(r_lo)
        lo, r_lo, hi = np.where(same, mid, lo), np.where(same, r_mid, r_lo), np.where(same, hi, mid)
    speed = 0.5 * (lo + hi)
    _, rho, p, j = _ringleb_state(speed, gamma)
    inv_k2 = 0.5 * (1.0 / speed**2 - 2.0 * rho * (x - 0.5 * j))
    sin_t = np.minimum(speed * np.sqrt(inv_k2), 1.0)
    cos_t = np.sqrt(1.0 - sin_t**2)
    return {
        "Density": rho,
        "VelocityX": speed * cos_t * np.sign(y),
        "VelocityY": speed * sin_t,
        "VelocityZ": np.zeros_like(rho),
        "Pressure": p,
    }


def ringleb(gamma: float = GAMMA) -> Oracle:
    """Ringleb flow, steady, transonic, isentropic and irrotational (2D)."""
    return Oracle(
        name="ringleb",
        description="Ringleb flow; HiOCFD1 C1.2 domain: streamlines k = 0.7 and 1.5, iso-speed V = 0.5.",
        physics="euler",
        dim=2,
        unsteady=False,
        gamma=gamma,
        gas_constant=1.0 / gamma,
        evaluate=lambda points, _t: ringleb_fields(points, gamma),
        parameters={"k_inner": 1.5, "k_outer": 0.7, "speed_inflow": 0.5},
        references=("chiocchia1985", "hiocfd_c12"),
    )


def isentropic_vortex(
    mach: float = 0.5,
    angle: float = 0.0,
    strength: float = 5.0,
    centre: tuple[float, float] = (0.0, 0.0),
    period: tuple[float, float] | None = (20.0, 20.0),
    gamma: float = GAMMA,
) -> Oracle:
    """Isentropic vortex convected by a uniform flow of Mach `mach` at `angle` rad (2D, unsteady).

    Velocity perturbation beta/(2 pi) exp((1 - r^2)/2) (-dy, dx), T = 1 - (gamma - 1) beta^2/(8 pi^2)
    exp(1 - r^2), rho = T^(1/(gamma-1)), p = rho T / gamma, with r measured from the convected centre
    (Shu 1998, rescaled so that the free-stream speed of sound is 1). On a periodic domain of size `period` the centre wraps and the
    nearest image is used; the default [-10, 10]^2 domain follows Spiegel, Huynh and DeBonis
    (AIAA 2015-2444), whose truncation error is below 1e-20.
    """

    u_inf = mach * math.cos(angle)
    v_inf = mach * math.sin(angle)

    def evaluate(points: FloatArray, time: float | None) -> dict[str, FloatArray]:
        t = 0.0 if time is None else time
        dx = points[:, 0] - (centre[0] + u_inf * t)
        dy = points[:, 1] - (centre[1] + v_inf * t)
        if period is not None:
            dx -= period[0] * np.round(dx / period[0])
            dy -= period[1] * np.round(dy / period[1])
        r2 = dx**2 + dy**2
        # Radial equilibrium dT/dr = (gamma - 1) u_theta^2 / r with T = c^2 (Shu's T = p/rho = c^2/gamma
        # carries an extra 1/gamma).
        temperature = 1.0 - (gamma - 1.0) * strength**2 / (8.0 * math.pi**2) * np.exp(1.0 - r2)
        bump = strength / (2.0 * math.pi) * np.exp(0.5 * (1.0 - r2))
        rho = temperature ** (1.0 / (gamma - 1.0))
        # Free-stream speed of sound sqrt(gamma p/rho) = 1 with p = rho T / gamma.
        return {
            "Density": rho,
            "VelocityX": u_inf - bump * dy,
            "VelocityY": v_inf + bump * dx,
            "VelocityZ": np.zeros_like(rho),
            "Pressure": rho * temperature / gamma,
        }

    return Oracle(
        name="isentropic-vortex",
        description=f"Isentropic vortex, strength {strength}, convected at Mach {mach}.",
        physics="euler",
        dim=2,
        unsteady=True,
        gamma=gamma,
        gas_constant=1.0 / gamma,
        evaluate=evaluate,
        parameters={"mach": mach, "angle": angle, "strength": strength},
        references=("shu1998", "spiegel2015"),
    )


SHOCK_TUBES = {
    # Toro (2009), Table 4.1, test 1 (Sod 1978).
    "sod": (State(1.0, 0.0, 1.0), State(0.125, 0.0, 0.1)),
    # Lax (1954), as tabulated by Toro (2009).
    "lax": (State(0.445, 0.698, 3.528), State(0.5, 0.0, 0.571)),
}


def shock_tube(name: str, diaphragm: float = 0.5, gamma: float = GAMMA) -> Oracle:
    """Exact solution of the `sod` or `lax` shock tube along x, diaphragm at x = `diaphragm`."""
    left, right = SHOCK_TUBES[name]

    def evaluate(points: FloatArray, time: float | None) -> dict[str, FloatArray]:
        x = points[:, 0]
        if not time:
            on_left = x <= diaphragm
            rho = np.where(on_left, left.rho, right.rho)
            u = np.where(on_left, left.u, right.u)
            p = np.where(on_left, left.p, right.p)
        else:
            rho, u, p = sample(left, right, gamma, (x - diaphragm) / time)
        zero = np.zeros_like(rho)
        return {"Density": rho, "VelocityX": u, "VelocityY": zero, "VelocityZ": zero, "Pressure": p}

    return Oracle(
        name=name,
        description=f"{name.capitalize()} shock tube, diaphragm at x = {diaphragm}, exact Riemann solution.",
        physics="euler",
        dim=1,
        unsteady=True,
        gamma=gamma,
        gas_constant=1.0 / gamma,
        evaluate=evaluate,
        parameters={
            "diaphragm": diaphragm,
            "rho_left": left.rho,
            "u_left": left.u,
            "p_left": left.p,
            "rho_right": right.rho,
            "u_right": right.u,
            "p_right": right.p,
        },
        references=("toro2009", "sod1978" if name == "sod" else "lax1954"),
    )
