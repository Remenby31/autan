"""Viscous reference flows: compressible Couette, plane Poiseuille, Blasius boundary layer."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from autan_vv.oracles import FloatArray, Oracle


def couette(
    wall_speed: float = 50.0,
    height: float = 1.0,
    t_bottom: float = 300.0,
    t_top: float = 320.0,
    pressure: float = 1.0e5,
    viscosity: float = 1.8e-5,
    prandtl: float = 0.72,
    gamma: float = 1.4,
    gas_constant: float = 287.0,
) -> Oracle:
    """Compressible plane Couette flow with viscous heating, constant mu and Pr (exact NS solution).

    Walls y = 0 (at rest, T_bottom) and y = H (moving at U in x, T_top), uniform pressure:
    u = U y/H, T = T_b + (T_t - T_b) y/H + Pr U^2/(2 cp) (y/H)(1 - y/H), rho = p/(R T), from
    k T'' + mu (du/dy)^2 = 0 with k = cp mu/Pr (White, Viscous Fluid Flow, section 3.2).
    """
    cp = gamma * gas_constant / (gamma - 1.0)

    def evaluate(points: FloatArray, _t: float | None) -> dict[str, FloatArray]:
        eta = points[:, 1] / height
        temperature = (
            t_bottom
            + (t_top - t_bottom) * eta
            + prandtl * wall_speed**2 / (2.0 * cp) * eta * (1.0 - eta)
        )
        zero = np.zeros_like(eta)
        return {
            "Density": pressure / (gas_constant * temperature),
            "VelocityX": wall_speed * eta,
            "VelocityY": zero,
            "VelocityZ": zero,
            "Pressure": np.full_like(eta, pressure),
            "Temperature": temperature,
        }

    return Oracle(
        name="couette",
        description="Compressible Couette flow with viscous heating between isothermal walls y = 0 and y = H.",
        physics="navier-stokes",
        dim=1,
        unsteady=False,
        gamma=gamma,
        gas_constant=gas_constant,
        evaluate=evaluate,
        parameters={
            "wall_speed": wall_speed,
            "height": height,
            "t_bottom": t_bottom,
            "t_top": t_top,
            "pressure": pressure,
            "viscosity": viscosity,
            "prandtl": prandtl,
        },
        references=("white2006",),
    )


def poiseuille(
    gradient: float = 1.0,
    height: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0e-2,
    pressure: float = 1.0e5,
) -> Oracle:
    """Plane Poiseuille flow, exact for the incompressible NS equations only.

    u = G/(2 mu) y (H - y) between walls y = 0 and y = H, p = p0 - G x. For the compressible solver
    it holds to O(M^2) at low Mach number.
    """

    def evaluate(points: FloatArray, _t: float | None) -> dict[str, FloatArray]:
        y = points[:, 1]
        zero = np.zeros_like(y)
        return {
            "Density": np.full_like(y, density),
            "VelocityX": gradient / (2.0 * viscosity) * y * (height - y),
            "VelocityY": zero,
            "VelocityZ": zero,
            "Pressure": pressure - gradient * points[:, 0],
        }

    return Oracle(
        name="poiseuille",
        description="Plane Poiseuille flow between walls y = 0 and y = H (incompressible).",
        physics="incompressible-navier-stokes",
        dim=1,
        unsteady=False,
        gamma=math.inf,
        gas_constant=math.nan,
        evaluate=evaluate,
        parameters={
            "gradient": gradient,
            "height": height,
            "density": density,
            "viscosity": viscosity,
        },
        references=("white2006",),
    )


BLASIUS_ETA_MAX = 20.0


@cache
def blasius_profile() -> tuple[float, float, Any]:
    """Solve f''' + f f''/2 = 0, f(0) = f'(0) = 0, f'(inf) = 1 by shooting on f''(0).

    Returns (f''(0), the displacement constant eta - f at infinity, the dense solution on
    [0, BLASIUS_ETA_MAX]); beyond it f' = 1 to within e^(-eta^2/4).
    """

    def rhs(_eta: float, f: FloatArray) -> list[float]:
        return [f[1], f[2], -0.5 * f[0] * f[2]]

    def solve(curvature: float) -> Any:
        return solve_ivp(
            rhs,
            (0.0, BLASIUS_ETA_MAX),
            [0.0, 0.0, curvature],
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            dense_output=True,
        )

    curvature = brentq(lambda s: solve(s).y[1, -1] - 1.0, 0.2, 0.5, xtol=1e-16, rtol=1e-15)
    sol = solve(curvature)
    return curvature, float(BLASIUS_ETA_MAX - sol.y[0, -1]), sol


def blasius(
    speed: float = 10.0,
    viscosity: float = 1.5e-5,
    density: float = 1.2,
    pressure: float = 1.0e5,
    leading_edge: float = 0.0,
) -> Oracle:
    """Blasius boundary layer on a flat plate y = 0 starting at x = `leading_edge` (x > leading edge).

    eta = y sqrt(U/(nu x)), u = U f'(eta), v = (1/2) sqrt(nu U/x) (eta f' - f); uniform density and
    pressure. A boundary-layer solution: it approximates NS for Re_x >> 1 and low Mach number.
    """
    nu = viscosity / density

    def evaluate(points: FloatArray, _t: float | None) -> dict[str, FloatArray]:
        _, displacement, sol = blasius_profile()
        x = points[:, 0] - leading_edge
        eta = points[:, 1] * np.sqrt(speed / (nu * x))
        inside = eta < BLASIUS_ETA_MAX
        f = eta - displacement
        fp = np.ones_like(eta)
        values = sol.sol(np.minimum(eta, BLASIUS_ETA_MAX))
        f = np.where(inside, values[0], f)
        fp = np.where(inside, values[1], fp)
        zero = np.zeros_like(eta)
        return {
            "Density": np.full_like(eta, density),
            "VelocityX": speed * fp,
            "VelocityY": 0.5 * np.sqrt(nu * speed / x) * (eta * fp - f),
            "VelocityZ": zero,
            "Pressure": np.full_like(eta, pressure),
        }

    return Oracle(
        name="blasius",
        description="Blasius flat-plate boundary layer (boundary-layer equations, Re_x >> 1, low Mach).",
        physics="boundary-layer",
        dim=2,
        unsteady=False,
        gamma=math.inf,
        gas_constant=math.nan,
        evaluate=evaluate,
        parameters={"speed": speed, "viscosity": viscosity, "density": density},
        references=("blasius1908", "ganapol2010"),
    )
