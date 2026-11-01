"""Shu-Osher shock/entropy-wave interaction: a fine-grid 1D reference, not yet an oracle.

The problem has no exact solution. The reference comes from a 1D finite-volume solve: MUSCL
reconstruction of primitive variables with the van Leer limiter, HLLC fluxes (Toro 2009, section
10.6) and the third-order SSP Runge-Kutta scheme (Shu and Osher 1988), on `cells` uniform cells.
`test_oracles.py` checks this solver against the exact Sod solution and the profile's exact parts
(the undisturbed flow ahead of the shock). The oscillatory profile behind the shock is confirmed by
no independent source yet, so the reference is not registered as an oracle (AGENTS.md, Tests): it
waits for independently published fine-grid data.
"""

from __future__ import annotations

import hashlib
import math
from functools import cache
from pathlib import Path

import numpy as np

from autan_vv.errors import VVError
from autan_vv.oracles import FloatArray, Oracle

GAMMA = 1.4
DOMAIN = (-5.0, 5.0)


def shu_osher_initial(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Shu and Osher (1989): a Mach 3 shock at x = -4 moving into rho = 1 + 0.2 sin(5x)."""
    left = x < -4.0
    rho = np.where(left, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x))
    u = np.where(left, 2.629369, 0.0)
    p = np.where(left, 10.33333, 1.0)
    return rho, u, p


def _hllc(ql: FloatArray, qr: FloatArray, gamma: float) -> FloatArray:
    """HLLC flux between primitive states (rho, u, p), wave speeds of Davis (Toro eq. 10.48)."""
    rl, ul, pl = ql
    rr, ur, pr = qr
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    sl = np.minimum(ul - al, ur - ar)
    sr = np.maximum(ul + al, ur + ar)
    s_star = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / (
        rl * (sl - ul) - rr * (sr - ur)
    )
    el = pl / (gamma - 1) + 0.5 * rl * ul**2
    er = pr / (gamma - 1) + 0.5 * rr * ur**2
    ulc = np.array([rl, rl * ul, el])
    urc = np.array([rr, rr * ur, er])
    fl = np.array([rl * ul, rl * ul**2 + pl, (el + pl) * ul])
    fr = np.array([rr * ur, rr * ur**2 + pr, (er + pr) * ur])

    def star(
        r: FloatArray, u: FloatArray, p: FloatArray, e: FloatArray, s: FloatArray
    ) -> FloatArray:
        c = r * (s - u) / (s - s_star)
        return c * np.array(  # type: ignore[no-any-return]
            [np.ones_like(r), s_star, e / r + (s_star - u) * (s_star + p / (r * (s - u)))]
        )

    fls = fl + sl * (star(rl, ul, pl, el, sl) - ulc)
    frs = fr + sr * (star(rr, ur, pr, er, sr) - urc)
    return np.where(sl >= 0, fl, np.where(s_star >= 0, fls, np.where(sr > 0, frs, fr)))


def solve_1d(
    rho: FloatArray,
    u: FloatArray,
    p: FloatArray,
    dx: float,
    t_end: float,
    gamma: float = GAMMA,
    cfl: float = 0.5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Advance a 1D Euler state to `t_end` with transmissive boundaries; returns (rho, u, p)."""
    cons = np.array([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u**2])

    def primitive(c: FloatArray) -> FloatArray:
        r = c[0]
        v = c[1] / r
        return np.array([r, v, (gamma - 1) * (c[2] - 0.5 * r * v**2)])

    def rate(c: FloatArray) -> FloatArray:
        q = primitive(c)
        q = np.concatenate([q[:, :1], q[:, :1], q, q[:, -1:], q[:, -1:]], axis=1)
        dl, dr = q[:, 1:-1] - q[:, :-2], q[:, 2:] - q[:, 1:-1]
        slope = np.where(
            dl * dr > 0, 2 * dl * dr / (dl + dr + np.where(dl + dr == 0, 1.0, 0.0)), 0.0
        )
        left_face = q[:, 1:-1] + 0.5 * slope
        right_face = q[:, 1:-1] - 0.5 * slope
        flux = _hllc(left_face[:, :-1], right_face[:, 1:], gamma)
        return -(flux[:, 1:] - flux[:, :-1]) / dx

    t = 0.0
    while t < t_end:
        q = primitive(cons)
        dt = min(cfl * dx / float(np.max(np.abs(q[1]) + np.sqrt(gamma * q[2] / q[0]))), t_end - t)
        c1 = cons + dt * rate(cons)
        c2 = 0.75 * cons + 0.25 * (c1 + dt * rate(c1))
        cons = cons / 3.0 + 2.0 / 3.0 * (c2 + dt * rate(c2))
        t += dt
    q = primitive(cons)
    return q[0], q[1], q[2]


def shu_osher(cells: int = 12800, t_end: float = 1.8, cache_dir: Path | None = None) -> Oracle:
    """Shu-Osher reference at `t_end`, piecewise-linear in x between the reference cell centres.

    The reference is computed on first evaluation and cached in `cache_dir` (default ~/.cache/autan-vv) under a name derived from the
    resolution, the end time and this module's source, so that a change of the solver recomputes it.
    """
    x0, x1 = DOMAIN
    dx = (x1 - x0) / cells
    xc = x0 + (np.arange(cells, dtype=np.float64) + 0.5) * dx
    digest = hashlib.sha256(Path(__file__).read_bytes() + f"{cells}:{t_end}".encode()).hexdigest()[
        :16
    ]
    path = (cache_dir or Path("~/.cache/autan-vv").expanduser()) / f"shu-osher-{digest}.npy"

    @cache
    def reference() -> FloatArray:
        if path.exists():
            return np.asarray(np.load(path))
        solution = np.array(solve_1d(*shu_osher_initial(xc), dx, t_end))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, solution)
        return solution

    def evaluate(points: FloatArray, time: float | None) -> dict[str, FloatArray]:
        if time is None or not math.isclose(time, t_end, rel_tol=1e-12):
            msg = f"the Shu-Osher reference exists at t = {t_end} only, not t = {time}"
            raise VVError(msg, code="V0707")
        x = points[:, 0]
        if not (np.all(np.isfinite(x)) and np.all(x >= DOMAIN[0]) and np.all(x <= DOMAIN[1])):
            msg = f"the Shu-Osher reference covers {DOMAIN[0]} <= x <= {DOMAIN[1]} only"
            raise VVError(msg, code="V0707")
        rho, u, p = reference()
        zero = np.zeros_like(x)
        return {
            "Density": np.interp(x, xc, rho),
            "VelocityX": np.interp(x, xc, u),
            "VelocityY": zero,
            "VelocityZ": zero,
            "Pressure": np.interp(x, xc, p),
        }

    return Oracle(
        name="shu-osher",
        description=f"Shu-Osher problem at t = {t_end}, 1D reference on {cells} cells (regression only).",
        physics="euler",
        axes=(0,),
        unsteady=True,
        gamma=GAMMA,
        gas_constant=1.0 / GAMMA,
        evaluate=evaluate,
        parameters={"cells": cells, "t_end": t_end},
        references=("shu1989", "toro2009"),
        reference_only=True,
    )
