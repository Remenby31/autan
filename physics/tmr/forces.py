"""Skin friction and forces of TMR runs, one line per run directory.

Usage: python physics/tmr/forces.py CASE RUN...   (in the uv environment of vv/; CASE as case.sh)

cf = tau_w,x / q_inf from the wall shear stress of solution.cgns (results.md, WallSolution), linearly
interpolated in x along the wall at the case's station (flatplate 0.970084071, bump 0.75; the
upper surface at x = 0.5 for naca0012); CL and CD are the last values of the `wall` forces monitor.
q_inf follows the case's freestream closure (case-schema.md), T 300 K, Sutherland's air.
"""

from __future__ import annotations

import sys
from pathlib import Path

import autan
import numpy as np
from autan_vv.harness import read_geometry

CASES = {  # Mach, Reynolds number per metre, cf station, wall x range
    "flatplate": (0.2, 5.0e6, 0.970084071, (0.0, 2.0)),
    "bump": (0.2, 3.0e6, 0.75, (0.0, 1.5)),
    "naca0012": (0.15, 6.0e6, 0.5, (0.0, 1.0)),
}


def q_inf(mach: float, re: float) -> float:
    g, r, t = 1.4, 8.314462618 / 0.0289647, 300.0
    mu = 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)
    v = mach * (g * r * t) ** 0.5
    return 0.5 * re * mu / v * v * v


def main() -> None:
    mach, re, station, (x0, x1) = CASES[sys.argv[1]]
    q = q_inf(mach, re)
    for path in sys.argv[2:]:
        r = autan.Results.open(Path(path))
        geometry = read_geometry(r).geometry
        faces, tau = r.wall("WallShearStressX")
        x = geometry.face_centroid[faces, 0]
        y = geometry.face_centroid[faces, 1]
        # Wall faces only (WallSolution also holds slip faces, whose shear is zero), the upper
        # surface of an airfoil.
        on = (x > x0) & (x < x1) & (np.abs(tau) > 0.0) & (y >= 0.0)
        order = np.argsort(x[on])
        cf = np.interp(station, x[on][order], tau[on][order] / q)
        forces = r.forces
        sys.stdout.write(
            f"{r.run.run_id} cells {geometry.cell_volume.size} iterations {r.run.iterations} "
            f"status {r.run.status} cf {cf:.6e} CL {forces.cl[-1]:.6e} CD {forces.cd[-1]:.6e}\n"
        )


if __name__ == "__main__":
    main()
