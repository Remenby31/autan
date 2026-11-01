"""Skin friction of TMR 2DZP runs: cf at x = 0.970084071 and CD, one line per run directory.

Usage: python physics/tmr/flatplate_cf.py RUN...   (in the uv environment of vv/)

cf = tau_w,x / q_inf from the wall shear stress of solution.cgns (results.md, WallSolution),
linearly interpolated between the plate faces on either side of the station; CD is the last value of
the `plate` forces monitor. q_inf follows the case's freestream closure (case-schema.md): M 0.2,
Re 5e6 per metre, T 300 K, Sutherland's air.
"""

from __future__ import annotations

import sys
from pathlib import Path

import autan
import numpy as np
from autan_vv.harness import read_geometry

STATION = 0.970084071


def q_inf() -> float:
    g, r, t, mach, re = 1.4, 8.314462618 / 0.0289647, 300.0, 0.2, 5.0e6
    mu = 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)
    v = mach * (g * r * t) ** 0.5
    rho = re * mu / v
    return 0.5 * rho * v * v


def main() -> None:
    q = q_inf()
    for path in sys.argv[1:]:
        r = autan.Results.open(Path(path))
        geometry = read_geometry(r).geometry
        faces, tau = r.wall("WallShearStressX")
        x = geometry.face_centroid[faces, 0]
        y = geometry.face_centroid[faces, 1]
        on_plate = (np.abs(y) < 1e-12) & (x > 0.0)
        order = np.argsort(x[on_plate])
        xs, cf = x[on_plate][order], tau[on_plate][order] / q
        cd = r.forces.cd[-1]
        sys.stdout.write(
            f"{r.run.run_id} cells {geometry.cell_volume.size} iterations {r.run.iterations} "
            f"cf(0.97) {np.interp(STATION, xs, cf):.6e} CD {cd:.6e}\n"
        )


if __name__ == "__main__":
    main()
