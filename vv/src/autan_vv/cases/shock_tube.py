"""The Sod shock tube of the `sod-order` gate (acceptance.md, M2): Toro (2009), Table 4.1, test 1, at
t = 0.2 on the `sod-tube` family (`autan_vv.meshes.tube`).

`case` builds the autan.case/1 case of one level, started from the oracle at t = 0 written by
`autan_vv.initial_state.write_initial_state`. The oracle's gas has R = 1/gamma, so the left state
(density 1, pressure 1) has T = gamma K. Every face is a symmetry plane: the ends reflect, but by
t = 0.2 no wave reaches them (the rarefaction's head is at 0.5 - c_L t = 0.26, c_L = sqrt(1.4), and
the shock at 0.5 + 1.75 t = 0.85), so the solution is the exact one on the whole tube.
"""

from __future__ import annotations

from typing import Literal

import autan
from autan.units import K, Pa, kg, mol, s

from autan_vv.harness import UNIVERSAL_GAS_CONSTANT
from autan_vv.meshes.families import BOX_ZONES
from autan_vv.oracles.inviscid import GAMMA, SHOCK_TUBES

END_TIME = 0.2
# Time steps per cell along x of the finest level: dt = 0.1 dx, a CFL number of about 0.2 for the
# fastest wave (|u| + c below 2.1 in the exact solution). Every level takes the finest level's
# step, so the levels solve one case and differ by their mesh.
STEPS_PER_CELL = 2


def case(
    mesh: str,
    n: int,
    initial_state: autan.StateFile,
    *,
    finest: int,
    precision: Literal["fp64", "fp32+acc64"] = "fp64",
) -> autan.Case:
    """The case of level `n` (n cells along the tube) whose mesh is `mesh`, in a family whose
    finest level is `finest`: the Sod tube at t = 0.2 with the finest level's fixed step, so that
    every level ends at t = 0.2 exactly."""
    result = autan.Case(f"sod-tube-n{n:03d}", mesh=mesh)
    gas = autan.IdealGas(
        name="sod",
        molar_mass=UNIVERSAL_GAS_CONSTANT * GAMMA * kg / mol,
        gamma=GAMMA,
        prandtl=0.72,
        viscosity=autan.ConstantViscosity(mu=1e-5 * Pa * s),
    )
    result.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision, gas=gas)
    left = SHOCK_TUBES["sod"][0]
    result.freestream(mach=0.5, pressure=left.p * Pa, temperature=GAMMA * left.p / left.rho * K)
    steps = STEPS_PER_CELL * finest
    result.time = autan.Unsteady(steps=steps, dt=END_TIME / steps * s)
    for zone in BOX_ZONES:
        result.boundary(zone).symmetry()
    result.initial_state = initial_state
    return result
