"""The compressible Couette case of the `couette-velocity-exact` and `couette-temperature-order`
gates (acceptance.md, M3; issue #596), on one level of the `couette` mesh family
(`autan_vv.meshes.couette`).

The solver has no moving wall and Navier-Stokes refuses periodic pairs, so the moving wall `ymax`
and the ends `xmin`, `xmax` impose the oracle's states (`oracle-state`): the gates verify the
interior scheme and the isothermal no-slip wall `ymin` only. The flow starts from the freestream
(U/2 along x, (T_b + T_t)/2, p).
"""

from __future__ import annotations

import math
from typing import Literal

import autan
from autan.units import K, Pa, kg, mol, s

from autan_vv.harness import UNIVERSAL_GAS_CONSTANT
from autan_vv.oracles.viscous import couette

MAX_ITERATIONS = 5000
RESIDUAL_ORDERS = 10.0


def case(
    mesh: str,
    n: int,
    states: str,
    states_sha256: str,
    *,
    precision: Literal["fp64", "fp32+acc64"] = "fp64",
) -> autan.Case:
    """The case of level `n` whose mesh is `mesh` and whose face states (`autan.face-states/1`,
    from `autan_vv.meshes.couette.write_face_states`) are `states` with SHA-256 `states_sha256`;
    the gas and the walls are the oracle's defaults."""
    oracle = couette()
    p = oracle.parameters
    result = autan.Case(f"couette-n{n:03d}", mesh=mesh)
    gas = autan.IdealGas(
        name="couette",
        molar_mass=UNIVERSAL_GAS_CONSTANT / oracle.gas_constant * kg / mol,
        gamma=oracle.gamma,
        prandtl=p["prandtl"],
        viscosity=autan.ConstantViscosity(mu=p["viscosity"] * Pa * s),
    )
    result.physics = autan.Compressible(turbulence=autan.Laminar(), precision=precision, gas=gas)
    temperature = 0.5 * (p["t_bottom"] + p["t_top"])
    sound = math.sqrt(oracle.gamma * oracle.gas_constant * temperature)
    result.freestream(
        mach=0.5 * p["wall_speed"] / sound,
        temperature=temperature * K,
        pressure=p["pressure"] * Pa,
    )
    result.time = autan.Steady(
        max_iterations=MAX_ITERATIONS,
        residual_orders=RESIDUAL_ORDERS,
        linear_solver="gauss-seidel",
    )
    result.boundary("ymin").wall(temperature=p["t_bottom"] * K)
    for zone in ("ymax", "xmin", "xmax"):
        result.boundary(zone).oracle_state(path=states, sha256=states_sha256)
    result.boundary("zmin").symmetry()
    result.boundary("zmax").symmetry()
    return result
