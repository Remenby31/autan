"""TMR 2DZP zero-pressure-gradient flat plate (M 0.2, Re 5e6 per metre, T 300 K), SA-neg, as an
``autan.Case`` script: the same case ``case.sh flatplate`` writes as TOML, for the mesh
``flatplate.msh`` beside this file (one layer of unit thickness between ``sides``, plate x = 0..2)."""

import autan
from autan.units import K, Pa, m

MACH = 0.2
RE_PER_METRE = 5.0e6
TEMPERATURE = 300.0
PLATE_LENGTH = 2.0
SPAN = 1.0

# p_inf = Re' mu R T / V; p0 = p_inf r^(g/(g-1)), T0 = T r with r = 1 + (g-1) M^2 / 2.
gas = autan.gases.AIR
R = 8.314462618 / gas.molar_mass.to("kg/mol").magnitude
gamma = gas.gamma
visc = gas.viscosity
mu = (
    visc.mu_ref.to("Pa*s").magnitude
    * (TEMPERATURE / visc.t_ref.to("K").magnitude) ** 1.5
    * (visc.t_ref.to("K").magnitude + visc.s.to("K").magnitude)
    / (TEMPERATURE + visc.s.to("K").magnitude)
)
speed = MACH * (gamma * R * TEMPERATURE) ** 0.5
p_inf = RE_PER_METRE * mu / speed * R * TEMPERATURE
ratio = 1 + 0.5 * (gamma - 1) * MACH**2

case = autan.Case("tmr-flatplate", mesh="flatplate.msh")
case.physics = autan.Compressible(
    turbulence=autan.SA(nu_tilde_ratio=3.0), precision="fp64", gas=gas
)
case.freestream(mach=MACH, re=RE_PER_METRE, ref_length=1.0 * m, temperature=TEMPERATURE * K)
case.time = autan.Steady(max_iterations=20000, residual_orders=10.0)
case.boundary("sides").symmetry()
case.boundary("plate").wall()
case.boundary("symmetry").symmetry()
case.boundary("inflow").inlet(
    total_pressure=p_inf * ratio ** (gamma / (gamma - 1)) * Pa,
    total_temperature=TEMPERATURE * ratio * K,
)
case.boundary("outflow").outlet(static_pressure=p_inf * Pa)
case.boundary("top").farfield()
case.monitor(autan.forces("plate", name="wall", ref_area=PLATE_LENGTH * SPAN * m**2), every=1)
