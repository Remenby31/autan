"""Two gases, one solid: hot air and cold CO2 ducts separated by a copper plate (CHT)."""

import autan
from autan.units import K, Pa, deg, kPa

case = autan.Case("heat-exchanger", mesh="heat-exchanger.msh")
case.physics = autan.Compressible(turbulence=autan.SST(), gas=autan.gases.AIR)
case.freestream(mach=0.1, aoa=0 * deg, pressure=101325 * Pa, temperature=300 * K)
case.cell_zone("co2").fluid(autan.gases.CO2)
case.cell_zone("plate").solid(autan.materials.COPPER)
case.boundary("air-inlet").inlet(total_pressure=103 * kPa, total_temperature=400 * K)
case.boundary("co2-inlet").inlet(total_pressure=103 * kPa, total_temperature=280 * K)
case.boundary("*-outlet").outlet(static_pressure=101325 * Pa)
