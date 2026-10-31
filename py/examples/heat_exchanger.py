"""Two gases, one solid: hot air and cold CO2 ducts separated by a copper plate (CHT)."""

import cfd
from cfd.units import K, Pa, deg, kPa

case = cfd.Case("heat-exchanger", mesh="heat-exchanger.msh")
case.physics = cfd.Compressible(turbulence=cfd.SST(), gas=cfd.gases.AIR)
case.freestream(mach=0.1, aoa=0 * deg, pressure=101325 * Pa, temperature=300 * K)
case.cell_zone("co2").fluid(cfd.gases.CO2)
case.cell_zone("plate").solid(cfd.materials.COPPER)
case.boundary("air-inlet").inlet(total_pressure=103 * kPa, total_temperature=400 * K)
case.boundary("co2-inlet").inlet(total_pressure=103 * kPa, total_temperature=280 * K)
case.boundary("*-outlet").outlet(static_pressure=101325 * Pa)
