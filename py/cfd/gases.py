"""Ideal-gas presets.

Air: molar mass, Sutherland law (1.716e-5 Pa s at 273.15 K, S = 110.4 K) from the
U.S. Standard Atmosphere 1976; gamma = 1.4 and Pr = 0.72 as in the NASA Turbulence
Modeling Resource. CO2: molar mass from IUPAC atomic weights; Sutherland constants
from White, *Viscous Fluid Flow*, 3rd ed. (2006), Table 1-2; Pr = 0.766 and
gamma = cp / (cp - R) with cp = 851 J/(kg K), both at 300 K from Incropera et al.,
*Fundamentals of Heat and Mass Transfer*, 7th ed. (2011), Table A.4.
"""

from cfd.schema import IdealGas, Sutherland
from cfd.units import K, Pa, kg, mol, s

AIR = IdealGas(
    name="air",
    molar_mass=0.0289647 * kg / mol,
    gamma=1.4,
    prandtl=0.72,
    viscosity=Sutherland(mu_ref=1.716e-5 * Pa * s, t_ref=273.15 * K, s=110.4 * K),
)

CO2 = IdealGas(
    name="co2",
    molar_mass=0.0440095 * kg / mol,
    gamma=1.285,
    prandtl=0.766,
    viscosity=Sutherland(mu_ref=1.370e-5 * Pa * s, t_ref=273.15 * K, s=222.0 * K),
)
