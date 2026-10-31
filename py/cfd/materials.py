"""Solid material presets: properties at 300 K from Incropera et al., *Fundamentals of
Heat and Mass Transfer*, 7th ed. (2011), Table A.1."""

from cfd.schema import Solid
from cfd.units import J, K, W, kg, m

COPPER = Solid(
    name="copper",
    density=8933.0 * kg / m**3,
    specific_heat=385.0 * J / kg / K,
    conductivity=401.0 * W / m / K,
)

ALUMINIUM = Solid(
    name="aluminium",
    density=2702.0 * kg / m**3,
    specific_heat=903.0 * J / kg / K,
    conductivity=237.0 * W / m / K,
)

STAINLESS_STEEL_304 = Solid(
    name="stainless-steel-304",
    density=7900.0 * kg / m**3,
    specific_heat=477.0 * J / kg / K,
    conductivity=14.9 * W / m / K,
)
