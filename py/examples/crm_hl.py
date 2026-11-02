"""CRM-HL high-lift configuration near CLmax (HLPW-5), as in docs/design/dx.html, with the
HLPW semispan reference area (297 360 in²)."""

import autan
from autan.units import deg, m

case = autan.Case("crm-hl", mesh="crm-hl.cgns")
case.physics = autan.Compressible(turbulence=autan.SA())
case.freestream(mach=0.2, aoa=19.57 * deg, re=5490000.0, ref_length=7.0 * m)
case.boundary("wing.*").wall()
case.boundary("symmetry").symmetry()
case.monitor(autan.forces("wing.*", ref_area=191.84 * m**2), every=10)
