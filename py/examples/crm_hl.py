"""CRM-HL high-lift configuration near CLmax (HLPW-5), as in docs/design/dx.html, with the
HLPW semispan reference area (297 360 in²)."""

import cfd
from cfd.units import deg, m

case = cfd.Case("crm-hl", mesh="crm-hl.cgns")
case.physics = cfd.Compressible(turbulence=cfd.SA(), precision="fp32+acc64")
case.freestream(mach=0.20, aoa=19.57 * deg, re=5.49e6, ref_length=7.0 * m)
case.boundary("wing.*").wall()
case.boundary("symmetry").symmetry()
case.monitor(cfd.forces("wing.*", ref_area=191.84 * m**2), every=10)
