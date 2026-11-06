"""#536: every intermediate array of naca0012.plane_grid, saved bit for bit."""

import math
import sys

import numpy as np
from autan_vv.meshes import naca0012 as nm

nc, out = int(sys.argv[1]), sys.argv[2]
st: dict[str, np.ndarray] = {}
kt = nm.KarmanTrefftz()
st["x_te"] = np.array(kt.x_te)
st["power"] = np.array(kt.power)
st["z2"] = np.array(kt.z2)
zeta_le = complex(kt.to_circle(np.array(0j)))
st["zeta_le"] = np.array(zeta_le)
centre = 0.5 * (zeta_le.real + 1.0)


def quasi(n: int, tag: str) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * math.pi * np.arange(n // 2 + 1) / n
    st[f"{tag}.theta"] = theta
    u = nm._surface_nodes(kt, centre, theta)
    st[f"{tag}.u"] = u
    upper = nm._upper_surface(kt, u)
    st[f"{tag}.upper"] = upper
    z = np.concatenate([upper, np.conj(upper[-2:0:-1])])
    return np.concatenate([theta, 2.0 * math.pi - theta[-2:0:-1]]), z


_, z_finest = quasi(nm.FINEST, "finest")
zeta_finest = kt.to_circle(z_finest)
st["finest.to_circle"] = zeta_finest
r1 = float(np.abs(np.roll(zeta_finest, -1) - zeta_finest).sum()) / (2.0 * math.pi)
st["r1"] = np.array(r1)
theta, z_surface = quasi(nc, "level")
r_surface = np.abs(kt.to_circle(z_surface) - centre)
st["level.r_surface"] = r_surface
radii = r1 * np.exp(2.0 * math.pi * np.arange(nc + 1) / nc)
st["radii"] = radii
r_far = radii[-1]
r = (r_surface[:, None] * (r_far - radii[None, :]) + r_far * (radii[None, :] - r1)) / (r_far - r1)
st["r"] = r
rot = np.exp(1j * theta)
st["exp_i_theta"] = rot
zeta = centre + r * rot[:, None]
st["zeta"] = zeta
st["to_airfoil"] = kt.to_airfoil(zeta)
z, _ = nm.plane_grid(nc)
st["plane_grid"] = z
st["grid.points"] = nm.grid(nc)[0]
np.savez(out, **st)
sys.stdout.write(f"saved {len(st)} arrays\n")
