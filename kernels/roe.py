"""First-order Roe flux of the 3D Euler equations, the nonlinear body of the AD spike.

Roe (1981), J. Comput. Phys. 43:357-372, eqs. 7-8, in the wave-strength form of Blazek,
Computational Fluid Dynamics: Principles and Applications, 3rd ed., 2015, eqs. 4.89-4.91, with
Harten's entropy fix (Harten 1983, J. Comput. Phys. 49:357-393, eq. 4.5) on every eigenvalue.
Conservative variables U = (rho, rho u, rho v, rho w, rho E), ideal gas.
"""

from autan_codegen.dsl import (
    Geom,
    Inc,
    Residual,
    State,
    Vec3,
    Vec5,
    cells,
    dot,
    face_loop,
    geometry,
    pointwise,
    sqrt,
)

GAMMA = 1.4
HARTEN = 0.1  # entropy-fix width as a fraction of the Roe-averaged sound speed

u_cons = cells.field("u_cons", State.Vec5)
res = cells.field("res", Residual.Vec5)


@pointwise
def harten(lam: State, delta: State) -> State:
    a = abs(lam)
    if a < delta:
        a = (lam * lam + delta * delta) / (2.0 * delta)
    return a


@pointwise
def physical_flux(u: State.Vec5, n: Geom.Vec3) -> State.Vec5:
    rho = u[0]
    vel = Vec3(u[1] / rho, u[2] / rho, u[3] / rho)
    p = (GAMMA - 1.0) * (u[4] - 0.5 * rho * dot(vel, vel))
    vn = dot(vel, n)
    return Vec5(rho * vn, u[1] * vn + p * n[0], u[2] * vn + p * n[1], u[3] * vn + p * n[2], (u[4] + p) * vn)


@pointwise
def roe_flux(ul: State.Vec5, ur: State.Vec5, n: Geom.Vec3, area: Geom) -> Residual.Vec5:
    rl, rr = ul[0], ur[0]
    vl = Vec3(ul[1] / rl, ul[2] / rl, ul[3] / rl)
    vr = Vec3(ur[1] / rr, ur[2] / rr, ur[3] / rr)
    pl = (GAMMA - 1.0) * (ul[4] - 0.5 * rl * dot(vl, vl))
    pr = (GAMMA - 1.0) * (ur[4] - 0.5 * rr * dot(vr, vr))
    hl = (ul[4] + pl) / rl
    hr = (ur[4] + pr) / rr
    s = sqrt(rr / rl)
    rho = s * rl
    v = (vl + s * vr) / (1.0 + s)
    h = (hl + s * hr) / (1.0 + s)
    vn = dot(v, n)
    q2 = dot(v, v)
    c = sqrt((GAMMA - 1.0) * (h - 0.5 * q2))
    dp = pr - pl
    drho = rr - rl
    dv = vr - vl
    dvn = dot(dv, n)
    delta = HARTEN * c
    l1 = harten(vn - c, delta)
    l2 = harten(vn, delta)
    l5 = harten(vn + c, delta)
    a1 = l1 * (dp - rho * c * dvn) / (2.0 * c * c)
    a5 = l5 * (dp + rho * c * dvn) / (2.0 * c * c)
    a2 = l2 * (drho - dp / (c * c))
    shear = dv - dvn * n
    d0 = a1 + a2 + a5
    dm = a1 * (v - c * n) + a2 * v + l2 * rho * shear + a5 * (v + c * n)
    de = a1 * (h - c * vn) + a2 * 0.5 * q2 + l2 * rho * (dot(v, dv) - vn * dvn) + a5 * (h + c * vn)
    central = 0.5 * (physical_flux(ul, n) + physical_flux(ur, n))
    return area * (central - 0.5 * Vec5(d0, dm[0], dm[1], dm[2], de))


flux = face_loop(
    roe_flux,
    ul=u_cons.left,
    ur=u_cons.right,
    n=geometry.face_normal,
    area=geometry.face_area_magnitude,
    out=Inc(res, left=+1, right=-1),
)
