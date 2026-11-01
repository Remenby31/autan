"""Viscous fluxes of the compressible Navier-Stokes equations, added to the Euler residual.

The residual of the Navier-Stokes equations is that of `euler.py` plus, over the faces of each cell,
-F_v . S, with the viscous flux F_v = (0, tau n, u . tau n + k grad T . n) of a Newtonian fluid
under Stokes' hypothesis and Fourier conduction (TMR, "Implementing turbulence models into the
compressible RANS equations"):

- tau = mu_e (grad u + grad u^T - 2/3 div u I), mu_e = mu(T) + mu_t, mu by Sutherland's law or
  constant, mu_t the eddy viscosity of the turbulence model (zero for laminar flow);
- k = c_p (mu / Pr + mu_t / Pr_t);
- face gradients: the average of the two cells' least-squares gradients, corrected along the line
  of centres so that its component along that line is the difference quotient of the cell values
  (Blazek 2015, eq. 5.69; Diskin et al. 2010, AIAA J. 48(7), "edge-normal" augmentation);
- face values: the average of both cells' linear reconstructions at the face centroid.

The loops share the datasets of `euler.py` (`prim`, `grad`, `prim_b`, `res`, `bc_kind`, `bc`): the
solver points both modules at the same arrays and runs these loops after the Euler loops.
"""

import sys
from pathlib import Path

from autan_codegen.dsl import (
    Geom,
    Grad,
    Inc,
    Jac,
    Jacobian,
    Param,
    Residual,
    State,
    Sum,
    Vec3,
    Vec5,
    Write,
    boundary_loop,
    cell_reduce,
    cells,
    dot,
    face_loop,
    faces,
    geometry,
    param,
    pointwise,
    rel,
    sqrt,
)

sys.path.insert(0, str(Path(__file__).parent))
from euler import (  # noqa: E402
    BC_SLIP,
    BC_WALL,
    bc,
    bc_kind,
    gamma,
    gas_constant,
    grad,
    p_ref,
    prim,
    prim_b,
    res,
    velocity,
)

mu_t = cells.field("mu_t", State)
mu_t_b = faces.field("mu_t_b", State)
lam_v = cells.field("lam_v", State)

mu_ref = param("mu_ref", Param)
t_ref = param("t_ref", Param)
sutherland_s = param("sutherland_s", Param)
sutherland = param("sutherland", Param)  # 1: Sutherland's law; 0: constant viscosity mu_ref
prandtl = param("prandtl", Param)
prandtl_t = param("prandtl_t", Param)


@pointwise
def temperature(w: State.Vec5, rgas: Param, pref: Param) -> State:
    return (w[4] + pref) / (w[0] * rgas)


@pointwise
def temperature_gradient(
    w: State.Vec5, gw: Grad.Mat53, rgas: Param, pref: Param
) -> Grad.Vec3:
    # T = p / (rho R): grad T = (grad p - R T grad rho) / (rho R).
    t = temperature(w, rgas, pref)
    gr = Vec3(gw[0, 0], gw[0, 1], gw[0, 2])
    gp = Vec3(gw[4, 0], gw[4, 1], gw[4, 2])
    return (gp - (rgas * t) * gr) / (w[0] * rgas)


@pointwise
def viscosity(t: State, mref: Param, tref: Param, s: Param, law: Param) -> State:
    # Sutherland (1893): mu = mu_ref (T / T_ref)^(3/2) (T_ref + S) / (T + S).
    q = t / tref
    return mref * q * sqrt(q) * (tref + s) / (t + s) if law > 0.5 else mref


@pointwise
def corrected(gbar: Grad.Vec3, dphi: State, e: Geom.Vec3, inv_d: Geom) -> Grad.Vec3:
    # Blazek (2015), eq. 5.69: replaces the component of the averaged gradient along the unit
    # vector e between the two values by their difference quotient.
    return gbar + (dphi * inv_d - dot(gbar, e)) * e


@pointwise
def viscous_flux(
    v: State.Vec3,
    gu: Grad.Vec3,
    gv: Grad.Vec3,
    gwz: Grad.Vec3,
    gt: Grad.Vec3,
    mu: State,
    k: State,
    n: Geom.Vec3,
) -> State.Vec5:
    # F_v . n with (tau n)_i = mu (grad u_i . n + d(u . n)/dx_i) - 2/3 mu div u n_i.
    div = gu[0] + gv[1] + gwz[2]
    tn = mu * (
        Vec3(dot(gu, n), dot(gv, n), dot(gwz, n)) + n[0] * gu + n[1] * gv + n[2] * gwz
    ) - (2.0 / 3.0 * mu * div) * n
    return Vec5(0.0, tn[0], tn[1], tn[2], dot(v, tn) + k * dot(gt, n))


@pointwise
def face_value(w: State.Vec5, gw: Grad.Mat53, r: Geom.Vec3, pref: Param) -> State.Vec5:
    # Linear reconstruction, the cell value where it would make density or pressure non-positive.
    wf = w + gw @ r
    return wf if wf[0] > 0.0 and wf[4] + pref > 0.0 else w


@pointwise
def viscous_face(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    ml: State,
    wr: State.Vec5,
    gr: Grad.Mat53,
    xr: Geom.Vec3,
    mr: State,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    g: Param,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
    prl: Param,
    prt: Param,
) -> Residual.Vec5:
    d = rel(xr, xl)
    dist = sqrt(dot(d, d))
    inv_d = 1.0 / dist
    e = inv_d * d
    wf = 0.5 * (face_value(wl, gl, rel(xf, xl), pref) + face_value(wr, gr, rel(xf, xr), pref))
    tl = temperature(wl, rgas, pref)
    tr = temperature(wr, rgas, pref)
    gm = 0.5 * (gl + gr)
    gu = corrected(Vec3(gm[1, 0], gm[1, 1], gm[1, 2]), wr[1] - wl[1], e, inv_d)
    gv = corrected(Vec3(gm[2, 0], gm[2, 1], gm[2, 2]), wr[2] - wl[2], e, inv_d)
    gwz = corrected(Vec3(gm[3, 0], gm[3, 1], gm[3, 2]), wr[3] - wl[3], e, inv_d)
    gtm = 0.5 * (temperature_gradient(wl, gl, rgas, pref) + temperature_gradient(wr, gr, rgas, pref))
    gt = corrected(gtm, tr - tl, e, inv_d)
    mu = viscosity(temperature(wf, rgas, pref), mref, tref, s, law)
    mt = 0.5 * (ml + mr)
    k = g * rgas / (g - 1.0) * (mu / prl + mt / prt)
    return -area * viscous_flux(velocity(wf), gu, gv, gwz, gt, mu + mt, k, n)


@pointwise
def viscous_boundary(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    wb: State.Vec5,
    mb: State,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    g: Param,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
    prl: Param,
    prt: Param,
) -> Residual.Vec5:
    # The face gradient joins the cell value to the boundary state at the face centroid. Slip walls
    # and symmetry planes carry no viscous flux; adiabatic walls no heat flux, and their
    # temperature is the cell's extrapolated to the face.
    r = rel(xf, xl)
    dist = sqrt(dot(r, r))
    inv_d = 1.0 / dist
    e = inv_d * r
    tl = temperature(wl, rgas, pref)
    gtl = temperature_gradient(wl, gl, rgas, pref)
    wall = kind > BC_WALL - 0.5
    adiabatic = wall and data[0] <= 0.0
    tb = tl + dot(gtl, r) if adiabatic else temperature(wb, rgas, pref)
    gu = corrected(Vec3(gl[1, 0], gl[1, 1], gl[1, 2]), wb[1] - wl[1], e, inv_d)
    gv = corrected(Vec3(gl[2, 0], gl[2, 1], gl[2, 2]), wb[2] - wl[2], e, inv_d)
    gwz = corrected(Vec3(gl[3, 0], gl[3, 1], gl[3, 2]), wb[3] - wl[3], e, inv_d)
    gt = corrected(gtl, tb - tl, e, inv_d)
    mu = viscosity(tb, mref, tref, s, law)
    k = 0.0 if adiabatic else g * rgas / (g - 1.0) * (mu / prl + mb / prt)
    f = -area * viscous_flux(velocity(wb), gu, gv, gwz, gt, mu + mb, k, n)
    slip = kind > BC_SLIP - 0.5 and kind < BC_SLIP + 0.5
    return 0.0 * f if slip else f


VISCOUS = {
    "g": gamma,
    "rgas": gas_constant,
    "pref": p_ref,
    "mref": mu_ref,
    "tref": t_ref,
    "s": sutherland_s,
    "law": sutherland,
    "prl": prandtl,
    "prt": prandtl_t,
}

FACE_ARGS = {
    "wl": prim.left,
    "gl": grad.left,
    "xl": geometry.cell_centroid.left,
    "ml": mu_t.left,
    "wr": prim.right,
    "gr": grad.right,
    "xr": geometry.cell_centroid.right,
    "mr": mu_t.right,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    **VISCOUS,
}

BOUNDARY_ARGS = {
    "wl": prim.left,
    "gl": grad.left,
    "xl": geometry.cell_centroid.left,
    "wb": prim_b,
    "mb": mu_t_b,
    "kind": bc_kind,
    "data": bc,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    **VISCOUS,
}

viscous_faces = face_loop(viscous_face, **FACE_ARGS, out=Inc(res, left=+1, right=-1))
viscous_boundaries = boundary_loop(viscous_boundary, **BOUNDARY_ARGS, out=Inc(res, left=+1))

# With the face Jacobians dF_v/dW_L and dF_v/dW_R, the cell gradients held fixed: the difference
# quotients of the face gradients give the compact part of the viscous Jacobian.
visc_jac = faces.field("visc_jac", Jac.shape(2, 5, 5))
visc_bnd_jac = faces.field("visc_bnd_jac", Jac.shape(5, 5))
viscous_faces_jacobian = face_loop(
    viscous_face,
    **FACE_ARGS,
    out=[Inc(res, left=+1, right=-1), Jacobian(visc_jac, wrt=("wl", "wr"))],
)
viscous_boundaries_jacobian = boundary_loop(
    viscous_boundary, **BOUNDARY_ARGS, out=[Inc(res, left=+1), Jacobian(visc_bnd_jac, wrt="wl")]
)


@pointwise
def viscous_wave(
    w: State.Vec5,
    mt: State,
    vol: Geom,
    area: Geom,
    g: Param,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
    prl: Param,
    prt: Param,
) -> State:
    # Viscous spectral radius of a face, Blazek (2015), eq. 6.22:
    # max(4 / (3 rho), gamma / rho) (mu / Pr + mu_t / Pr_t) A^2 / V.
    mu = viscosity(temperature(w, rgas, pref), mref, tref, s, law)
    return max(4.0 / 3.0, g) / w[0] * (mu / prl + mt / prt) * area * area / vol


viscous_spectral_radius = cell_reduce(
    viscous_wave,
    w=prim.self_,
    mt=mu_t.self_,
    vol=geometry.cell_volume.self_,
    area=geometry.face_area_magnitude,
    **VISCOUS,
    acc=Sum(lam_v),
)


@pointwise
def viscous_traction(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    wb: State.Vec5,
    mb: State,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    g: Param,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
    prl: Param,
    prt: Param,
) -> State.Vec5:
    # Per unit area: components 1-3 are -tau n, the viscous force on the wall.
    f = viscous_boundary(
        wl, gl, xl, wb, mb, kind, data, xf, n, area, g, rgas, pref, mref, tref, s, law, prl, prt
    )
    return f / area


traction = faces.field("traction", State.Vec5)
viscous_tractions = boundary_loop(
    viscous_traction, **BOUNDARY_ARGS, out=Write(traction), frozen=True
)
