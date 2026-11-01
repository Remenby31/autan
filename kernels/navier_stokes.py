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
    STENCIL,
    Dataset,
    Geom,
    Grad,
    Inc,
    Jac,
    Jacobian,
    MeshSet,
    Param,
    Residual,
    State,
    Sum,
    Vec3,
    Vec5,
    Write,
    boundary_loop,
    cell_loop,
    cell_reduce,
    cells,
    dot,
    face_loop,
    faces,
    geometry,
    outer,
    param,
    pointwise,
    rel,
    sqrt,
)

sys.path.insert(0, str(Path(__file__).parent))
from euler import (
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
def temperature_gradient(w: State.Vec5, gw: Grad.Mat53, rgas: Param, pref: Param) -> Grad.Vec3:
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
def corrected_boundary(g: Grad.Vec3, dphi: State, e: Geom.Vec3, inv_d: Geom) -> Grad.Vec3:
    # One-sided derivative along e at the face from the boundary value, the cell value and the
    # cell gradient, exact for a quadratic along e: 2 (phi_b - phi_c) / d - grad phi_c . e.
    return g + 2.0 * (dphi * inv_d - dot(g, e)) * e


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
    tn = (
        mu * (Vec3(dot(gu, n), dot(gv, n), dot(gwz, n)) + n[0] * gu + n[1] * gv + n[2] * gwz)
        - (2.0 / 3.0 * mu * div) * n
    )
    return Vec5(0.0, tn[0], tn[1], tn[2], dot(v, tn) + k * dot(gt, n))


@pointwise
def warp_work(gu: Grad.Vec3, gv: Grad.Vec3, gwz: Grad.Vec3, m: Geom.Mat3, mu: State) -> State:
    # Work of the stress through the fan triangles t of a warped face beyond its centroid value:
    # sum_t (u(c_t) - u(x_f)) . tau S_t = tau : (grad u M), M = sum_t (c_t - x_f) (x) S_t, zero
    # on a planar face. The stress and the heat flux are uniform over the face to this order.
    m0 = Vec3(m[0, 0], m[1, 0], m[2, 0])
    m1 = Vec3(m[0, 1], m[1, 1], m[2, 1])
    m2 = Vec3(m[0, 2], m[1, 2], m[2, 2])
    div = gu[0] + gv[1] + gwz[2]
    t00 = mu * (2.0 * gu[0] - 2.0 / 3.0 * div)
    t11 = mu * (2.0 * gv[1] - 2.0 / 3.0 * div)
    t22 = mu * (2.0 * gwz[2] - 2.0 / 3.0 * div)
    t01 = mu * (gu[1] + gv[0])
    t02 = mu * (gu[2] + gwz[0])
    t12 = mu * (gv[2] + gwz[1])
    return (
        t00 * dot(gu, m0)
        + t01 * dot(gu, m1)
        + t02 * dot(gu, m2)
        + t01 * dot(gv, m0)
        + t11 * dot(gv, m1)
        + t12 * dot(gv, m2)
        + t02 * dot(gwz, m0)
        + t12 * dot(gwz, m1)
        + t22 * dot(gwz, m2)
    )


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
    ql: Geom,
    l0: Grad.Mat53,
    l1: Grad.Mat53,
    l2: Grad.Mat53,
    l3: Grad.Mat53,
    l4: Grad.Mat53,
    l5: Grad.Mat53,
    l6: Grad.Mat53,
    wr: State.Vec5,
    gr: Grad.Mat53,
    xr: Geom.Vec3,
    mr: State,
    qr: Geom,
    r0: Grad.Mat53,
    r1: Grad.Mat53,
    r2: Grad.Mat53,
    r3: Grad.Mat53,
    r4: Grad.Mat53,
    r5: Grad.Mat53,
    r6: Grad.Mat53,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
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
    tgl, ugl, vgl, wgl = side_gradients(
        wl, gl, ql, l0, l1, l2, l3, l4, l5, l6, rel(xf, xl), rgas, pref
    )
    tgr, ugr, vgr, wgr = side_gradients(
        wr, gr, qr, r0, r1, r2, r3, r4, r5, r6, rel(xf, xr), rgas, pref
    )
    gu = corrected(0.5 * (ugl + ugr), wr[1] - wl[1], e, inv_d)
    gv = corrected(0.5 * (vgl + vgr), wr[2] - wl[2], e, inv_d)
    gwz = corrected(0.5 * (wgl + wgr), wr[3] - wl[3], e, inv_d)
    gt = corrected(0.5 * (tgl + tgr), tr - tl, e, inv_d)
    mu = viscosity(temperature(wf, rgas, pref), mref, tref, s, law)
    mt = 0.5 * (ml + mr)
    k = g * rgas / (g - 1.0) * (mu / prl + mt / prt)
    f = area * viscous_flux(velocity(wf), gu, gv, gwz, gt, mu + mt, k, n)
    return -(f + Vec5(0.0, 0.0, 0.0, 0.0, warp_work(gu, gv, gwz, m, mu + mt)))


# Boundary cells: weighted least-squares fit of a cubic polynomial to T and the velocity over the
# cell's two-ring stencil, with the boundary faces whose condition gives T and the velocity
# (manufactured, walls). Its gradient at a boundary face is exact for cubic fields: the face's
# viscous flux error is O(h^3), so that boundary faces add no lower-order error than interior faces
# (docs/design/ns-sa.md, Boundary conditions). Per stencil entry, the fit's 19 weights theta_k
# (gradient, Hessian, third derivatives; NavierStokes<P>, zero for the other cells, whose `quad` is
# 0) in seven 3-vectors: 0 the gradient, 1 (Hxx, Hyy, Hzz), 2 (Hxy, Hxz, Hyz), 3 (Cxxx, Cyyy, Czzz),
# 4 (Cxxy, Cxxz, Cxyy), 5 (Cyyz, Cxzz, Cyzz), 6 (Cxyz, 0, 0).
q0 = Dataset("q0", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q1 = Dataset("q1", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q2 = Dataset("q2", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q3 = Dataset("q3", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q4 = Dataset("q4", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q5 = Dataset("q5", MeshSet.STENCIL, Geom.Vec3, geometry=True)
q6 = Dataset("q6", MeshSet.STENCIL, Geom.Vec3, geometry=True)
quad = cells.field("quad", Geom)
ut = cells.field("ut", State.Vec5)  # (T, u, v, w, 0)
ut_b = faces.field("ut_b", State.Vec5)
fit0 = cells.field("fit0", Grad.Mat53)
fit1 = cells.field("fit1", Grad.Mat53)
fit2 = cells.field("fit2", Grad.Mat53)
fit3 = cells.field("fit3", Grad.Mat53)
fit4 = cells.field("fit4", Grad.Mat53)
fit5 = cells.field("fit5", Grad.Mat53)
fit6 = cells.field("fit6", Grad.Mat53)


@pointwise
def temperature_velocity(w: State.Vec5, rgas: Param, pref: Param) -> State.Vec5:
    return Vec5(temperature(w, rgas, pref), w[1], w[2], w[3], 0.0)


@pointwise
def boundary_temperature(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    wb: State.Vec5,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    rgas: Param,
    pref: Param,
) -> State:
    # The boundary state's temperature; at adiabatic walls, the cell's extrapolated to the face.
    adiabatic = kind > BC_WALL - 0.5 and data[0] <= 0.0
    tl = temperature(wl, rgas, pref)
    return (
        tl + dot(temperature_gradient(wl, gl, rgas, pref), rel(xf, xl))
        if adiabatic
        else temperature(wb, rgas, pref)
    )


@pointwise
def boundary_temperature_velocity(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    wb: State.Vec5,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    rgas: Param,
    pref: Param,
) -> State.Vec5:
    tb = boundary_temperature(wl, gl, xl, wb, kind, data, xf, rgas, pref)
    return Vec5(tb, wb[1], wb[2], wb[3], 0.0)


@pointwise
def cubic_term(
    uc: State.Vec5,
    un: State.Vec5,
    a0: Geom.Vec3,
    a1: Geom.Vec3,
    a2: Geom.Vec3,
    a3: Geom.Vec3,
    a4: Geom.Vec3,
    a5: Geom.Vec3,
    a6: Geom.Vec3,
) -> tuple[Grad.Mat53, Grad.Mat53, Grad.Mat53, Grad.Mat53, Grad.Mat53, Grad.Mat53, Grad.Mat53]:
    du = un - uc
    return (
        outer(du, a0),
        outer(du, a1),
        outer(du, a2),
        outer(du, a3),
        outer(du, a4),
        outer(du, a5),
        outer(du, a6),
    )


@pointwise
def cubic_parts(
    g: Grad.Vec3,
    hd: Grad.Vec3,
    ho: Grad.Vec3,
    c0: Grad.Vec3,
    c1: Grad.Vec3,
    c2: Grad.Vec3,
    c3: Grad.Vec3,
    r: Geom.Vec3,
) -> tuple[Grad.Vec3, Grad]:
    # Gradient at r of the fitted cubic and its increment phi(r) - phi(0). Monomials: x^3/6,
    # y^3/6, z^3/6, x^2 y/2, x^2 z/2, x y^2/2, y^2 z/2, x z^2/2, y z^2/2, x y z.
    x = r[0]
    y = r[1]
    z = r[2]
    hr = Vec3(
        hd[0] * x + ho[0] * y + ho[1] * z,
        ho[0] * x + hd[1] * y + ho[2] * z,
        ho[1] * x + ho[2] * y + hd[2] * z,
    )
    grad_c = Vec3(
        0.5 * x * x * c0[0]
        + x * y * c1[0]
        + x * z * c1[1]
        + 0.5 * y * y * c1[2]
        + 0.5 * z * z * c2[1]
        + y * z * c3[0],
        0.5 * y * y * c0[1]
        + 0.5 * x * x * c1[0]
        + x * y * c1[2]
        + y * z * c2[0]
        + 0.5 * z * z * c2[2]
        + x * z * c3[0],
        0.5 * z * z * c0[2]
        + 0.5 * x * x * c1[1]
        + 0.5 * y * y * c2[0]
        + x * z * c2[1]
        + y * z * c2[2]
        + x * y * c3[0],
    )
    value_c = (
        (x * x * x * c0[0] + y * y * y * c0[1] + z * z * z * c0[2]) / 6.0
        + 0.5 * (x * x * y * c1[0] + x * x * z * c1[1] + x * y * y * c1[2])
        + 0.5 * (y * y * z * c2[0] + x * z * z * c2[1] + y * z * z * c2[2])
        + x * y * z * c3[0]
    )
    return g + hr + grad_c, dot(g, r) + 0.5 * dot(hr, r) + value_c


@pointwise
def cubic_face_gradient(
    g: Grad.Vec3,
    hd: Grad.Vec3,
    ho: Grad.Vec3,
    c0: Grad.Vec3,
    c1: Grad.Vec3,
    c2: Grad.Vec3,
    c3: Grad.Vec3,
    dphi: State,
    r: Geom.Vec3,
    e: Geom.Vec3,
    inv_d: Geom,
) -> Grad.Vec3:
    # The fitted cubic's gradient at the face, its component along e replaced by the secant slope
    # of the values, which the cubic gives as (phi(r) - phi(0)) / |r|: the correction vanishes on
    # cubic fields.
    gf, rise = cubic_parts(g, hd, ho, c0, c1, c2, c3, r)
    return gf + (dphi - rise) * inv_d * e


@pointwise
def side_gradients(
    w: State.Vec5,
    gw: Grad.Mat53,
    q: Geom,
    f0: Grad.Mat53,
    f1: Grad.Mat53,
    f2: Grad.Mat53,
    f3: Grad.Mat53,
    f4: Grad.Mat53,
    f5: Grad.Mat53,
    f6: Grad.Mat53,
    r: Geom.Vec3,
    rgas: Param,
    pref: Param,
) -> tuple[Grad.Vec3, Grad.Vec3, Grad.Vec3, Grad.Vec3]:
    # Gradients of T, u, v, w that one cell gives a face: its cubic fit's at the face where it
    # carries one, its least-squares gradient otherwise.
    if q > 0.5:
        gt, a0 = cubic_parts(
            Vec3(f0[0, 0], f0[0, 1], f0[0, 2]),
            Vec3(f1[0, 0], f1[0, 1], f1[0, 2]),
            Vec3(f2[0, 0], f2[0, 1], f2[0, 2]),
            Vec3(f3[0, 0], f3[0, 1], f3[0, 2]),
            Vec3(f4[0, 0], f4[0, 1], f4[0, 2]),
            Vec3(f5[0, 0], f5[0, 1], f5[0, 2]),
            Vec3(f6[0, 0], f6[0, 1], f6[0, 2]),
            r,
        )
        gu, a1 = cubic_parts(
            Vec3(f0[1, 0], f0[1, 1], f0[1, 2]),
            Vec3(f1[1, 0], f1[1, 1], f1[1, 2]),
            Vec3(f2[1, 0], f2[1, 1], f2[1, 2]),
            Vec3(f3[1, 0], f3[1, 1], f3[1, 2]),
            Vec3(f4[1, 0], f4[1, 1], f4[1, 2]),
            Vec3(f5[1, 0], f5[1, 1], f5[1, 2]),
            Vec3(f6[1, 0], f6[1, 1], f6[1, 2]),
            r,
        )
        gv, a2 = cubic_parts(
            Vec3(f0[2, 0], f0[2, 1], f0[2, 2]),
            Vec3(f1[2, 0], f1[2, 1], f1[2, 2]),
            Vec3(f2[2, 0], f2[2, 1], f2[2, 2]),
            Vec3(f3[2, 0], f3[2, 1], f3[2, 2]),
            Vec3(f4[2, 0], f4[2, 1], f4[2, 2]),
            Vec3(f5[2, 0], f5[2, 1], f5[2, 2]),
            Vec3(f6[2, 0], f6[2, 1], f6[2, 2]),
            r,
        )
        gwz, a3 = cubic_parts(
            Vec3(f0[3, 0], f0[3, 1], f0[3, 2]),
            Vec3(f1[3, 0], f1[3, 1], f1[3, 2]),
            Vec3(f2[3, 0], f2[3, 1], f2[3, 2]),
            Vec3(f3[3, 0], f3[3, 1], f3[3, 2]),
            Vec3(f4[3, 0], f4[3, 1], f4[3, 2]),
            Vec3(f5[3, 0], f5[3, 1], f5[3, 2]),
            Vec3(f6[3, 0], f6[3, 1], f6[3, 2]),
            r,
        )
    else:
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        gt = temperature_gradient(w, gw, rgas, pref)
        gu = Vec3(gw[1, 0], gw[1, 1], gw[1, 2])
        gv = Vec3(gw[2, 0], gw[2, 1], gw[2, 2])
        gwz = Vec3(gw[3, 0], gw[3, 1], gw[3, 2])
    return gt, gu, gv, gwz


@pointwise
def viscous_boundary(
    wl: State.Vec5,
    gl: Grad.Mat53,
    xl: Geom.Vec3,
    f0: Grad.Mat53,
    f1: Grad.Mat53,
    f2: Grad.Mat53,
    f3: Grad.Mat53,
    f4: Grad.Mat53,
    f5: Grad.Mat53,
    f6: Grad.Mat53,
    q: Geom,
    wb: State.Vec5,
    mb: State,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
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
    tb = boundary_temperature(wl, gl, xl, wb, kind, data, xf, rgas, pref)
    if q > 0.5:
        gt = cubic_face_gradient(
            Vec3(f0[0, 0], f0[0, 1], f0[0, 2]),
            Vec3(f1[0, 0], f1[0, 1], f1[0, 2]),
            Vec3(f2[0, 0], f2[0, 1], f2[0, 2]),
            Vec3(f3[0, 0], f3[0, 1], f3[0, 2]),
            Vec3(f4[0, 0], f4[0, 1], f4[0, 2]),
            Vec3(f5[0, 0], f5[0, 1], f5[0, 2]),
            Vec3(f6[0, 0], f6[0, 1], f6[0, 2]),
            tb - tl,
            r,
            e,
            inv_d,
        )
        gu = cubic_face_gradient(
            Vec3(f0[1, 0], f0[1, 1], f0[1, 2]),
            Vec3(f1[1, 0], f1[1, 1], f1[1, 2]),
            Vec3(f2[1, 0], f2[1, 1], f2[1, 2]),
            Vec3(f3[1, 0], f3[1, 1], f3[1, 2]),
            Vec3(f4[1, 0], f4[1, 1], f4[1, 2]),
            Vec3(f5[1, 0], f5[1, 1], f5[1, 2]),
            Vec3(f6[1, 0], f6[1, 1], f6[1, 2]),
            wb[1] - wl[1],
            r,
            e,
            inv_d,
        )
        gv = cubic_face_gradient(
            Vec3(f0[2, 0], f0[2, 1], f0[2, 2]),
            Vec3(f1[2, 0], f1[2, 1], f1[2, 2]),
            Vec3(f2[2, 0], f2[2, 1], f2[2, 2]),
            Vec3(f3[2, 0], f3[2, 1], f3[2, 2]),
            Vec3(f4[2, 0], f4[2, 1], f4[2, 2]),
            Vec3(f5[2, 0], f5[2, 1], f5[2, 2]),
            Vec3(f6[2, 0], f6[2, 1], f6[2, 2]),
            wb[2] - wl[2],
            r,
            e,
            inv_d,
        )
        gwz = cubic_face_gradient(
            Vec3(f0[3, 0], f0[3, 1], f0[3, 2]),
            Vec3(f1[3, 0], f1[3, 1], f1[3, 2]),
            Vec3(f2[3, 0], f2[3, 1], f2[3, 2]),
            Vec3(f3[3, 0], f3[3, 1], f3[3, 2]),
            Vec3(f4[3, 0], f4[3, 1], f4[3, 2]),
            Vec3(f5[3, 0], f5[3, 1], f5[3, 2]),
            Vec3(f6[3, 0], f6[3, 1], f6[3, 2]),
            wb[3] - wl[3],
            r,
            e,
            inv_d,
        )
    else:
        gt = corrected_boundary(gtl, tb - tl, e, inv_d)
        gu = corrected_boundary(Vec3(gl[1, 0], gl[1, 1], gl[1, 2]), wb[1] - wl[1], e, inv_d)
        gv = corrected_boundary(Vec3(gl[2, 0], gl[2, 1], gl[2, 2]), wb[2] - wl[2], e, inv_d)
        gwz = corrected_boundary(Vec3(gl[3, 0], gl[3, 1], gl[3, 2]), wb[3] - wl[3], e, inv_d)
    mu = viscosity(tb, mref, tref, s, law)
    k = 0.0 if adiabatic else g * rgas / (g - 1.0) * (mu / prl + mb / prt)
    fv = area * viscous_flux(velocity(wb), gu, gv, gwz, gt, mu + mb, k, n)
    f = -(fv + Vec5(0.0, 0.0, 0.0, 0.0, warp_work(gu, gv, gwz, m, mu + mb)))
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
    "ql": quad.left,
    "l0": fit0.left,
    "l1": fit1.left,
    "l2": fit2.left,
    "l3": fit3.left,
    "l4": fit4.left,
    "l5": fit5.left,
    "l6": fit6.left,
    "wr": prim.right,
    "gr": grad.right,
    "xr": geometry.cell_centroid.right,
    "mr": mu_t.right,
    "qr": quad.right,
    "r0": fit0.right,
    "r1": fit1.right,
    "r2": fit2.right,
    "r3": fit3.right,
    "r4": fit4.right,
    "r5": fit5.right,
    "r6": fit6.right,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    "m": geometry.face_moment,
    **VISCOUS,
}

BOUNDARY_ARGS = {
    "wl": prim.left,
    "gl": grad.left,
    "xl": geometry.cell_centroid.left,
    "f0": fit0.left,
    "f1": fit1.left,
    "f2": fit2.left,
    "f3": fit3.left,
    "f4": fit4.left,
    "f5": fit5.left,
    "f6": fit6.left,
    "q": quad.left,
    "wb": prim_b,
    "mb": mu_t_b,
    "kind": bc_kind,
    "data": bc,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    "m": geometry.face_moment,
    **VISCOUS,
}

temperatures_velocities = cell_loop(
    temperature_velocity, w=prim, rgas=gas_constant, pref=p_ref, out=Write(ut)
)
boundary_temperatures_velocities = boundary_loop(
    boundary_temperature_velocity,
    wl=prim.left,
    gl=grad.left,
    xl=geometry.cell_centroid.left,
    wb=prim_b,
    kind=bc_kind,
    data=bc,
    xf=geometry.face_centroid,
    rgas=gas_constant,
    pref=p_ref,
    out=Write(ut_b),
    frozen=True,
)
cubic_fits = cell_reduce(
    cubic_term,
    over=STENCIL,
    uc=ut.self_,
    un=ut.other(boundary=ut_b),
    a0=q0,
    a1=q1,
    a2=q2,
    a3=q3,
    a4=q4,
    a5=q5,
    a6=q6,
    acc=[Sum(fit0), Sum(fit1), Sum(fit2), Sum(fit3), Sum(fit4), Sum(fit5), Sum(fit6)],
)

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
    f0: Grad.Mat53,
    f1: Grad.Mat53,
    f2: Grad.Mat53,
    f3: Grad.Mat53,
    f4: Grad.Mat53,
    f5: Grad.Mat53,
    f6: Grad.Mat53,
    q: Geom,
    wb: State.Vec5,
    mb: State,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
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
        wl,
        gl,
        xl,
        f0,
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        q,
        wb,
        mb,
        kind,
        data,
        xf,
        n,
        area,
        m,
        g,
        rgas,
        pref,
        mref,
        tref,
        s,
        law,
        prl,
        prt,
    )
    return f / area


traction = faces.field("traction", State.Vec5)
viscous_tractions = boundary_loop(
    viscous_traction, **BOUNDARY_ARGS, out=Write(traction), frozen=True
)
