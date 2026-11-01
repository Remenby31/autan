"""Spalart-Allmaras turbulence model, negative variant (SA-neg), conservative form.

Allmaras, Johnson and Spalart (2012), ICCFD7-1902, as given on the TMR SA page
(https://tmbwg.github.io/turbmodels/spalart.html, "SA-neg" and the conservative form of its note):

    d(rho nt)/dt + div(rho u nt) = rho P - rho D
        + 1/sigma [div(rho (nu + nt f_n) grad nt) + rho cb2 |grad nt|^2 - (nu + nt f_n) grad rho . grad nt]

with nt the working variable nu-tilde, nu = mu / rho, P and D as below, f_n = 1 when nt >= 0, and the
eddy viscosity mu_t = rho nt f_v1 (zero when nt < 0). The cell state is rho nt (`rnu`).

Discretisation, as the mean flow (`navier_stokes.py`): least-squares gradients of nt over the cell
stencil; convection by the face mass flux of the averaged linear reconstructions, upwinded with the
linearly reconstructed nt of the upwind cell (second order); diffusion through the corrected face
gradient; sources at cell centroids. Boundary values of nt per face (`nt_b`): zero at walls, the
freestream value at farfield and inlets, the interior value at outlets and slip faces, given in
`nt_bc` for manufactured solutions.
"""

import sys
from pathlib import Path

from autan_codegen.dsl import (
    STENCIL,
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
    Write,
    boundary_loop,
    cell_loop,
    cell_reduce,
    cells,
    dot,
    exp,
    face_loop,
    faces,
    geometry,
    log,
    param,
    pointwise,
    rel,
    sqrt,
)

sys.path.insert(0, str(Path(__file__).parent))
from euler import (
    BC_FARFIELD,
    BC_INLET,
    BC_STATE,
    BC_WALL,
    bc_kind,
    gas_constant,
    grad,
    p_ref,
    prim,
    prim_b,
    velocity,
)
from navier_stokes import (
    corrected,
    corrected_boundary,
    face_value,
    mu_ref,
    mu_t,
    mu_t_b,
    sutherland,
    sutherland_s,
    t_ref,
    temperature,
    viscosity,
)

CB1 = 0.1355
SIGMA = 2.0 / 3.0
CB2 = 0.622
KAPPA = 0.41
CW2 = 0.3
CW3 = 2.0
CV1 = 7.1
CT3 = 1.2
CT4 = 0.5
C2 = 0.7
C3 = 0.9
CN1 = 16.0
CW1 = CB1 / (KAPPA * KAPPA) + (1.0 + CB2) / SIGMA
R_MAX = 10.0

rnu = cells.field("rnu", State)
nt_grad = cells.field("nt_grad", Grad.Vec3)
nt_b = faces.field("nt_b", State)
nt_bc = faces.field("nt_bc", State)
wall_distance = cells.field("wall_distance", Geom)
nt_source = cells.field("nt_source", State)
nt_res = cells.field("nt_res", Residual)

ft2_on = param("ft2_on", Param)  # 1: SA-neg; 0: SA-noft2 (manufactured solutions only)


@pointwise
def lam_nu(
    w: State.Vec5, rgas: Param, pref: Param, mref: Param, tref: Param, s: Param, law: Param
) -> State:
    return viscosity(temperature(w, rgas, pref), mref, tref, s, law) / w[0]


@pointwise
def fv1(chi: State) -> State:
    c3 = chi * chi * chi
    return c3 / (c3 + CV1 * CV1 * CV1)


@pointwise
def fn(chi: State) -> State:
    # Allmaras et al. (2012), eq. 21: the diffusion coefficient's factor for nt < 0.
    c3 = chi * chi * chi
    return (CN1 + c3) / (CN1 - c3) if chi < 0.0 else 1.0


@pointwise
def vorticity(gw: Grad.Mat53) -> State:
    wx = gw[3, 1] - gw[2, 2]
    wy = gw[1, 2] - gw[3, 0]
    wz = gw[2, 0] - gw[1, 1]
    return sqrt(wx * wx + wy * wy + wz * wz)


@pointwise
def sa_source(nt: State, nu: State, omega: State, d: Geom, ft2s: Param) -> State:
    # rho-free source P - D of SA-neg (TMR SA page): standard SA with the S_tilde of Allmaras
    # et al. (2012), Note 1(c), for nt >= 0; P = cb1 (1 - ct3) Omega nt, D = -cw1 (nt / d)^2 for
    # nt < 0.
    chi = nt / nu
    kd2 = KAPPA * KAPPA * d * d
    f1 = fv1(chi)
    fv2 = 1.0 - chi / (1.0 + chi * f1)
    sbar = nt * fv2 / kd2
    if sbar >= -C2 * omega:
        st = omega + sbar
    else:
        st = omega + omega * (C2 * C2 * omega + C3 * sbar) / ((C3 - 2.0 * C2) * omega - sbar)
    r = min(nt / (st * kd2), R_MAX) if st > 0.0 else R_MAX
    g = r + CW2 * (r**6 - r)
    c6 = CW3**6
    fw = g * exp(log((1.0 + c6) / (g**6 + c6)) / 6.0)
    ft2 = ft2s * CT3 * exp(-CT4 * chi * chi)
    pos = CB1 * (1.0 - ft2) * st * nt - (CW1 * fw - CB1 / (KAPPA * KAPPA) * ft2) * (nt / d) * (
        nt / d
    )
    neg = CB1 * (1.0 - CT3) * omega * nt + CW1 * (nt / d) * (nt / d)
    return pos if nt >= 0.0 else neg


@pointwise
def nt_value(rn: State, w: State.Vec5) -> State:
    return rn / w[0]


# ---------------------------------------------------------------------------- loops

VISCOSITY = {
    "rgas": gas_constant,
    "pref": p_ref,
    "mref": mu_ref,
    "tref": t_ref,
    "s": sutherland_s,
    "law": sutherland,
}


@pointwise
def eddy_viscosity(
    rn: State,
    w: State.Vec5,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
) -> State:
    nt = rn / w[0]
    chi = nt / lam_nu(w, rgas, pref, mref, tref, s, law)
    return rn * fv1(chi) if nt > 0.0 else 0.0 * rn


eddy_viscosities = cell_loop(eddy_viscosity, rn=rnu, w=prim, **VISCOSITY, out=Write(mu_t))


@pointwise
def boundary_nt(rn: State, w: State.Vec5, kind: Geom, data: State) -> State:
    # Zero at walls; the given value at manufactured, farfield and inlet faces; the interior value
    # elsewhere (outlets, slip walls, symmetry planes).
    nt = rn / w[0]
    given = kind < BC_STATE + 0.5 or (kind > BC_FARFIELD - 0.5 and kind < BC_FARFIELD + 0.5)
    inflow = kind > BC_INLET - 0.5 and kind < BC_INLET + 0.5
    if kind > BC_WALL - 0.5:
        v = 0.0 * nt
    elif given or inflow:
        v = data
    else:
        v = nt
    return v


boundary_values = boundary_loop(
    boundary_nt, rn=rnu.left, w=prim.left, kind=bc_kind, data=nt_bc, out=Write(nt_b), frozen=True
)


@pointwise
def boundary_eddy_viscosity(
    nt: State,
    w: State.Vec5,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
) -> State:
    chi = nt / lam_nu(w, rgas, pref, mref, tref, s, law)
    return w[0] * nt * fv1(chi) if nt > 0.0 else 0.0 * nt


boundary_eddy_viscosities = boundary_loop(
    boundary_eddy_viscosity, nt=nt_b, w=prim_b, **VISCOSITY, out=Write(mu_t_b), frozen=True
)


nt = cells.field("nt", State)
nt_values = cell_loop(nt_value, rn=rnu, w=prim, out=Write(nt))


@pointwise
def nt_lsq(tc: State, tn: State, w: Geom.Vec3) -> Grad.Vec3:
    return (tn - tc) * w


# Over the mean flow's stencil and weights (Topology::stencil, geometry.stencil_lsq_weight).
nt_gradients = cell_reduce(
    nt_lsq,
    over=STENCIL,
    tc=nt.self_,
    tn=nt.other(boundary=nt_b),
    w=geometry.stencil_lsq_weight,
    acc=Sum(nt_grad),
)


@pointwise
def nt_face(
    rl: State,
    wl: State.Vec5,
    gl: Grad.Mat53,
    tl: Grad.Vec3,
    xl: Geom.Vec3,
    rr: State,
    wr: State.Vec5,
    gr: Grad.Mat53,
    tr: Grad.Vec3,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
) -> Residual:
    ntl = rl / wl[0]
    ntr = rr / wr[0]
    d = rel(xr, xl)
    dist = sqrt(dot(d, d))
    inv_d = 1.0 / dist
    e = inv_d * d
    sl = rel(xf, xl)
    sr = rel(xf, xr)
    wf = 0.5 * (face_value(wl, gl, sl, pref) + face_value(wr, gr, sr, pref))
    mdot = wf[0] * dot(velocity(wf), n) * area
    ntf_l = ntl + dot(tl, sl)
    ntf_r = ntr + dot(tr, sr)
    conv = mdot * ntf_l if mdot > 0.0 else mdot * ntf_r
    ntf = 0.5 * (ntf_l + ntf_r)
    nu = lam_nu(wf, rgas, pref, mref, tref, s, law)
    gn = corrected(0.5 * (tl + tr), ntr - ntl, e, inv_d)
    diff = wf[0] * (nu + ntf * fn(ntf / nu)) / SIGMA * dot(gn, n) * area
    return conv - diff


@pointwise
def nt_boundary(
    rl: State,
    wl: State.Vec5,
    gl: Grad.Mat53,
    tl: Grad.Vec3,
    xl: Geom.Vec3,
    ntb: State,
    wb: State.Vec5,
    kind: Geom,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
) -> Residual:
    ntl = rl / wl[0]
    r = rel(xf, xl)
    dist = sqrt(dot(r, r))
    inv_d = 1.0 / dist
    e = inv_d * r
    wf = 0.5 * (face_value(wl, gl, r, pref) + wb)
    mdot = wf[0] * dot(velocity(wf), n) * area
    ntf_l = ntl + dot(tl, r)
    conv = mdot * ntf_l if mdot > 0.0 else mdot * ntb
    nu = lam_nu(wb, rgas, pref, mref, tref, s, law)
    gn = corrected_boundary(tl, ntb - ntl, e, inv_d)
    diff = wb[0] * (nu + ntb * fn(ntb / nu)) / SIGMA * dot(gn, n) * area
    wall = kind > BC_WALL - 0.5
    return 0.0 * conv - diff if wall else conv - diff


@pointwise
def nt_cell_source(
    rc: State,
    w: State.Vec5,
    gw: Grad.Mat53,
    tg: Grad.Vec3,
    d: Geom,
    vol: Geom,
    ext: State,
    rgas: Param,
    pref: Param,
    mref: Param,
    tref: Param,
    s: Param,
    law: Param,
    ft2s: Param,
) -> Residual:
    # -V (rho (P - D) + [rho cb2 |grad nt|^2 - (nu + nt f_n) grad rho . grad nt] / sigma + ext).
    nt = rc / w[0]
    nu = lam_nu(w, rgas, pref, mref, tref, s, law)
    grho = Vec3(gw[0, 0], gw[0, 1], gw[0, 2])
    q = (
        w[0] * sa_source(nt, nu, vorticity(gw), d, ft2s)
        + (w[0] * CB2 * dot(tg, tg) - (nu + nt * fn(nt / nu)) * dot(grho, tg)) / SIGMA
    )
    return -vol * (q + ext)


TRANSPORT = {**VISCOSITY}

nt_sources = cell_loop(
    nt_cell_source,
    rc=rnu,
    w=prim,
    gw=grad,
    tg=nt_grad,
    d=wall_distance,
    vol=geometry.cell_volume,
    ext=nt_source,
    **TRANSPORT,
    ft2s=ft2_on,
    out=Write(nt_res),
)

FACE = {
    "rl": rnu.left,
    "wl": prim.left,
    "gl": grad.left,
    "tl": nt_grad.left,
    "xl": geometry.cell_centroid.left,
    "rr": rnu.right,
    "wr": prim.right,
    "gr": grad.right,
    "tr": nt_grad.right,
    "xr": geometry.cell_centroid.right,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    **TRANSPORT,
}
BOUNDARY = {
    "rl": rnu.left,
    "wl": prim.left,
    "gl": grad.left,
    "tl": nt_grad.left,
    "xl": geometry.cell_centroid.left,
    "ntb": nt_b,
    "wb": prim_b,
    "kind": bc_kind,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    **TRANSPORT,
}

nt_faces = face_loop(nt_face, **FACE, out=Inc(nt_res, left=+1, right=-1))
nt_boundaries = boundary_loop(nt_boundary, **BOUNDARY, out=Inc(nt_res, left=+1))

# Scalar Jacobians d/d(rho nt) for the implicit step, the gradients held fixed.
nt_face_jac = faces.field("nt_face_jac", Jac.shape(2, 1, 1))
nt_bnd_jac = faces.field("nt_bnd_jac", Jac.shape(1, 1))
nt_cell_jac = cells.field("nt_cell_jac", Jac.shape(1, 1))
nt_sources_jacobian = cell_loop(
    nt_cell_source,
    rc=rnu,
    w=prim,
    gw=grad,
    tg=nt_grad,
    d=wall_distance,
    vol=geometry.cell_volume,
    ext=nt_source,
    **TRANSPORT,
    ft2s=ft2_on,
    out=[Write(nt_res), Jacobian(nt_cell_jac, wrt="rc")],
)
nt_faces_jacobian = face_loop(
    nt_face,
    **FACE,
    out=[Inc(nt_res, left=+1, right=-1), Jacobian(nt_face_jac, wrt=("rl", "rr"))],
)
nt_boundaries_jacobian = boundary_loop(
    nt_boundary, **BOUNDARY, out=[Inc(nt_res, left=+1), Jacobian(nt_bnd_jac, wrt="rl")]
)
