"""Compressible Euler residual, second order, on arbitrary polyhedra.

State: conserved U = (rho, rho u, rho v, rho w, rho E) per cell; reconstruction in primitive
variables W = (rho, u, v, w, p), ideal gas with ratio of specific heats `gamma`.

Residual R_c = sum over the faces of c of F(W_L, W_R) . S - s_c V_c, with
- gradients of W by weighted least squares over face neighbours and boundary states,
- the Venkatakrishnan (1995) limiter with bounds over the gradient stencil, J. Comput. Phys. 118:120-130, eq. 5, with the per-component
  threshold eps^2 = scale_k V_c (scale_k = K^3 q_k^2 / L^3, set by the caller),
- face states W_f = W_c + phi (grad W) (x_f - x_c),
- one of three numerical fluxes: Roe with Harten's entropy fix, HLLC, AUSM+-up,
- boundary states per face from its condition code (`BC_*` below) and the face data `bc`.

Face normals point from the owner (left) to the neighbour (right) and out of the domain on boundary
faces.
"""

from autan_codegen.dsl import (
    STENCIL,
    Geom,
    Grad,
    Inc,
    Max,
    Min,
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
    exp,
    face_loop,
    faces,
    geometry,
    log,
    outer,
    param,
    pointwise,
    rel,
    sqrt,
    transpose,
)

# Boundary condition codes of the face dataset `bc_kind`.
BC_STATE = 0.0  # boundary state given in `bc` as W (manufactured solutions, exact data)
BC_FARFIELD = 1.0  # `bc` = freestream W; Riemann invariants normal to the face
BC_SLIP = 2.0  # slip wall and symmetry plane: no flux but pressure
BC_INLET = 3.0  # subsonic inlet: `bc` = (total pressure, total temperature, 0, 0, 0), flow normal
BC_OUTLET = 4.0  # subsonic outlet: `bc` = (0, 0, 0, 0, static pressure)

HARTEN = 0.1  # entropy-fix width as a fraction of the Roe-averaged sound speed
AUSM_KP = 0.25  # AUSM+-up pressure diffusion, Liou (2006), section 3
AUSM_KU = 0.75  # AUSM+-up velocity diffusion
AUSM_SIGMA = 1.0
AUSM_BETA = 0.125

u = cells.field("u", State.Vec5)
prim = cells.field("prim", State.Vec5)
grad = cells.field("grad", Grad.Mat53)
wmin = cells.field("wmin", State.Vec5)
wmax = cells.field("wmax", State.Vec5)
phi = cells.field("phi", State.Vec5)
source = cells.field("source", State.Vec5)
res = cells.field("res", Residual.Vec5)

prim_b = faces.field("prim_b", State.Vec5)
bc_kind = faces.field("bc_kind", Geom)
bc = faces.field("bc", State.Vec5)

gamma = param("gamma", Param)
gas_constant = param("gas_constant", Param)
mach_ref = param("mach_ref", Param)
limiter_scale = param("limiter_scale", Param.Vec5)


# ---------------------------------------------------------------------------- thermodynamics


@pointwise
def to_primitive(q: State.Vec5, g: Param) -> State.Vec5:
    rho = q[0]
    vel = Vec3(q[1] / rho, q[2] / rho, q[3] / rho)
    return Vec5(rho, vel[0], vel[1], vel[2], (g - 1.0) * (q[4] - 0.5 * rho * dot(vel, vel)))


@pointwise
def velocity(w: State.Vec5) -> State.Vec3:
    return Vec3(w[1], w[2], w[3])


@pointwise
def total_enthalpy(w: State.Vec5, g: Param) -> State:
    v = velocity(w)
    return g / (g - 1.0) * w[4] / w[0] + 0.5 * dot(v, v)


@pointwise
def physical_flux(w: State.Vec5, n: Geom.Vec3, g: Param) -> State.Vec5:
    v = velocity(w)
    vn = dot(v, n)
    m = w[0] * vn
    return Vec5(
        m,
        m * v[0] + w[4] * n[0],
        m * v[1] + w[4] * n[1],
        m * v[2] + w[4] * n[2],
        m * total_enthalpy(w, g),
    )


# ---------------------------------------------------------------------------- numerical fluxes


@pointwise
def harten(lam: State, delta: State) -> State:
    a = abs(lam)
    if a < delta:
        a = (lam * lam + delta * delta) / (2.0 * delta)
    return a


@pointwise
def roe(wl: State.Vec5, wr: State.Vec5, n: Geom.Vec3, g: Param) -> State.Vec5:
    # Roe (1981), J. Comput. Phys. 43:357-372, in the wave-strength form of Blazek (2015),
    # eqs. 4.89-4.91; Harten (1983), J. Comput. Phys. 49:357-393, eq. 4.5, on the acoustic
    # eigenvalues only, so that stationary contacts and shear layers stay exact.
    rl, rr = wl[0], wr[0]
    vl, vr = velocity(wl), velocity(wr)
    hl, hr = total_enthalpy(wl, g), total_enthalpy(wr, g)
    s = sqrt(rr / rl)
    rho = s * rl
    v = (vl + s * vr) / (1.0 + s)
    h = (hl + s * hr) / (1.0 + s)
    vn = dot(v, n)
    q2 = dot(v, v)
    c = sqrt((g - 1.0) * (h - 0.5 * q2))
    dp = wr[4] - wl[4]
    drho = rr - rl
    dv = vr - vl
    dvn = dot(dv, n)
    delta = HARTEN * c
    l1 = harten(vn - c, delta)
    l2 = abs(vn)
    l5 = harten(vn + c, delta)
    a1 = l1 * (dp - rho * c * dvn) / (2.0 * c * c)
    a5 = l5 * (dp + rho * c * dvn) / (2.0 * c * c)
    a2 = l2 * (drho - dp / (c * c))
    shear = dv - dvn * n
    d0 = a1 + a2 + a5
    dm = a1 * (v - c * n) + a2 * v + l2 * rho * shear + a5 * (v + c * n)
    de = a1 * (h - c * vn) + a2 * 0.5 * q2 + l2 * rho * (dot(v, dv) - vn * dvn) + a5 * (h + c * vn)
    central = 0.5 * (physical_flux(wl, n, g) + physical_flux(wr, n, g))
    return central - 0.5 * Vec5(d0, dm[0], dm[1], dm[2], de)


@pointwise
def hllc_star(w: State.Vec5, n: Geom.Vec3, sk: State, sm: State, g: Param) -> State.Vec5:
    # Toro (2009), Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed., eq. 10.39,
    # in the face-normal frame: the star state on the side of wave speed sk.
    v = velocity(w)
    vn = dot(v, n)
    rs = w[0] * (sk - vn) / (sk - sm)
    e = w[4] / ((g - 1.0) * w[0]) + 0.5 * dot(v, v)
    ms = rs * (v + (sm - vn) * n)
    es = rs * (e + (sm - vn) * (sm + w[4] / (w[0] * (sk - vn))))
    return Vec5(rs, ms[0], ms[1], ms[2], es)


@pointwise
def conserved(w: State.Vec5, g: Param) -> State.Vec5:
    v = velocity(w)
    return Vec5(
        w[0],
        w[0] * v[0],
        w[0] * v[1],
        w[0] * v[2],
        w[4] / (g - 1.0) + 0.5 * w[0] * dot(v, v),
    )


@pointwise
def hllc(wl: State.Vec5, wr: State.Vec5, n: Geom.Vec3, g: Param) -> State.Vec5:
    # Toro, Spruce and Speares (1994), Shock Waves 4:25-34; wave speeds of Einfeldt (1988) as in
    # Batten et al. (1997), SIAM J. Sci. Comput. 18:1553-1570, eqs. 15 (S_M) and 13 (S_L, S_R).
    rl, rr = wl[0], wr[0]
    vl, vr = velocity(wl), velocity(wr)
    vnl, vnr = dot(vl, n), dot(vr, n)
    cl = sqrt(g * wl[4] / rl)
    cr = sqrt(g * wr[4] / rr)
    s = sqrt(rr / rl)
    v = (vl + s * vr) / (1.0 + s)
    h = (total_enthalpy(wl, g) + s * total_enthalpy(wr, g)) / (1.0 + s)
    c = sqrt((g - 1.0) * (h - 0.5 * dot(v, v)))
    vn = dot(v, n)
    sl = min(vnl - cl, vn - c)
    sr = max(vnr + cr, vn + c)
    sm = (wr[4] - wl[4] + rl * vnl * (sl - vnl) - rr * vnr * (sr - vnr)) / (
        rl * (sl - vnl) - rr * (sr - vnr)
    )
    fl = physical_flux(wl, n, g)
    fr = physical_flux(wr, n, g)
    if sl >= 0.0:
        f = fl
    elif sm >= 0.0:
        f = fl + sl * (hllc_star(wl, n, sl, sm, g) - conserved(wl, g))
    elif sr > 0.0:
        f = fr + sr * (hllc_star(wr, n, sr, sm, g) - conserved(wr, g))
    else:
        f = fr
    return f


@pointwise
def m4_plus(m: State) -> State:
    # Liou (2006), J. Comput. Phys. 214:137-170: split Mach polynomials M_(4)^+ and P_(5)^+ with
    # beta = 1/8 and alpha = 3/16 (-4 + 5 fa^2).
    mp = 0.25 * (m + 1.0) * (m + 1.0)
    mm = -0.25 * (m - 1.0) * (m - 1.0)
    return 0.5 * (m + abs(m)) if abs(m) >= 1.0 else mp * (1.0 - 16.0 * AUSM_BETA * mm)


@pointwise
def m4_minus(m: State) -> State:
    mp = 0.25 * (m + 1.0) * (m + 1.0)
    mm = -0.25 * (m - 1.0) * (m - 1.0)
    return 0.5 * (m - abs(m)) if abs(m) >= 1.0 else mm * (1.0 + 16.0 * AUSM_BETA * mp)


@pointwise
def p5_plus(m: State, alpha: State) -> State:
    mp = 0.25 * (m + 1.0) * (m + 1.0)
    mm = -0.25 * (m - 1.0) * (m - 1.0)
    supersonic = 1.0 if m > 0.0 else 0.0
    return supersonic if abs(m) >= 1.0 else mp * ((2.0 - m) - 16.0 * alpha * m * mm)


@pointwise
def p5_minus(m: State, alpha: State) -> State:
    mp = 0.25 * (m + 1.0) * (m + 1.0)
    mm = -0.25 * (m - 1.0) * (m - 1.0)
    supersonic = 1.0 if m < 0.0 else 0.0
    return supersonic if abs(m) >= 1.0 else mm * ((-2.0 - m) + 16.0 * alpha * m * mp)


@pointwise
def ausm_up(wl: State.Vec5, wr: State.Vec5, n: Geom.Vec3, g: Param, minf: Param) -> State.Vec5:
    # Liou (2006): AUSM+-up with the interface sound speed from the critical speed of sound
    # (Liou 1996, J. Comput. Phys. 129:364-382) and the low-Mach scaling fa(M_o), M_co = minf.
    rl, rr = wl[0], wr[0]
    vl, vr = velocity(wl), velocity(wr)
    vnl, vnr = dot(vl, n), dot(vr, n)
    hl, hr = total_enthalpy(wl, g), total_enthalpy(wr, g)
    csl = sqrt(2.0 * (g - 1.0) / (g + 1.0) * hl)
    csr = sqrt(2.0 * (g - 1.0) / (g + 1.0) * hr)
    al = csl * csl / max(csl, vnl)
    ar = csr * csr / max(csr, -vnr)
    a = min(al, ar)
    ml = vnl / a
    mr = vnr / a
    mbar2 = 0.5 * (vnl * vnl + vnr * vnr) / (a * a)
    mo2 = min(1.0, max(mbar2, minf * minf))
    mo = sqrt(mo2)
    fa = mo * (2.0 - mo)
    alpha = 0.1875 * (-4.0 + 5.0 * fa * fa)
    rho = 0.5 * (rl + rr)
    mp = -AUSM_KP / fa * max(1.0 - AUSM_SIGMA * mbar2, 0.0) * (wr[4] - wl[4]) / (rho * a * a)
    m = m4_plus(ml) + m4_minus(mr) + mp
    pu = -AUSM_KU * p5_plus(ml, alpha) * p5_minus(mr, alpha) * (rl + rr) * fa * a * (vnr - vnl)
    p = p5_plus(ml, alpha) * wl[4] + p5_minus(mr, alpha) * wr[4] + pu
    mdot = a * m * rl if m > 0.0 else a * m * rr
    vu = vl if m > 0.0 else vr
    hu = hl if m > 0.0 else hr
    return Vec5(
        mdot, mdot * vu[0] + p * n[0], mdot * vu[1] + p * n[1], mdot * vu[2] + p * n[2], mdot * hu
    )


# ---------------------------------------------------------------------------- boundary states


@pointwise
def farfield_state(wi: State.Vec5, wf: State.Vec5, n: Geom.Vec3, g: Param) -> State.Vec5:
    # Riemann invariants normal to the face (Blazek 2015, section 8.4.1; Whitfield and Janus
    # 1984, AIAA 84-1552): R+ from the interior, R- from the freestream, entropy and tangential
    # velocity from the upwind side; supersonic faces take the upwind state.
    vi, vf = velocity(wi), velocity(wf)
    vni, vnf = dot(vi, n), dot(vf, n)
    ci = sqrt(g * wi[4] / wi[0])
    cf = sqrt(g * wf[4] / wf[0])
    rp = vni + 2.0 * ci / (g - 1.0)
    rm = vnf - 2.0 * cf / (g - 1.0)
    vnb = 0.5 * (rp + rm)
    cb = 0.25 * (g - 1.0) * (rp - rm)
    if vnb < 0.0:
        ent = wf[4] / exp(g * log(wf[0]))
        vt = vf - vnf * n
    else:
        ent = wi[4] / exp(g * log(wi[0]))
        vt = vi - vni * n
    rb = exp(log(cb * cb / (g * ent)) / (g - 1.0))
    vb = vt + vnb * n
    sub = Vec5(rb, vb[0], vb[1], vb[2], rb * cb * cb / g)
    if vni >= ci:
        w = wi
    elif vnf <= -cf:
        w = wf
    else:
        w = sub
    return w


@pointwise
def inlet_state(
    wi: State.Vec5, data: State.Vec5, n: Geom.Vec3, g: Param, rgas: Param
) -> State.Vec5:
    # Subsonic inflow normal to the face from total pressure and temperature: the outgoing
    # invariant R+ = vn + 2c/(g-1) and the total enthalpy cp T0 = c^2/(g-1) + vn^2/2 give c_b
    # (larger root); Carlson (2011), NASA/TM-2011-217181, section 2.
    p0, t0 = data[0], data[1]
    gm = g - 1.0
    vi = velocity(wi)
    ci = sqrt(g * wi[4] / wi[0])
    r = dot(vi, n) + 2.0 * ci / gm
    h0 = g * rgas * t0 / gm
    disc = max((gm + 2.0) * h0 - 0.5 * gm * r * r, 0.0)
    cb = gm * (r + sqrt(disc)) / (gm + 2.0)
    vnb = min(r - 2.0 * cb / gm, 0.0)
    tb = t0 - 0.5 * gm * vnb * vnb / (g * rgas)
    pb = p0 * exp(g / gm * log(tb / t0))
    rb = pb / (rgas * tb)
    return Vec5(rb, vnb * n[0], vnb * n[1], vnb * n[2], pb)


@pointwise
def outlet_state(wi: State.Vec5, data: State.Vec5, n: Geom.Vec3, g: Param) -> State.Vec5:
    # Subsonic outflow at a given static pressure, the other variables from the outgoing
    # characteristics (Blazek 2015, section 8.4.2); supersonic outflow extrapolates everything.
    vi = velocity(wi)
    ci = sqrt(g * wi[4] / wi[0])
    pb = data[4]
    dp = wi[4] - pb
    rb = wi[0] - dp / (ci * ci)
    vb = vi + dp / (wi[0] * ci) * n
    sub = Vec5(rb, vb[0], vb[1], vb[2], pb)
    return wi if dot(vi, n) >= ci else sub


@pointwise
def mirror_state(wi: State.Vec5, n: Geom.Vec3) -> State.Vec5:
    vi = velocity(wi)
    vb = vi - 2.0 * dot(vi, n) * n
    return Vec5(wi[0], vb[0], vb[1], vb[2], wi[4])


@pointwise
def boundary_value(
    wi: State.Vec5, kind: Geom, data: State.Vec5, n: Geom.Vec3, g: Param, rgas: Param
) -> State.Vec5:
    # Codes are whole numbers stored in FP64: compare with the midpoints.
    if kind < BC_FARFIELD - 0.5:
        w = data
    elif kind < BC_SLIP - 0.5:
        w = farfield_state(wi, data, n, g)
    elif kind < BC_INLET - 0.5:
        w = mirror_state(wi, n)
    elif kind < BC_OUTLET - 0.5:
        w = inlet_state(wi, data, n, g, rgas)
    else:
        w = outlet_state(wi, data, n, g)
    return w


# ---------------------------------------------------------------------------- loops


@pointwise
def primitive(q: State.Vec5, g: Param) -> State.Vec5:
    return to_primitive(q, g)


primitives = cell_loop(primitive, q=u, g=gamma, out=Write(prim))


@pointwise
def boundary_primitive(
    wi: State.Vec5, kind: Geom, data: State.Vec5, n: Geom.Vec3, g: Param, rgas: Param
) -> State.Vec5:
    return boundary_value(wi, kind, data, n, g, rgas)


boundary_states = boundary_loop(
    boundary_primitive,
    wi=prim.left,
    kind=bc_kind,
    data=bc,
    n=geometry.face_normal,
    g=gamma,
    rgas=gas_constant,
    out=Write(prim_b),
    frozen=True,
)


@pointwise
def lsq_term(wc: State.Vec5, wn: State.Vec5, w: Geom.Vec3) -> Grad.Mat53:
    return outer(wn - wc, w)


# Least squares over the cell stencil (vertex or face2 neighbours, Topology::stencil): over face
# neighbours alone the second-order scheme is linearly unstable on tetrahedra (docs/design/
# euler.md). The limiter's bounds are taken over the same stencil.
gradients = cell_reduce(
    lsq_term,
    over=STENCIL,
    wc=prim.self_,
    wn=prim.other(boundary=prim_b),
    w=geometry.stencil_lsq_weight,
    acc=Sum(grad),
)


@pointwise
def neighbour_bounds(wn: State.Vec5) -> tuple[State.Vec5, State.Vec5]:
    return wn, wn


@pointwise
def include_own_value(
    lo: Residual.Vec5, hi: Residual.Vec5, wc: State.Vec5
) -> tuple[State.Vec5, State.Vec5]:
    return min(lo, wc), max(hi, wc)


bounds = cell_reduce(
    neighbour_bounds,
    over=STENCIL,
    wn=prim.other(boundary=prim_b),
    acc=[Min(wmin), Max(wmax)],
    finalise=include_own_value,
    finalise_args={"wc": prim},
)


@pointwise
def venkat(d: State, dmax: State, dmin: State, eps2: State) -> State:
    # Venkatakrishnan (1995), eq. 5: smooth limiter of the extrapolated increment d.
    dp = dmax if d > 0.0 else dmin
    return (dp * dp + eps2 + 2.0 * d * dp) / (dp * dp + 2.0 * d * d + d * dp + eps2)


@pointwise
def limiter_term(
    wc: State.Vec5,
    gc: Grad.Mat53,
    lo: State.Vec5,
    hi: State.Vec5,
    vol: Geom,
    xf: Geom.Vec3,
    xc: Geom.Vec3,
    scale: Param.Vec5,
) -> State.Vec5:
    d = gc @ rel(xf, xc)
    return Vec5(
        venkat(d[0], hi[0] - wc[0], lo[0] - wc[0], scale[0] * vol),
        venkat(d[1], hi[1] - wc[1], lo[1] - wc[1], scale[1] * vol),
        venkat(d[2], hi[2] - wc[2], lo[2] - wc[2], scale[2] * vol),
        venkat(d[3], hi[3] - wc[3], lo[3] - wc[3], scale[3] * vol),
        venkat(d[4], hi[4] - wc[4], lo[4] - wc[4], scale[4] * vol),
    )


@pointwise
def at_most_one(p: Residual.Vec5) -> State.Vec5:
    return Vec5(min(p[0], 1.0), min(p[1], 1.0), min(p[2], 1.0), min(p[3], 1.0), min(p[4], 1.0))


limiter = cell_reduce(
    limiter_term,
    wc=prim.self_,
    gc=grad.self_,
    lo=wmin.self_,
    hi=wmax.self_,
    vol=geometry.cell_volume.self_,
    xf=geometry.face_centroid,
    xc=geometry.cell_centroid.self_,
    scale=limiter_scale,
    acc=Min(phi),
    finalise=at_most_one,
)


@pointwise
def volume_source(s: State.Vec5, vol: Geom) -> Residual.Vec5:
    return -vol * s


sources = cell_loop(volume_source, s=source, vol=geometry.cell_volume, out=Write(res))


@pointwise
def reconstruct(w: State.Vec5, gw: Grad.Mat53, lim: State.Vec5, r: Geom.Vec3) -> State.Vec5:
    d = gw @ r
    return w + Vec5(lim[0] * d[0], lim[1] * d[1], lim[2] * d[2], lim[3] * d[3], lim[4] * d[4])


@pointwise
def times(a: State.Vec5, b: State.Vec5) -> State.Vec5:
    return Vec5(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3], a[4] * b[4])


@pointwise
def flux_increment(w: State.Vec5, dw: State.Vec5, e: Geom.Vec3, g: Param) -> State.Vec5:
    # Linearisation of the physical flux through direction e, in primitive variables.
    v = velocity(w)
    dv = velocity(dw)
    ve = dot(v, e)
    dve = dot(dv, e)
    rh = g / (g - 1.0) * w[4] + 0.5 * w[0] * dot(v, v)
    drh = g / (g - 1.0) * dw[4] + 0.5 * dw[0] * dot(v, v) + w[0] * dot(v, dv)
    dm = dw[0] * ve + w[0] * dve
    mom = dm * v + (w[0] * ve) * dv + dw[4] * e
    return Vec5(dm, mom[0], mom[1], mom[2], drh * ve + rh * dve)


@pointwise
def warp(
    w: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    gr: Grad.Mat53,
    pr: State.Vec5,
    m: Geom.Mat3,
    g: Param,
) -> State.Vec5:
    # Flux of the linear reconstruction through the fan triangles t of a warped face beyond its
    # centroid value: sum_j dF_j[W; grad W . M[:, j]], M = sum_t (c_t - x_f) (x) S_t, zero on a
    # planar face; gradients averaged over both sides.
    m0 = Vec3(m[0, 0], m[1, 0], m[2, 0])
    m1 = Vec3(m[0, 1], m[1, 1], m[2, 1])
    m2 = Vec3(m[0, 2], m[1, 2], m[2, 2])
    d0 = 0.5 * (times(pl, gl @ m0) + times(pr, gr @ m0))
    d1 = 0.5 * (times(pl, gl @ m1) + times(pr, gr @ m1))
    d2 = 0.5 * (times(pl, gl @ m2) + times(pr, gr @ m2))
    return (
        flux_increment(w, d0, Vec3(1.0, 0.0, 0.0), g)
        + flux_increment(w, d1, Vec3(0.0, 1.0, 0.0), g)
        + flux_increment(w, d2, Vec3(0.0, 0.0, 1.0), g)
    )


@pointwise
def wall_warp(gl: Grad.Mat53, pl: State.Vec5, m: Geom.Mat3) -> State.Vec5:
    dp = pl[4] * (transpose(m) @ Vec3(gl[4, 0], gl[4, 1], gl[4, 2]))
    return Vec5(0.0, dp[0], dp[1], dp[2], 0.0)


# One face loop and one boundary loop per numerical flux; the solver calls the pair it uses.


@pointwise
def roe_face(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    wr: State.Vec5,
    gr: Grad.Mat53,
    pr: State.Vec5,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
) -> Residual.Vec5:
    fl = reconstruct(wl, gl, pl, rel(xf, xl))
    fr = reconstruct(wr, gr, pr, rel(xf, xr))
    return area * roe(fl, fr, n, g) + warp(0.5 * (fl + fr), gl, pl, gr, pr, m, g)


@pointwise
def hllc_face(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    wr: State.Vec5,
    gr: Grad.Mat53,
    pr: State.Vec5,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
) -> Residual.Vec5:
    fl = reconstruct(wl, gl, pl, rel(xf, xl))
    fr = reconstruct(wr, gr, pr, rel(xf, xr))
    return area * hllc(fl, fr, n, g) + warp(0.5 * (fl + fr), gl, pl, gr, pr, m, g)


@pointwise
def ausm_face(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    wr: State.Vec5,
    gr: Grad.Mat53,
    pr: State.Vec5,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
    minf: Param,
) -> Residual.Vec5:
    fl = reconstruct(wl, gl, pl, rel(xf, xl))
    fr = reconstruct(wr, gr, pr, rel(xf, xr))
    return area * ausm_up(fl, fr, n, g, minf) + warp(0.5 * (fl + fr), gl, pl, gr, pr, m, g)


FACE_ARGS = {
    "wl": prim.left,
    "gl": grad.left,
    "pl": phi.left,
    "xl": geometry.cell_centroid.left,
    "wr": prim.right,
    "gr": grad.right,
    "pr": phi.right,
    "xr": geometry.cell_centroid.right,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    "m": geometry.face_moment,
    "g": gamma,
}

roe_faces = face_loop(roe_face, **FACE_ARGS, out=Inc(res, left=+1, right=-1))
hllc_faces = face_loop(hllc_face, **FACE_ARGS, out=Inc(res, left=+1, right=-1))
ausm_faces = face_loop(ausm_face, **FACE_ARGS, minf=mach_ref, out=Inc(res, left=+1, right=-1))


@pointwise
def wall_flux(wf: State.Vec5, n: Geom.Vec3) -> State.Vec5:
    return Vec5(0.0, wf[4] * n[0], wf[4] * n[1], wf[4] * n[2], 0.0)


@pointwise
def roe_boundary(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
    rgas: Param,
) -> Residual.Vec5:
    wf = reconstruct(wl, gl, pl, rel(xf, xl))
    wb = boundary_value(wf, kind, data, n, g, rgas)
    if kind > BC_SLIP - 0.5 and kind < BC_SLIP + 0.5:
        f = area * wall_flux(wf, n) + wall_warp(gl, pl, m)
    else:
        f = area * roe(wf, wb, n, g) + warp(wf, gl, pl, gl, pl, m, g)
    return f


@pointwise
def hllc_boundary(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
    rgas: Param,
) -> Residual.Vec5:
    wf = reconstruct(wl, gl, pl, rel(xf, xl))
    wb = boundary_value(wf, kind, data, n, g, rgas)
    if kind > BC_SLIP - 0.5 and kind < BC_SLIP + 0.5:
        f = area * wall_flux(wf, n) + wall_warp(gl, pl, m)
    else:
        f = area * hllc(wf, wb, n, g) + warp(wf, gl, pl, gl, pl, m, g)
    return f


@pointwise
def ausm_boundary(
    wl: State.Vec5,
    gl: Grad.Mat53,
    pl: State.Vec5,
    xl: Geom.Vec3,
    kind: Geom,
    data: State.Vec5,
    xf: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    m: Geom.Mat3,
    g: Param,
    rgas: Param,
    minf: Param,
) -> Residual.Vec5:
    wf = reconstruct(wl, gl, pl, rel(xf, xl))
    wb = boundary_value(wf, kind, data, n, g, rgas)
    if kind > BC_SLIP - 0.5 and kind < BC_SLIP + 0.5:
        f = area * wall_flux(wf, n) + wall_warp(gl, pl, m)
    else:
        f = area * ausm_up(wf, wb, n, g, minf) + warp(wf, gl, pl, gl, pl, m, g)
    return f


BOUNDARY_ARGS = {
    "wl": prim.left,
    "gl": grad.left,
    "pl": phi.left,
    "xl": geometry.cell_centroid.left,
    "kind": bc_kind,
    "data": bc,
    "xf": geometry.face_centroid,
    "n": geometry.face_normal,
    "area": geometry.face_area_magnitude,
    "m": geometry.face_moment,
    "g": gamma,
    "rgas": gas_constant,
}

roe_boundaries = boundary_loop(roe_boundary, **BOUNDARY_ARGS, out=Inc(res, left=+1))
hllc_boundaries = boundary_loop(hllc_boundary, **BOUNDARY_ARGS, out=Inc(res, left=+1))
ausm_boundaries = boundary_loop(
    ausm_boundary, **BOUNDARY_ARGS, minf=mach_ref, out=Inc(res, left=+1)
)


lam = cells.field("lam", State)


@pointwise
def wave_speed(wc: State.Vec5, wn: State.Vec5, n: Geom.Vec3, area: Geom, g: Param) -> State:
    # Convective spectral radius of a face, (|v . n| + c) A at the arithmetic mean state
    # (Blazek 2015, chapter 6): its sum over the faces of a cell bounds the local time step.
    w = 0.5 * (wc + wn)
    return (abs(dot(velocity(w), n)) + sqrt(g * w[4] / w[0])) * area


spectral_radius = cell_reduce(
    wave_speed,
    wc=prim.self_,
    wn=prim.other(boundary=prim_b),
    n=geometry.face_normal,
    area=geometry.face_area_magnitude,
    g=gamma,
    acc=Sum(lam),
)
