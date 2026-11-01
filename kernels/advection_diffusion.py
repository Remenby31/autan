"""Steady scalar advection-diffusion, second order: div(a q) - div(nu grad q) = s.

Residual R_c = sum over faces of (convective + diffusive) flux - s_c V_c. Gradients by weighted
least squares (weights 1/|d|^2 over face neighbours and boundary face centroids). Convection: upwind
value reconstructed with its cell gradient, integrated exactly for linear fields over warped faces
through the face moment. Diffusion: face gradient with alpha damping, Nishikawa (2010), eq. 12,
alpha = 4/3 inside and 2 on Dirichlet boundary faces. Boundary values are Dirichlet.
"""

from autan_codegen.dsl import (
    Geom,
    Grad,
    Inc,
    Param,
    Residual,
    State,
    Sum,
    Write,
    boundary_loop,
    cell_loop,
    cell_reduce,
    cells,
    dot,
    face_loop,
    faces,
    geometry,
    param,
    pointwise,
    rel,
)

q = cells.field("q", State)
q_b = faces.field("q_b", State)
grad = cells.field("grad", Grad.Vec3)
source = cells.field("source", State)
res = cells.field("res", Residual)
velocity = param("velocity", Param.Vec3)
nu = param("nu", Param)


@pointwise
def lsq_term(qc: State, qn: State, w: Geom.Vec3) -> Grad.Vec3:
    return w * (qn - qc)


lsq_gradient = cell_reduce(
    lsq_term,
    qc=q.self_,
    qn=q.other(boundary=q_b),
    w=geometry.lsq_weight,
    acc=Sum(grad),
)


@pointwise
def volume_source(s: State, vol: Geom) -> Residual:
    return -s * vol


init_residual = cell_loop(volume_source, s=source, vol=geometry.cell_volume, out=Write(res))


@pointwise
def convective(qup: State, gup: Grad.Vec3, un: Param, m: Geom.Mat3, a: Param.Vec3) -> Residual:
    # Flux of (qup + gup . (x - x_f)) a through the fan triangles of the face.
    return un * qup + dot(gup, m @ a)


@pointwise
def interior_flux(
    ql: State,
    gl: Grad.Vec3,
    xl: Geom.Vec3,
    qr: State,
    gr: Grad.Vec3,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    s: Geom.Vec3,
    m: Geom.Mat3,
    a: Param.Vec3,
    nu: Param,
) -> Residual:
    un = dot(a, s)
    qfl = ql + dot(gl, rel(xf, xl))
    qfr = qr + dot(gr, rel(xf, xr))
    conv = convective(qfl, gl, un, m, a) if un >= 0.0 else convective(qfr, gr, un, m, a)
    d = rel(xr, xl)
    gbar = 0.5 * (gl + gr)
    alpha = 4.0 / 3.0
    gf = gbar + (alpha * (qr - ql - dot(gbar, d)) / dot(d, d)) * d
    return conv - nu * dot(gf, s)


flux = face_loop(
    interior_flux,
    ql=q.left,
    gl=grad.left,
    xl=geometry.cell_centroid.left,
    qr=q.right,
    gr=grad.right,
    xr=geometry.cell_centroid.right,
    xf=geometry.face_centroid,
    s=geometry.face_area,
    m=geometry.face_moment,
    a=velocity,
    nu=nu,
    out=Inc(res, left=+1, right=-1),
)


@pointwise
def dirichlet_flux(
    ql: State,
    gl: Grad.Vec3,
    xl: Geom.Vec3,
    qb: State,
    xf: Geom.Vec3,
    s: Geom.Vec3,
    m: Geom.Mat3,
    a: Param.Vec3,
    nu: Param,
) -> Residual:
    un = dot(a, s)
    qfl = ql + dot(gl, rel(xf, xl))
    conv = convective(qfl, gl, un, m, a) if un >= 0.0 else convective(qb, gl, un, m, a)
    d = rel(xf, xl)
    alpha = 2.0
    gf = gl + (alpha * (qb - ql - dot(gl, d)) / dot(d, d)) * d
    return conv - nu * dot(gf, s)


boundary_flux = boundary_loop(
    dirichlet_flux,
    ql=q.left,
    gl=grad.left,
    xl=geometry.cell_centroid.left,
    qb=q_b,
    xf=geometry.face_centroid,
    s=geometry.face_area,
    m=geometry.face_moment,
    a=velocity,
    nu=nu,
    out=Inc(res, left=+1),
)
