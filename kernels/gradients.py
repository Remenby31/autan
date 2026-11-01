"""Cell gradients of a scalar: least squares with limiter bounds, and corrected Green-Gauss.

Least squares: weights 1/|d|^2 over face neighbours and boundary face centroids (Mavriplis 2003),
precomputed per incidence; the same reduction gathers the min and max of the neighbour values,
the bounds of a Venkatakrishnan or Barth-Jespersen limiter. Green-Gauss: face values
reconstructed from both cells with the previous gradient, the iterative skewness correction of
Syrakos et al. (2017), section 2.2; the face moment makes each face integral exact for linear
fields on warped faces. Exact for linear fields at the fixed point.
"""

from autan_codegen.dsl import (
    Geom,
    Grad,
    Inc,
    Max,
    Min,
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
    pointwise,
    rel,
    transpose,
)

q = cells.field("q", State)
q_b = faces.field("q_b", State)
grad = cells.field("grad", Grad.Vec3)
qmin = cells.field("qmin", State)
qmax = cells.field("qmax", State)
gg_sum = cells.field("gg_sum", Residual.Vec3)


@pointwise
def lsq_bounds_term(qc: State, qn: State, w: Geom.Vec3) -> tuple[Grad.Vec3, State, State]:
    return w * (qn - qc), qn, qn


@pointwise
def include_own_value(
    g: Residual.Vec3, lo: Residual, hi: Residual, qc: State
) -> tuple[Grad.Vec3, State, State]:
    return g, min(lo, qc), max(hi, qc)


lsq_gradient_bounds = cell_reduce(
    lsq_bounds_term,
    qc=q.self_,
    qn=q.other(boundary=q_b),
    w=geometry.lsq_weight,
    acc=[Sum(grad), Min(qmin), Max(qmax)],
    finalise=include_own_value,
    finalise_args={"qc": q},
)


@pointwise
def gg_interior(
    ql: State,
    gl: Grad.Vec3,
    xl: Geom.Vec3,
    qr: State,
    gr: Grad.Vec3,
    xr: Geom.Vec3,
    xf: Geom.Vec3,
    s: Geom.Vec3,
    m: Geom.Mat3,
) -> Residual.Vec3:
    qf = 0.5 * (ql + dot(gl, rel(xf, xl)) + qr + dot(gr, rel(xf, xr)))
    return qf * s + transpose(m) @ (0.5 * (gl + gr))


@pointwise
def gg_boundary(qb: State, gl: Grad.Vec3, s: Geom.Vec3, m: Geom.Mat3) -> Residual.Vec3:
    return qb * s + transpose(m) @ gl


@pointwise
def gg_divide(total: Residual.Vec3, vol: Geom) -> Grad.Vec3:
    return total / vol


green_gauss_interior = face_loop(
    gg_interior,
    ql=q.left,
    gl=grad.left,
    xl=geometry.cell_centroid.left,
    qr=q.right,
    gr=grad.right,
    xr=geometry.cell_centroid.right,
    xf=geometry.face_centroid,
    s=geometry.face_area,
    m=geometry.face_moment,
    out=Inc(gg_sum, left=+1, right=-1),
)

green_gauss_boundary = boundary_loop(
    gg_boundary,
    qb=q_b,
    gl=grad.left,
    s=geometry.face_area,
    m=geometry.face_moment,
    out=Inc(gg_sum, left=+1),
)

green_gauss_finalise = cell_loop(gg_divide, total=gg_sum, vol=geometry.cell_volume, out=Write(grad))
