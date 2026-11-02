"""Test kernel: Euler's gradients and limiter pattern on a scalar, separate and chained.

`gradients` sums the stencil least-squares gradient of q and the bounds of q over the stencil,
stored without a finalise: the FP64 sum is converted to Grad (FP32 under fp32+acc64) where it is
stored, so a chain that skipped that conversion would change the limiter's bits; `limiter` reads its cell's gradient
and bounds through `.self_` and takes, over the cell's faces, the minimum of a smooth limiter of
the extrapolated increment. `fused` chains them: the bounds stay in registers, the gradient is
stored. The tests compare grad and phi of `fused` with those of the two loops, bit for bit.
"""

from autan_codegen.dsl import (
    STENCIL,
    Geom,
    Grad,
    Max,
    Min,
    Residual,
    State,
    Sum,
    cell_reduce,
    cells,
    chain,
    faces,
    geometry,
    pointwise,
    rel,
)

q = cells.field("q", State)
q_b = faces.field("q_b", State)
grad = cells.field("grad", Grad.Vec3)
lo = cells.field("lo", State)
hi = cells.field("hi", State)
phi = cells.field("phi", State)


@pointwise
def lsq(qc: State, qn: State, w: Geom.Vec3) -> tuple[Grad.Vec3, State, State]:
    return (qn - qc) * w, qn, qn


gradients = cell_reduce(
    lsq,
    over=STENCIL,
    qc=q.self_,
    qn=q.other(boundary=q_b),
    w=geometry.stencil_lsq_weight,
    acc=[Sum(grad), Min(lo), Max(hi)],
)


@pointwise
def smooth(qc: State, g: Grad.Vec3, lo_: State, hi_: State, xf: Geom.Vec3, xc: Geom.Vec3) -> State:
    d = g[0] * rel(xf, xc)[0] + g[1] * rel(xf, xc)[1] + g[2] * rel(xf, xc)[2]
    dp = hi_ - qc if d > 0.0 else lo_ - qc
    return (dp * dp + 2.0 * d * dp + 1e-3) / (dp * dp + 2.0 * d * d + d * dp + 1e-3)


@pointwise
def at_most_one(p: Residual) -> State:
    return min(p, 1.0)


limiter = cell_reduce(
    smooth,
    qc=q.self_,
    g=grad.self_,
    lo_=lo.self_,
    hi_=hi.self_,
    xf=geometry.face_centroid,
    xc=geometry.cell_centroid.self_,
    acc=Min(phi),
    finalise=at_most_one,
)

fused = chain(gradients, limiter, keep=[grad])
