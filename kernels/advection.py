"""Second-order upwind advection face loop of research section 3.3, the bandwidth benchmark.

F = |S| (a . n) q_up with q_up reconstructed to the face from the upwind cell: q + grad q . r.
"""

from autan_codegen.dsl import (
    Geom,
    Grad,
    Inc,
    Param,
    Residual,
    State,
    cells,
    dot,
    face_loop,
    geometry,
    param,
    pointwise,
)

q = cells.field("q", State)
grad = cells.field("grad", Grad.Vec3)
res = cells.field("res", Residual)
velocity = param("velocity", Param.Vec3)


@pointwise
def upwind(
    qc: State,
    gc: Grad.Vec3,
    rc: Geom.Vec3,
    qn: State,
    gn: Grad.Vec3,
    rn: Geom.Vec3,
    n: Geom.Vec3,
    area: Geom,
    a: Param.Vec3,
) -> Residual:
    ql = qc + dot(gc, rc)
    qr = qn + dot(gn, rn)
    un = dot(a, n)
    return area * (un * ql if un > 0.0 else un * qr)


residual = face_loop(
    upwind,
    qc=q.left,
    gc=grad.left,
    rc=geometry.face_r_left,
    qn=q.right,
    gn=grad.right,
    rn=geometry.face_r_right,
    n=geometry.face_normal,
    area=geometry.face_area_magnitude,
    a=velocity,
    out=Inc(res, left=+1, right=-1),
)
