"""Test kernel: a cell reduction that reads data of the incidence's face.

w_c = sum over the faces f of c of q_other * |S_f| * |x_f - x_c|^2, q_other the value across f
(q_b on boundary faces). The tests compare it, its tangent and its adjoint with sums computed
from the mesh arrays.
"""

from autan_codegen.dsl import (
    Geom,
    State,
    Sum,
    cell_reduce,
    cells,
    dot,
    faces,
    geometry,
    pointwise,
    rel,
)

q = cells.field("q", State)
q_b = faces.field("q_b", State)
w = cells.field("w", State)


@pointwise
def weighted(qn: State, area: Geom, xf: Geom.Vec3, xc: Geom.Vec3) -> State:
    r = rel(xf, xc)
    return qn * area * dot(r, r)


face_weighted = cell_reduce(
    weighted,
    qn=q.other(boundary=q_b),
    area=geometry.face_area_magnitude,
    xf=geometry.face_centroid,
    xc=geometry.cell_centroid.self_,
    acc=Sum(w),
)
