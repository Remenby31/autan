"""Test kernel: face data written per face, flags among them, then read by a face loop.

On interior faces, in the face's global orientation (left l, right r, unit normal n):
keep = (q_l - q_r > 0, n_x > 0), jump = q_r - q_l. On boundary faces: keep = (q_l > 0, n_x > 0).
pick then sends the value keep[0] selects, q_l or q_r, into res: +v on l, -v on r.
"""

from autan_codegen.dsl import (
    Flag,
    Geom,
    Inc,
    Residual,
    State,
    Vec2,
    Write,
    boundary_loop,
    cells,
    face_loop,
    faces,
    geometry,
    pointwise,
)

q = cells.field("q", State)
keep = faces.field("keep", Flag.Vec2)
jump = faces.field("jump", State)
res = cells.field("res", Residual)


@pointwise
def flags(ql: State, qr: State, n: Geom.Vec3) -> tuple[Flag.Vec2, State]:
    return Vec2(ql - qr, n[0]), qr - ql


@pointwise
def boundary_flags(ql: State, n: Geom.Vec3) -> Flag.Vec2:
    return Vec2(ql, n[0])


@pointwise
def pick(k: Flag.Vec2, ql: State, qr: State) -> Residual:
    return ql if k[0] > 0.5 else qr


interior_flags = face_loop(
    flags, ql=q.left, qr=q.right, n=geometry.face_normal, out=[Write(keep), Write(jump)],
    frozen=True,
)  # fmt: skip
boundary_flags_loop = boundary_loop(
    boundary_flags, ql=q.left, n=geometry.face_normal, out=Write(keep), frozen=True
)
picked = face_loop(pick, k=keep, ql=q.left, qr=q.right, out=Inc(res, left=1, right=-1))
