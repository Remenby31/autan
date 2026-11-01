"""Block-sparse linear algebra of the implicit solvers: 5 x 5 blocks per cell and per face.

The matrix has one diagonal block per cell (`diag`) and, per interior face, the derivatives of
the face flux with respect to its owner's and its neighbour's states (`face_jac`, [0] and [1]):
the block coupling the owner to the neighbour is +face_jac[1], the neighbour to the owner
-face_jac[0] (physics BlockMatrix: upper and lower). The diagonal blocks are inverted in the
accumulation type (FP64 under fp32+acc64), with the operation order of the host BlockMatrix
(physics/src/block_matrix.cpp), so a device solve matches the host one to rounding.
"""

from autan_codegen.dsl import (
    Geom,
    Jac,
    JacAccum,
    Residual,
    Sum,
    Write,
    cell_loop,
    cell_reduce,
    cells,
    faces,
    geometry,
    inverse,
    pointwise,
)

diag = cells.field("diag", JacAccum.Mat5)
diag_inv = cells.field("diag_inv", JacAccum.Mat5)
face_jac = faces.field("face_jac", Jac.shape(2, 5, 5))
x = cells.field("x", Residual.Vec5)
y = cells.field("y", Residual.Vec5)


@pointwise(compute="accum")
def invert(a: JacAccum.Mat5) -> JacAccum.Mat5:
    return inverse(a)


invert_diagonal = cell_loop(invert, a=diag, out=Write(diag_inv))


@pointwise
def coupling(b: Jac.Mat5, s: Geom, xo: Residual.Vec5) -> Residual.Vec5:
    # Owner: +face_jac[1] x_neighbour; neighbour: -face_jac[0] x_owner; boundary face: 0.
    return s * (b @ xo)


@pointwise
def with_diagonal(acc: Residual.Vec5, a: JacAccum.Mat5, xc: Residual.Vec5) -> Residual.Vec5:
    return acc + a @ xc


# y = A x: the off-diagonal blocks of each cell in incidence order, then its diagonal block.
multiply = cell_reduce(
    coupling,
    b=face_jac.far,
    s=geometry.incidence_side,
    xo=x.other(None),
    acc=Sum(y),
    finalise=with_diagonal,
    finalise_args={"a": diag, "xc": x},
)
