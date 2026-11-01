"""Block-sparse linear algebra of the implicit solvers: 5 x 5 blocks per cell and per face.

The diagonal blocks are inverted in the accumulation type (FP64 under fp32+acc64), with the
operation order of the host BlockMatrix (physics/src/block_matrix.cpp), so a device solve
matches the host one to rounding.
"""

from autan_codegen.dsl import JacAccum, Write, cell_loop, cells, inverse, pointwise

diag = cells.field("diag", JacAccum.Mat5)
diag_inv = cells.field("diag_inv", JacAccum.Mat5)


@pointwise(compute="accum")
def invert(a: JacAccum.Mat5) -> JacAccum.Mat5:
    return inverse(a)


invert_diagonal = cell_loop(invert, a=diag, out=Write(diag_inv))
