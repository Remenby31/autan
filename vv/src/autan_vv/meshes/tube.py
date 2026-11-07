"""The shock tube of the `sod-order` gate (acceptance.md, M2): [0, 1] along x, n hexahedra, one
cell across a square cross-section of side WIDTH.

The order study of a 1D oracle needs one layer across the other axes (harness.md: h is the cell
measure along the oracle's axes), and WIDTH is the same at every level, so that only the cells along
x refine. The zones are the box faces xmin, xmax, ymin, ymax, zmin and zmax. The family is written
as Fluent ASCII without Gmsh, so it is generated on any platform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from autan_vv.meshes.facemesh import FloatArray, IntArray

KIND = "sod-tube"
LENGTH = 1.0
WIDTH = 0.01


def grid(n: int) -> tuple[FloatArray, dict[str, IntArray]]:
    """Points and hexahedra (Gmsh node order) of the level with `n` cells along x, n even so
    that the diaphragm x = 0.5 is a face."""
    if n < 2 or n % 2:
        msg = f"{KIND} needs an even n >= 2, so that the diaphragm x = 0.5 is a face, got n = {n}"
        raise VVError(msg, code="V0309")
    x = LENGTH * np.arange(n + 1) / n
    # Node (i, j, k): x index i, y and z indices j, k in {0, 1}.
    points = np.array([(xi, WIDTH * j, WIDTH * k) for k in (0, 1) for j in (0, 1) for xi in x])

    def node(i: IntArray, j: int, k: int) -> IntArray:
        return (k * 2 + j) * (n + 1) + i

    i = np.arange(n)
    hexes = np.stack(
        [
            node(i, 0, 0),
            node(i + 1, 0, 0),
            node(i + 1, 1, 0),
            node(i, 1, 0),
            node(i, 0, 1),
            node(i + 1, 0, 1),
            node(i + 1, 1, 1),
            node(i, 1, 1),
        ],
        axis=1,
    )
    return points, {"hex": hexes}
