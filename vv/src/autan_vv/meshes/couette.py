"""The column of the Couette gates (acceptance.md, M3; issue #596): n hexahedra along y over the
channel height H, one cell of side WIDTH across x and z.

The oracle is 1D along y, so its order study needs one layer across x and z, and one domain at
every level (harness.md): WIDTH (m) is the same at every level, H/128, so that the cells of the
finest level n = 128 are cubes. The zones are the box faces: `ymin` is the wall at rest, `ymax`
(the moving wall) and the `xmin`/`xmax` ends take the oracle's states, `zmin`/`zmax` are symmetry
planes. Written as Fluent ASCII without Gmsh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from autan_vv.errors import VVError
from autan_vv.meshes import face_states
from autan_vv.oracles.viscous import couette

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray

KIND = "couette"
WIDTH = 1.0 / 128.0
STATE_ZONES = ("ymax", "xmin", "xmax")


def grid(n: int) -> tuple[FloatArray, dict[str, IntArray]]:
    """Points and hexahedra (Gmsh node order) of the level with `n` cells along y."""
    if n < 1:
        msg = f"{KIND} needs n >= 1, got n = {n}"
        raise VVError(msg, code="V0309")
    height = couette().parameters["height"]
    y = height * np.arange(n + 1) / n
    # Node (i, j, k): x and z indices i, k in {0, 1}, y index j.
    points = np.array([(WIDTH * i, yj, WIDTH * k) for k in (0, 1) for i in (0, 1) for yj in y])

    def node(i: int, j: IntArray, k: int) -> IntArray:
        return (k * 2 + i) * (n + 1) + j

    j = np.arange(n)
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    hexes = np.stack([node(i, j + dj, k) for i, dj, k in corners], axis=1)
    return points, {"hex": hexes}


def write_face_states(mesh: FaceMesh, mesh_file: Path, out: Path) -> str:
    """Write the exact Couette flow on the STATE_ZONES of `mesh`, read from `mesh_file`, to `out`
    as autan.face-states/1; return its SHA-256."""
    oracle = couette()
    return face_states.write(mesh, mesh_file, out, oracle.name, STATE_ZONES, oracle.fields)
