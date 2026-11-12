"""Meshes of the Ringleb domain (HiOCFD1 C1.2) and their face states (docs/contracts/face-states.md).

The domain is mapped from the hodograph: streamline parameter k in [0.7, 1.5] and flow angle
theta in [asin(0.5/k), pi - asin(0.5/k)], with speed V = k sin(theta), so that the lines of
the grid are streamlines and isoclines and the cells vary smoothly. One layer of hexahedra,
THICKNESS thick at every level, carries the 2D flow.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from autan_vv.meshes import face_states
from autan_vv.oracles.inviscid import RINGLEB_K, RINGLEB_SPEED_MIN, _ringleb_state, ringleb_fields

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray

# Every in-plane side takes the exact state; the two planes of the layer are symmetry planes.
STATE_ZONES = ("k-min", "k-max", "inflow", "outflow")
# The same at every level, so that a family covers one domain; about the in-plane cell of n = 16.
THICKNESS = 0.1
SYMMETRY_ZONES = ("back", "front")


def plane_points(n: int) -> tuple[FloatArray, int, int]:
    """Grid points (x, y) of level n: n cells across the streamlines, 2n along them."""
    nk, nt = n, 2 * n
    k = np.linspace(RINGLEB_K[0], RINGLEB_K[1], nk + 1)[:, None]
    t0 = np.arcsin(RINGLEB_SPEED_MIN / k)
    theta = t0 + (math.pi - 2.0 * t0) * np.linspace(0.0, 1.0, nt + 1)[None, :]
    speed = k * np.sin(theta)
    _, rho, _, j = _ringleb_state(speed)
    # HiOCFD1 C1.2, eqs. 1-2, with cos(theta) = +-sqrt(1 - V^2/k^2) carrying the sign of y.
    x = (2.0 / k**2 - 1.0 / speed**2) / (2.0 * rho) - 0.5 * j
    y = np.cos(theta) / (k * rho * speed)
    return np.stack([x, y], axis=-1), nk, nt


def grid(n: int) -> tuple[FloatArray, dict[str, IntArray], dict[str, IntArray]]:
    """Points, hexahedra and boundary faces (zones of STATE_ZONES and SYMMETRY_ZONES) of level n."""
    xy, nk, nt = plane_points(n)
    dz = THICKNESS
    pts = np.concatenate(
        [
            np.concatenate([xy.reshape(-1, 2), np.full(((nk + 1) * (nt + 1), 1), z)], axis=1)
            for z in (0.0, dz)
        ]
    )
    layer = (nk + 1) * (nt + 1)

    def node(i: IntArray, j: IntArray, z: int) -> IntArray:
        return z * layer + i * (nt + 1) + j

    i, j = (a.ravel() for a in np.meshgrid(np.arange(nk), np.arange(nt), indexing="ij"))
    quad = [(0, 0), (1, 0), (1, 1), (0, 1)]
    bottom = np.stack([node(i + a, j + b, 0) for a, b in quad], axis=1)
    top = bottom + layer
    # Orientation: the (k, theta) grid may map with either handedness.
    p = pts[bottom[0]]
    if np.cross(p[1] - p[0], p[3] - p[0])[2] < 0.0:
        bottom, top = bottom[:, ::-1], top[:, ::-1]
    hexes = np.concatenate([bottom, top], axis=1)

    def side(ii: IntArray, jj: IntArray, di: int, dj: int) -> IntArray:
        a, b = node(ii, jj, 0), node(ii + di, jj + dj, 0)
        return np.stack([a, b, b + layer, a + layer], axis=1)

    ar_t, ar_k = np.arange(nt), np.arange(nk)
    boundary = {
        "k-min": side(np.zeros(nt, int), ar_t, 0, 1),
        "k-max": side(np.full(nt, nk), ar_t, 0, 1),
        "inflow": side(ar_k, np.zeros(nk, int), 1, 0),
        "outflow": side(ar_k, np.full(nk, nt), 1, 0),
        "back": bottom,
        "front": top,
    }
    return pts, {"hex": hexes}, boundary


def write_face_states(mesh: FaceMesh, mesh_file: Path, out: Path) -> str:
    """Write the exact Ringleb flow on the STATE_ZONES of `mesh`, read from `mesh_file`, to `out`
    as autan.face-states/1; return its SHA-256."""
    return face_states.write(
        mesh, mesh_file, out, "ringleb", STATE_ZONES, lambda x: ringleb_fields(x, extended=True)
    )
