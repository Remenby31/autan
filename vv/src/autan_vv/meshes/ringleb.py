"""Meshes of the Ringleb domain (HiOCFD1 C1.2) and their face states (docs/contracts/face-states.md).

The domain is mapped from the hodograph: streamline parameter k in [0.7, 1.5] and flow angle
theta in [asin(0.5/k), pi - asin(0.5/k)], with speed V = k sin(theta), so that the lines of
the grid are streamlines and isoclines and the cells vary smoothly. One layer of hexahedra,
as thick as the mean in-plane cell, carries the 2D flow.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.oracles.inviscid import RINGLEB_K, RINGLEB_SPEED_MIN, _ringleb_state, ringleb_fields

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray

FACE_STATES_SCHEMA = "autan.face-states/1"
VARIABLES = ("density", "velocity-x", "velocity-y", "velocity-z", "pressure")
_ORACLE_FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")
# Every in-plane side takes the exact state; the two planes of the layer are symmetry planes.
STATE_ZONES = ("k-min", "k-max", "inflow", "outflow")
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
    lengths = (
        np.linalg.norm(np.diff(xy, axis=0), axis=-1).mean()
        * np.linalg.norm(np.diff(xy, axis=1), axis=-1).mean()
    )
    dz = math.sqrt(float(lengths))
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


def face_states(mesh: FaceMesh, mesh_sha256: str) -> dict[str, Any]:
    """The autan.face-states/1 document of the exact Ringleb flow on the STATE_ZONES of `mesh`."""
    g = mesh.geometry
    zones: dict[str, Any] = {}
    for zone in mesh.zones:
        if zone.name not in STATE_ZONES:
            continue
        faces = zone.start + np.arange(zone.count)
        centroids = g.face_centroid[faces]
        fields = ringleb_fields(centroids, extended=True)
        values = np.stack([fields[name] for name in _ORACLE_FIELDS], axis=1)
        zones[zone.name] = {
            "faces": int(zone.count),
            "centroids": centroids.tolist(),
            "values": values.tolist(),
        }
    return {
        "schema": FACE_STATES_SCHEMA,
        "oracle": "ringleb",
        "mesh_sha256": mesh_sha256,
        "variables": list(VARIABLES),
        "zones": zones,
    }


def write_face_states(mesh: FaceMesh, mesh_file: Path, out: Path) -> str:
    """Write the face states of `mesh`, read from `mesh_file`, to `out`; return their SHA-256."""
    doc = face_states(mesh, hashlib.sha256(mesh_file.read_bytes()).hexdigest())
    text = json.dumps(doc, allow_nan=False) + "\n"
    out.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
