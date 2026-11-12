"""Face states of an oracle on some boundary zones of a mesh (docs/contracts/face-states.md)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray

SCHEMA = "autan.face-states/1"
VARIABLES = ("density", "velocity-x", "velocity-y", "velocity-z", "pressure")
_ORACLE_FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")


def document(
    mesh: FaceMesh,
    mesh_sha256: str,
    oracle: str,
    zones: tuple[str, ...],
    fields: Callable[[FloatArray], dict[str, FloatArray]],
) -> dict[str, Any]:
    """The autan.face-states/1 document of `fields` (oracle field name to values at points) at
    the face centroids of the `zones` of `mesh`."""
    g = mesh.geometry
    out: dict[str, Any] = {}
    for zone in mesh.zones:
        if zone.name not in zones:
            continue
        faces = zone.start + np.arange(zone.count)
        centroids = g.face_centroid[faces]
        values = fields(centroids)
        out[zone.name] = {
            "faces": int(zone.count),
            "centroids": centroids.tolist(),
            "values": np.stack([values[name] for name in _ORACLE_FIELDS], axis=1).tolist(),
        }
    return {
        "schema": SCHEMA,
        "oracle": oracle,
        "mesh_sha256": mesh_sha256,
        "variables": list(VARIABLES),
        "zones": out,
    }


def write(
    mesh: FaceMesh,
    mesh_file: Path,
    out: Path,
    oracle: str,
    zones: tuple[str, ...],
    fields: Callable[[FloatArray], dict[str, FloatArray]],
) -> str:
    """Write the face states of `mesh`, read from `mesh_file`, to `out`; return their SHA-256."""
    doc = document(mesh, hashlib.sha256(mesh_file.read_bytes()).hexdigest(), oracle, zones, fields)
    text = json.dumps(doc, allow_nan=False) + "\n"
    out.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
