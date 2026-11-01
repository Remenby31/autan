"""Initial-state files (``autan.initial-state/1``, docs/contracts/initial-state.md) from an oracle
evaluated at the cell centroids of a mesh."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from autan import initial_state

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh
    from autan_vv.oracles import Oracle

_FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")


def write_initial_state(
    oracle: Oracle, mesh: FaceMesh, mesh_file: Path, out: Path, time: float | None = None
) -> str:
    """Write to `out` the state of `oracle` at `time` (its start without one) at every cell
    centroid of `mesh`, the mesh `mesh_file` holds, and return the file's SHA-256 for the case.
    The state is the oracle's value at the centroid, with no projection onto the cell."""
    centroids = mesh.geometry.cell_centroid
    fields = oracle.evaluate(centroids, time)
    values = np.stack([np.broadcast_to(fields[name], (mesh.n_cells,)) for name in _FIELDS], axis=1)
    return initial_state.write(
        out,
        oracle=oracle.name,
        mesh_sha256=hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
        centroids=centroids.tolist(),
        values=values.tolist(),
    )
