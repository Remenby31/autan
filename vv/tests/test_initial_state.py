"""The initial-state exporter: an oracle at the cell centroids of a mesh, as the SDK reads it."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from autan import initial_state
from autan_vv.initial_state import write_initial_state
from autan_vv.meshes.families import cartesian
from autan_vv.oracles.inviscid import isentropic_vortex

if TYPE_CHECKING:
    from pathlib import Path


def test_the_vortex_is_written_at_every_cell_centroid(tmp_path: Path) -> None:
    mesh = cartesian((4, 4, 1), (-10.0, -10.0, 0.0), (10.0, 10.0, 1.0))
    mesh_file = tmp_path / "box.msh"
    mesh_file.write_bytes(b"the mesh the case names")
    out = tmp_path / "vortex.json"
    oracle = isentropic_vortex()

    digest = write_initial_state(oracle, mesh, mesh_file, out)

    data = out.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    document = initial_state.parse(data)
    assert not isinstance(document, str), document
    assert document.oracle == "isentropic-vortex"
    assert document.mesh_sha256 == hashlib.sha256(mesh_file.read_bytes()).hexdigest()
    # Cell (i, j) of the 5 m grid has its centroid at (-7.5 + 5 i, -7.5 + 5 j, 0.5).
    centres = -7.5 + 5.0 * np.arange(4)
    expected = np.array(sorted((x, y, 0.5) for x in centres for y in centres))
    rows = np.array(document.centroids)
    # The centroids are computed (mesh.md), so they agree to round-off of the 20 m extent.
    order = np.lexsort((np.round(rows[:, 1], 6), np.round(rows[:, 0], 6)))
    tolerance = 64 * np.finfo(float).eps * 20.0
    np.testing.assert_allclose(rows[order], expected, rtol=0, atol=tolerance)
    state = oracle.evaluate(rows, None)
    names = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")
    np.testing.assert_array_equal(
        np.array(document.values), np.stack([state[n] for n in names], axis=1)
    )
