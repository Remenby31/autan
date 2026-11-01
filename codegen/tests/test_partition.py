"""Generated loops on a partition with two ghost layers (mesh contract, invariant 6)."""

from __future__ import annotations

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, eps, refused

DEVICE_POLICY = {"cpu": "fp64", "gpu": "fp64", "metal": "fp32+acc64"}
WHERE = ["host", *(pytest.param(x, marks=pytest.mark.gpu) for x in s.STRATEGIES)]


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize("kind", KINDS)
def test_face_loops_skip_halo_faces_of_a_two_layer_partition(
    backend: str, kind: str, where: str
) -> None:
    # A one-layer partition has no halo faces: its residual on owned cells and its partial sums
    # on ghost cells (the faces they share with owned cells) are what the two-layer partition
    # must give. Faces between two ghosts and boundary faces of ghosts add nothing.
    pol = DEVICE_POLICY[backend] if where != "host" else "fp64"
    if where != "host" and refused(backend, pol, where):
        pytest.skip(f"{backend} refuses {pol} with {where}")
    full = s.mesh(meshes.build(kind, 6, 0.15, seed=21))
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    rng = np.random.default_rng(22)
    q, src = rng.standard_normal(full.n_cells), rng.standard_normal(full.n_cells)
    q_b = rng.standard_normal(full.n_faces)

    def residual(m: s.Mesh) -> dict[int, float]:
        cells, faces = s.global_ids(m)
        args = (q[cells], q_b[faces], src[cells], mms.VELOCITY, mms.NU)
        if where == "host":
            res, _ = s.advdiff_residual(m, pol, *args)
        else:
            res, _, _ = s.gpu_advdiff_residual(m, pol, where, 64, *args)
        return dict(zip(cells.tolist(), res.tolist(), strict=True))

    for rank in (0, 1):
        two = s.partition(full, parts, 2, rank, layers=2)
        one = s.partition(full, parts, 2, rank, layers=1)
        assert two.n_faces > one.n_faces  # the halo faces this test is about
        r2, r1 = residual(two), residual(one)
        scale = max(abs(v) for v in r1.values())
        for g, v in r2.items():
            assert abs(v - r1.get(g, 0.0)) <= 500 * eps(pol) * scale, g
