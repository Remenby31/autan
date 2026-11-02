"""Face data written per face (face_loop and boundary_loop with Write outputs), Flag datasets
among them, and read back by a face loop: codegen/tests/kernels/face_flags.py."""

from __future__ import annotations

import autan_spike as s
import meshes
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps, refused


@pytest.mark.parametrize("policy", PRECISIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_face_writes_store_each_face_once_in_its_global_orientation(
    backend: str, kind: str, policy: str
) -> None:
    where = None if backend == "cpu" else "gather-recompute"
    full = s.mesh(meshes.build(kind, 6, 0.15, seed=41))
    q = np.random.default_rng(42).uniform(-1.0, 1.0, full.n_cells)
    code = refused(backend, policy, where) if where is not None else None
    if code is not None:
        with pytest.raises(s.SpikeError) as err:
            s.face_flags(full, policy, q, where)
        assert str(err.value).startswith(code)
        return
    keep, jump, res = s.face_flags(full, policy, q, where, read_back=True)
    assert res is not None
    # Flags from the definition, on q as the policy stores it: q_l - q_r > 0 is q_l > q_r.
    qs = q.astype(np.float64 if policy == "fp64" else np.float32)
    ni, o, nb = full.n_interior, full.owner, full.neighbour
    first = np.concatenate([qs[o[:ni]] > qs[nb], qs[o[ni:]] > 0])
    assert np.array_equal(keep[:, 0], first.astype(np.uint8))
    nx = full.face_area[:, 0] / np.linalg.norm(full.face_area, axis=1)
    sure = np.abs(nx) > 1e-6  # the sign of a zero normal component is round-off
    assert np.array_equal(keep[sure, 1], (nx[sure] > 0).astype(np.uint8))
    want = qs[nb].astype(np.float64) - qs[o[:ni]]
    assert np.all(np.abs(jump[:ni] - want) <= 4 * eps(policy) * np.abs(want))
    # The face loop reading keep: +v on the left cell, -v on the right one.
    v = np.where(first[:ni], qs[o[:ni]], qs[nb]).astype(np.float64)
    expected = np.zeros(full.n_cells)
    np.add.at(expected, o[:ni], v)
    np.add.at(expected, nb, -v)
    assert np.all(np.abs(res - expected) <= 50 * eps(policy) * max(1.0, np.abs(expected).max()))
    if backend == "metal":
        return  # the global order is a DeviceMesh option (CUDA/HIP)

    # Ordered by the global numbering, a 2-rank partition writes on each interior face touching
    # an owned cell the bits a 1-rank partition writes there, and leaves halo faces alone.
    def written(m: s.Mesh) -> dict[int, bytes]:
        s.set_global_order(m, None)
        cells, faces = s.global_ids(m)
        k, j, _ = s.face_flags(m, policy, q[cells], where)
        n = m.n_interior
        touches = (m.owner[:n] < m.n_owned) | (m.neighbour < m.n_owned)
        assert np.all(k[:n][~touches] == 255)
        return {int(faces[f]): k[f].tobytes() + j[f].tobytes() for f in np.flatnonzero(touches)}

    reference = written(s.partition(full, np.zeros(full.n_cells, np.int32), 1, 0, layers=2))
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    for rank in (0, 1):
        got = written(s.partition(full, parts, 2, rank, layers=2))
        assert got.keys() <= reference.keys()
        assert all(got[g] == reference[g] for g in got), (kind, rank)
