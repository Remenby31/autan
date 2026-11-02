"""Face data written per face (face_loop and boundary_loop with Write outputs), Flag datasets
among them, and read back by a face loop: codegen/tests/kernels/face_flags.py."""

from __future__ import annotations

import autan_spike as s
import meshes
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps, refused


def _nx(m: s.Mesh) -> np.ndarray:  # type: ignore[type-arg]
    return m.face_area[:, 0] / np.linalg.norm(m.face_area, axis=1)


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
    # The definitions on the full mesh, whose faces are the global ones in their global
    # orientation, on q as the policy stores it (q_l - q_r > 0 is q_l > q_r).
    qs = q.astype(np.float64 if policy == "fp64" else np.float32)
    ni, o, nb = full.n_interior, full.owner, full.neighbour
    first = qs[o[:ni]] > qs[nb]
    jump = qs[nb].astype(np.float64) - qs[o[:ni]]
    nx = _nx(full)
    v = np.where(first, qs[o[:ni]], qs[nb]).astype(np.float64)
    res = np.zeros(full.n_cells)
    np.add.at(res, o[:ni], v)
    np.add.at(res, nb, -v)
    scale = max(1.0, np.abs(res).max())

    def check(m: s.Mesh) -> dict[int, bytes]:
        """Asserts m's writes against the definitions; returns the bits of its written faces."""
        cells, faces = s.global_ids(m)
        k, j, r = s.face_flags(m, policy, q[cells], where, read_back=True)
        assert r is not None
        n = m.n_interior
        # Interior faces touching an owned cell: written once, in the global orientation.
        touches = (m.owner[:n] < m.n_owned) | (m.neighbour < m.n_owned)
        f = np.flatnonzero(touches)
        g = faces[f]
        assert np.array_equal(k[f, 0], first[g].astype(np.uint8))
        sure = np.abs(nx[g]) > 1e-6  # the sign of a zero normal component is round-off
        assert np.array_equal(k[f, 1][sure], (nx[g][sure] > 0).astype(np.uint8))
        assert np.all(np.abs(j[f] - jump[g]) <= 4 * eps(policy) * np.abs(jump[g]))
        # Halo faces, between two ghosts: untouched.
        assert np.all(k[:n][~touches] == 255)
        assert np.all(j[:n][~touches] == 0)
        # Every boundary face, of owned cells and of ghosts, outward.
        b = np.arange(n, m.n_faces)
        assert np.array_equal(k[b, 0], (qs[cells[m.owner[b]]] > 0).astype(np.uint8))
        nxb = _nx(m)[b]
        sure = np.abs(nxb) > 1e-6
        assert np.array_equal(k[b, 1][sure], (nxb[sure] > 0).astype(np.uint8))
        # The face loop reading keep: owned cells hold the sum over all their faces.
        own = cells[: m.n_owned]
        assert np.all(np.abs(r[: m.n_owned] - res[own]) <= 50 * eps(policy) * scale)
        return {int(faces[x]): k[x].tobytes() + j[x].tobytes() for x in f}

    check(full)
    if backend == "metal":
        return  # the global order is a DeviceMesh option (CUDA/HIP)
    # Ordered by the global numbering, a partition evaluates each face in its global orientation:
    # 2 ranks write the bits 1 rank writes.
    one = s.partition(full, np.zeros(full.n_cells, np.int32), 1, 0, layers=2)
    s.set_global_order(one, None)
    reference = check(one)
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    for rank in (0, 1):
        two = s.partition(full, parts, 2, rank, layers=2)
        s.set_global_order(two, None)
        got = check(two)
        assert got.keys() <= reference.keys()
        assert all(got[g] == reference[g] for g in got), (kind, rank)
