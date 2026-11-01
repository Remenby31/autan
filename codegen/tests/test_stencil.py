"""Cell reductions over a cell stencil (over=STENCIL): stencil contents, exactness, derivatives
and the linear stability they restore on tetrahedra (Diskin and Thomas, AIAA J. 49(4), 2011)."""

from __future__ import annotations

from pathlib import Path

import autan_spike as s
import meshes
import mms
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS, eps

DATA = Path(__file__).parent / "data"
STENCILS = ["vertex", "face2"]
G = np.array([0.3, -1.2, 0.7])


def _cell_nodes(fa: meshes.FaceArrays) -> tuple[list[set[int]], list[tuple[int, set[int]]]]:
    """Node sets of each cell and of each boundary face, from the input arrays."""
    nodes: list[set[int]] = [set() for _ in range(fa.n_cells)]
    boundary = []
    for f in range(len(fa.cell0)):
        poly = {int(n) for n in fa.face_nodes[fa.face_offsets[f] : fa.face_offsets[f + 1]]}
        nodes[fa.cell0[f]] |= poly
        if fa.cell1[f] >= 0:
            nodes[fa.cell1[f]] |= poly
        else:
            boundary.append((f, poly))
    return nodes, boundary


@pytest.mark.parametrize("kind", KINDS)
def test_vertex_stencil_holds_the_cells_and_boundary_faces_sharing_a_node(kind: str) -> None:
    fa = meshes.build(kind, 4, 0.15, seed=1)
    m = s.mesh(fa)
    offsets, entries = s.stencil_arrays(m, s.set_stencil(m, "vertex"))
    nodes, boundary = _cell_nodes(fa)
    for c in range(m.n_cells):
        row = entries[offsets[c] : offsets[c + 1]]
        cells = row[row >= 0]
        expected = sorted(o for o in range(m.n_cells) if o != c and nodes[o] & nodes[c])
        assert cells.tolist() == expected
        assert int((row < 0).sum()) == sum(1 for _, poly in boundary if poly & nodes[c])


@pytest.mark.parametrize("kind", KINDS)
def test_face2_stencil_holds_face_neighbours_and_theirs(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=2))
    offsets, entries = s.stencil_arrays(m, s.set_stencil(m, "face2"))
    nbrs: list[set[int]] = [set() for _ in range(m.n_cells)]
    bfaces: list[set[int]] = [set() for _ in range(m.n_cells)]
    for f in range(m.n_faces):
        o = int(m.owner[f])
        if f < m.n_interior:
            nb = int(m.neighbour[f])
            nbrs[o].add(nb)
            nbrs[nb].add(o)
        else:
            bfaces[o].add(f)
    for c in range(m.n_cells):
        ring = set(nbrs[c]).union(*(nbrs[n] for n in nbrs[c])) - {c}
        faces = set(bfaces[c]).union(*(bfaces[n] for n in nbrs[c]))
        row = entries[offsets[c] : offsets[c + 1]]
        assert row[row >= 0].tolist() == sorted(ring)
        assert sorted(-row[row < 0] - 1) == sorted(faces)


@pytest.mark.parametrize("stencil", STENCILS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_stencil_gradient_and_residual_are_exact_for_linear_fields(
    kind: str, stencil: str, policy: str
) -> None:
    m = s.mesh(meshes.build(kind, 5, 0.15, seed=4))
    s.set_stencil(m, stencil)
    q = 1.0 + m.cell_centroid @ G
    q_b = np.zeros(m.n_faces)
    q_b[m.n_interior :] = 1.0 + m.face_centroid[m.n_interior :] @ G
    src = np.full(m.n_cells, np.dot(mms.VELOCITY, G))
    res, grad = s.advdiff_residual(m, policy, q, q_b, src, mms.VELOCITY, mms.NU)
    assert np.abs(grad - G).max() <= 400 * eps(policy) * 3.0 * 5
    scale = np.abs(m.face_area).sum(axis=1).max() * 3.0 / m.h
    assert np.abs(res).max() <= 2000 * eps(policy) * scale


@pytest.mark.parametrize("stencil", STENCILS)
@pytest.mark.parametrize("policy", PRECISIONS)
def test_stencil_residual_adjoint_passes_the_dot_product_test(stencil: str, policy: str) -> None:
    m = s.mesh(meshes.build("tet", 5, 0.15, seed=6))
    s.set_stencil(m, stencil)
    rng = np.random.default_rng(3)
    q, q_b, src = (
        rng.standard_normal(m.n_cells),
        rng.standard_normal(m.n_faces),
        rng.standard_normal(m.n_cells),
    )
    x, y = rng.standard_normal(m.n_cells), rng.standard_normal(m.n_cells)
    jx = s.advdiff_derivative(m, policy, "tangent", q, q_b, src, mms.VELOCITY, mms.NU, x)
    jty = s.advdiff_derivative(m, policy, "adjoint", q, q_b, src, mms.VELOCITY, mms.NU, y)
    gap = abs(y @ jx - jty @ x) / (
        np.linalg.norm(y) * np.linalg.norm(jx) + np.linalg.norm(jty) * np.linalg.norm(x)
    )
    assert gap <= (1e-13 if policy == "fp64" else 1e-5)


def _spectrum(m: s.Mesh) -> np.ndarray:
    """Eigenvalues of V^-1 dR/dq for pure advection (nu = 0), columns from the tangent."""
    n = m.n_cells
    zero = np.zeros(n)
    jac = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        jac[:, j] = s.advdiff_derivative(
            m, "fp64", "tangent", zero, np.zeros(m.n_faces), zero, mms.VELOCITY, 0.0, e
        )
    return np.linalg.eigvals(jac / m.cell_volume[:, None])


@pytest.mark.slow
def test_stencil_gradients_make_unlimited_advection_stable_on_tetrahedra() -> None:
    # V&V tet family, n = 5 (Gmsh Delaunay; tests/data/tet-n005.manifest.json). With face-neighbour
    # LSQ the semi-discrete operator -V^-1 J has eigenvalues in the right half plane; with either
    # wider stencil it has none.
    m = s.read(str(DATA / "tet-n005.msh"))
    face = _spectrum(m)
    assert face.real.min() < -1e-3 * np.abs(face).max()
    for stencil in STENCILS:
        s.set_stencil(m, stencil)
        ev = _spectrum(m)
        assert ev.real.min() >= -1e-10 * np.abs(ev).max(), stencil
