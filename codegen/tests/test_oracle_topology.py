"""Topology and orientation derived independently (ADR 0003, check 1): from each cell's own outward
polygons, before `to_faces` merges them, by the contract's convention (docs/contracts/mesh.md).
The derivation must agree with the mesh the kernels read, and refuse the cells and corruptions the
ADR names."""

from fractions import Fraction as Fr

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import topology as t

CASES = [(k, p) for k in ("hex", "tet", "hybrid", "poly") for p in (0.0, 0.15)]


@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_the_derivation_agrees_with_the_assembled_mesh(kind: str, perturbation: float) -> None:
    pts, cells, agglomerate = meshes.cells(kind, 4, perturbation, seed=3)
    derived = t.derive(pts, cells, agglomerate)
    m = s.mesh(meshes.to_faces(pts, cells, agglomerate))
    assert t.compare(derived, t.read_mesh(m)) == []


def test_build_is_to_faces_of_cells() -> None:
    a = meshes.build("hybrid", 4, 0.15, seed=3)
    b = meshes.to_faces(*meshes.cells("hybrid", 4, 0.15, seed=3))
    for field in ("points", "face_offsets", "face_nodes", "cell0", "cell1"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.n_cells == b.n_cells


def _cube(
    origin: tuple[int, int, int], outward: bool, base: int
) -> tuple[list[list[float]], list[list[int]]]:
    x0, y0, z0 = origin
    pts = [[x0 + (i & 1), y0 + ((i >> 1) & 1), z0 + ((i >> 2) & 1)] for i in range(8)]
    faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    polys = [[base + v for v in (f if outward else f[::-1])] for f in faces]
    return pts, polys


def test_two_shells_of_opposite_volume_are_refused() -> None:
    # +8 and -1: closed, edges cancel, total volume positive, yet not one outward component.
    big_pts, big = _cube((0, 0, 0), True, 0)
    big_pts = [[2 * c for c in p] for p in big_pts]
    small_pts, small = _cube((10, 0, 0), False, 8)
    pts = np.array(big_pts + small_pts, dtype=np.float64)
    with pytest.raises(t.Refused, match="component"):
        t.derive(pts, [big + small], None)


def test_a_reversed_polygon_is_refused() -> None:
    cpts, polys = _cube((0, 0, 0), True, 0)
    polys[2] = polys[2][::-1]
    with pytest.raises(t.Refused, match="closed"):
        t.derive(np.array(cpts, dtype=np.float64), [polys], None)


def test_an_inward_cell_is_refused() -> None:
    cpts, polys = _cube((0, 0, 0), False, 0)
    with pytest.raises(t.Refused, match="volume"):
        t.derive(np.array(cpts, dtype=np.float64), [polys], None)


def test_the_cell_volumes_are_exact() -> None:
    cpts, polys = _cube((0, 0, 0), True, 0)
    derived = t.derive(np.array(cpts, dtype=np.float64), [polys], None)
    assert derived.volumes == [Fr(1)]


def test_corrupted_owner_and_incidences_are_reported() -> None:
    pts, cells, agglomerate = meshes.cells("hex", 3, 0.15, seed=3)
    derived = t.derive(pts, cells, agglomerate)
    read = t.read_mesh(s.mesh(meshes.to_faces(pts, cells, agglomerate)))
    swapped = read.copy()
    swapped.owner = read.owner.copy()
    swapped.neighbour = read.neighbour.copy()
    swapped.owner[0], swapped.neighbour[0] = read.neighbour[0], read.owner[0]  # control 9
    assert t.compare(derived, swapped)
    dropped = read.copy()
    dropped.cell_faces = np.delete(read.cell_faces, 0)  # control 3
    dropped.cell_face_offsets = read.cell_face_offsets.copy()
    dropped.cell_face_offsets[1:] -= 1
    assert t.compare(derived, dropped)
