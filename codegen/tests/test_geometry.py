"""Packed geometry built by autan::kernels::pack_geometry (kernel-generator.md, Data)."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import TYPE_CHECKING

import autan_spike as s
import meshes
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

EPS = float(np.finfo(np.float64).eps)
Vec = tuple[Fraction, Fraction, Fraction]


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _exact(points: npt.NDArray[np.float64]) -> tuple[Vec, Vec, list[list[Fraction]]]:
    # Area vector, centroid and moment of a polygon in exact arithmetic, from their definitions
    # (core Geometry; kernel-generator.md, Data): fan triangles t = (m, p_i, p_i+1) about the node
    # mean m, centroid weighted by the area of each t projected on the face's area vector,
    # M = sum_t (c_t - x_f) (x) S_t.
    p = [tuple(Fraction(float(c)) for c in q) for q in points]
    n = len(p)
    m = tuple(sum(q[k] for q in p) / n for k in range(3))
    rel = [tuple(q[k] - m[k] for k in range(3)) for q in p]
    tri = [(rel[i], rel[(i + 1) % n], _cross(rel[i], rel[(i + 1) % n])) for i in range(n)]
    area = tuple(sum(c[k] for _, _, c in tri) / 2 for k in range(3))
    w = [sum(c[k] * area[k] for k in range(3)) for _, _, c in tri]
    xf = tuple(
        m[k] + sum(wt * (a[k] + b[k]) for wt, (a, b, _) in zip(w, tri, strict=True)) / (3 * sum(w))
        for k in range(3)
    )
    moment = [[Fraction(0)] * 3 for _ in range(3)]
    for a, b, c in tri:
        r = [m[k] + (a[k] + b[k]) / 3 - xf[k] for k in range(3)]
        for u in range(3):
            for v in range(3):
                moment[u][v] += r[u] * c[v] / 2
    return area, xf, moment


def _check_moments(fa: meshes.FaceArrays) -> npt.NDArray[np.bool_]:
    # Every face's stored M against the exact one, matched by centroid; returns which faces the
    # exact test proves planar (a triangle, or nodes sharing one coordinate): those hold zeros.
    m = s.mesh(fa)
    proven = np.zeros(m.n_faces, dtype=bool)
    for g in range(len(fa.face_offsets) - 1):
        pts = fa.points[fa.face_nodes[fa.face_offsets[g] : fa.face_offsets[g + 1]]]
        area, xf, exact = _exact(pts)
        f = int(np.argmin(np.linalg.norm(m.face_centroid - [float(c) for c in xf], axis=1)))
        sign = 1.0 if np.dot(m.face_area[f], [float(c) for c in area]) > 0 else -1.0
        scale = np.linalg.norm(m.face_area[f]) ** 1.5
        np.testing.assert_allclose(
            m.face_moment[f], sign * np.array(exact, dtype=float), rtol=0, atol=64 * EPS * scale
        )
        proven[f] = len(pts) == 3 or bool((pts == pts[0]).all(axis=0).any())
        if proven[f]:
            assert np.all(m.face_moment[f] == 0.0)
    return proven


def test_face_moments_are_exactly_zero_where_planarity_is_exact() -> None:
    # M is zero on a planar face. The geometry stores exact zeros where planarity is exact (a
    # triangle, or nodes sharing one coordinate) and the fan sum on every other face, planar or
    # warped (#412). The oracle is M of the stored node coordinates in exact arithmetic.
    assert _check_moments(meshes.build("tet", 3, 0.15, seed=3)).all()

    # Warped hexahedra, then uneven y and z and x' = x + y: the faces on x = 0 lie exactly in the
    # oblique plane x' = y and nothing proves them planar, so they keep a round-off fan sum.
    _check_moments(meshes.build("hex", 3, 0.15, seed=3))
    fa = meshes.build("hex", 4)
    pts = fa.points.copy()
    pts[:, 1:] += 0.03 * np.sin(7.0 * pts[:, 1:])
    pts[:, 0] += pts[:, 1]
    sheared = dataclasses.replace(fa, points=pts)
    proven = _check_moments(sheared)
    m = s.mesh(sheared)
    in_plane = ~proven & (np.abs(m.face_centroid[:, 0] - m.face_centroid[:, 1]) < 1e-12)
    assert in_plane.sum() == 16
    # Their round-off is kept: it is exactly zero on 1 of the 16 by chance.
    assert (np.abs(m.face_moment[in_plane]).reshape(-1, 9).max(axis=1) > 0.0).sum() >= 12

    # A saddle quad whose area vector is along z although it is warped: not proven planar.
    fa = meshes.build("hex", 1)
    pts = fa.points.copy()
    pts[(pts[:, 2] == 1.0) & (pts[:, 0] != pts[:, 1]), 2] += 2.0**-20
    assert not _check_moments(dataclasses.replace(fa, points=pts)).all()

    # Partitioned and in the global order, a face proven planar holds +0 in its global orientation
    # on every rank: -0 where its local orientation is reversed.
    full = s.mesh(meshes.build("hybrid", 4, 0.15, seed=5))
    parts = (full.cell_centroid[:, 0] > 0.5).astype(np.int32)
    reversed_planar = 0
    for rank in (0, 1):
        local = s.partition(full, parts, 2, rank, layers=1)
        s.set_global_order(local, None)
        _, faces = s.global_ids(local)
        moments = s.face_moments(local)
        zero = np.all(moments.reshape(-1, 9) == 0.0, axis=1)
        reversed_ = np.einsum("ij,ij->i", local.face_area, full.face_area[faces]) < 0.0
        reversed_planar += int((zero & reversed_).sum())
        assert np.all(np.signbit(moments[zero]).all(axis=(1, 2)) == reversed_[zero])
    assert reversed_planar > 0
