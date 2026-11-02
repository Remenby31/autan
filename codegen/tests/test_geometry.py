"""Packed geometry built by autan::kernels::pack_geometry (kernel-generator.md, Data)."""

from __future__ import annotations

import dataclasses

import autan_spike as s
import meshes
import numpy as np

EPS = float(np.finfo(np.float64).eps)


def _axis_plane(m: s.Mesh) -> np.ndarray:
    # On these meshes, a face whose nodes share one coordinate is the face whose area vector has
    # two components exactly zero.
    return (m.face_area == 0.0).sum(axis=1) == 2


def test_face_moments_are_exactly_zero_where_planarity_is_exact() -> None:
    # M = sum_t (c_t - x_f) (x) S_t is zero on a planar face. The geometry stores exact zeros where
    # planarity is exact (a triangle, or nodes sharing one coordinate) and keeps the fan sum on
    # every other face, planar or warped (#412).
    tet = s.mesh(meshes.build("tet", 4, 0.15, seed=3))
    assert np.all(tet.face_moment == 0.0)
    assert not np.signbit(tet.face_moment).any()

    # Uneven y and z, then x' = x + y: the faces on x = 0 lie exactly in the oblique plane x' = y
    # and the other faces normal to x nearly so (x + y rounds). Nothing proves them planar: their
    # fan sum is kept, round-off of a few eps.
    fa = meshes.build("hex", 4)
    pts = fa.points.copy()
    pts[:, 1:] += 0.03 * np.sin(7.0 * pts[:, 1:])
    pts[:, 0] += pts[:, 1]
    sheared = s.mesh(dataclasses.replace(fa, points=pts))
    flat = _axis_plane(sheared)
    assert np.all(sheared.face_moment[flat] == 0.0)
    oblique = np.abs(sheared.face_moment[~flat]).reshape(-1, 9).max(axis=1)
    scale = np.linalg.norm(sheared.face_area[~flat], axis=1) ** 1.5
    in_plane = np.abs(sheared.face_centroid[~flat, 0] - sheared.face_centroid[~flat, 1]) < 1e-12
    assert in_plane.sum() == 16
    assert oblique[in_plane].max() > 0.0
    assert np.all(oblique <= 64 * EPS * scale)

    # Perturbed hexahedra: boundary faces stay in the cube's planes, interior faces are warped.
    warped = s.mesh(meshes.build("hex", 4, 0.15, seed=3))
    flat = _axis_plane(warped)
    assert np.all(warped.face_moment[flat] == 0.0)
    inner = np.abs(warped.face_moment[~flat]).reshape(-1, 9).max(axis=1)
    assert np.all(inner > 1e-6 * np.linalg.norm(warped.face_area[~flat], axis=1) ** 1.5)
