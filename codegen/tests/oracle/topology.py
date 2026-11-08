"""Topology and orientation derived independently of the code under test (ADR 0003, check 1).

The derivation reads each cell's own outward polygons, as the builders of `meshes.py` return them
before `to_faces` merges them, and applies the convention of docs/contracts/mesh.md: a face's
right-hand-rule normal points from owner to neighbour, interior faces have owner < neighbour, and a
face is outward for c iff owner[f] == c. Every cell must be closed (each edge once in each
direction), one connected component, with a positive exact signed volume; the domain's boundary
surface must be closed and consistently oriented, enclosing the sum of the cells' volumes. Volumes
are exact rationals: each polygon is fanned about its node mean, as the contract defines face areas.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from fractions import Fraction

import autan_spike as s
import numpy as np
import numpy.typing as npt

Poly = tuple[int, ...]


class Refused(ValueError):
    """The cells' polygons do not form a valid, outward, closed mesh."""


@dataclasses.dataclass
class Derived:
    """Faces by sorted node key: (owner, neighbour or -1, node order seen from the owner)."""

    points: npt.NDArray[np.float64]
    faces: dict[Poly, tuple[int, int, Poly]]
    volumes: list[Fraction]
    n_cells: int


@dataclasses.dataclass
class ReadMesh:
    """The topology the kernels read, in their numbering."""

    points: npt.NDArray[np.float64]
    face_offsets: npt.NDArray[np.int64]
    face_nodes: npt.NDArray[np.int32]
    owner: npt.NDArray[np.int32]
    neighbour: npt.NDArray[np.int32]
    cell_face_offsets: npt.NDArray[np.int64]
    cell_faces: npt.NDArray[np.int32]
    incidence_side: npt.NDArray[np.float64]
    n_cells: int

    def copy(self) -> ReadMesh:
        return dataclasses.replace(self)


def _volume(pts: list[list[Fraction]], polys: list[Poly]) -> Fraction:
    """(1/6) sum over polygons and fan triangles (m, p_i, p_i+1) of det(m, p_i, p_i+1)."""
    total = Fraction(0)
    for poly in polys:
        n = len(poly)
        m = [sum((pts[v][a] for v in poly), Fraction(0)) / n for a in range(3)]
        for i in range(n):
            p, q = pts[poly[i]], pts[poly[(i + 1) % n]]
            total += (
                m[0] * (p[1] * q[2] - p[2] * q[1])
                - m[1] * (p[0] * q[2] - p[2] * q[0])
                + m[2] * (p[0] * q[1] - p[1] * q[0])
            )
    return total / 6


def _edges(poly: Poly) -> list[tuple[int, int]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _same_cycle(a: Poly, b: Poly) -> bool:
    if len(a) != len(b) or a[0] not in b:
        return False
    k = b.index(a[0])
    return a == b[k:] + b[:k]


def _components(polys: list[Poly]) -> int:
    parent = list(range(len(polys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_edge: dict[tuple[int, int], int] = {}
    for i, poly in enumerate(polys):
        for a, b in _edges(poly):
            key = (min(a, b), max(a, b))
            if key in by_edge:
                parent[find(i)] = find(by_edge[key])
            else:
                by_edge[key] = i
    return len({find(i) for i in range(len(polys))})


def derive(
    points: npt.NDArray[np.float64], cells: list[list[list[int]]], agglomerate: list[int] | None
) -> Derived:
    pts = [[Fraction(float(v)) for v in row] for row in points]
    cell_of = agglomerate or list(range(len(cells)))
    n_cells = max(cell_of) + 1
    final: list[list[Poly]] = [[] for _ in range(n_cells)]
    for i, cell in enumerate(cells):
        final[cell_of[i]].extend(tuple(p) for p in cell)
    volumes = []
    owners: dict[Poly, list[tuple[int, Poly]]] = {}
    for c, parts in enumerate(final):
        # Polygons shared by two parts of an agglomerated cell, in opposite orders, are internal.
        keys = Counter(tuple(sorted(p)) for p in parts)
        polys = [p for p in parts if keys[tuple(sorted(p))] == 1]
        if any(n > 2 for n in keys.values()):
            msg = f"cell {c}: a polygon appears more than twice"
            raise Refused(msg)
        directed = Counter(e for p in polys for e in _edges(p))
        if any(n != 1 or directed[(b, a)] != 1 for (a, b), n in directed.items()):
            msg = f"cell {c} is not closed: an edge is not traversed once in each direction"
            raise Refused(msg)
        k = _components(polys)
        if k != 1:
            msg = f"cell {c} has {k} connected components, not one"
            raise Refused(msg)
        v = _volume(pts, polys)
        if v <= 0:
            msg = f"cell {c}: its signed volume {v} is not positive"
            raise Refused(msg)
        volumes.append(v)
        for p in polys:
            owners.setdefault(tuple(sorted(p)), []).append((c, p))
    faces: dict[Poly, tuple[int, int, Poly]] = {}
    boundary: list[Poly] = []
    for key, occ in owners.items():
        if len(occ) == 1:
            c, p = occ[0]
            faces[key] = (c, -1, p)
            boundary.append(p)
        elif len(occ) == 2:
            (c1, p1), (c2, p2) = sorted(occ)
            if not _same_cycle(p1, p2[::-1]):
                msg = f"cells {c1} and {c2} list their common face with the same orientation"
                raise Refused(msg)
            faces[key] = (c1, c2, p1)
        else:
            msg = f"a face is shared by {len(occ)} cells"
            raise Refused(msg)
    directed = Counter(e for p in boundary for e in _edges(p))
    if any(directed[(b, a)] != n for (a, b), n in directed.items()):
        msg = "the boundary surface is not closed and consistently oriented"
        raise Refused(msg)
    if _volume(pts, boundary) != sum(volumes, Fraction(0)):
        msg = "the boundary surface does not enclose the sum of the cells' volumes"
        raise Refused(msg)
    return Derived(np.asarray(points), faces, volumes, n_cells)


def read_mesh(m: s.Mesh) -> ReadMesh:
    pts, offsets, nodes = s.mesh_nodes(m)
    return ReadMesh(
        pts, offsets, nodes, m.owner.copy(), m.neighbour.copy(), m.cell_face_offsets.copy(),
        m.cell_faces.copy(), m.incidence_side.copy(), m.n_cells,
    )  # fmt: skip


def compare(derived: Derived, read: ReadMesh) -> list[str]:
    """Every difference between the derived topology and the one the kernels read."""
    issues: list[str] = []
    ids = {tuple(float(v) for v in row): i for i, row in enumerate(derived.points)}
    found = [ids.get(tuple(float(v) for v in row)) for row in read.points]
    nid = [i for i in found if i is not None]
    if len(nid) != len(found):
        return ["a node of the kernels' mesh is not a node of the cells"]
    if read.n_cells != derived.n_cells:
        issues.append(f"{read.n_cells} cells read, {derived.n_cells} derived")
    nf, ni = len(read.owner), len(read.neighbour)
    seen = set()
    for f in range(nf):
        span = read.face_nodes[read.face_offsets[f] : read.face_offsets[f + 1]]
        poly = tuple(nid[int(v)] for v in span)
        key = tuple(sorted(poly))
        if key not in derived.faces:
            issues.append(f"face {f} is no face of the cells")
            continue
        owner, neighbour, order = derived.faces[key]
        seen.add(key)
        if int(read.owner[f]) != owner:
            issues.append(f"face {f}: owner {int(read.owner[f])}, derived {owner}")
        read_neighbour = int(read.neighbour[f]) if f < ni else -1
        if read_neighbour != neighbour:
            issues.append(f"face {f}: neighbour {read_neighbour}, derived {neighbour}")
        if not _same_cycle(poly, order):
            issues.append(f"face {f}: its node order does not point from owner to neighbour")
    if len(seen) != len(derived.faces):
        issues.append(f"{len(derived.faces) - len(seen)} derived faces are missing")
    expected: list[list[int]] = [[] for _ in range(read.n_cells)]
    for f in range(nf):
        expected[int(read.owner[f])].append(f)
        if f < ni:
            expected[int(read.neighbour[f])].append(f)
    if len(read.cell_face_offsets) != read.n_cells + 1:
        issues.append("cell_face_offsets has the wrong size")
        return issues
    for c in range(read.n_cells):
        k0, k1 = int(read.cell_face_offsets[c]), int(read.cell_face_offsets[c + 1])
        got = [int(f) for f in read.cell_faces[k0:k1]]
        if got != sorted(expected[c]):
            issues.append(f"cell {c}: incidences {got}, derived {sorted(expected[c])}")
            continue
        for k, f in zip(range(k0, k1), got, strict=True):
            side = 1.0 if int(read.owner[f]) == c else -1.0
            if k < len(read.incidence_side) and float(read.incidence_side[k]) != side:
                issues.append(f"cell {c}, face {f}: incidence side {read.incidence_side[k]}")
    return issues
