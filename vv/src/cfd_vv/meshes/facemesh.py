"""Face-based polyhedral mesh and its FP64 geometry, with the same conventions as the C++ core.

Faces are polygons stored as CSR (`face_offsets`, `face_nodes`). Interior faces come first; their
right-hand-rule normal points from `owner` to `neighbour`. Boundary faces follow, grouped by zone,
with outward normals. Face geometry uses a triangle fan around the face mean point; cell volumes and
centroids use the tetrahedra (cell centre, face mean point, edge) of that fan, which is exact for the
triangulated polyhedron.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from cfd_vv.errors import VVError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Zone:
    """A boundary face zone: a contiguous range of boundary faces sharing a name and BC type."""

    name: str
    bc_type: str
    start: int
    count: int


@dataclass(frozen=True)
class Geometry:
    """FP64 geometry of a FaceMesh."""

    face_area: FloatArray  # (n_faces, 3) area vectors
    face_centroid: FloatArray  # (n_faces, 3)
    cell_volume: FloatArray  # (n_cells,)
    cell_centroid: FloatArray  # (n_cells, 3)
    min_subvolume: FloatArray  # (n_cells,) smallest signed tetrahedron of the decomposition


@dataclass(frozen=True, eq=False)
class FaceMesh:
    """Unstructured mesh of arbitrary polyhedra stored by faces (see module docstring)."""

    points: FloatArray
    face_offsets: IntArray
    face_nodes: IntArray
    owner: IntArray
    neighbour: IntArray
    n_cells: int
    zones: tuple[Zone, ...]
    cell_kinds: tuple[str, ...] = field(default=())

    @property
    def n_faces(self) -> int:
        return int(self.owner.size)

    @property
    def n_interior(self) -> int:
        return int(self.neighbour.size)

    def face(self, f: int) -> IntArray:
        return self.face_nodes[self.face_offsets[f] : self.face_offsets[f + 1]]

    @cached_property
    def geometry(self) -> Geometry:
        pts = self.points
        sizes = np.diff(self.face_offsets)
        face_of_node = np.repeat(np.arange(self.n_faces), sizes)
        mean = np.zeros((self.n_faces, 3))
        np.add.at(mean, face_of_node, pts[self.face_nodes])
        mean /= sizes[:, None]
        # Edge i -> i+1 of each face, cyclic.
        nxt = np.arange(self.face_nodes.size) + 1
        last = self.face_offsets[1:] - 1
        nxt[last] = self.face_offsets[:-1]
        a = pts[self.face_nodes] - mean[face_of_node]
        b = pts[self.face_nodes[nxt]] - mean[face_of_node]
        tri_area = 0.5 * np.cross(a, b)
        face_area = np.zeros((self.n_faces, 3))
        np.add.at(face_area, face_of_node, tri_area)
        unit = face_area / np.linalg.norm(face_area, axis=1, keepdims=True)
        weight = np.einsum("ij,ij->i", tri_area, unit[face_of_node])
        tri_centroid = mean[face_of_node] + (a + b) / 3.0
        face_centroid = np.zeros((self.n_faces, 3))
        np.add.at(face_centroid, face_of_node, weight[:, None] * tri_centroid)
        wsum = np.zeros(self.n_faces)
        np.add.at(wsum, face_of_node, weight)
        face_centroid /= wsum[:, None]

        # Cell reference point: mean of its face mean points.
        ref = np.zeros((self.n_cells, 3))
        count = np.zeros(self.n_cells)
        np.add.at(ref, self.owner, mean)
        np.add.at(count, self.owner, 1.0)
        np.add.at(ref, self.neighbour, mean[: self.n_interior])
        np.add.at(count, self.neighbour, 1.0)
        ref /= count[:, None]

        volume = np.zeros(self.n_cells)
        moment = np.zeros((self.n_cells, 3))
        min_sub = np.full(self.n_cells, np.inf)
        apex_owner = ref[self.owner][face_of_node]
        # Tetrahedron (ref, mean, p_i, p_i+1); positive when the face normal points away from ref.
        tet_owner = np.einsum("ij,ij->i", tri_area, mean[face_of_node] - apex_owner) / 3.0
        cen_owner = 0.25 * (apex_owner + mean[face_of_node] + pts[self.face_nodes] + pts[self.face_nodes[nxt]])
        owner_of_tri = self.owner[face_of_node]
        np.add.at(volume, owner_of_tri, tet_owner)
        np.add.at(moment, owner_of_tri, tet_owner[:, None] * cen_owner)
        np.minimum.at(min_sub, owner_of_tri, tet_owner)
        interior_tris = face_of_node < self.n_interior
        fi = face_of_node[interior_tris]
        apex_nb = ref[self.neighbour[fi]]
        tet_nb = -np.einsum("ij,ij->i", tri_area[interior_tris], mean[fi] - apex_nb) / 3.0
        cen_nb = 0.25 * (
            apex_nb + mean[fi] + pts[self.face_nodes[interior_tris]] + pts[self.face_nodes[nxt][interior_tris]]
        )
        nb_of_tri = self.neighbour[fi]
        np.add.at(volume, nb_of_tri, tet_nb)
        np.add.at(moment, nb_of_tri, tet_nb[:, None] * cen_nb)
        np.minimum.at(min_sub, nb_of_tri, tet_nb)
        return Geometry(face_area, face_centroid, volume, moment / volume[:, None], min_sub)

    def check(self, rtol: float = 1e-12) -> None:
        """Raise V0301 unless every cell is closed and has a positive volume.

        Sub-tetrahedra of the decomposition may be negative for strongly non-planar faces (median
        duals); `min_subvolume_ratio` reports that as a quality measure instead.
        """
        g = self.geometry
        closure = np.zeros((self.n_cells, 3))
        np.add.at(closure, self.owner, g.face_area)
        np.add.at(closure, self.neighbour, -g.face_area[: self.n_interior])
        face_scale = np.zeros(self.n_cells)
        mag = np.linalg.norm(g.face_area, axis=1)
        np.add.at(face_scale, self.owner, mag)
        np.add.at(face_scale, self.neighbour, mag[: self.n_interior])
        rel = np.linalg.norm(closure, axis=1) / face_scale
        if rel.max() > rtol:
            msg = f"cell {int(rel.argmax())} is not closed: relative closure error {rel.max():.3g}"
            raise VVError(msg, code="V0301")
        if g.cell_volume.min() <= 0.0:
            msg = f"cell {int(g.cell_volume.argmin())} has a non-positive volume"
            raise VVError(msg, code="V0301")

    def min_subvolume_ratio(self) -> float:
        """Smallest (sub-tetrahedron volume / cell volume): positive iff every cell is star-shaped."""
        g = self.geometry
        return float((g.min_subvolume / g.cell_volume).min())

    def sha256(self) -> str:
        """Hash of the topology and coordinates, independent of file format."""
        h = hashlib.sha256()
        for arr in (self.points, self.face_offsets, self.face_nodes, self.owner, self.neighbour):
            h.update(np.ascontiguousarray(arr).tobytes())
        for z in self.zones:
            h.update(f"{z.name}:{z.bc_type}:{z.start}:{z.count};".encode())
        return h.hexdigest()


def _pad(polys: IntArray, width: int) -> IntArray:
    out = np.full((polys.shape[0], width), -1, dtype=np.int64)
    out[:, : polys.shape[1]] = polys
    return out


def from_faces(
    points: FloatArray,
    n_cells: int,
    interior: tuple[IntArray, IntArray, IntArray],
    boundary: dict[str, tuple[str, IntArray, IntArray]],
    cell_kinds: tuple[str, ...] = (),
) -> FaceMesh:
    """Build a FaceMesh from oriented polygons given as rows padded with -1.

    `interior` is (polygons, owner, neighbour) with normals from owner to neighbour; `boundary` maps a
    zone name to (BC type, polygons, owner) with outward normals. Faces are sorted by (owner,
    neighbour) inside the interior block and by owner inside each zone, so equal inputs give equal
    meshes.
    """
    polys_in, own_in, nb_in = interior
    width = max([polys_in.shape[1]] + [b[1].shape[1] for b in boundary.values()])
    order = np.lexsort((nb_in, own_in))
    blocks = [_pad(polys_in[order], width)]
    owners = [own_in[order]]
    zones = []
    start = polys_in.shape[0]
    for name, (bc_type, zpolys, zown) in boundary.items():
        zorder = np.argsort(zown, kind="stable")
        blocks.append(_pad(zpolys[zorder], width))
        owners.append(zown[zorder])
        zones.append(Zone(name, bc_type, start, zpolys.shape[0]))
        start += zpolys.shape[0]
    polys = np.concatenate(blocks)
    valid = polys >= 0
    return FaceMesh(
        points=np.ascontiguousarray(points, dtype=np.float64),
        face_offsets=np.concatenate([[0], np.cumsum(valid.sum(axis=1))]).astype(np.int64),
        face_nodes=polys[valid].astype(np.int64),
        owner=np.concatenate(owners).astype(np.int64),
        neighbour=np.asarray(nb_in[order], dtype=np.int64),
        n_cells=n_cells,
        zones=tuple(zones),
        cell_kinds=cell_kinds,
    )
