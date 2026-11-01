"""Conversion of a one-layer hexahedral slab (MSH 4.1) to SU2's native mesh format.

SU2 reads the same mesh Autan reads, either in 2D (the slab's back plane, the planar zones dropped)
or as the 3D slab itself. The record (`autan.su2-mesh/1`) ties the converted file to its source by
their SHA-256 and by `plane_sha256`, a hash of the in-plane points and quadrangles that is computed
the same way from the source slab and from the converted file, so that a reader can check that the
two hold the same grid bit for bit.

SU2's native format: SU2 documentation, "Mesh File" (su2code.github.io/docs_v7/Mesh-File);
element types are VTK's (line 3, quadrilateral 9, hexahedron 12).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from autan_vv.errors import VVError
from autan_vv.su2 import msh

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.su2.msh import FloatArray, IntArray

RECORD_SCHEMA = "autan.su2-mesh/1"
VTK_LINE, VTK_QUAD, VTK_HEX = 3, 9, 12
Dimension = Literal[2, 3]


@dataclass(frozen=True)
class Slab:
    """A one-layer hexahedral mesh between the planes y = y0 and y = y1, read from MSH 4.1."""

    points: FloatArray
    hexes: IntArray
    zones: dict[str, IntArray]
    y0: float
    y1: float


@dataclass(frozen=True)
class Plane:
    """The slab's grid in the (x, z) plane: points (n, 2), counterclockwise quadrangles, and each
    lateral zone as edges (k, 2). `source` maps each plane point to its slab point."""

    points: FloatArray
    quads: IntArray
    markers: dict[str, IntArray]
    source: IntArray


def _fail(what: str) -> VVError:
    return VVError(what, code="V0902")


def read_slab(path: Path) -> Slab:
    """The slab in `path`: exactly one 3D group of hexahedra, every node on y0 or y1 (y0 < y1),
    each hexahedron with four nodes on each plane, the y1 nodes above the y0 ones. V0902 otherwise."""
    m = msh.read(path)
    volumes = [n for n, d in m.dims.items() if d == 3]
    if len(volumes) != 1:
        msg = f"{path}: expected one 3D physical group, got {volumes}"
        raise _fail(msg)
    hexes = m.groups[volumes[0]]
    if hexes.shape[1] != 8:
        msg = f"{path}: the 3D group is not made of hexahedra"
        raise _fail(msg)
    y = m.points[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    if not y0 < y1 or np.any((y != y0) & (y != y1)):
        msg = f"{path}: the nodes do not lie on two planes y = const"
        raise _fail(msg)
    zones = {n: q for n, q in m.groups.items() if m.dims[n] == 2}
    slab = Slab(m.points, hexes, zones, y0, y1)
    _bottom_and_top(slab)
    return slab


# Faces of a hexahedron in Gmsh's (and VTK's) node order, each a cycle, and the node joined by an
# edge to each of nodes 0-7 across the opposite face (Gmsh reference manual, "Node ordering").
_HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_HEX_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7),
              (5, 6), (6, 7))  # fmt: skip


def _bottom_and_top(slab: Slab) -> tuple[IntArray, IntArray]:
    """Each hexahedron's face on y0, as a cycle, and the y1 node joined by an edge to each of its
    nodes, as (M, 4) arrays."""
    on_y0 = slab.points[slab.hexes, 1] == slab.y0
    bottom = np.full((len(slab.hexes), 4), -1, dtype=np.int64)
    top = np.full((len(slab.hexes), 4), -1, dtype=np.int64)
    for face in _HEX_FACES:
        rest = [k for k in range(8) if k not in face]
        hit = on_y0[:, list(face)].all(axis=1) & ~on_y0[:, rest].any(axis=1)
        across = [next(b if a == k else a for a, b in _HEX_EDGES if k in (a, b) and
                       (b if a == k else a) in rest) for k in face]  # fmt: skip
        bottom[hit] = slab.hexes[hit][:, list(face)]
        top[hit] = slab.hexes[hit][:, across]
    if np.any(bottom < 0):
        msg = "a hexahedron does not have one face on each plane of the slab"
        raise _fail(msg)
    xz = [0, 2]
    if not np.array_equal(slab.points[bottom][..., xz], slab.points[top][..., xz]):
        msg = "a hexahedron is not a straight extrusion between the two planes"
        raise _fail(msg)
    return bottom, top


def to_plane(slab: Slab) -> tuple[Plane, list[str]]:
    """The plane grid and the names of the planar zones dropped (those lying on y0 or y1).

    Every other zone must be lateral: each of its quadrangles has exactly two nodes on y0, and
    its edges are boundary edges of the plane grid. Every boundary edge must belong to exactly one
    lateral zone. V0902 otherwise.
    """
    bottom, _ = _bottom_and_top(slab)
    used = np.unique(bottom)
    renumber = np.full(len(slab.points), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    pts = slab.points[used][:, [0, 2]]
    quads = renumber[bottom]
    a, b, c, d = (pts[quads[:, k]] for k in range(4))
    u, v = c - a, d - b
    area2 = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    if np.any(area2 == 0.0):
        msg = "a quadrangle of the plane has zero area"
        raise _fail(msg)
    quads = np.where((area2 > 0.0)[:, None], quads, quads[:, ::-1])

    edges = np.sort(np.concatenate([quads[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1)
    keys, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = {tuple(e) for e in keys[counts == 1].tolist()}
    if np.any(counts > 2):
        msg = "an edge of the plane is shared by more than two quadrangles"
        raise _fail(msg)

    markers: dict[str, IntArray] = {}
    dropped: list[str] = []
    seen: set[tuple[int, ...]] = set()
    for name, faces in slab.zones.items():
        if faces.shape[1] != 4:
            msg = f"zone {name} is not made of quadrangles"
            raise _fail(msg)
        low = slab.points[faces, 1] == slab.y0
        high = slab.points[faces, 1] == slab.y1
        if low.all() or high.all():
            dropped.append(name)
            continue
        if not np.all(low.sum(axis=1) == 2):
            msg = f"zone {name} mixes planar and lateral faces"
            raise _fail(msg)
        pairs = renumber[faces[low].reshape(-1, 2)]
        keyed = [tuple(sorted(p)) for p in pairs.tolist()]
        if any(k not in boundary for k in keyed) or seen.intersection(keyed):
            msg = f"zone {name} has an edge that is interior or in another zone"
            raise _fail(msg)
        seen.update(keyed)
        markers[name] = pairs
    if seen != boundary:
        msg = "a boundary edge of the plane belongs to no lateral zone"
        raise _fail(msg)
    return Plane(pts, quads, markers, used), dropped


def plane_sha256(points: FloatArray, quads: IntArray, markers: dict[str, IntArray]) -> str:
    """Hash of a plane grid independent of node numbering and marker order: the sorted
    quadrangles and edges written as coordinate tuples (FP64 bytes), each quadrangle rotated to
    start at its smallest point."""

    def canonical(cells: IntArray) -> list[bytes]:
        xy = points[cells]
        out = []
        for poly in xy:
            keys = [tuple(p) for p in poly.tolist()]
            start = keys.index(min(keys))
            out.append(np.asarray(keys[start:] + keys[:start], dtype="<f8").tobytes())
        return sorted(out)

    h = hashlib.sha256()
    for item in canonical(quads):
        h.update(item)
    for name in sorted(markers):
        h.update(name.encode())
        for item in canonical(np.sort(markers[name], axis=1)):
            h.update(item)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(
    path: Path,
    dim: int,
    points: FloatArray,
    elements: tuple[int, IntArray],
    markers: dict[str, tuple[int, IntArray]],
) -> None:
    vtk, conn = elements
    with path.open("w") as f:
        f.write(f"NDIME= {dim}\nNELEM= {len(conn)}\n")
        ids = np.arange(len(conn))[:, None]
        np.savetxt(f, np.hstack([np.full_like(ids, vtk), conn, ids]), fmt="%d")
        f.write(f"NPOIN= {len(points)}\n")
        # 17 significant digits: every FP64 value reads back bit for bit.
        np.savetxt(f, points, fmt="%.17g")
        f.write(f"NMARK= {len(markers)}\n")
        for name, (mtype, mconn) in markers.items():
            f.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(mconn)}\n")
            np.savetxt(f, np.hstack([np.full((len(mconn), 1), mtype), mconn]), fmt="%d")


@dataclass(frozen=True)
class Record:
    """What `convert` wrote and from what (schema autan.su2-mesh/1)."""

    schema: str
    source: str
    source_sha256: str
    file: str
    file_sha256: str
    dimension: int
    points: int
    elements: int
    markers: dict[str, int]
    dropped_zones: list[str]
    plane_sha256: str


def convert(source: Path, out: Path, dimension: Dimension) -> Record:
    """Write the slab `source` to `out` in SU2 format, with `out`.json holding its Record.

    In 2D the plane (x, z) becomes SU2's (x, y): lift along +z in Autan is lift along +y in SU2,
    and the planar zones are dropped. In 3D every zone is kept and the hexahedra are written in
    the source's node order (VTK and Gmsh order a hexahedron's nodes alike).
    """
    slab = read_slab(source)
    plane, dropped = to_plane(slab)
    if dimension == 2:
        _write(
            out,
            2,
            plane.points,
            (VTK_QUAD, plane.quads),
            {n: (VTK_LINE, e) for n, e in plane.markers.items()},
        )
        n_points, n_elements = len(plane.points), len(plane.quads)
        markers = {n: len(e) for n, e in plane.markers.items()}
        dropped_zones = dropped
    else:
        _write(
            out,
            3,
            slab.points,
            (VTK_HEX, slab.hexes),
            {n: (VTK_QUAD, q) for n, q in slab.zones.items()},
        )
        n_points, n_elements = len(slab.points), len(slab.hexes)
        markers = {n: len(q) for n, q in slab.zones.items()}
        dropped_zones = []
    record = Record(
        schema=RECORD_SCHEMA,
        source=source.name,
        source_sha256=file_sha256(source),
        file=out.name,
        file_sha256=file_sha256(out),
        dimension=dimension,
        points=n_points,
        elements=n_elements,
        markers=markers,
        dropped_zones=dropped_zones,
        plane_sha256=plane_sha256(plane.points, plane.quads, plane.markers),
    )
    out.with_suffix(".json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    return record
