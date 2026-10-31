"""Median-dual polyhedral mesh of a tetrahedral mesh.

One polyhedron per primal vertex. The face between the cells of vertices a and b is the median-dual
face of edge ab: the ring of primal face centroids and tetrahedron centroids around the edge, closed
through the edge midpoint when the edge lies on the boundary (Barth, "Aspects of unstructured grids
and finite-volume solvers for the Euler and Navier-Stokes equations", VKI LS 1994-05). Boundary faces
of a cell are the fans of edge midpoints and boundary-triangle centroids around its vertex, one per
boundary zone, closed through the vertex itself where the zone ends (edges and corners of the box).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np

from cfd_vv.errors import VVError
from cfd_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray, from_faces

_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _newell(points: FloatArray) -> FloatArray:
    nxt = np.roll(points, -1, axis=0)
    return 0.5 * np.cross(points, nxt).sum(axis=0)


def _chain(pairs: list[tuple[int, int, int]], start: int | None) -> list[tuple[int, int, int]]:
    """Order (item, u, v) links into a path or cycle where consecutive links share a vertex."""
    by_vertex: dict[int, list[int]] = defaultdict(list)
    for i, (_, u, v) in enumerate(pairs):
        by_vertex[u].append(i)
        by_vertex[v].append(i)
    current = start if start is not None else pairs[0][1]
    out: list[tuple[int, int, int]] = []
    used = [False] * len(pairs)
    for _ in pairs:
        nxt = next((i for i in by_vertex[current] if not used[i]), None)
        if nxt is None:
            break
        used[nxt] = True
        item, u, v = pairs[nxt]
        other = v if u == current else u
        out.append((item, current, other))
        current = other
    if len(out) != len(pairs):
        msg = "the cells around an edge or vertex do not form a single fan"
        raise VVError(msg, code="V0307")
    return out


def median_dual(
    points: FloatArray,
    tets: IntArray,
    boundary: dict[str, IntArray],
    bc_types: dict[str, str],
) -> FaceMesh:
    """Build the median-dual polyhedral mesh; `boundary` maps zone names to boundary triangles."""
    nv = points.shape[0]
    nt = tets.shape[0]
    tet_centroid = points[tets].mean(axis=1)

    face_nodes = np.sort(tets[:, list(_TET_FACES)].reshape(-1, 3), axis=1)
    faces, face_of_tet = np.unique(face_nodes, axis=0, return_inverse=True)
    face_of_tet = face_of_tet.reshape(nt, 4)
    face_id = {tuple(f): i for i, f in enumerate(faces.tolist())}
    face_centroid = points[faces].mean(axis=1)
    tet_of_face = np.full(faces.shape[0], -1, dtype=np.int64)
    tet_of_face[face_of_tet.ravel()] = np.repeat(np.arange(nt), 4)

    # Dual points: primal vertices, edge midpoints (added on demand), face centroids, tet centroids.
    extra: list[FloatArray] = []
    midpoint_id: dict[tuple[int, int], int] = {}
    base_face = nv
    base_tet = nv + faces.shape[0]
    base_mid = base_tet + nt

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in midpoint_id:
            midpoint_id[key] = base_mid + len(extra)
            extra.append(0.5 * (points[a] + points[b]))
        return midpoint_id[key]

    # Boundary triangles, oriented outward using the tetrahedron that owns them.
    zone_of_face: dict[int, str] = {}
    for name, tris in boundary.items():
        for tri in tris[:, :3].tolist():
            f = face_id.get(tuple(sorted(tri)))
            if f is None:
                msg = f"boundary triangle {tri} of zone {name} is not a tetrahedron face"
                raise VVError(msg, code="V0302")
            zone_of_face[f] = name
    boundary_faces = np.array(sorted(zone_of_face), dtype=np.int64)
    outward = face_centroid[boundary_faces] - tet_centroid[tet_of_face[boundary_faces]]

    all_points = np.concatenate([points, face_centroid, tet_centroid])

    # Interior dual faces, one per primal edge.
    edges_local = tets[:, [e[0] for e in _TET_EDGES]], tets[:, [e[1] for e in _TET_EDGES]]
    lo = np.minimum(*edges_local).ravel()
    hi = np.maximum(*edges_local).ravel()
    others = np.array([[k for k in range(4) if k not in e] for e in _TET_EDGES])
    third = tets[:, others[:, 0]].ravel()
    fourth = tets[:, others[:, 1]].ravel()
    tet_idx = np.repeat(np.arange(nt), 6)
    order = np.lexsort((hi, lo))
    lo, hi, third, fourth, tet_idx = (
        lo[order],
        hi[order],
        third[order],
        fourth[order],
        tet_idx[order],
    )
    starts = np.flatnonzero(np.r_[True, (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1]), True])

    polys: list[list[int]] = []
    owners: list[int] = []
    neighbours: list[int] = []
    for s, e in pairwise(starts):
        a, b = int(lo[s]), int(hi[s])
        links = [(int(tet_idx[i]), int(third[i]), int(fourth[i])) for i in range(s, e)]
        degree: dict[int, int] = defaultdict(int)
        for _, u, v in links:
            degree[u] += 1
            degree[v] += 1
        ends = [v for v, d in degree.items() if d == 1]
        path = _chain(links, ends[0] if ends else None)
        ring: list[int] = []
        for tet, u, _ in path:
            ring.append(base_face + face_id[tuple(sorted((a, b, u)))])
            ring.append(base_tet + tet)
        if ends:
            last = path[-1][2]
            ring.append(base_face + face_id[tuple(sorted((a, b, last)))])
            ring.append(midpoint(a, b))
        polys.append(ring)
        owners.append(a)
        neighbours.append(b)

    # Boundary dual faces: per (vertex, zone), the fan of boundary triangles around the vertex.
    fans: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for f in boundary_faces.tolist():
        tri = faces[f].tolist()
        for k in range(3):
            v = tri[k]
            u, w = tri[(k + 1) % 3], tri[(k + 2) % 3]
            fans[(v, zone_of_face[f])].append((f, u, w))
    zone_polys: dict[str, list[list[int]]] = {name: [] for name in boundary}
    zone_owner: dict[str, list[int]] = {name: [] for name in boundary}
    zone_normal: dict[str, list[FloatArray]] = {name: [] for name in boundary}
    outward_of = {int(f): outward[i] for i, f in enumerate(boundary_faces)}
    for (v, name), links in sorted(fans.items()):
        degree = defaultdict(int)
        for _, u, w in links:
            degree[u] += 1
            degree[w] += 1
        ends = [p for p, d in degree.items() if d == 1]
        path = _chain(links, ends[0] if ends else None)
        ring = [v] if ends else []
        for f, u, _ in path:
            ring.append(midpoint(v, u))
            ring.append(base_face + f)
        if ends:
            ring.append(midpoint(v, path[-1][2]))
        zone_polys[name].append(ring)
        zone_owner[name].append(v)
        zone_normal[name].append(sum(outward_of[f] for f, _, _ in path))

    all_points = np.concatenate([all_points, np.array(extra).reshape(-1, 3)])

    def oriented(rings: list[list[int]], direction: list[FloatArray]) -> IntArray:
        width = max(len(r) for r in rings)
        out = np.full((len(rings), width), -1, dtype=np.int64)
        for i, (r, d) in enumerate(zip(rings, direction, strict=True)):
            rr = r if float(_newell(all_points[r]) @ d) > 0.0 else r[::-1]
            out[i, : len(rr)] = rr
        return out

    edge_dir = [points[b] - points[a] for a, b in zip(owners, neighbours, strict=True)]
    interior = (oriented(polys, edge_dir), np.array(owners), np.array(neighbours))
    zones = {
        name: (
            bc_types[name],
            oriented(zone_polys[name], zone_normal[name]),
            np.array(zone_owner[name]),
        )
        for name in boundary
    }
    # Drop dual points no face uses (interior primal vertices), keeping the order of the others.
    used = np.zeros(all_points.shape[0], dtype=bool)
    for arr in [interior[0], *(z[1] for z in zones.values())]:
        used[arr[arr >= 0]] = True
    remap = np.cumsum(used) - 1

    def renumber(arr: IntArray) -> IntArray:
        return np.where(arr >= 0, remap[np.maximum(arr, 0)], -1)

    interior = (renumber(interior[0]), interior[1], interior[2])
    zones = {k: (t, renumber(p), o) for k, (t, p, o) in zones.items()}
    return from_faces(all_points[used], nv, interior, zones, ("polyhedron",) * nv)
