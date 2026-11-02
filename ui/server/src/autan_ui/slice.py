"""Plane slices of a solution.cgns (results.md#solutioncgns) for the viewer."""

from collections import defaultdict

import h5py
import numpy as np
import numpy.typing as npt

from autan_ui.confine import Folder
from autan_ui.surface import Zone, encode
from autan_ui.wall import _data, _ear_clip, solution

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]

# Connectivity entries read from the file at once: bounds the memory a slice needs to a few
# arrays of this length, whatever the mesh size.
CHUNK = 1 << 22


def slice_surface(folder: Folder, origin: F64, normal: F64) -> bytes:
    """The cut of every cell of the run's solution by the plane through `origin` normal to
    `normal`, as an `autan-ui.surface/1` message with one zone, `slice`: the section of each cell
    the plane crosses, carrying that cell's `FlowSolution` values (cell-centred, so constant on
    the section). A plane through mesh nodes is moved off them by at most 1e-7 of the mesh size,
    to the side where its section is the larger: a plane along mesh faces gives those faces
    once, with the values of the cells on that side. ValueError when the plane meets no cell or
    the normal is zero."""
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length == 0 or not np.isfinite(origin).all():
        msg = "the plane needs a finite origin and a non-zero normal"
        raise ValueError(msg)
    unit = normal / length
    with solution(folder) as tree:
        zone = tree["Base"]["Zone"]
        coords = np.stack(
            [_data(zone["GridCoordinates"][f"Coordinate{a}"]) for a in "XYZ"], axis=1
        ).astype(np.float64)
        faces = Connectivity(zone["NGonElements"])
        cells = Connectivity(zone["NFaceElements"])
        cut = _section(coords, faces, cells, origin, unit)
        if cut is None:
            msg = "the plane meets no cell of the mesh"
            raise ValueError(msg)
        cell, points, triangles = cut
        node_cell = np.repeat(cell, [len(p) for p in points])
        fields = {
            name: _data(array)[node_cell].astype(np.float32)
            for name, array in zone["FlowSolution"].items()
            if isinstance(array, h5py.Group) and name != "GridLocation"
        }
    flat = np.concatenate(points)
    centre = (flat.min(axis=0) + flat.max(axis=0)) / 2
    zone_out = Zone("slice", (flat - centre).astype(np.float32), triangles, fields)
    return encode([zone_out], origin=(float(centre[0]), float(centre[1]), float(centre[2])))


class Connectivity:
    """An NGON or NFACE element section: offsets in memory, entries read in slices."""

    def __init__(self, section: h5py.Group) -> None:
        self.start = _data(section["ElementStartOffset"]).astype(np.int64)
        self.entries = section["ElementConnectivity"][" data"]

    def __len__(self) -> int:
        return len(self.start) - 1

    def chunks(self) -> "list[tuple[int, int]]":
        # Element ranges of about CHUNK entries each.
        bounds = np.searchsorted(self.start, np.arange(0, self.start[-1], CHUNK), side="right") - 1
        edges = np.unique(np.append(bounds, len(self)))
        return list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))

    def read(self, first: int, last: int) -> tuple[I64, I64]:
        """The offsets of elements first..last from their first entry, and their entries
        (0-based, unsigned)."""
        offsets = self.start[first : last + 1]
        values = np.asarray(self.entries[offsets[0] : offsets[-1]]).ravel()
        return offsets - offsets[0], np.abs(values.astype(np.int64, copy=False)) - 1


Cut = tuple[I64, list[F64], npt.NDArray[np.uint32]]
# Per segment: its cell, the names of its two points, and the points.
Segments = tuple[I64, I64, I64, F64, F64]


def _section(
    coords: F64, faces: Connectivity, cells: Connectivity, origin: F64, unit: F64
) -> Cut | None:
    if len(coords) >= 1 << 31:
        msg = "a slice needs fewer than 2^31 nodes"
        raise ValueError(msg)
    size = float(np.ptp(coords, axis=0).max()) or 1.0
    distance = (coords - origin) @ unit
    on_plane = np.abs(distance) < 1e-12 * size
    if not on_plane.any():
        return _cut(coords, faces, cells, distance, unit)
    # Nodes on the plane: it is moved off them, to each side by less than half the distance to
    # the nearest other node, so that every crossing is inside an edge. Moved to one side, it also
    # cuts slivers off the cells that only touch the plane from that side: a cell keeps its
    # section there only if it has nodes beyond the plane, or a face lying in it (the section is
    # then that face).
    others = np.abs(distance[~on_plane])
    shift = min(1e-7 * size, float(others.min()) / 2 if len(others) else size)
    above, below, lying = _sides(faces, cells, distance > 0, distance < 0, on_plane)
    cuts = [
        _cut(coords, faces, cells, distance - shift, unit, (below | lying, on_plane)),
        _cut(coords, faces, cells, distance + shift, unit, (above | lying, on_plane)),
    ]
    found = [c for c in cuts if c is not None]
    return max(found, key=_area) if found else None


def _sides(
    faces: Connectivity,
    cells: Connectivity,
    above: npt.NDArray[np.bool_],
    below: npt.NDArray[np.bool_],
    on_plane: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # Per cell: a node strictly above the plane, one strictly below, a face lying in it.
    face_flags = np.zeros(len(faces), dtype=np.uint8)
    for first, last in faces.chunks():
        offsets, node = faces.read(first, last)
        count = np.diff(offsets)
        starts = offsets[:-1]
        flags = np.where(np.add.reduceat(above[node], starts, dtype=np.int32) > 0, 1, 0)
        flags |= np.where(np.add.reduceat(below[node], starts, dtype=np.int32) > 0, 2, 0)
        flags |= np.where(np.add.reduceat(on_plane[node], starts, dtype=np.int32) == count, 4, 0)
        face_flags[first:last] = flags
    cell_flags = np.zeros(len(cells), dtype=np.uint8)
    for first, last in cells.chunks():
        offsets, cell_face = cells.read(first, last)
        cell_flags[first:last] = np.bitwise_or.reduceat(face_flags[cell_face], offsets[:-1])
    return (cell_flags & 1) > 0, (cell_flags & 2) > 0, (cell_flags & 4) > 0


def _area(cut: Cut) -> float:
    points = np.concatenate(cut[1])
    a, b, c = (points[cut[2][:, k].astype(np.int64)] for k in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def _cut(
    coords: F64,
    faces: Connectivity,
    cells: Connectivity,
    distance: F64,
    unit: F64,
    moved: tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]] | None = None,
) -> Cut | None:
    # `moved`: the plane was moved off nodes (`moved[1]`); only the cells `moved[0]` keep their
    # section, and a crossing on an edge from such a node is that node.
    above = distance > 0
    n_nodes = len(coords)

    # Every face edge the plane crosses, face by face, in the face's node order; a crossing point
    # is named by its edge.
    crossed: list[tuple[I64, I64, I64]] = []
    for first, last in faces.chunks():
        offsets, node = faces.read(first, last)
        node_above = above[node]
        count = np.diff(offsets)
        # The faces with nodes on both sides, then their edges.
        ups = np.add.reduceat(node_above, offsets[:-1], dtype=np.int32)
        cut = np.flatnonzero((ups > 0) & (ups < count))
        n = count[cut]
        within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        begin = np.repeat(offsets[cut], n)
        here = begin + within
        there = begin + (within + 1) % np.repeat(n, n)
        crossing = node_above[here] != node_above[there]
        face = np.repeat(first + cut, n)
        crossed.append((face[crossing], node[here[crossing]], node[there[crossing]]))
    face = np.concatenate([c[0] for c in crossed])
    a = np.concatenate([c[1] for c in crossed])
    b = np.concatenate([c[2] for c in crossed])
    if not len(face):
        return None
    t = distance[a] / (distance[a] - distance[b])
    point = coords[a] + t[:, None] * (coords[b] - coords[a])
    key = np.minimum(a, b) * n_nodes + np.maximum(a, b)
    raw_key, raw_point = key.copy(), point.copy()
    if moved is not None:
        at = np.where(moved[1][a], a, np.where(moved[1][b], b, -1))
        snapped = at >= 0
        point[snapped] = coords[at[snapped]]
        key[snapped] = n_nodes * n_nodes + at[snapped]
    # Paired by the crossings as computed: snapped ones can coincide.
    seg_face, seg_p, seg_q = _segments(face, raw_point)

    # The cells holding each crossed face, read cell by cell from NFACE.
    is_crossed = np.zeros(len(faces), dtype=bool)
    is_crossed[seg_face] = True
    pairs: list[tuple[I64, I64]] = []
    for first, last in cells.chunks():
        offsets, cell_face = cells.read(first, last)
        hit = np.flatnonzero(is_crossed[cell_face])
        pairs.append((first + np.searchsorted(offsets, hit, side="right") - 1, cell_face[hit]))
    pair_cell = np.concatenate([p[0] for p in pairs])
    pair_face = np.concatenate([p[1] for p in pairs])
    # Each cell's segments: its crossed faces' segments.
    order = np.argsort(seg_face, kind="stable")
    lo = np.searchsorted(seg_face[order], pair_face)
    hi = np.searchsorted(seg_face[order], pair_face, side="right")
    taken = hi - lo
    cell = np.repeat(pair_cell, taken)
    pick = order[
        np.repeat(lo, taken) + np.arange(taken.sum()) - np.repeat(np.cumsum(taken) - taken, taken)
    ]
    p, q = seg_p[pick], seg_q[pick]
    raw: Segments | None = None
    if moved is not None:
        kept = moved[0][cell]
        cell, p, q = cell[kept], p[kept], q[kept]
        raw = (cell, raw_key[p], raw_key[q], raw_point[p], raw_point[q])
        # Segments between two distinct points.
        mine = key[p] != key[q]
        cell, p, q = cell[mine], p[mine], q[mine]
        if not len(cell):
            return None
    return _polygons(cell, key[p], key[q], point[p], point[q], unit, raw)


def _segments(face: I64, point: F64) -> tuple[I64, I64, I64]:
    # The pieces of each face inside the plane: its crossing points, two by two. A face crossed
    # twice (every convex face) gives one segment; a concave face crossed 2k times gives k, its
    # points paired in order along the line the plane cuts it.
    boundaries = np.flatnonzero(np.diff(face)) + 1
    starts = np.concatenate([[0], boundaries])
    counts = np.diff(np.append(starts, len(face)))
    two = counts == 2
    seg_face = [face[starts[two]]]
    seg_p = [starts[two]]
    seg_q = [starts[two] + 1]
    for s, n in zip(starts[~two].tolist(), counts[~two].tolist(), strict=True):
        if n < 2:
            continue
        index = np.arange(s, s + n)
        direction = point[index] - point[s]
        line = direction[np.argmax(np.linalg.norm(direction, axis=1))]
        index = index[np.argsort(direction @ line, kind="stable")]
        seg_face.append(np.full(n // 2, face[s]))
        seg_p.append(index[0 : 2 * (n // 2) : 2])
        seg_q.append(index[1 : 2 * (n // 2) : 2])
    return np.concatenate(seg_face), np.concatenate(seg_p), np.concatenate(seg_q)


def _polygons(
    cell: I64, kp: I64, kq: I64, pp: F64, pq: F64, unit: F64, raw: "Segments | None" = None
) -> tuple[I64, list[F64], npt.NDArray[np.uint32]] | None:
    # Most cells cut in one convex loop: their points, ordered by angle about their mean, are
    # checked to run along their segments, and fanned, all at once. The others (a concave or
    # multiply connected section, a cell touching the plane) are chained one by one.
    u = np.cross(unit, [1.0, 0.0, 0.0] if abs(unit[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(unit, u)
    lo, hi = np.minimum(kp, kq), np.maximum(kp, kq)
    seg = np.unique(np.stack([cell, lo, hi], axis=1), axis=0)
    ends = np.concatenate([np.stack([cell, kp], 1), np.stack([cell, kq], 1)])
    where = np.concatenate([pp, pq])
    pts, first, degree = np.unique(ends, axis=0, return_index=True, return_counts=True)
    pt_cell, pt_key, pt_xyz = pts[:, 0], pts[:, 1], where[first]
    cells, pt_start = np.unique(pt_cell, return_index=True)
    n_pts = np.diff(np.append(pt_start, len(pt_cell)))
    n_seg = np.diff(np.append(np.searchsorted(seg[:, 0], cells), len(seg)))
    rank = np.searchsorted(cells, pt_cell)
    bad = np.zeros(len(cells), dtype=bool)
    np.logical_or.at(bad, rank, degree != 2)
    bad |= (n_pts != n_seg) | (n_pts < 3)

    # Angle order about each cell's mean, in the plane.
    mean = np.stack([np.bincount(rank, pt_xyz @ axis) for axis in (u, v)], axis=1) / n_pts[:, None]
    flat = np.stack([pt_xyz @ u, pt_xyz @ v], axis=1) - mean[rank]
    order = np.lexsort((np.arctan2(flat[:, 1], flat[:, 0]), rank))
    rank, pt_key, pt_xyz, flat = rank[order], pt_key[order], pt_xyz[order], flat[order]
    nxt = np.arange(len(rank)) + 1
    last = np.append(pt_start[1:], len(rank)) - 1
    nxt[last] = pt_start
    # The order runs along the cell's segments: its consecutive pairs are exactly its segments.
    walked = np.stack(
        [cells[rank], np.minimum(pt_key, pt_key[nxt]), np.maximum(pt_key, pt_key[nxt])], axis=1
    )
    walked = walked[np.lexsort(walked.T[::-1])]
    same = np.zeros(len(cells), dtype=bool)
    if len(walked) == len(seg):
        same_row = (walked == seg).all(axis=1)
        mismatch = np.zeros(len(cells), dtype=bool)
        np.logical_or.at(mismatch, np.searchsorted(cells, seg[:, 0]), ~same_row)
        same = ~mismatch
    else:
        bad[:] = True
    # Convex with a positive area (counter-clockwise about the normal).
    b, c = flat[nxt], flat[nxt[nxt]]
    turn = (b[:, 0] - flat[:, 0]) * (c[:, 1] - flat[:, 1]) - (b[:, 1] - flat[:, 1]) * (
        c[:, 0] - flat[:, 0]
    )
    cross = flat[:, 0] * b[:, 1] - b[:, 0] * flat[:, 1]
    area = 0.5 * np.bincount(rank, cross, minlength=len(cells))
    extent = np.maximum.reduceat(np.abs(flat).max(axis=1), pt_start) * 2
    small = 1e-9 * extent * extent
    concave = np.zeros(len(cells), dtype=bool)
    np.logical_or.at(concave, rank, turn < -1e-12 * (extent * extent)[rank])
    fast = ~bad & same & ~concave & (area > small)

    keep = fast[rank]
    out_cells = cells[fast]
    out_points = list(np.split(pt_xyz[keep], np.cumsum(n_pts[fast])[:-1])) if fast.any() else []
    count = n_pts[fast]
    begin = np.cumsum(count) - count
    fan = np.repeat(np.arange(len(count)), count - 2)
    step = np.arange(len(fan)) - np.repeat(np.cumsum(count - 2) - (count - 2), count - 2)
    head = begin[fan]
    triangles = np.stack([head, head + step + 1, head + step + 2], axis=1)

    rest = ~fast & (area > small) | bad | ~same
    rest_rows = np.isin(cell, cells[rest])
    if raw is not None:
        raw_rows = np.isin(raw[0], cells[rest])
        raw = (
            raw[0][raw_rows],
            raw[1][raw_rows],
            raw[2][raw_rows],
            raw[3][raw_rows],
            raw[4][raw_rows],
        )
    slow = _triangulated(
        _sections(
            (cell[rest_rows], kp[rest_rows], kq[rest_rows], pp[rest_rows], pq[rest_rows]), raw
        ),
        unit,
    )
    if slow is not None:
        out_cells = np.concatenate([out_cells, slow[0]])
        triangles = np.concatenate([triangles, slow[2].astype(np.int64) + int(count.sum())])
        out_points += slow[1]
    if not len(out_cells):
        return None
    return out_cells.astype(np.int64), out_points, triangles.astype(np.uint32)


def _sections(segments: "Segments", raw: "Segments | None") -> list[tuple[int, F64]]:
    # Each cell's segments chained into closed loops; a loop is a section polygon. Where crossings
    # were snapped to on-plane nodes and that leaves a node on more than two segments (a concave
    # cell cut through two of its reflex nodes), the cell is chained from its crossings unsnapped.
    snapped = _by_cell(segments)
    unsnapped = _by_cell(raw) if raw is not None else {}
    out: list[tuple[int, F64]] = []
    for c, mine in snapped.items():
        neighbours, where = _graph(mine)
        if c in unsnapped and any(len(n) != 2 for n in neighbours.values()):
            neighbours, where = _graph(unsnapped[c])
        out += [(c, loop) for loop in _loops(neighbours, where)]
    return out


def _by_cell(segments: "Segments") -> dict[int, list[tuple[int, int, F64, F64]]]:
    cell, kp, kq, pp, pq = segments
    by_cell: dict[int, list[tuple[int, int, F64, F64]]] = defaultdict(list)
    for c, a, b, x, y in zip(cell.tolist(), kp.tolist(), kq.tolist(), pp, pq, strict=True):
        by_cell[c].append((a, b, x, y))
    return by_cell


def _graph(
    segments: list[tuple[int, int, F64, F64]],
) -> tuple[dict[int, list[int]], dict[int, F64]]:
    neighbours: dict[int, list[int]] = defaultdict(list)
    where: dict[int, F64] = {}
    for a, b, x, y in segments:
        if b in neighbours[a]:
            continue
        neighbours[a].append(b)
        neighbours[b].append(a)
        where[a], where[b] = x, y
    return neighbours, where


def _loops(neighbours: dict[int, list[int]], where: dict[int, F64]) -> list[F64]:
    # The closed loops of three points or more, each through nodes on exactly two segments.
    seen: set[int] = set()
    out = []
    for begin in list(neighbours):
        if begin in seen or len(neighbours[begin]) != 2:
            continue
        loop = [begin]
        seen.add(begin)
        previous, current = begin, neighbours[begin][0]
        while current != begin and current not in seen and len(neighbours[current]) == 2:
            loop.append(current)
            seen.add(current)
            a, b = neighbours[current]
            previous, current = current, b if a == previous else a
        if current == begin and len(loop) >= 3:
            out.append(np.array([where[k] for k in loop]))
    return out


def _triangulated(
    sections: list[tuple[int, F64]], unit: F64
) -> tuple[I64, list[F64], npt.NDArray[np.uint32]] | None:
    # Counter-clockwise about the normal; fans for convex sections, ear clipping otherwise. A
    # section of no area (a cell cut along one of its faces' edges) is left out.
    u = np.cross(unit, [1.0, 0.0, 0.0] if abs(unit[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(unit, u)
    cells: list[int] = []
    points: list[F64] = []
    triangles: list[I64] = []
    offset = 0
    for c, loop in sections:
        polygon = loop
        flat = np.stack([polygon @ u, polygon @ v], axis=1)
        x, y = flat[:, 0] - flat[0, 0], flat[:, 1] - flat[0, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        extent = float(np.ptp(flat, axis=0).max())
        if abs(area) <= 1e-9 * extent * extent:
            continue
        if area < 0:
            polygon, flat = polygon[::-1], flat[::-1]
        n = len(polygon)
        a, b, c2 = flat, np.roll(flat, -1, axis=0), np.roll(flat, -2, axis=0)
        turn = (b[:, 0] - a[:, 0]) * (c2[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c2[:, 0] - a[:, 0]
        )
        if (turn >= -1e-12 * extent * extent).all():
            local = np.stack(
                [np.zeros(n - 2, np.int64), np.arange(1, n - 1), np.arange(2, n)], axis=1
            )
        else:
            local = _ear_clip(np.arange(n), polygon, unit)
        cells.append(c)
        points.append(polygon)
        triangles.append(local + offset)
        offset += n
    if not cells:
        return None
    return np.array(cells, np.int64), points, np.concatenate(triangles).astype(np.uint32)
