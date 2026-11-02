"""Plane slices of a solution.cgns (results.md#solutioncgns) for the viewer."""

import os

import h5py
import numpy as np
import numpy.typing as npt

from autan_ui.confine import Folder
from autan_ui.surface import Zone, encode
from autan_ui.wall import _data

I64 = npt.NDArray[np.int64]
F64 = npt.NDArray[np.float64]


def slice_surface(folder: Folder, origin: F64, normal: F64) -> bytes:
    """The cut of every cell of the run's solution by the plane through `origin` normal to
    `normal`, as an `autan-ui.surface/1` message with one zone, `slice`: one polygon per cell
    the plane crosses, carrying that cell's `FlowSolution` values (cell-centred, so constant on
    the polygon). ValueError when the plane meets no cell or the normal is zero."""
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length == 0 or not np.isfinite(origin).all():
        msg = "the plane needs a finite origin and a non-zero normal"
        raise ValueError(msg)
    unit = normal / length
    with os.fdopen(folder.open("solution.cgns"), "rb") as file, h5py.File(file, "r") as tree:
        zone = tree["Base"]["Zone"]
        coords = np.stack(
            [_data(zone["GridCoordinates"][f"Coordinate{a}"]) for a in "XYZ"], axis=1
        ).astype(np.float64)
        ngon, nface = zone["NGonElements"], zone["NFaceElements"]
        face_start = _data(ngon["ElementStartOffset"]).astype(np.int64)
        face_nodes = _data(ngon["ElementConnectivity"]).astype(np.int64) - 1
        cell_start = _data(nface["ElementStartOffset"]).astype(np.int64)
        # Signed face element numbers (results.md): faces are elements 1..F.
        cell_faces = np.abs(_data(nface["ElementConnectivity"]).astype(np.int64)) - 1
        solution = zone["FlowSolution"]
        fields = {
            name: _data(array).astype(np.float64)
            for name, array in solution.items()
            if isinstance(array, h5py.Group) and name != "GridLocation"
        }
    polygons = _cut(coords, face_start, face_nodes, cell_start, cell_faces, origin, unit)
    if polygons is None:
        msg = "the plane meets no cell of the mesh"
        raise ValueError(msg)
    cells, points, start = polygons
    count = np.diff(start)
    # Fan triangles from each polygon's first point; nodes are per polygon, so each carries
    # its own cell's values.
    polygon = np.repeat(np.arange(len(count)), count - 2)
    step = np.arange(len(polygon)) - np.repeat(np.cumsum(count - 2) - (count - 2), count - 2)
    first = start[:-1][polygon]
    triangles = np.stack([first, first + step + 1, first + step + 2], axis=1)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    node_cell = np.repeat(cells, count)
    zone_out = Zone(
        "slice",
        (points - centre).astype(np.float32),
        triangles.astype(np.uint32),
        {name: values[node_cell].astype(np.float32) for name, values in fields.items()},
    )
    return encode([zone_out], origin=(float(centre[0]), float(centre[1]), float(centre[2])))


def _cut(
    coords: F64,
    face_start: I64,
    face_nodes: I64,
    cell_start: I64,
    cell_faces: I64,
    origin: F64,
    unit: F64,
) -> tuple[I64, F64, I64] | None:
    # The polygon each crossed cell leaves on the plane: the points where the plane crosses the
    # cell's edges, ordered by angle around their mean in the plane (the cells of a finite
    # volume mesh are convex enough for this to give their section).
    size = float(np.ptp(coords, axis=0).max()) or 1.0
    distance = (coords - origin) @ unit
    # A node on the plane is moved just off it, so that every crossing is an edge's interior.
    distance[np.abs(distance) < 1e-12 * size] = 1e-12 * size
    above = distance > 0

    count = np.diff(face_start)
    face = np.repeat(np.arange(len(count)), count)
    position = np.arange(len(face_nodes)) - np.repeat(face_start[:-1], count)
    following = face_nodes[face_start[:-1][face] + (position + 1) % count[face]]
    crossing = above[face_nodes] != above[following]
    a, b, edge_face = face_nodes[crossing], following[crossing], face[crossing]
    if not len(a):
        return None
    t = distance[a] / (distance[a] - distance[b])
    point = coords[a] + t[:, None] * (coords[b] - coords[a])
    width = len(coords)
    edge = np.minimum(a, b) * width + np.maximum(a, b)

    # Each cell's crossing edges, through its faces; an edge shared by two of its faces once.
    order = np.argsort(edge_face, kind="stable")
    by_face = edge_face[order]
    lo = np.searchsorted(by_face, np.arange(len(count)))
    hi = np.searchsorted(by_face, np.arange(len(count)), side="right")
    per_face = hi - lo
    cell_count = np.diff(cell_start)
    cell_of = np.repeat(np.arange(len(cell_count)), cell_count)
    taken = per_face[cell_faces]
    pair_cell = np.repeat(cell_of, taken)
    first = np.repeat(lo[cell_faces], taken)
    within = np.arange(len(pair_cell)) - np.repeat(np.cumsum(taken) - taken, taken)
    crossing_edge = order[first + within]
    key = pair_cell * (int(edge.max()) + 1) + edge[crossing_edge]
    _, keep = np.unique(key, return_index=True)
    cell = pair_cell[keep]
    pts = point[crossing_edge[keep]]

    # Polygons of at least three points, each ordered by angle in the plane.
    corners = np.bincount(cell, minlength=len(cell_count))
    keep_cell = corners[cell] >= 3
    cell, pts = cell[keep_cell], pts[keep_cell]
    if not len(cell):
        return None
    mean = (
        np.stack(
            [np.bincount(cell, pts[:, k], minlength=len(cell_count)) for k in range(3)], axis=1
        )
        / np.maximum(corners, 1)[:, None]
    )
    u = np.cross(unit, [1.0, 0.0, 0.0] if abs(unit[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(unit, u)
    relative = pts - mean[cell]
    angle = np.arctan2(relative @ v, relative @ u)
    ordered = np.lexsort((angle, cell))
    cell, pts = cell[ordered], pts[ordered]
    cells, starts = np.unique(cell, return_index=True)
    start = np.append(starts, len(cell)).astype(np.int64)
    return cells.astype(np.int64), pts, start
