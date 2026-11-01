"""Wall surfaces of a solution.cgns (results.md#solutioncgns) for the viewer."""

import json
import os
import tomllib
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from autan.errors import Location
from autan.schema import CaseSpec
from autan.solver_input import freestream_state

from autan_ui.confine import Folder
from autan_ui.surface import Zone, encode


def _data(group: h5py.Group) -> npt.NDArray[Any]:
    array: npt.NDArray[Any] = np.asarray(group[" data"][()])
    return array.T if array.ndim > 1 else array


def _text(group: h5py.Group) -> str:
    return bytes(np.asarray(_data(group), dtype=np.int8).astype(np.uint8)).decode("ascii")


def wall_surface(folder: Folder) -> bytes:
    """The wall and slip zones of the run's solution as an `autan-ui.surface/1` message.

    Fields per node, from `WallSolution` face values averaged with face-area weights: `Cp`
    = (p - p_inf)/q_inf, `Cf` = |tau|/q_inf (0 on slip faces), `Pressure` (Pa), with q_inf =
    rho_inf V_inf^2 / 2 from
    the stored case's freestream. ValueError when the solution has no wall face.
    """
    case = CaseSpec.from_document(
        tomllib.loads(folder.read("case.toml").decode()), Location(file="case.toml")
    )
    state = freestream_state(case)
    q_inf = 0.5 * state.density * state.speed**2
    with os.fdopen(folder.open("solution.cgns"), "rb") as file, h5py.File(file, "r") as tree:
        base = tree["Base"]
        zone = base["Zone"]
        names = json.loads(_text(base["ZoneNames"]))["boundary"]
        coords = np.stack(
            [_data(zone["GridCoordinates"][f"Coordinate{a}"]) for a in "XYZ"], axis=1
        ).astype(np.float64)
        ngon = zone["NGonElements"]
        start = _data(ngon["ElementStartOffset"]).astype(np.int64)
        connectivity = _data(ngon["ElementConnectivity"]).astype(np.int64) - 1
        wall = zone["WallSolution"]
        wall_faces = _data(wall["PointList"]).astype(np.int64).ravel() - 1
        pressure = _data(wall["Pressure"]).astype(np.float64)
        shear = np.stack([_data(wall[f"WallShearStress{a}"]) for a in "XYZ"], axis=1).astype(
            np.float64
        )
        regions = [
            (names.get(node_name, node_name), _data(bc["PointList"]).astype(np.int64).ravel() - 1)
            for node_name, bc in zone["ZoneBC"].items()
            if isinstance(bc, h5py.Group) and "PointList" in bc
        ]
    row = {int(face): k for k, face in enumerate(wall_faces)}
    origin = (coords.min(axis=0) + coords.max(axis=0)) / 2
    zones = []
    for name, faces in sorted(regions):
        if not len(faces) or any(int(f) not in row for f in faces):
            continue
        rows = np.array([row[int(f)] for f in faces])
        values = {
            "Cp": (pressure[rows] - state.pressure) / q_inf,
            "Cf": np.linalg.norm(shear[rows], axis=1) / q_inf,
            "Pressure": pressure[rows],
        }
        zones.append(polygon_zone(name, faces, start, connectivity, coords, values, origin))
    if not zones:
        msg = "the solution has no wall face"
        raise ValueError(msg)
    return encode(zones, origin=(float(origin[0]), float(origin[1]), float(origin[2])))


def polygon_zone(
    name: str,
    faces: npt.NDArray[np.int64],
    start: npt.NDArray[np.int64],
    connectivity: npt.NDArray[np.int64],
    coords: npt.NDArray[np.float64],
    face_values: dict[str, npt.NDArray[np.float64]],
    origin: npt.NDArray[np.float64],
) -> Zone:
    """Faces triangulated (`_triangulate`), node values averaged from face values with the face
    areas of results.md (half the norm of the summed edge cross products), positions relative
    to `origin` computed in FP64 before the FP32 cast."""
    first, count = start[faces], start[faces + 1] - start[faces]
    vertex_face = np.repeat(np.arange(len(faces)), count)
    position = np.arange(len(vertex_face)) - np.repeat(np.cumsum(count) - count, count)
    vertices = connectivity[first[vertex_face] + position]
    following = connectivity[first[vertex_face] + (position + 1) % count[vertex_face]]
    # Relative to each face's first vertex: absolute coordinates far from the origin would
    # cancel in the cross products.
    anchor = coords[connectivity[first]][vertex_face]
    cross = np.cross(coords[vertices] - anchor, coords[following] - anchor)
    normal = np.stack(
        [np.bincount(vertex_face, cross[:, k], minlength=len(faces)) for k in range(3)], axis=1
    )
    face_area = 0.5 * np.linalg.norm(normal, axis=1)

    nodes, local = np.unique(vertices, return_inverse=True)
    weight = np.bincount(local, face_area[vertex_face], minlength=len(nodes))
    fields = {
        key: (
            np.bincount(local, (face_area * value)[vertex_face], minlength=len(nodes))
            / np.where(weight > 0, weight, 1)
        ).astype(np.float32)
        for key, value in face_values.items()
    }
    triangles = _triangulate(first, count, connectivity, coords, normal)
    lookup = np.searchsorted(nodes, triangles)
    positions = (coords[nodes] - origin).astype(np.float32)
    return Zone(name, positions, lookup.astype(np.uint32), fields)


# The two axes kept when projecting along axis k, in cyclic order so the projection keeps the
# handedness of the normal's k component.
PLANE = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


def _triangulate(
    first: npt.NDArray[np.int64],
    count: npt.NDArray[np.int64],
    connectivity: npt.NDArray[np.int64],
    coords: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Triangles of every face with its winding: batched fans for faces that are convex in the
    plane of their Newell normal, ear clipping for the concave ones (a fan would cross their
    notch)."""
    out = []
    for n in np.unique(count):
        group = np.flatnonzero(count == n)
        polygons = connectivity[first[group][:, None] + np.arange(n)]
        if n == 3:
            out.append(polygons)
            continue
        convex = np.ones(len(group), dtype=bool)
        for axis, (u, v) in PLANE.items():
            mine = np.argmax(np.abs(normal[group]), axis=1) == axis
            if not mine.any():
                continue
            flat = coords[polygons[mine]][:, :, [u, v]]
            sign = np.where(normal[group][mine, axis] >= 0, 1.0, -1.0)[:, None]
            a, b, c = flat, np.roll(flat, -1, axis=1), np.roll(flat, -2, axis=1)
            turn = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
                c[..., 0] - a[..., 0]
            )
            convex[mine] = (sign * turn >= 0).all(axis=1)
        fans = polygons[convex]
        if len(fans):
            out.append(
                np.stack(
                    [
                        np.repeat(fans[:, 0], n - 2),
                        fans[:, 1:-1].ravel(),
                        fans[:, 2:].ravel(),
                    ],
                    axis=1,
                )
            )
        for polygon, face_normal in zip(polygons[~convex], normal[group][~convex], strict=True):
            out.append(_ear_clip(polygon, coords, face_normal))
    return np.concatenate(out)


def _ear_clip(
    polygon: npt.NDArray[np.int64], coords: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Ear clipping of one concave polygon in the plane of its Newell normal."""
    axis = int(np.argmax(np.abs(normal)))
    u, v = PLANE[axis]
    flat = coords[polygon][:, [u, v]] - coords[polygon[0]][[u, v]]
    sign = 1.0 if normal[axis] >= 0 else -1.0

    def turn(a: int, b: int, c: int) -> float:
        (ax, ay), (bx, by), (cx, cy) = flat[a], flat[b], flat[c]
        return float(sign * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))

    remaining = list(range(len(polygon)))
    out: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        m = len(remaining)
        for k in range(m):
            a, b, c = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
            if turn(a, b, c) <= 0:
                continue
            inside = any(
                turn(a, b, q) >= 0 and turn(b, c, q) >= 0 and turn(c, a, q) >= 0
                for q in remaining
                if q not in (a, b, c)
            )
            if not inside:
                out.append((a, b, c))
                remaining.pop(k)
                break
        else:  # degenerate polygon: no ear, close it with a fan
            out.extend((remaining[0], remaining[j], remaining[j + 1]) for j in range(1, m - 1))
            remaining = []
    if len(remaining) == 3:
        out.append((remaining[0], remaining[1], remaining[2]))
    triangles: npt.NDArray[np.int64] = polygon[np.array(out)]
    return triangles
