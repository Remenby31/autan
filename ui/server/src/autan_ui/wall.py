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
        zones.append(_zone(name, faces, start, connectivity, coords, values, origin))
    if not zones:
        msg = "the solution has no wall face"
        raise ValueError(msg)
    return encode(zones, origin=(float(origin[0]), float(origin[1]), float(origin[2])))


def _zone(
    name: str,
    faces: npt.NDArray[np.int64],
    start: npt.NDArray[np.int64],
    connectivity: npt.NDArray[np.int64],
    coords: npt.NDArray[np.float64],
    face_values: dict[str, npt.NDArray[np.float64]],
    origin: npt.NDArray[np.float64],
) -> Zone:
    """Faces triangulated (`_triangles`), node values averaged from face values with the face
    areas of results.md (half the norm of the summed edge cross products), positions relative
    to `origin` computed in FP64 before the FP32 cast."""
    first, count = start[faces], start[faces + 1] - start[faces]
    polygons = [connectivity[s : s + n] for s, n in zip(first, count, strict=True)]
    vertices = np.concatenate(polygons)
    vertex_face = np.repeat(np.arange(len(faces)), count)
    following = np.concatenate([np.roll(p, -1) for p in polygons])
    cross = np.cross(coords[vertices], coords[following])
    vector = np.stack(
        [np.bincount(vertex_face, cross[:, k], minlength=len(faces)) for k in range(3)], axis=1
    )
    face_area = 0.5 * np.linalg.norm(vector, axis=1)

    nodes, local = np.unique(vertices, return_inverse=True)
    weight = np.bincount(local, face_area[vertex_face], minlength=len(nodes))
    fields = {
        key: (
            np.bincount(local, (face_area * value)[vertex_face], minlength=len(nodes))
            / np.where(weight > 0, weight, 1)
        ).astype(np.float32)
        for key, value in face_values.items()
    }
    triangles = np.concatenate(
        [_triangles(p, coords, n) for p, n in zip(polygons, vector, strict=True)]
    )
    lookup = np.searchsorted(nodes, triangles)
    positions = (coords[nodes] - origin).astype(np.float32)
    return Zone(name, positions, lookup.astype(np.uint32), fields)


def _triangles(
    polygon: npt.NDArray[np.int64], coords: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Triangles of one polygon with its winding: a fan when it is convex, ear clipping in the
    plane of its Newell normal otherwise (a fan crosses the notch of a concave face)."""
    n = len(polygon)
    if n == 3:
        return polygon.reshape(1, 3)
    axis = int(np.argmax(np.abs(normal)))
    u, v = [k for k in range(3) if k != axis]
    flat = coords[polygon][:, [u, v]]
    sign = 1.0 if normal[axis] >= 0 else -1.0

    def turn(a: int, b: int, c: int) -> float:
        (ax, ay), (bx, by), (cx, cy) = flat[a], flat[b], flat[c]
        return float(sign * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)))

    if all(turn(k, (k + 1) % n, (k + 2) % n) >= 0 for k in range(n)):
        return np.stack([np.full(n - 2, polygon[0]), polygon[1:-1], polygon[2:]], axis=1)
    remaining = list(range(n))
    out = []
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
            break
    if len(remaining) == 3:
        out.append((remaining[0], remaining[1], remaining[2]))
    triangles: npt.NDArray[np.int64] = polygon[np.array(out)]
    return triangles
