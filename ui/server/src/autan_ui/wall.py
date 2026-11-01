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
        zones.append(_zone(name, faces, start, connectivity, coords, values))
    if not zones:
        msg = "the solution has no wall face"
        raise ValueError(msg)
    low, high = coords.min(axis=0), coords.max(axis=0)
    origin = (low + high) / 2
    for z in zones:
        z.positions -= origin.astype(np.float32)
    return encode(zones, origin=(float(origin[0]), float(origin[1]), float(origin[2])))


def _zone(
    name: str,
    faces: npt.NDArray[np.int64],
    start: npt.NDArray[np.int64],
    connectivity: npt.NDArray[np.int64],
    coords: npt.NDArray[np.float64],
    face_values: dict[str, npt.NDArray[np.float64]],
) -> Zone:
    """Fan-triangulated faces with node values averaged from face values by face area."""
    first, count = start[faces], start[faces + 1] - start[faces]
    owner = np.repeat(np.arange(len(faces)), count - 2)
    corner = np.arange(len(owner)) - np.repeat(np.cumsum(count - 2) - (count - 2), count - 2)
    a = connectivity[first[owner]]
    b = connectivity[first[owner] + corner + 1]
    c = connectivity[first[owner] + corner + 2]
    area = 0.5 * np.linalg.norm(np.cross(coords[b] - coords[a], coords[c] - coords[a]), axis=1)
    face_area = np.bincount(owner, area, minlength=len(faces))

    vertex_face = np.repeat(np.arange(len(faces)), count)
    vertices = np.concatenate([connectivity[s : s + n] for s, n in zip(first, count, strict=True)])
    nodes, local = np.unique(vertices, return_inverse=True)
    weight = np.bincount(local, face_area[vertex_face], minlength=len(nodes))
    fields = {
        key: (
            np.bincount(local, (face_area * value)[vertex_face], minlength=len(nodes))
            / np.where(weight > 0, weight, 1)
        ).astype(np.float32)
        for key, value in face_values.items()
    }
    lookup = np.searchsorted(nodes, np.stack([a, b, c], axis=1))
    return Zone(name, coords[nodes].astype(np.float32), lookup.astype(np.uint32), fields)
