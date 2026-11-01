import json
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from autan_ui.surface import encode
from autan_ui.wall import polygon_zone

SCHEMA = "autan.surface/1"
BUFFERS: dict[str, np.dtype[Any]] = {
    "positions": np.dtype("<f8"),
    "face_offsets": np.dtype("<u4"),
    "face_nodes": np.dtype("<u4"),
}
DTYPE_NAMES = {"positions": "f64", "face_offsets": "u32", "face_nodes": "u32"}


def _broken(message: str) -> ValueError:
    return ValueError(f"the surface breaks {SCHEMA}: {message}")


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def mesh_surface(data: bytes, mesh_info: dict[str, Any]) -> bytes:
    """The boundary of a mesh, an `autan.surface/1` file (mesh.md#boundary-surface) written by
    `autan-mesh-info --surface`, as an `autan-ui.surface/1` message with one zone per boundary
    zone, checked against the mesh's MeshInfo; ValueError when either contract is broken."""
    header, buffers = _read(data)
    source = header.get("source")
    if not isinstance(source, dict) or source.get("sha256") != mesh_info["source"]["sha256"]:
        msg = "the surface is not of this mesh: its source.sha256 differs from the MeshInfo's"
        raise ValueError(msg)
    zones = _zones(header, mesh_info)
    positions = buffers["positions"].reshape(-1, 3)
    offsets = buffers["face_offsets"].astype(np.int64)
    nodes = buffers["face_nodes"].astype(np.int64)
    faces = header.get("faces")
    if not isinstance(faces, int) or not _count(faces):
        msg = "its face count is not a count"
        raise _broken(msg)
    if (
        len(positions) != header.get("nodes")
        or len(offsets) != faces + 1
        or sum(z["faces"] for z in zones) != faces
        or offsets[0] != 0
        or offsets[-1] != len(nodes)
        or np.any(np.diff(offsets) < 3)
        or (len(nodes) and nodes.max() >= len(positions))
    ):
        msg = "its counts, offsets or node indices are inconsistent"
        raise _broken(msg)
    if not np.isfinite(positions).all():
        msg = "a node position is not finite"
        raise _broken(msg)
    _check_polygons(offsets, nodes)
    origin = (positions.min(axis=0) + positions.max(axis=0)) / 2 if len(positions) else np.zeros(3)
    return encode(
        [
            polygon_zone(
                zone["name"],
                np.arange(zone["first_face"], zone["first_face"] + zone["faces"]),
                offsets,
                nodes,
                positions,
                {},
                origin,
            )
            for zone in zones
            if zone["faces"]
        ],
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
    )


def _zones(header: dict[str, Any], mesh_info: dict[str, Any]) -> list[dict[str, Any]]:
    # The MeshInfo's boundary zones, in order and with the same fields, contiguous in faces.
    zones = header.get("zones")
    keys = ("name", "type", "native_type", "faces")
    expected = [{k: z.get(k) for k in keys} for z in mesh_info["boundary_zones"]]
    if (
        not isinstance(zones, list)
        or not all(isinstance(z, dict) for z in zones)
        or [{k: z.get(k) for k in keys} for z in zones] != expected
    ):
        msg = "its zones are not the MeshInfo's boundary zones"
        raise _broken(msg)
    first = 0
    for zone in zones:
        if not _count(zone["faces"]) or zone.get("first_face") != first:
            msg = f"zone {zone['name']!r} does not start where the previous one ends"
            raise _broken(msg)
        first += zone["faces"]
    return zones


def _check_polygons(offsets: npt.NDArray[np.int64], nodes: npt.NDArray[np.int64]) -> None:
    # Each face a polygon of distinct consecutive nodes; the boundary closed and consistently
    # oriented: every edge traversed as often in each direction (mesh.md#boundary-surface).
    count = np.diff(offsets)
    face = np.repeat(np.arange(len(count)), count)
    position = np.arange(len(nodes)) - np.repeat(offsets[:-1], count)
    following = nodes[offsets[:-1][face] + (position + 1) % count[face]]
    if np.any(nodes == following):
        msg = "a face repeats a node along its boundary"
        raise _broken(msg)
    width = int(nodes.max()) + 1 if len(nodes) else 1
    forward = nodes * width + following
    backward = following * width + nodes
    edges, there = np.unique(forward, return_counts=True)
    back = dict(zip(*np.unique(backward, return_counts=True), strict=True))
    if any(back.get(edge, 0) != n for edge, n in zip(edges, there, strict=True)):
        msg = "the boundary is not closed with a consistent orientation"
        raise _broken(msg)


def _read(data: bytes) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    if len(data) < 4:
        msg = "the file is truncated"
        raise _broken(msg)
    (length,) = struct.unpack_from("<I", data)
    start = 4 + length
    if start % 8 or start > len(data):
        msg = "the header is not padded to 8 bytes or overruns the file"
        raise _broken(msg)
    try:
        header = json.loads(data[4:start].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"the header is not JSON: {error}"
        raise _broken(msg) from error
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        msg = f"its header does not name {SCHEMA}"
        raise _broken(msg)
    described = header.get("buffers")
    if not isinstance(described, list) or not all(isinstance(b, dict) for b in described):
        msg = "its buffers are not a list of descriptors"
        raise _broken(msg)
    names = [b.get("name") for b in described]
    if sorted(map(str, names)) != sorted(BUFFERS):
        msg = "its buffers are not exactly positions, face_offsets and face_nodes"
        raise _broken(msg)
    buffers = {}
    for buffer in described:
        name, offset, count = buffer["name"], buffer.get("offset"), buffer.get("length")
        dtype = BUFFERS[name]
        if (
            buffer.get("dtype") != DTYPE_NAMES[name]
            or not _count(offset)
            or not _count(count)
            or offset % 8
            or start + offset + count * dtype.itemsize > len(data)
        ):
            msg = f"buffer {name!r} has the wrong dtype or overruns the file"
            raise _broken(msg)
        buffers[name] = np.frombuffer(data, dtype, count, start + offset)
    if len(buffers["positions"]) % 3:
        msg = "positions do not hold three coordinates per node"
        raise _broken(msg)
    return header, buffers
