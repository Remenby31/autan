import json
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from autan_ui.surface import encode
from autan_ui.wall import polygon_zone

SCHEMA = "autan.surface/1"
DTYPES: dict[str, np.dtype[Any]] = {"f64": np.dtype("<f8"), "u32": np.dtype("<u4")}


def mesh_surface(data: bytes, mesh_info: dict[str, Any]) -> bytes:
    """The boundary of a mesh, an `autan.surface/1` file (mesh.md#boundary-surface) written by
    `autan-mesh-info --surface`, as an `autan-ui.surface/1` message with one zone per boundary
    zone, checked against the mesh's MeshInfo; ValueError when either contract is broken."""
    header, buffers = _read(data)
    if header.get("schema") != SCHEMA:
        msg = f"not an {SCHEMA} file"
        raise ValueError(msg)
    if header.get("source", {}).get("sha256") != mesh_info["source"]["sha256"]:
        msg = "the surface is not of this mesh: its source.sha256 differs from the MeshInfo's"
        raise ValueError(msg)
    zones = header.get("zones")
    expected = [(z["name"], z["faces"]) for z in mesh_info["boundary_zones"]]
    if not isinstance(zones, list) or [(z.get("name"), z.get("faces")) for z in zones] != expected:
        msg = "the surface's zones are not the MeshInfo's boundary zones"
        raise ValueError(msg)
    positions = buffers["positions"].astype(np.float64).reshape(-1, 3)
    offsets = buffers["face_offsets"].astype(np.int64)
    nodes = buffers["face_nodes"].astype(np.int64)
    faces = header.get("faces")
    if (
        len(positions) != header.get("nodes")
        or len(offsets) != (faces if isinstance(faces, int) else -1) + 1
        or sum(z["faces"] for z in zones) != faces
        or offsets[0] != 0
        or offsets[-1] != len(nodes)
        or np.any(np.diff(offsets) < 3)
        or (len(nodes) and (nodes.min() < 0 or nodes.max() >= len(positions)))
    ):
        msg = "the surface's counts, offsets or node indices are inconsistent"
        raise ValueError(msg)
    first = 0
    for zone in zones:
        if zone.get("first_face") != first:
            msg = f"zone {zone['name']!r} does not start where the previous one ends"
            raise ValueError(msg)
        first += zone["faces"]
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


def _read(data: bytes) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    if len(data) < 4:
        msg = "the surface file is truncated"
        raise ValueError(msg)
    (length,) = struct.unpack_from("<I", data)
    start = 4 + length
    if start % 8 or start > len(data):
        msg = "the surface header is not padded to 8 bytes or overruns the file"
        raise ValueError(msg)
    header = json.loads(data[4:start].decode())
    if not isinstance(header, dict):
        msg = "the surface header is not a JSON object"
        raise ValueError(msg)  # noqa: TRY004 -- a broken file contract, as every other check
    buffers = {}
    for buffer in header.get("buffers", []):
        dtype = DTYPES.get(buffer.get("dtype"))
        offset, count = buffer.get("offset"), buffer.get("length")
        if (
            dtype is None
            or not isinstance(offset, int)
            or not isinstance(count, int)
            or offset % 8
            or start + offset + count * dtype.itemsize > len(data)
        ):
            msg = f"surface buffer {buffer.get('name')!r} is malformed or overruns the file"
            raise ValueError(msg)
        buffers[buffer["name"]] = np.frombuffer(data, dtype, count, start + offset)
    if set(buffers) != {"positions", "face_offsets", "face_nodes"}:
        msg = "the surface has not exactly the buffers positions, face_offsets and face_nodes"
        raise ValueError(msg)
    return header, buffers
