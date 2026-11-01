import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from autan_ui.app import create_app
from fastapi.testclient import TestClient

# Written by `autan-mesh-info --surface` and `--json` from tests/data/mesh/hybrid in four
# formats (ui/scripts/surface-fixtures.sh): the producer's own output.
DATA = Path(__file__).parent / "data" / "surface"
FORMATS = ["hybrid-ascii-msh", "hybrid-cgns", "hybrid-fluent-msh", "polyMesh"]


def _client(tmp_path: Path, name: str, surface: bytes | None = None) -> TestClient:
    meshes = tmp_path / "meshes"
    meshes.mkdir(exist_ok=True)
    shutil.copy(DATA / f"{name}.meshinfo.json", meshes / f"{name}.meshinfo.json")
    (meshes / f"{name}.surface").write_bytes(
        surface if surface is not None else (DATA / f"{name}.surface").read_bytes()
    )
    app = create_app(tmp_path / "runs", static=None, hosts=("testserver",), meshes_root=meshes)
    return TestClient(app)


def _message(data: bytes) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    (length,) = struct.unpack_from("<I", data)
    header = json.loads(data[4 : 4 + length])
    dtypes = {"f32": np.float32, "u32": np.uint32}
    buffers = {
        b["name"]: np.frombuffer(data, dtypes[b["dtype"]], b["length"], 4 + length + b["offset"])
        for b in header["buffers"]
    }
    return header, buffers


def _polygons(name: str) -> tuple[npt.NDArray[np.int64], list[dict[str, Any]]]:
    data = (DATA / f"{name}.surface").read_bytes()
    (length,) = struct.unpack_from("<I", data)
    header = json.loads(data[4 : 4 + length])
    offsets = next(b for b in header["buffers"] if b["name"] == "face_offsets")
    starts = np.frombuffer(data, "<u4", offsets["length"], 4 + length + offsets["offset"])
    wide = starts.astype(np.int64)
    counts: npt.NDArray[np.int64] = wide[1:] - wide[:-1]
    return counts, header["zones"]


@pytest.mark.parametrize("name", FORMATS)
def test_a_mesh_boundary_reaches_the_viewer_zone_by_zone(tmp_path: Path, name: str) -> None:
    response = _client(tmp_path, name).get(f"/api/meshes/{name}/surface")

    assert response.status_code == 200
    header, buffers = _message(response.content)
    mesh_info = json.loads((DATA / f"{name}.meshinfo.json").read_text())
    counts, zones = _polygons(name)
    # Every boundary zone, in MeshInfo order, with a triangle per polygon side beyond two.
    assert [z["name"] for z in header["zones"]] == [
        z["name"] for z in mesh_info["boundary_zones"] if z["faces"]
    ]
    for shown, zone in zip(header["zones"], [z for z in zones if z["faces"]], strict=True):
        first = zone["first_face"]
        assert shown["triangles"] == int((counts[first : first + zone["faces"]] - 2).sum())
    # tests/data/mesh/README.md: the hybrid mesh fills the unit cube [0, 1]³. Its boundary has
    # area 6 and, oriented outward, ∮ x·n dA = 3V = 3 (divergence theorem).
    points = buffers["positions"].astype(np.float64).reshape(-1, 3) + header["origin"]
    a, b, c = (points[buffers["indices"].reshape(-1, 3)[:, k]] for k in range(3))
    area = np.cross(b - a, c - a) / 2
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(points.min(axis=0), 0, atol=tolerance)
    np.testing.assert_allclose(points.max(axis=0), 1, atol=tolerance)
    assert np.linalg.norm(area, axis=1).sum() == pytest.approx(6, rel=tolerance)
    assert np.einsum("ij,ij->", (a + b + c) / 3, area) == pytest.approx(3, rel=tolerance)


def _parts(name: str) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    data = (DATA / f"{name}.surface").read_bytes()
    (length,) = struct.unpack_from("<I", data)
    header = json.loads(data[4 : 4 + length])
    dtypes = {"f64": "<f8", "u32": "<u4"}
    arrays = {
        b["name"]: np.frombuffer(data, dtypes[b["dtype"]], b["length"], 4 + length + b["offset"])
        for b in header["buffers"]
    }
    return header, arrays


def _pack(header: dict[str, Any], arrays: dict[str, npt.NDArray[Any]]) -> bytes:
    # The layout of mesh.md#boundary-surface, with the buffers given.
    body, layout = b"", []
    for b in header["buffers"]:
        array = arrays.get(b["name"], np.zeros(0))
        layout.append({**b, "offset": len(body), "length": array.size})
        body += array.tobytes() + b"\0" * (-array.nbytes % 8)
    text = json.dumps({**header, "buffers": layout}).encode()
    text += b" " * (-(4 + len(text)) % 8)
    return struct.pack("<I", len(text)) + text + body


def _mutated(change: str) -> bytes:
    header, arrays = _parts("polyMesh")
    arrays = {k: v.copy() for k, v in arrays.items()}
    if change == "wrong dtype":
        header["buffers"][0]["dtype"] = "u32" if header["buffers"][0]["dtype"] == "f64" else "f64"
    elif change == "duplicate buffer":
        header["buffers"].append(dict(header["buffers"][0]))
    elif change == "null source":
        header["source"] = None
    elif change == "another mesh's hash":
        header["source"]["sha256"] = "0" * 64
    elif change == "null zone":
        header["zones"][0] = None
    elif change == "zone type":
        header["zones"][0]["type"] = "inlet"
    elif change == "infinite position":
        arrays["positions"][0] = np.inf
    elif change == "repeated node":
        nodes, offsets = arrays["face_nodes"], arrays["face_offsets"]
        nodes[offsets[0] + 1] = nodes[offsets[0]]
    elif change == "retraced node":
        nodes, offsets = arrays["face_nodes"], arrays["face_offsets"]
        first = nodes[offsets[0] : offsets[1]].copy()
        spiked = np.concatenate([first[:2], first[:2], first[2:]]).astype(np.uint32)
        arrays["face_nodes"] = np.concatenate([nodes[: offsets[0]], spiked, nodes[offsets[1] :]])
        arrays["face_offsets"] = np.concatenate([offsets[:1], offsets[1:] + 2]).astype(np.uint32)
    elif change == "collapsed positions":
        arrays["positions"][:] = 0
    elif change == "every face reversed":
        nodes, offsets = arrays["face_nodes"], arrays["face_offsets"]
        for f in range(len(offsets) - 1):
            nodes[offsets[f] : offsets[f + 1]] = nodes[offsets[f] : offsets[f + 1]][::-1].copy()
    elif change == "float first face":
        header["zones"][0]["first_face"] = 0.0
    elif change == "reversed face":
        nodes, offsets = arrays["face_nodes"], arrays["face_offsets"]
        nodes[offsets[0] : offsets[1]] = nodes[offsets[0] : offsets[1]][::-1].copy()
    return _pack(header, arrays)


@pytest.mark.parametrize(
    "change",
    [
        "wrong dtype",
        "duplicate buffer",
        "null source",
        "another mesh's hash",
        "null zone",
        "zone type",
        "infinite position",
        "repeated node",
        "reversed face",
        "retraced node",
        "collapsed positions",
        "every face reversed",
        "float first face",
    ],
)
def test_a_surface_that_breaks_its_contract_is_refused(tmp_path: Path, change: str) -> None:
    assert (
        _client(tmp_path, "polyMesh", _pack(*_parts("polyMesh")))
        .get("/api/meshes/polyMesh/surface")
        .status_code
        == 200
    )  # The repacked original passes: the change is what is refused.

    response = _client(tmp_path, "polyMesh", _mutated(change)).get("/api/meshes/polyMesh/surface")

    assert response.status_code == 422, change


def test_a_surface_beside_an_invalid_meshinfo_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path, "polyMesh")
    (tmp_path / "meshes" / "polyMesh.meshinfo.json").write_text('{"schema": "autan.mesh-info/1"}')

    assert client.get("/api/meshes/polyMesh/surface").status_code in (404, 422)


def test_a_thin_mesh_keeps_its_small_faces(tmp_path: Path) -> None:
    # The unit cube flattened to 1e-6 × 1e-6 × 1: every face keeps an area and the boundary
    # its orientation, though the smallest faces are 1e-12 of the largest.
    header, arrays = _parts("polyMesh")
    positions = arrays["positions"].reshape(-1, 3) * [1e-6, 1e-6, 1]

    response = _client(
        tmp_path, "polyMesh", _pack(header, {**arrays, "positions": positions.ravel()})
    ).get("/api/meshes/polyMesh/surface")

    assert response.status_code == 200


def test_a_truncated_surface_is_refused(tmp_path: Path) -> None:
    data = (DATA / "polyMesh.surface").read_bytes()

    response = _client(tmp_path, "polyMesh", data[: len(data) - 8]).get(
        "/api/meshes/polyMesh/surface"
    )

    assert response.status_code == 422


def test_a_mesh_without_its_surface_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path, "polyMesh")
    (tmp_path / "meshes" / "polyMesh.surface").unlink()

    assert client.get("/api/meshes/polyMesh/surface").status_code == 404
    assert client.get("/api/meshes/missing/surface").status_code == 404
