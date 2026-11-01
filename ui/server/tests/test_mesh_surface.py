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
    # The hybrid mesh is a closed box: its triangles' area vectors cancel, to FP32 round-off.
    points = buffers["positions"].astype(np.float64).reshape(-1, 3)
    a, b, c = (points[buffers["indices"].reshape(-1, 3)[:, k]] for k in range(3))
    area = np.cross(b - a, c - a) / 2
    total = np.linalg.norm(area, axis=1).sum()
    assert np.linalg.norm(area.sum(axis=0)) <= 64 * np.finfo(np.float32).eps * total


def test_a_surface_of_another_mesh_is_refused(tmp_path: Path) -> None:
    other = (DATA / "polyMesh.surface").read_bytes()
    mesh_info = json.loads((DATA / "hybrid-cgns.meshinfo.json").read_text())
    assert mesh_info["source"]["sha256"] not in other.decode(errors="ignore")

    response = _client(tmp_path, "hybrid-cgns", other).get("/api/meshes/hybrid-cgns/surface")

    assert response.status_code == 422


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
