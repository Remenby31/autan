import json
import struct
from pathlib import Path
from typing import Any

import autan
import h5py
import numpy as np
import numpy.typing as npt
import pytest
from autan_ui.app import create_app
from fastapi.testclient import TestClient

PY = Path(__file__).parents[3] / "py"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    # A run of the SDK's fake backend: its solution is the unit cube in 8 by 8 by 8 hexahedra.
    case = autan.load(PY / "examples" / "crm_hl.py")
    mesh_info = autan.MeshInfo.load(PY / "tests" / "fixtures" / "crm-hl.meshinfo.json")
    return case.run(mesh_info, backend=autan.FakeBackend(iterations=3), runs=tmp_path / "runs").path


def _client(run: Path) -> TestClient:
    return TestClient(create_app(run.parent, static=None, hosts=("testserver",)))


def _manufacture(run: Path) -> None:
    # Density = x + 2y + 3z at each cell's centroid (the mean of its nodes, a hexahedron's
    # centroid), computed here from the connectivity alone.
    with h5py.File(run / "solution.cgns", "r+") as tree:
        zone = tree["Base"]["Zone"]
        coords = np.stack(
            [zone["GridCoordinates"][f"Coordinate{a}"][" data"][()] for a in "XYZ"], 1
        )
        face_start = zone["NGonElements"]["ElementStartOffset"][" data"][()]
        face_nodes = zone["NGonElements"]["ElementConnectivity"][" data"][()] - 1
        cell_start = zone["NFaceElements"]["ElementStartOffset"][" data"][()]
        cell_faces = np.abs(zone["NFaceElements"]["ElementConnectivity"][" data"][()]) - 1
        density = zone["FlowSolution"]["Density"][" data"]
        values = np.empty(len(cell_start) - 1)
        for cell in range(len(values)):
            faces = cell_faces[cell_start[cell] : cell_start[cell + 1]]
            nodes = np.unique(
                np.concatenate([face_nodes[face_start[f] : face_start[f + 1]] for f in faces])
            )
            centroid = coords[nodes].mean(axis=0)
            values[cell] = centroid @ [1.0, 2.0, 3.0]
        density[...] = values


def _decode(
    message: bytes,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], dict[str, npt.NDArray[Any]]]:
    (length,) = struct.unpack_from("<I", message)
    header = json.loads(message[4 : 4 + length])
    body = memoryview(message)[4 + length :]
    dtypes = {"f32": np.float32, "u32": np.uint32}
    buffers = {
        b["name"]: np.frombuffer(body, dtypes[b["dtype"]], b["length"], b["offset"])
        for b in header["buffers"]
    }
    positions = buffers["positions"].astype(np.float64).reshape(-1, 3) + header["origin"]
    triangles = buffers["indices"].reshape(-1, 3)
    fields = {k[len("field:") :]: v for k, v in buffers.items() if k.startswith("field:")}
    return positions, triangles, fields


def _areas(positions: npt.NDArray[Any], triangles: npt.NDArray[Any]) -> npt.NDArray[Any]:
    a, b, c = (positions[triangles[:, k]] for k in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2


def test_a_slice_across_the_cube_is_its_section_with_each_cells_value(run: Path) -> None:
    _manufacture(run)
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0.4,0,0&normal=1,0,0")

    assert response.status_code == 200
    positions, triangles, fields = _decode(response.content)
    area = _areas(positions, triangles)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(positions[:, 0], 0.4, atol=tolerance)
    assert area.sum() == pytest.approx(1.0, rel=tolerance)
    # The cells the plane crosses span x ∈ [0.375, 0.5], centroid x 0.4375; y and z average
    # to 0.5 over the section: ∫ (x + 2y + 3z) dA = 0.4375 + 1 + 1.5.
    density = fields["Density"][triangles[:, 0]]
    assert (area @ density) == pytest.approx(2.9375, rel=tolerance)
    # Each polygon carries one cell's value: the three nodes of a triangle agree.
    np.testing.assert_array_equal(fields["Density"][triangles], density[:, None].repeat(3, 1))


def test_an_oblique_slice_through_the_centre_is_the_regular_hexagon(run: Path) -> None:
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0.5,0.5,0.5&normal=1,1,1")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(positions @ np.ones(3), 1.5, atol=4 * tolerance)
    # The unit cube cut through its centre normal to a diagonal: a regular hexagon of side
    # √2/2, area 3√3/4.
    assert _areas(positions, triangles).sum() == pytest.approx(3 * np.sqrt(3) / 4, rel=tolerance)


@pytest.mark.parametrize(
    ("query", "status"),
    [
        ("origin=0.5,0.5,0.5&normal=0,0,0", 422),
        ("origin=5,5,5&normal=1,0,0", 422),
        ("origin=0.5,0.5&normal=1,0,0", 422),
        ("origin=a,b,c&normal=1,0,0", 422),
        ("origin=nan,0,0&normal=1,0,0", 422),
    ],
)
def test_a_plane_that_is_not_one_or_misses_the_mesh_is_refused(
    run: Path, query: str, status: int
) -> None:
    assert _client(run).get(f"/api/runs/{run.name}/slice?{query}").status_code == status


def test_a_running_run_has_no_slice_yet(run: Path) -> None:
    document = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(
        json.dumps(document | {"status": "running", "finished": None, "monitors": None})
    )
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0.4,0,0&normal=1,0,0")
    assert response.status_code == 409
