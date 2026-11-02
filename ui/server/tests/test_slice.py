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


@pytest.mark.parametrize(
    ("origin", "normal"),
    [
        ("0.5,0,0", "1,0,0"),
        ("0.5,0,0", "-1,0,0"),
        ("0,0,0", "1,0,0"),
        ("1,0,0", "-1,0,0"),
        ("0,0,0", "-1,0,0"),
        ("1,0,0", "1,0,0"),
    ],
)
def test_a_plane_along_mesh_faces_gives_each_face_once(run: Path, origin: str, normal: str) -> None:
    # x = 0.5 is a plane of faces inside the 8x8x8 cube, x = 0 and x = 1 its boundary.
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin={origin}&normal={normal}")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        positions[:, 0], float(origin.split(",", maxsplit=1)[0]), atol=tolerance
    )
    assert _areas(positions, triangles).sum() == pytest.approx(1.0, rel=tolerance)


def test_a_plane_through_mesh_nodes_leaves_no_empty_triangle(run: Path) -> None:
    # x + y + z = 1.5 passes through the cube's nodes (multiples of 1/8): cells touching it at a
    # node or along an edge have no section.
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0.5,0.5,0.5&normal=1,1,1")

    positions, triangles, _ = _decode(response.content)
    area = _areas(positions, triangles)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    assert area.sum() == pytest.approx(3 * np.sqrt(3) / 4, rel=tolerance)
    assert area.min() > 1e-3 / 64


def _mesh(
    run: Path, coords: npt.NDArray[Any], faces: list[list[int]], cells: list[list[int]]
) -> None:
    # The run's whole solution mesh replaced by these nodes, faces and cells (0-based), every
    # field 1 in every cell.
    face_start = np.cumsum([0] + [len(f) for f in faces])
    cell_start = np.cumsum([0] + [len(c) for c in cells])
    with h5py.File(run / "solution.cgns", "r+") as tree:
        zone = tree["Base"]["Zone"]
        arrays = {
            ("GridCoordinates", f"Coordinate{a}"): coords[:, k] for k, a in enumerate("XYZ")
        } | {
            ("NGonElements", "ElementStartOffset"): face_start,
            ("NGonElements", "ElementConnectivity"): np.concatenate(faces) + 1,
            ("NFaceElements", "ElementStartOffset"): cell_start,
            ("NFaceElements", "ElementConnectivity"): np.concatenate(cells) + 1,
        }
        for name, group in zone["FlowSolution"].items():
            if isinstance(group, h5py.Group) and name != "GridLocation":
                arrays[("FlowSolution", name)] = np.ones(len(cells))
        for (parent, name), values in arrays.items():
            group = zone[parent][name]
            dtype = group[" data"].dtype
            del group[" data"]
            group.create_dataset(" data", data=np.asarray(values).astype(dtype))


def _single_cell(run: Path, base: list[tuple[float, float]]) -> None:
    # One prism over the polygon `base`, z from 0 to 1.
    n = len(base)
    coords = np.array([(x, y, z) for z in (0.0, 1.0) for x, y in base])
    faces = [list(range(n))[::-1], list(range(n, 2 * n))]
    faces += [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]
    _mesh(run, coords, faces, [list(range(len(faces)))])


def _cubes(run: Path, corners: list[tuple[int, int, int]]) -> None:
    # Unit cubes at these integer corners, sharing their common nodes and faces.
    nodes: dict[tuple[int, int, int], int] = {}
    faces: dict[frozenset[int], int] = {}
    face_nodes: list[list[int]] = []
    cells = []
    sides = [
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    ]
    for x, y, z in corners:
        cell = []
        for side in sides:
            ids = [nodes.setdefault((x + i, y + j, z + k), len(nodes)) for i, j, k in side]
            cell.append(faces.setdefault(frozenset(ids), len(face_nodes)))
            if cell[-1] == len(face_nodes):
                face_nodes.append(ids)
        cells.append(cell)
    coords = np.array(sorted(nodes, key=nodes.__getitem__), dtype=np.float64)
    _mesh(run, coords, face_nodes, cells)


def test_a_concave_cell_is_cut_along_its_notch(run: Path) -> None:
    # An L-shaped prism: its section at z = 0.5 is the L, area 3; a fan from one corner would
    # cover the notch too (area 4).
    _single_cell(run, [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0,0,0.5&normal=0,0,1")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    assert _areas(positions, triangles).sum() == pytest.approx(3.0, rel=tolerance)
    assert len(triangles) == 4


def test_a_slice_is_read_only_from_the_solution_file(run: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.bin"
    with h5py.File(run / "solution.cgns", "r+") as tree:
        density = tree["Base"]["Zone"]["FlowSolution"]["Density"]
        values = density[" data"][()]
        del density[" data"]
        secret.write_bytes(values.astype("<f8").tobytes())
        density.create_dataset(
            " data", shape=values.shape, dtype="<f8", external=[(str(secret), 0, values.nbytes)]
        )

    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0.4,0,0&normal=1,0,0")

    assert response.status_code == 422


def test_slices_asked_while_one_is_cut_leave_only_the_newest(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A plane dragged across the view: requests pile up while the first is cut, more of them
    # than the server has worker threads. The server keeps answering; only the newest waiting
    # slice is cut next, the ones it superseded are refused without cutting.
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    import autan_ui.app  # noqa: PLC0415

    started, release = threading.Event(), threading.Event()
    cut = autan_ui.app.slice_surface
    calls: list[float] = []

    def held(folder: Any, origin: Any, normal: Any) -> bytes:
        calls.append(float(origin[0]))
        started.set()
        release.wait(30)
        return cut(folder, origin, normal)

    monkeypatch.setattr(autan_ui.app, "slice_surface", held)
    url = f"/api/runs/{run.name}/slice?normal=1,0,0&origin="
    waiting_count = 48
    # One event loop serves every request, as uvicorn does.
    with _client(run) as client, ThreadPoolExecutor(waiting_count + 2) as pool:
        first = pool.submit(client.get, url + "0.1,0,0")
        assert started.wait(10)
        waiting = [pool.submit(client.get, url + f"0.{k + 2},0,0") for k in range(waiting_count)]
        # Every one has taken its turn (the server's count of slices asked) while the first is
        # still being cut.
        deadline = time.monotonic() + 10
        while client.app.state.slices_asked.get(run.name, 0) < waiting_count + 1:  # type: ignore[attr-defined]
            assert time.monotonic() < deadline, "requests waiting for a slice starve the server"
            time.sleep(0.01)
        assert pool.submit(client.get, "/api/materials").result(timeout=10).status_code == 200
        release.set()
        codes = [first.result().status_code] + [w.result().status_code for w in waiting]

    assert codes[0] == 200
    assert sorted(codes[1:]) == [200] + [409] * (waiting_count - 1)
    assert len(calls) == 2


@pytest.mark.parametrize("normal", ["0,0,1", "0,0,-1"])
def test_faces_in_the_plane_count_once_whichever_side_the_mesh_is(run: Path, normal: str) -> None:
    # Three unit cubes: two side by side on z = 0, one under the first. z = 0 holds an interior
    # face (between the first and the lower cube) and a boundary face (under the second).
    _cubes(run, [(0, 0, 0), (1, 0, 0), (0, 0, -1)])
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0,0,0&normal={normal}")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(positions[:, 2], 0.0, atol=tolerance)
    assert _areas(positions, triangles).sum() == pytest.approx(2.0, rel=tolerance)


@pytest.mark.parametrize("normal", ["0,0,1", "0,0,-1"])
def test_a_face_in_the_plane_split_in_triangles_is_the_whole_face(run: Path, normal: str) -> None:
    # A unit cube whose top is two triangles: the plane of its top gives the unit square.
    coords = np.array([(x, y, z) for z in (0.0, 1.0) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))])
    faces = [[3, 2, 1, 0], [4, 5, 6], [4, 6, 7]]
    faces += [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)]
    _mesh(run, coords, faces, [list(range(len(faces)))])
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0,0,1&normal={normal}")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    assert _areas(positions, triangles).sum() == pytest.approx(1.0, rel=tolerance)


@pytest.mark.parametrize("normal", ["1,0,0", "-1,0,0"])
def test_a_plane_through_a_cell_and_along_its_face_gives_both(run: Path, normal: str) -> None:
    # The L-shaped prism with a node at (1, 0): x = 1 crosses its foot (y from 0 to 1) and lies
    # along its notch (y from 1 to 2), the section of the cell on its x < 1 side: area 2.
    _single_cell(run, [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=1,0,0.5&normal={normal}")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(positions[:, 0], 1.0, atol=tolerance)
    assert _areas(positions, triangles).sum() == pytest.approx(2.0, rel=tolerance)


def test_a_concave_cell_cut_through_its_reflex_nodes_keeps_its_section(run: Path) -> None:
    # A prism over a hexagon whose two reflex corners, (-1, 0) and (1, 0), lie on y = 0: its
    # section there is the full 4 by 1 rectangle.
    _single_cell(run, [(-2, -1), (-1, 0), (2, -1), (2, 1), (1, 0), (-2, 1)])
    response = _client(run).get(f"/api/runs/{run.name}/slice?origin=0,0,0.5&normal=0,1,0")

    assert response.status_code == 200
    positions, triangles, _ = _decode(response.content)
    tolerance = 64 * float(np.finfo(np.float32).eps)
    assert _areas(positions, triangles).sum() == pytest.approx(4.0, rel=1e-6)
    np.testing.assert_allclose(positions[:, 1], 0.0, atol=tolerance)
