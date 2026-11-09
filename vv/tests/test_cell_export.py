"""The primal-cell export (cell_export.py) at the CLI seam, on tet and a small poly only.

Checked against the mesh file the CLI wrote, read back without the generator's own conversion code:
for tet, Gmsh's nodes, its tetrahedra in file order and its surface triangles; for poly, the Fluent
reader of test_meshes. Without the oracle that will read the export: it holds the file's points,
cell i is the file's cell i, every primal cell closes with a positive volume equal to the file
cell's, the polygons cover each file face exactly (once from each side inside, the file's boundary
faces once), and the export changes no byte of the family it accompanies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest
from autan_vv.cli import main
from test_meshes import _read_fluent

if TYPE_CHECKING:
    from pathlib import Path

gmsh = pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")

LEVEL, PERTURBATION, SEED = 3, 0.15, 5
Face = frozenset[int]


@dataclass(frozen=True)
class FileMesh:
    """A mesh file read back: its points, each cell's faces (vertex sets, in file cell order) and
    volume, its boundary faces, and, when the reader gives them, each cell's outward polygons."""

    points: npt.NDArray[np.float64]
    cell_faces: list[set[Face]]
    volume: npt.NDArray[np.float64]
    boundary: set[Face]
    outward: list[set[tuple[int, ...]]] | None


def _generate(kind: str, out: Path, *, export: bool) -> None:
    args = ["meshes", kind, "--levels", str(LEVEL), "--perturbation", str(PERTURBATION)]
    args += ["--seed", str(SEED), "--out", str(out)]
    assert main([*args, "--export-cells"] if export else args) == 0


def _fan(points: npt.NDArray[np.float64], poly: list[int]) -> tuple[npt.NDArray[np.float64], float]:
    """Area vector and divergence-theorem volume term of a polygon, by its fan from its mean point."""
    p = points[poly]
    mean = p.mean(axis=0)
    a, b = p - mean, np.roll(p, -1, axis=0) - mean
    tri = 0.5 * np.cross(a, b)
    return tri.sum(axis=0), float((tri @ mean).sum() / 3.0)


def _key(poly: list[int]) -> tuple[int, ...]:
    """The polygon from its smallest node, orientation kept."""
    k = poly.index(min(poly))
    return tuple(poly[k:] + poly[:k])


def _read_tet(path: Path) -> FileMesh:
    """Gmsh's nodes (by tag), tetrahedra (in file order) and surface triangles: no generator code."""
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(sorted(int(t) for t in node_tags))}
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[np.argsort(node_tags)]
        types, _, conn = gmsh.model.mesh.getElements(3)
        assert [int(t) for t in types] == [4]
        tets = [[index[int(v)] for v in c] for c in np.asarray(conn[0]).reshape(-1, 4)]
        types2, _, conn2 = gmsh.model.mesh.getElements(2)
        assert [int(t) for t in types2] == [2]
        boundary = {
            frozenset(index[int(v)] for v in c) for c in np.asarray(conn2[0]).reshape(-1, 3)
        }
    finally:
        gmsh.finalize()
    faces = [{frozenset(t) - {v} for v in t} for t in tets]
    volume = np.array(
        [abs(np.linalg.det(points[t[1:]] - points[t[0]])) / 6.0 for t in tets], dtype=np.float64
    )
    return FileMesh(points, faces, volume, boundary, None)


def _read_poly(path: Path) -> FileMesh:
    """The test Fluent reader's mesh: faces, outward polygons and volumes per file cell."""
    back = _read_fluent(path)
    faces: list[set[Face]] = [set() for _ in range(back.n_cells)]
    outward: list[set[tuple[int, ...]]] = [set() for _ in range(back.n_cells)]
    for f in range(back.n_faces):
        nodes = [int(v) for v in back.face(f)]
        faces[int(back.owner[f])].add(frozenset(nodes))
        outward[int(back.owner[f])].add(_key(nodes))
        if f < back.n_interior:
            faces[int(back.neighbour[f])].add(frozenset(nodes))
            outward[int(back.neighbour[f])].add(_key(nodes[::-1]))
    boundary = {
        frozenset(int(v) for v in back.face(f)) for f in range(back.n_interior, back.n_faces)
    }
    return FileMesh(back.points, faces, back.geometry.cell_volume, boundary, outward)


@pytest.fixture(scope="module", params=["tet", "poly"])
def level(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path, Path, dict[str, Any], FileMesh]:
    kind = request.param
    out = tmp_path_factory.mktemp(kind)
    _generate(kind, out, export=True)
    mesh_file = next(out.glob("*.msh"))
    doc = json.loads(mesh_file.with_name(f"{mesh_file.stem}.cells.json").read_text())
    back = _read_tet(mesh_file) if kind == "tet" else _read_poly(mesh_file)
    return kind, out, mesh_file, doc, back


Level = tuple[str, "Path", "Path", dict[str, Any], FileMesh]


def test_the_export_names_its_file_holds_its_points_and_maps_cell_i_to_cell_i(level: Level) -> None:
    kind, _, mesh_file, doc, back = level
    assert doc["schema"] == "autan.vv-cell-export/1"
    assert (doc["kind"], doc["n"], doc["perturbation"], doc["seed"]) == (
        kind, LEVEL, PERTURBATION, SEED,
    )  # fmt: skip
    assert doc["mesh"]["file"] == mesh_file.name
    assert doc["mesh"]["sha256"] == hashlib.sha256(mesh_file.read_bytes()).hexdigest()
    assert doc["mesh"]["n_cells"] == len(back.cell_faces)
    assert np.array_equal(np.array(doc["points"]), back.points)
    assert doc["map"] == list(range(len(back.cell_faces)))
    assert len(doc["cells"]) == len(back.cell_faces)


def test_each_primal_cell_has_the_faces_of_its_file_cell(level: Level) -> None:
    kind, _, _, doc, back = level
    for i, polys in enumerate(doc["cells"]):
        # Distinct vertices, at least 3 (exactly 3 for tet): repeated vertices would keep the vertex
        # sets, areas and volumes below unchanged.
        assert all(len(p) == len(set(p)) >= 3 for p in polys)
        if kind == "tet":
            assert all(len(p) == 3 for p in polys)
        assert len(polys) == len(back.cell_faces[i])
        assert {frozenset(p) for p in polys} == back.cell_faces[i]
        if back.outward is not None:
            assert {_key(p) for p in polys} == back.outward[i]


def test_every_primal_cell_closes_with_the_volume_of_its_file_cell(level: Level) -> None:
    _, _, _, doc, back = level
    points = np.array(doc["points"], dtype=np.float64)
    for i, polys in enumerate(doc["cells"]):
        terms = [_fan(points, p) for p in polys]
        total = np.sum([a for a, _ in terms], axis=0)
        scale = sum(float(np.linalg.norm(a)) for a, _ in terms)
        assert float(np.linalg.norm(total)) <= 1e-13 * scale
        volume = sum(v for _, v in terms)
        assert volume > 0.0
        assert volume == pytest.approx(float(back.volume[i]), rel=1e-12)


def test_the_polygons_cover_each_file_face_once_from_each_side(level: Level) -> None:
    _, _, _, doc, back = level
    keys = [_key(p) for polys in doc["cells"] for p in polys]
    assert len(keys) == len(set(keys))
    seen = set(keys)
    one_sided = {frozenset(k) for k in keys if _key(list(k)[::-1]) not in seen}
    assert one_sided == back.boundary
    interior = {frozenset(f) for faces in back.cell_faces for f in faces} - back.boundary
    assert len(keys) == len(back.boundary) + 2 * len(interior)


def test_the_export_changes_no_byte_of_the_family(level: Level, tmp_path: Path) -> None:
    kind, out, _, _, _ = level
    _generate(kind, tmp_path, export=False)
    without = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with_export = {
        p.name: p.read_bytes() for p in out.iterdir() if not p.name.endswith(".cells.json")
    }
    assert with_export == without
    assert sorted(p.name for p in out.iterdir() if p.name.endswith(".cells.json")) == [
        f"{kind}-dirty-n{LEVEL:03d}.cells.json"
    ]


def test_a_kind_without_primal_cells_to_export_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["meshes", "hex", "--levels", "3", "--out", str(tmp_path), "--export-cells"]) == 2
    assert "--export-cells applies to tet, poly and agglomerated" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
