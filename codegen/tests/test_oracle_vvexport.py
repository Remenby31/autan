"""Stage 2's source (ADR 0003): a mesh read from its file, its primal cells from a vv export.

The files are made here from the spike's own small meshes (hex, tet, hybrid and the agglomerated
poly of meshes.py), never from vv's agglomerated family: the merged mesh written as Fluent ASCII as
vv writes it (fluent.py of autan_vv), with every point the cells use, and the export of the cells
`meshes.cells` gives over those points. Check 1 must find no issue on them; each fault put into a
copy must be refused for its own reason.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import heldout as h
from oracle import kernels as kr
from oracle import topology as t
from oracle import vvexport as vx

if TYPE_CHECKING:
    from pathlib import Path

N, P, SEED = 3, 0.15, 3
KINDS = ["hex", "tet", "hybrid", "poly"]


def fluent(points: list[list[float]], read: t.ReadMesh, index: list[int]) -> str:
    """The merged mesh as vv writes it: one polyhedral cell zone, interior faces then one wall
    zone; node lists reversed (Fluent's normal points toward c0), 1-based, hexadecimal. `index`
    sends the read mesh's node numbers into `points`."""
    lines = ['(0 "oracle test mesh")', "(2 3)"]
    npts = len(points)
    lines += [f"(10 (0 1 {npts:x} 0 3))", f"(10 (1 1 {npts:x} 1 3)("]
    lines += [f"{x!r} {y!r} {z!r}" for x, y, z in points]
    lines += ["))", f"(12 (0 1 {read.n_cells:x} 0))", f"(12 (2 1 {read.n_cells:x} 1 7))"]
    nf, ni = len(read.owner), len(read.neighbour)
    lines.append(f"(13 (0 1 {nf:x} 0))")
    for zone, bc, lo, hi in ((3, 2, 0, ni), (4, 3, ni, nf)):
        lines.append(f"(13 ({zone:x} {lo + 1:x} {hi:x} {bc:x} 0)(")
        for f in range(lo, hi):
            span = read.face_nodes[read.face_offsets[f] : read.face_offsets[f + 1]]
            nodes = [index[int(v)] + 1 for v in span][::-1]
            c0 = int(read.owner[f]) + 1
            c1 = int(read.neighbour[f]) + 1 if f < ni else 0
            lines.append(f"{len(nodes):x} " + " ".join(f"{v:x}" for v in nodes) + f" {c0:x} {c1:x}")
        lines.append("))")
    lines += ["(45 (2 fluid fluid)())", "(45 (3 interior interior)())", "(45 (4 wall walls)())"]
    return "\n".join(lines) + "\n"


def make(kind: str, d: Path) -> tuple[Path, Path, dict[str, Any]]:
    """A mesh file and its export from the spike's `kind` mesh: the file holds the cells' points,
    in their order, so the export's points are the file's vertices."""
    read = t.read_mesh(s.mesh(meshes.build(kind, N, P, seed=SEED)))
    pts, cells, ag = meshes.cells(kind, N, P, seed=SEED)
    points = [[float(v) for v in row] for row in pts]
    at = {tuple(row): i for i, row in enumerate(points)}
    index = [at[tuple(float(v) for v in row)] for row in read.points]
    mesh_file = d / f"{kind}.msh"
    mesh_file.write_text(fluent(points, read, index))
    doc = {
        "schema": vx.SCHEMA, "kind": kind, "n": N, "perturbation": P, "seed": SEED,
        "mesh": {"file": mesh_file.name, "sha256": vx.sha256(mesh_file.read_bytes()),
                 "n_cells": read.n_cells},
        "points": points,
        "cells": [[[int(v) for v in poly] for poly in cell] for cell in cells],
        "map": list(ag) if ag is not None else list(range(len(cells))),
    }  # fmt: skip
    cells_file = d / f"{kind}.cells.json"
    cells_file.write_text(json.dumps(doc))
    return mesh_file, cells_file, doc


def check1(mesh_file: Path, doc: dict[str, Any]) -> list[str]:
    """As heldout does for a vv-export instance: load, check the primal cells, read the file,
    require the export's points to be its vertices, then derive and compare."""
    pts, cells, ag = vx.load_bytes(json.dumps(doc).encode(), vx.sha256(mesh_file.read_bytes()))
    vx.check_primal(pts, cells)
    read = t.read_mesh(s.read(str(mesh_file)))
    assert np.array_equal(read.points, pts)
    return t.compare(t.derive(pts, cells, ag), read)


@pytest.fixture(scope="module", params=KINDS)
def files(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Path, Path, dict[str, Any]]:
    kind = request.param
    return (kind, *make(kind, tmp_path_factory.mktemp(kind)))


Files = tuple[str, "Path", "Path", dict[str, Any]]


def test_check1_finds_no_issue_on_a_file_and_its_export(files: Files) -> None:
    _, mesh_file, _, doc = files
    assert check1(mesh_file, doc) == []


def test_check1_finds_two_cells_relabelled(files: Files) -> None:
    # Every primal cell of one mesh cell sent to another and back: the derivation still succeeds
    # (each label is a whole cell), the comparison must report the cells it moved.
    _, mesh_file, _, doc = files
    bad = copy.deepcopy(doc)
    a, b = 0, max(bad["map"])
    bad["map"] = [b if c == a else a if c == b else c for c in bad["map"]]
    pts, cells, ag = vx.load_bytes(json.dumps(bad).encode(), doc["mesh"]["sha256"])
    vx.check_primal(pts, cells)
    derived = t.derive(pts, cells, ag)
    assert t.compare(derived, t.read_mesh(s.read(str(mesh_file)))) != []


def _keys(doc: dict[str, Any]) -> dict[tuple[int, ...], list[tuple[int, int]]]:
    out: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i, cell in enumerate(doc["cells"]):
        for k, poly in enumerate(cell):
            out.setdefault(tuple(sorted(poly)), []).append((i, k))
    return out


def test_a_reversed_face_of_a_mesh_cell_is_refused(files: Files) -> None:
    # A polygon no other primal cell shares: a face the derivation keeps.
    _, _, _, doc = files
    i, k = next(occ[0] for occ in _keys(doc).values() if len(occ) == 1)
    bad = copy.deepcopy(doc)
    bad["cells"][i][k] = bad["cells"][i][k][::-1]
    pts, cells, _ = vx.load_bytes(json.dumps(bad).encode(), doc["mesh"]["sha256"])
    with pytest.raises(vx.Invalid, match="not closed and consistently oriented"):
        vx.check_primal(pts, cells)


def test_a_reversed_polygon_inside_an_agglomerated_cell_is_refused(tmp_path: Path) -> None:
    # poly: two primal cells of one mesh cell share it; derive alone would cancel it unseen.
    _, _, doc = make("poly", tmp_path)
    inner = next(
        occ for occ in _keys(doc).values()
        if len(occ) == 2 and doc["map"][occ[0][0]] == doc["map"][occ[1][0]]
    )  # fmt: skip
    i, k = inner[0]
    bad = copy.deepcopy(doc)
    bad["cells"][i][k] = bad["cells"][i][k][::-1]
    pts, cells, _ = vx.load_bytes(json.dumps(bad).encode(), doc["mesh"]["sha256"])
    with pytest.raises(vx.Invalid):
        vx.check_primal(pts, cells)


def test_a_polygon_listed_twice_by_one_primal_cell_is_refused(files: Files) -> None:
    _, _, _, doc = files
    bad = copy.deepcopy(doc)
    bad["cells"][0] += [bad["cells"][0][0], bad["cells"][0][0][::-1]]
    pts, cells, _ = vx.load_bytes(json.dumps(bad).encode(), doc["mesh"]["sha256"])
    with pytest.raises(vx.Invalid, match="lists a polygon twice"):
        vx.check_primal(pts, cells)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda d: d.update(schema="other"), "schema"),
        (lambda d: d["points"].__setitem__(0, [0.0, 0.0]), "malformed point 0"),
        (lambda d: d["cells"][0].__setitem__(0, d["cells"][0][0][:2]), "malformed polygon"),
        (lambda d: d["map"].pop(), "the map"),
        (lambda d: d["mesh"].update(sha256="0" * 64), "another mesh file"),
    ],
    ids=["schema", "point", "polygon", "map", "mesh"],
)
def test_one_malformed_field_makes_the_export_invalid(
    files: Files, mutate: Any, reason: str
) -> None:
    _, _, _, doc = files
    bad = copy.deepcopy(doc)
    mutate(bad)
    with pytest.raises(vx.Invalid, match=reason):
        vx.load_bytes(json.dumps(bad).encode(), doc["mesh"]["sha256"])


def _spec(kind: str, mesh_file: Path, cells_file: Path) -> dict[str, Any]:
    return {
        "id": f"vv-{kind}", "source": h.VV_SOURCE, "kind": kind, "n": N, "perturbation": P,
        "seed": SEED, "fields": [{"a": 1.0, "G": [0.3, -1.2, 0.7]}], "policies": ["fp64"],
        "methods": ["lsq"],
        "files": {
            "mesh": {"name": mesh_file.name, "sha256": vx.sha256(mesh_file.read_bytes())},
            "cells": {"name": cells_file.name, "sha256": vx.sha256(cells_file.read_bytes())},
        },
    }  # fmt: skip


@pytest.fixture(scope="module")
def library() -> str:
    digest = kr.library_sha256()
    assert digest is not None
    return digest


@pytest.mark.parametrize("kind", ["tet", "poly"])
def test_a_vv_instance_runs_every_check_and_passes(tmp_path: Path, library: str, kind: str) -> None:
    mesh_file, cells_file, _ = make(kind, tmp_path)
    spec = _spec(kind, mesh_file, cells_file)
    assert h.spec_issues(spec) == []
    rec = h.run_instance(spec, library, "0" * 64, data_dir=tmp_path)
    assert rec["invalid"] == []
    assert rec["check1"] == []
    assert rec["check2"] == []
    assert rec["stage"] == "done"
    assert h.passed(rec, spec)


def test_a_vv_instance_without_a_data_directory_is_invalid(tmp_path: Path, library: str) -> None:
    mesh_file, cells_file, _ = make("tet", tmp_path)
    spec = _spec("tet", mesh_file, cells_file)
    rec = h.run_instance(spec, library, "0" * 64)
    assert rec["invalid"] == ["invalid: a vv-export instance needs a data directory"]
    assert not h.passed(rec, spec)


def test_a_vv_instance_with_a_changed_file_is_invalid(tmp_path: Path, library: str) -> None:
    mesh_file, cells_file, _ = make("tet", tmp_path)
    spec = _spec("tet", mesh_file, cells_file)
    mesh_file.write_text(mesh_file.read_text() + "\n")
    rec = h.run_instance(spec, library, "0" * 64, data_dir=tmp_path)
    assert rec["invalid"] == ["invalid: the mesh file is not the declared one"]


def test_a_node_moved_by_one_ulp_in_the_file_is_refused(tmp_path: Path, library: str) -> None:
    mesh_file, cells_file, doc = make("tet", tmp_path)
    lines = mesh_file.read_text().splitlines()
    k = lines.index(next(x for x in lines if x.startswith("(10 (1 "))) + 1
    x, y, z = (float(v) for v in lines[k].split())
    moved = float(np.nextafter(x, np.inf))
    lines[k] = f"{moved!r} {y!r} {z!r}"
    mesh_file.write_text("\n".join(lines) + "\n")
    # The reader keeps the moved coordinate exactly.
    assert float(t.read_mesh(s.read(str(mesh_file))).points[0][0]) == moved
    doc["mesh"]["sha256"] = vx.sha256(mesh_file.read_bytes())
    cells_file.write_text(json.dumps(doc))
    spec = _spec("tet", mesh_file, cells_file)
    rec = h.run_instance(spec, library, "0" * 64, data_dir=tmp_path)
    assert rec["invalid"] == ["invalid: the export's points are not the mesh's vertices"]


@pytest.mark.parametrize(
    "change",
    [
        {"source": "elsewhere"},
        {"kind": "hex"},
        {"files": {"mesh": {"name": "../x.msh", "sha256": "0" * 64},
                   "cells": {"name": "a.cells.json", "sha256": "1" * 64}}},
        {"files": {"mesh": {"name": "a.msh", "sha256": "z" * 64},
                   "cells": {"name": "a.cells.json", "sha256": "1" * 64}}},
    ],
    ids=["source", "kind", "path", "digest"],
)  # fmt: skip
def test_a_malformed_vv_spec_is_invalid(change: dict[str, Any]) -> None:
    spec = {
        "id": "vv-x", "source": h.VV_SOURCE, "kind": "agglomerated", "n": N, "perturbation": P,
        "seed": SEED, "fields": [{"a": 1.0, "G": [0.0, 0.0, 0.0]}], "policies": ["fp64"],
        "methods": ["lsq"],
        "files": {"mesh": {"name": "a.msh", "sha256": "0" * 64},
                  "cells": {"name": "a.cells.json", "sha256": "1" * 64}},
    }  # fmt: skip
    assert h.spec_issues(spec) == []
    assert h.spec_issues({**spec, **change}) != []
