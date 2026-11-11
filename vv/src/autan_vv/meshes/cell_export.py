"""A level's primal cells, exported for a checker that reads them apart from the mesh file.

`tet`, `poly` and `agglomerated` levels can write, next to their mesh, one JSON file (schema
autan.vv-cell-export/1) holding each primal cell's outward polygons over the level's points and the
map from each primal cell to the cell of the written mesh. The map is the identity except for
`agglomerated`, whose cells merge cells of the median dual: there the primal cells are the median
dual's, and each merged cell is the union of the primal cells the map sends to it. The points are
the mesh file's vertices, the exact coordinates it holds, so the polygons and the mesh share them.
The export is opt-in: without it, a family's files are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, IntArray

CELL_EXPORT_SCHEMA = "autan.vv-cell-export/1"


def cell_polygons(mesh: FaceMesh) -> list[list[list[int]]]:
    """Each cell's polygons, oriented outward, in face order: a face as stored for its owner (its
    normal points from owner to neighbour), reversed for its neighbour."""
    cells: list[list[list[int]]] = [[] for _ in range(mesh.n_cells)]
    for f in range(mesh.n_faces):
        nodes = [int(v) for v in mesh.face(f)]
        cells[int(mesh.owner[f])].append(nodes)
        if f < mesh.n_interior:
            cells[int(mesh.neighbour[f])].append(nodes[::-1])
    return cells


def write_cell_export(
    path: Path,
    *,
    kind: str,
    n: int,
    perturbation: float,
    seed: int,
    mesh_path: Path,
    primal: FaceMesh,
    cell_map: IntArray,
    n_cells: int,
) -> None:
    """Write the export of a level whose mesh file is `mesh_path`: `primal`'s cells and `cell_map`
    (primal cell -> cell of the mesh file, `n_cells` of them)."""
    doc: dict[str, Any] = {
        "schema": CELL_EXPORT_SCHEMA,
        "kind": kind,
        "n": n,
        "perturbation": perturbation,
        "seed": seed,
        "mesh": {
            "file": mesh_path.name,
            "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "n_cells": n_cells,
        },
        "points": primal.points.tolist(),
        "cells": cell_polygons(primal),
        "map": [int(c) for c in cell_map],
    }
    path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
