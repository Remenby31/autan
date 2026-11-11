"""A vv mesh level's primal cells, read from its export (schema autan.vv-cell-export/1), for ADR
0003's check 1 on a mesh read from its file (stage 2: the agglomerated family).

The export (docs/validation/meshes.md) holds the mesh file's points, each primal cell's outward
polygons and the map from each primal cell to the cell of the mesh file. `load_bytes` checks its
shape and that it names the mesh file by SHA-256; `check_primal` checks what `topology.derive`
assumes of its input but does not check itself, since derive cancels any polygon two parts of an
agglomerated cell share: every primal cell is closed, consistently oriented and of positive exact
volume, and every polygon cancelled inside a mesh cell is one face of two different primal cells
listed from both sides. Both refuse (Invalid) rather than repair anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from oracle import topology as t

SCHEMA = "autan.vv-cell-export/1"


class Invalid(ValueError):
    """The export is unreadable, malformed, names another mesh file, or lists cells that are not
    closed, outward primal cells."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_bytes(
    data: bytes, mesh_sha256: str
) -> tuple[npt.NDArray[np.float64], list[list[list[int]]], list[int]]:
    """The points, each primal cell's polygons and the map of the export `data`, which must name
    the mesh file of SHA-256 `mesh_sha256`. The caller hashes the same bytes it passes here."""
    try:
        doc = json.loads(data)
    except ValueError as e:
        msg = f"unreadable export: {e}"
        raise Invalid(msg) from e
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        msg = f"not an export of schema {SCHEMA}"
        raise Invalid(msg)
    mesh = doc.get("mesh")
    if not isinstance(mesh, dict) or mesh.get("sha256") != mesh_sha256:
        msg = "the export names another mesh file"
        raise Invalid(msg)
    n_cells = mesh.get("n_cells")
    if not _int(n_cells) or n_cells <= 0:
        msg = "the export's mesh has no positive cell count"
        raise Invalid(msg)
    points, cells, cell_map = doc.get("points"), doc.get("cells"), doc.get("map")
    if not isinstance(points, list) or not points:
        msg = "the export has no points"
        raise Invalid(msg)
    for i, p in enumerate(points):
        if (
            not isinstance(p, list)
            or len(p) != 3
            or not all(isinstance(v, float) and math.isfinite(v) for v in p)
        ):
            msg = f"malformed point {i}: {p!r}"
            raise Invalid(msg)
    if not isinstance(cells, list) or not cells:
        msg = "the export has no cells"
        raise Invalid(msg)
    n_points = len(points)
    for i, polys in enumerate(cells):
        if not isinstance(polys, list) or not polys:
            msg = f"cell {i} has no polygon"
            raise Invalid(msg)
        for poly in polys:
            if (
                not isinstance(poly, list)
                or len(poly) < 3
                or not all(_int(v) and 0 <= v < n_points for v in poly)
                or len(set(poly)) != len(poly)
            ):
                msg = f"cell {i}: malformed polygon {poly!r}"
                raise Invalid(msg)
    if (
        not isinstance(cell_map, list)
        or len(cell_map) != len(cells)
        or not all(_int(c) and 0 <= c < n_cells for c in cell_map)
        or set(cell_map) != set(range(n_cells))
    ):
        msg = "the map does not send each primal cell to one cell and reach every cell"
        raise Invalid(msg)
    return np.array(points, dtype=np.float64), cells, cell_map


def check_primal(points: npt.NDArray[np.float64], cells: list[list[list[int]]]) -> None:
    """Every primal cell closed (each edge once in each direction), with no polygon twice and a
    positive exact volume; every polygon two primal cells of one mesh cell share listed once by
    each, in opposite orders; no polygon in more than two primal cells."""
    pts = [[Fraction(float(v)) for v in row] for row in points]
    by_key: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for i, polys in enumerate(cells):
        keys = Counter(tuple(sorted(p)) for p in polys)
        if any(n > 1 for n in keys.values()):
            msg = f"primal cell {i} lists a polygon twice"
            raise Invalid(msg)
        directed = Counter(e for p in polys for e in t._edges(tuple(p)))
        if any(n != 1 or directed[(b, a)] != 1 for (a, b), n in directed.items()):
            msg = f"primal cell {i} is not closed and consistently oriented"
            raise Invalid(msg)
        if t._volume(pts, [tuple(p) for p in polys]) <= 0:
            msg = f"primal cell {i} has no positive volume"
            raise Invalid(msg)
        for p in polys:
            by_key.setdefault(tuple(sorted(p)), []).append((i, tuple(p)))
    for occ in by_key.values():
        if len(occ) > 2:
            msg = f"a polygon is listed by {len(occ)} primal cells"
            raise Invalid(msg)
        if len(occ) == 2:
            (i, p), (j, q) = occ
            if not t._same_cycle(p, q[::-1]):
                msg = f"primal cells {i} and {j} list their common polygon in the same order"
                raise Invalid(msg)


def used_points(
    points: npt.NDArray[np.float64], cells: list[list[list[int]]], cell_map: list[int]
) -> set[tuple[float, float, float]]:
    """The export points the merged mesh's faces use. Within one mesh cell, a polygon two of its
    primal parts share is internal and cancelled (as `topology.derive` cancels it); the nodes of
    every polygon left are used. A mesh file may keep the others (a vv agglomerated file keeps every
    primal node), and the reader drops them."""
    per_cell: dict[int, Counter[tuple[int, ...]]] = {}
    for i, polys in enumerate(cells):
        per_cell.setdefault(cell_map[i], Counter()).update(tuple(sorted(p)) for p in polys)
    used = {v for keys in per_cell.values() for k, n in keys.items() if n == 1 for v in k}
    return {(float(points[v][0]), float(points[v][1]), float(points[v][2])) for v in used}


def vertex_issues(
    read_points: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    cells: list[list[list[int]]],
    cell_map: list[int],
) -> list[str]:
    """Why the mesh reader's vertices are not exactly the export points its faces use (empty if
    they are): coordinates compared exactly, no point twice, none missing, none extra."""
    got = [(float(r[0]), float(r[1]), float(r[2])) for r in read_points]
    want = used_points(points, cells, cell_map)
    issues = []
    # Two export points at one position, one of them used, would make the mapping by coordinates
    # ambiguous: refused.
    seen = Counter((float(p[0]), float(p[1]), float(p[2])) for p in points)
    if shared := sum(1 for xyz in want if seen[xyz] > 1):
        issues.append(f"{shared} export points the faces use share their coordinates with another")
    if len(set(got)) != len(got):
        issues.append(f"the reader gives {len(got) - len(set(got))} vertices twice")
    if missing := len(want - set(got)):
        issues.append(f"{missing} export points the faces use are not vertices of the mesh")
    if extra := len(set(got) - want):
        issues.append(f"{extra} vertices of the mesh are not export points the faces use")
    return issues
