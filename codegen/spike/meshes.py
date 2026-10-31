"""Small unit-cube meshes for generator tests: hex, tet, hybrid and polyhedral, optionally perturbed.

Each builder returns cells as lists of outward polygons over a point array; `to_faces` merges
coincident polygons into the face-based arrays that `autan::assemble` takes. The order study uses
the V&V mesh families instead; these exist so that tests run without Gmsh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Cell = list[list[int]]


@dataclass
class FaceArrays:
    points: npt.NDArray[np.float64]
    face_offsets: npt.NDArray[np.int64]
    face_nodes: npt.NDArray[np.int32]
    cell0: npt.NDArray[np.int32]
    cell1: npt.NDArray[np.int32]
    n_cells: int


def to_faces(
    points: npt.NDArray[np.float64], cells: list[Cell], agglomerate: list[int] | None = None
) -> FaceArrays:
    """Merge the polygons of `cells`; `agglomerate[i]` is the final cell of input cell i."""
    cell_of = agglomerate or list(range(len(cells)))
    seen: dict[tuple[int, ...], int] = {}
    faces: list[list[int]] = []
    c0: list[int] = []
    c1: list[int] = []
    for i, cell in enumerate(cells):
        for poly in cell:
            key = tuple(sorted(poly))
            if key in seen:
                f = seen.pop(key)
                c1[f] = cell_of[i]
            else:
                seen[key] = len(faces)
                faces.append(poly)
                c0.append(cell_of[i])
                c1.append(-1)
    keep = [f for f in range(len(faces)) if c0[f] != c1[f]]
    offsets = np.zeros(len(keep) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(faces[f]) for f in keep])
    nodes = np.array([n for f in keep for n in faces[f]], dtype=np.int32)
    used = np.unique(nodes)
    remap = np.full(len(points), -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return FaceArrays(
        np.ascontiguousarray(points[used]),
        offsets,
        remap[nodes],
        np.array([c0[f] for f in keep], dtype=np.int32),
        np.array([c1[f] for f in keep], dtype=np.int32),
        max(cell_of) + 1,
    )


def _grid(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    g = np.linspace(0.0, 1.0, n + 1)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    return pts, idx


def _corners(idx: npt.NDArray[np.int64], i: int, j: int, k: int) -> list[int]:
    """Corners of cube (i, j, k), numbered by bits (dx, dy, dz) = (b & 1, b >> 1 & 1, b >> 2)."""
    return [int(idx[i + (b & 1), j + (b >> 1 & 1), k + (b >> 2)]) for b in range(8)]


# Outward quads of a cube in the bit numbering above.
CUBE_FACES = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]]


def _orient(
    pts: npt.NDArray[np.float64], poly: list[int], inside: npt.NDArray[np.float64]
) -> list[int]:
    p = pts[poly]
    m = p.mean(axis=0)
    area = np.zeros(3)
    for a in range(len(poly)):
        area += np.cross(p[a] - m, p[(a + 1) % len(poly)] - m)
    return poly if np.dot(area, m - inside) > 0 else poly[::-1]


def _tet(pts: npt.NDArray[np.float64], t: list[int]) -> Cell:
    c = pts[t].mean(axis=0)
    return [
        _orient(pts, [t[a], t[b], t[d]], c)
        for a, b, d in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    ]


def _kuhn(pts: npt.NDArray[np.float64], v: list[int]) -> list[Cell]:
    """Six tetrahedra along the main diagonal 0 -> 7; every face diagonal joins its min and max corner."""
    out = []
    for a, b in ((1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)):
        out.append(_tet(pts, [v[0], v[a], v[a | b], v[7]]))
    return out


def _split_face(q: list[int]) -> list[list[int]]:
    """Split quad q (cube-local corner ids, cyclic) along the diagonal from its min to max corner."""
    lo = q.index(min(q))
    r = q[lo:] + q[:lo]
    return [[r[0], r[1], r[2]], [r[0], r[2], r[3]]]


def _perturb(
    pts: npt.NDArray[np.float64], h: float, alpha: float, seed: int
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    move = rng.uniform(-alpha * h, alpha * h, pts.shape)
    on_boundary = (np.abs(pts) < 1e-12) | (np.abs(pts - 1.0) < 1e-12)
    move[on_boundary] = 0.0  # keep the cube: boundary nodes slide only within their faces
    return pts + move


def build(kind: str, n: int, perturbation: float = 0.0, seed: int = 0) -> FaceArrays:
    """Mesh of the unit cube with about n cells per edge; kinds: hex, tet, hybrid, poly."""
    pts, idx = _grid(n)
    h = 1.0 / n
    cells: list[Cell] = []
    agglomerate: list[int] | None = None
    extra: list[npt.NDArray[np.float64]] = []
    cubes = [(i, j, k) for k in range(n) for j in range(n) for i in range(n)]
    if kind in ("hex", "poly"):
        for i, j, k in cubes:
            v = _corners(idx, i, j, k)
            cells.append([[v[a] for a in f] for f in CUBE_FACES])
        if kind == "poly":
            agglomerate = _bricks(n)
    elif kind == "tet":
        for i, j, k in cubes:
            cells.extend(_kuhn(pts, _corners(idx, i, j, k)))
    elif kind == "hybrid":
        n_pts = len(pts)
        for i, j, k in cubes:
            v = _corners(idx, i, j, k)
            slab = 4 * k // n
            if slab == 0:
                cells.append([[v[a] for a in f] for f in CUBE_FACES])
            elif slab == 1:
                centre = n_pts + len(extra)
                extra.append(pts[v].mean(axis=0))
                cells.extend(_transition(v, centre))
            elif slab == 2:
                cells.extend(_kuhn(pts, v))
            else:
                cells.extend(_prisms(v))
        pts = np.vstack([pts, *extra]) if extra else pts
    else:
        raise ValueError(kind)
    cells = [[_orient(pts, p, _cell_centre(pts, c)) for p in c] for c in cells]
    if perturbation:
        # Cube corners move; cell-centre nodes of the hybrid transition layer stay put.
        n_grid = (n + 1) ** 3
        pts = np.vstack([_perturb(pts[:n_grid], h, perturbation, seed), pts[n_grid:]])
    return to_faces(pts, cells, agglomerate)


def _cell_centre(pts: npt.NDArray[np.float64], cell: Cell) -> npt.NDArray[np.float64]:
    return pts[sorted({n for p in cell for n in p})].mean(axis=0)


def _transition(v: list[int], centre: int) -> list[Cell]:
    """A pyramid on the bottom quad and ten tetrahedra on the split other faces, apex at the centre."""
    out: list[Cell] = []
    bottom = [v[a] for a in CUBE_FACES[4]]
    out.append([bottom, *[[bottom[a], bottom[(a + 1) % 4], centre] for a in range(4)]])
    for f in (CUBE_FACES[0], CUBE_FACES[1], CUBE_FACES[2], CUBE_FACES[3], CUBE_FACES[5]):
        for tri in _split_face(list(f)):
            t = [v[a] for a in tri]
            out.append([t, *[[t[a], t[(a + 1) % 3], centre] for a in range(3)]])
    return out


def _prisms(v: list[int]) -> list[Cell]:
    """Two prisms split by the vertical plane through the xy diagonal from corner 0 to corner 3."""
    out = []
    for tri in ([0, 1, 3], [0, 3, 2]):
        b = [v[a] for a in tri]
        t = [v[a + 4] for a in tri]
        out.append([b, t, *[[b[a], b[(a + 1) % 3], t[(a + 1) % 3], t[a]] for a in range(3)]])
    return out


def _bricks(n: int) -> list[int]:
    """Agglomeration of an n^3 cube grid: 2x1x1 bricks offset on alternate rows, and for n >= 8
    4x4x4 blocks plus one protruding cube: non-convex cells with 100 faces."""
    cell = -np.ones((n, n, n), dtype=np.int64)
    nxt = 0
    for k in range(n):
        for j in range(n):
            i = (j + k) % 2
            if i == 1:
                cell[0, j, k] = nxt
                nxt += 1
            while i < n:
                cell[i : i + 2, j, k] = nxt
                nxt += 1
                i += 2
    # One block per 8^3 cubes keeps the share of large cells equal on every level.
    for bi in range(2, n - 5, 8):
        for bj in range(2, n - 5, 8):
            for bk in range(2, n - 5, 8):
                cell[bi : bi + 4, bj : bj + 4, bk : bk + 4] = nxt
                cell[bi + 4, bj, bk] = nxt  # a protruding cube makes the cell non-convex
                nxt += 1
    _, dense = np.unique(cell.transpose(2, 1, 0).ravel(), return_inverse=True)
    return [int(c) for c in dense]
