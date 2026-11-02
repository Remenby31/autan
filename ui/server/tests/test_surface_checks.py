from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from autan_ui.mesh_surface import _broken, _check_polygons


# The surface checks as they were before they were vectorised (#351), kept verbatim as the
# reference: the current checks must give the same verdict, with the same message, on every
# surface below.
def _previous_check_polygons(
    offsets: npt.NDArray[np.int64], nodes: npt.NDArray[np.int64], positions: npt.NDArray[Any]
) -> None:
    # Each face a polygon of distinct nodes with an area; the boundary closed and consistently
    # oriented (every edge traversed as often in each direction) and outward: the volume it
    # encloses, ∮ x·n dA / 3, is positive (mesh.md#boundary-surface).
    count = np.diff(offsets)
    face = np.repeat(np.arange(len(count)), count)
    width = int(nodes.max()) + 1 if len(nodes) else 1
    if len(np.unique(face * width + nodes)) != len(nodes):
        msg = "a face repeats a node"
        raise _broken(msg)
    position = np.arange(len(nodes)) - np.repeat(offsets[:-1], count)
    following = nodes[offsets[:-1][face] + (position + 1) % count[face]]
    edges, there = np.unique(nodes * width + following, return_counts=True)
    back = dict(zip(*np.unique(following * width + nodes, return_counts=True), strict=True))
    if any(back.get(edge, 0) != n for edge, n in zip(edges, there, strict=True)):
        msg = "the boundary is not closed with a consistent orientation"
        raise _broken(msg)
    points = positions.astype(np.float64)
    # Each face's normal from its own first node, so that the area of a small face far from the
    # origin does not drown in cancellation; compared with the face's longest edge.
    corner = points[nodes[offsets[:-1]]][face]
    a, b = points[nodes] - corner, points[following] - corner
    normal = (
        np.stack(
            [np.bincount(face, np.cross(a, b)[:, k], minlength=len(count)) for k in range(3)],
            axis=1,
        )
        / 2
    )
    area = np.linalg.norm(normal, axis=1)
    edge = np.maximum.reduceat(np.linalg.norm(b - a, axis=1), offsets[:-1])
    if not (np.isfinite(area).all() and np.isfinite(edge).all()):
        msg = "a face's geometry overflows double precision"
        raise _broken(msg)
    if np.any(area <= 1e3 * np.finfo(np.float64).eps * edge**2):
        msg = "a face has no area"
        raise _broken(msg)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    centroid = (
        np.stack(
            [
                np.bincount(face, points[nodes][:, k] - centre[k], minlength=len(count))
                for k in range(3)
            ],
            axis=1,
        )
        / count[:, None]
    )
    volume = np.einsum("ij,ij->", centroid, normal)
    if not np.isfinite(volume) or volume <= 0:
        msg = "the boundary is not oriented outward: the volume it encloses is not positive"
        raise _broken(msg)


def _cube(n: int, rng: np.random.Generator) -> tuple[list[list[int]], npt.NDArray[np.float64]]:
    # A closed cube surface, outward: n² cells per side, some split into two triangles, nodes
    # shared and numbered in a random order.
    g = np.linspace(0.0, 1.0, n + 1)
    index: dict[tuple[float, ...], int] = {}
    points: list[npt.NDArray[np.float64]] = []

    def node(p: npt.NDArray[np.float64]) -> int:
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(points)
            points.append(p)
        return index[key]

    faces = []
    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for side in (0.0, 1.0):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = side, g[i + di], g[j + dj]
                        quad.append(node(p))
                    # Seen from outside, counter-clockwise: (u, v, axis) right-handed or not.
                    right = (u, v, axis) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
                    if (side == 1.0) != right:
                        quad.reverse()
                    if rng.random() < 0.3:
                        faces += [quad[:3], [quad[0], quad[2], quad[3]]]
                    else:
                        faces.append(quad)
    order = rng.permutation(len(points))
    faces = [[int(order[k]) for k in face] for face in faces]
    positions = np.empty((len(points), 3))
    positions[order] = np.array(points)
    return faces, positions


MUTATIONS = [
    "none",
    "rotate",
    "reverse",
    "drop",
    "duplicate",
    "repeat",
    "swap",
    "collapse",
    "flatten",
    "scale",
    "reverse all",
]


def _mutate(
    faces: list[list[int]],
    positions: npt.NDArray[np.float64],
    change: str,
    rng: np.random.Generator,
) -> tuple[list[list[int]], npt.NDArray[np.float64]]:
    faces = [list(f) for f in faces]
    positions = positions.copy()
    k = int(rng.integers(len(faces)))
    if change == "rotate":  # Still valid: the same polygon from another first node.
        faces[k] = faces[k][1:] + faces[k][:1]
    elif change == "reverse":
        faces[k] = faces[k][::-1]
    elif change == "drop":
        del faces[k]
    elif change == "duplicate":
        faces.append(list(faces[k]))
    elif change == "repeat":
        faces[k][1] = faces[k][0]
    elif change == "swap":
        faces[k][0], faces[k][1] = faces[k][1], faces[k][0]
    elif change == "collapse":
        positions[faces[k][1]] = positions[faces[k][0]]
    elif change == "flatten":
        positions[faces[k][2]] = (positions[faces[k][0]] + positions[faces[k][1]]) / 2
    elif change == "scale":  # Still valid: small faces on a large body (#340).
        positions *= np.array([1e-6, 1e-6, 1.0])
    elif change == "reverse all":
        faces = [f[::-1] for f in faces]
    return faces, positions


def _verdict(check: Any, faces: list[list[int]], positions: npt.NDArray[np.float64]) -> str:
    offsets = np.concatenate([[0], np.cumsum([len(f) for f in faces])]).astype(np.int64)
    nodes = np.array([k for f in faces for k in f], dtype=np.int64)
    try:
        check(offsets, nodes, positions)
    except ValueError as error:
        return str(error)
    return "accepted"


@pytest.mark.parametrize("change", MUTATIONS)
def test_the_checks_judge_every_surface_as_before(change: str) -> None:
    rng = np.random.default_rng(MUTATIONS.index(change))
    verdicts = set()
    for _ in range(25):
        faces, positions = _cube(int(rng.integers(1, 6)), rng)
        faces, positions = _mutate(faces, positions, change, rng)
        expected = _verdict(_previous_check_polygons, faces, positions)
        assert _verdict(_check_polygons, faces, positions) == expected, change
        verdicts.add(expected)
    # The generated surfaces do exercise the change: a valid one is never refused, a breaking
    # one is refused at least once (collapsing a quad's node can leave a valid polygon).
    if change in ("none", "rotate", "scale"):
        assert verdicts == {"accepted"}, verdicts
    else:
        assert verdicts - {"accepted"}, verdicts
