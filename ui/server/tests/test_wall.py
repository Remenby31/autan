"""Wall surface geometry: triangles cover each polygon exactly, areas follow results.md."""

import numpy as np
import numpy.typing as npt
from autan_ui.wall import _triangles, _zone

# A C-shaped octagon in z = 0 (notch on the right), counter-clockwise.
C_SHAPE = np.array(
    [[0, 0, 0], [3, 0, 0], [3, 1, 0], [1, 1, 0], [1, 2, 0], [3, 2, 0], [3, 3, 0], [0, 3, 0]],
    dtype=np.float64,
)
C_AREA = 7.0


def shoelace(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_a_concave_polygon_is_covered_without_crossing_its_notch() -> None:
    polygon = np.arange(8)
    triangles = _triangles(polygon, C_SHAPE, np.array([0.0, 0.0, 2 * C_AREA]))
    areas = [shoelace(C_SHAPE[t]) for t in triangles]
    assert len(triangles) == 6
    assert all(a > 0 for a in areas)  # same winding as the polygon
    assert sum(areas) == C_AREA  # no overlap, no gap: the notch (area 2) is not covered


def test_face_areas_and_node_values_follow_the_contract() -> None:
    start = np.array([0, 8], dtype=np.int64)
    rotated = np.roll(np.arange(8, dtype=np.int64), 3)
    for connectivity in (np.arange(8, dtype=np.int64), rotated):
        zone = _zone(
            "c",
            np.array([0]),
            start,
            connectivity,
            C_SHAPE,
            {"Cp": np.array([0.5])},
            np.zeros(3),
        )
        assert np.allclose(zone.fields["Cp"], 0.5)
        corners = zone.positions[zone.triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        assert abs(0.5 * cross[:, 2].sum() - C_AREA) < 1e-5


def test_positions_far_from_the_origin_keep_their_shape() -> None:
    far = C_SHAPE + 1e9
    zone = _zone(
        "c",
        np.array([0]),
        np.array([0, 8], dtype=np.int64),
        np.arange(8, dtype=np.int64),
        far,
        {"Cp": np.array([0.0])},
        np.full(3, 1e9),
    )
    assert np.allclose(zone.positions, C_SHAPE, atol=1e-6)
