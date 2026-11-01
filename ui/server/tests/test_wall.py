"""Wall surface geometry: triangles cover each polygon exactly, areas follow results.md."""

import numpy as np
import numpy.typing as npt
import pytest
from autan_ui.wall import _ear_clip, _zone

# A C-shaped octagon in z = 0 (notch on the right), counter-clockwise.
C_SHAPE = np.array(
    [[0, 0, 0], [3, 0, 0], [3, 1, 0], [1, 1, 0], [1, 2, 0], [3, 2, 0], [3, 3, 0], [0, 3, 0]],
    dtype=np.float64,
)
C_AREA = 7.0


def shoelace(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@pytest.mark.parametrize("plane", ["xy", "yz", "zx"])
def test_a_concave_polygon_is_covered_without_crossing_its_notch(plane: str) -> None:
    # The same polygon in each coordinate plane, counter-clockwise about +z, +x, +y.
    order = {"xy": [0, 1, 2], "yz": [2, 0, 1], "zx": [1, 2, 0]}[plane]
    shape = C_SHAPE[:, order]
    zone = _zone(
        "c", np.array([0]), np.array([0, 8]), np.arange(8), shape, {"Cp": np.zeros(1)}, np.zeros(3)
    )
    corners = zone.positions[zone.triangles].astype(np.float64)
    vector = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) / 2
    normal = np.zeros(3)
    normal[order.index(2)] = 1.0
    areas = vector @ normal
    assert len(zone.triangles) == 6
    assert (areas > 0).all()  # same winding as the polygon
    assert abs(areas.sum() - C_AREA) < 1e-6  # no overlap, no gap: the notch is not covered


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


def test_ear_clipping_alone_covers_the_polygon() -> None:
    triangles = _ear_clip(np.arange(8), C_SHAPE, np.array([0.0, 0.0, 2 * C_AREA]))
    assert sum(shoelace(C_SHAPE[t]) for t in triangles) == C_AREA


def test_positions_far_from_the_origin_keep_their_shape() -> None:
    far = C_SHAPE + 1e9
    zone = _zone(
        "c",
        np.array([0]),
        np.array([0, 8], dtype=np.int64),
        np.arange(8, dtype=np.int64),
        far,
        {"Cp": np.array([0.25])},
        np.full(3, 1e9),
    )
    assert np.allclose(zone.positions, C_SHAPE, atol=1e-6)
    assert np.allclose(zone.fields["Cp"], 0.25)
