import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from autan_vv.errors import VVError
from autan_vv.meshes import cgrid, naca0012

ROOT = Path(__file__).resolve().parents[2]


def _plot3d() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot3d", ROOT / "physics" / "tmr" / "plot3d.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _naca0012() -> tuple[np.ndarray, np.ndarray]:
    # The sharp-trailing-edge NACA 0012 of Vassberg and Jameson, scaled to unit chord.
    x_te = naca0012.trailing_edge()
    x = x_te * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 81)))
    y = naca0012.half_thickness(x)
    y[-1] = 0.0
    upper = np.stack([x, y], axis=1) / x_te
    lower = np.stack([x, -y], axis=1) / x_te
    return upper, lower


@pytest.fixture(scope="module")
def grid() -> cgrid.CGrid:
    upper, lower = _naca0012()
    return cgrid.c_grid(upper, lower, 96, 16, 32, 1.0e-4, 50.0)


def test_a_c_grid_marches_orthogonal_positive_cells_to_its_outer_boundary(
    grid: cgrid.CGrid,
) -> None:
    q = cgrid.quality(grid)
    assert q["min_area"] > 0.0
    assert q["wall_angle_deg"] <= 2.0
    ratio = cgrid._ratio(1.0e-4, 50.0, 32)
    assert q["max_stretch"] <= 1.05 * ratio
    first = np.hypot(grid.x[1] - grid.x[0], grid.y[1] - grid.y[0])
    np.testing.assert_allclose(first[16:-16], 1.0e-4, rtol=0.05)
    radius = np.hypot(grid.x[-1] - 0.5, grid.y[-1])
    assert radius.min() >= 40.0
    # The two halves of the wake cut coincide node for node.
    w = grid.wake_cells
    np.testing.assert_array_equal(grid.x[0, : w + 1], grid.x[0, ::-1][: w + 1])
    np.testing.assert_array_equal(grid.y[0, : w + 1], grid.y[0, ::-1][: w + 1])


def test_coarser_levels_are_every_other_line(grid: cgrid.CGrid) -> None:
    coarse = grid.coarsened()
    np.testing.assert_array_equal(coarse.x, grid.x[::2, ::2])
    assert coarse.x.shape == (17, 65)
    assert coarse.wake_cells == 8
    assert cgrid.quality(coarse)["min_area"] > 0.0


def test_the_converted_mesh_is_closed_with_the_wake_cut_interior(
    grid: cgrid.CGrid, tmp_path: Path
) -> None:
    path = tmp_path / "g.p2dfmt"
    cgrid.write_plot3d(grid, path)
    plot3d = _plot3d()
    x, y = plot3d.read_plot3d_2d(path)
    np.testing.assert_array_equal(x, grid.x)
    mesh = plot3d.extrude(x, y, 1.0, plot3d.ZONES["airfoil"], "xz")
    assert mesh.n_cells == 32 * 128
    zones = {z.name: z for z in mesh.zones}
    assert set(zones) == {"airfoil", "farfield", "sides"}
    # Only the airfoil's faces remain on j = 0: the wake's are interior.
    assert zones["airfoil"].count == 96


def test_open_surfaces_and_impossible_spacings_are_refused() -> None:
    upper, lower = _naca0012()
    shifted = lower.copy()
    shifted[-1, 1] += 1e-3
    with pytest.raises(VVError) as error:
        cgrid.c_grid(upper, shifted, 96, 16, 32, 1.0e-4, 50.0)
    assert error.value.code == "V0311"
    with pytest.raises(VVError) as error:
        cgrid.c_grid(upper, lower, 96, 16, 32, 10.0, 50.0)
    assert error.value.code == "V0310"


def test_the_rae2822_section_is_the_measured_columns_with_the_lower_surface_negative(
    tmp_path: Path,
) -> None:
    # File 1121 holds X/C, YUS/C, YLS/C and the two design columns (the tape's write-up).
    tape = tmp_path / "f8621.txt"
    tape.write_text(
        "----------------- FILE NUMBER  1121-----------------\n"
        " 0.100000E 01 0.600000E-01 0.500000E-01 0.700000E-01 0.900000E-01\n"
        " 0.000000E 00 0.000000E 00 0.000000E 00 0.000000E 00 0.000000E 00\n"
        "     0     0    10     0     0\n"
        "  5000 10000 10000 10000 10000\n"
        " 10000     0    10     0     0\n"
        "-------------- END OF FILE NUMBER  1121-------------\n"
    )
    upper, lower = cgrid.rae2822(tape, ())
    np.testing.assert_allclose(upper[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(upper[1], [0.5, 0.06])
    np.testing.assert_allclose(lower[1], [0.5, -0.05])
    # Leading and trailing edges: the mean of the two surfaces' values.
    np.testing.assert_allclose(upper[0], [0.0, -0.5 * 0.05e-3])
    np.testing.assert_array_equal(upper[0], lower[0])
    np.testing.assert_array_equal(upper[-1], lower[-1])
