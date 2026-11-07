import importlib.util
import math
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from autan_vv.cases import fetch, get_case
from autan_vv.errors import VVError
from autan_vv.meshes import construct2d

ROOT = Path(__file__).resolve().parents[2]


def _plot3d() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plot3d", ROOT / "physics" / "tmr" / "plot3d.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _polar(ni: int, nj: int, r0: float, q: float, dtheta: float) -> construct2d.CGrid:
    # Nodes on circles r_j = r0 q^j and rays theta_i = -i dtheta: an orthogonal grid, right-handed
    # in (i, j), whose every criterion has a closed form.
    r = r0 * q ** np.arange(nj)
    theta = -dtheta * np.arange(ni)
    return construct2d.CGrid(np.outer(r, np.cos(theta)), np.outer(r, np.sin(theta)), 0)


def test_the_criteria_of_an_orthogonal_polar_grid_take_their_closed_forms() -> None:
    ni, nj, r0, q, dtheta = 41, 9, 1.0, 1.3, math.pi / 60
    m = construct2d.criteria(_polar(ni, nj, r0, q, dtheta))
    r_out = r0 * q ** (nj - 1)
    chord = 2.0 * r_out * math.sin(dtheta / 2)
    tol = 1e3 * np.finfo(np.float64).eps
    assert m["min_area"] > 0.0
    assert m["n4_min_angle_deg"] == pytest.approx(90.0, abs=1e-6)
    assert m["wall_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert m["n2_outer_min_over_mean"] == pytest.approx(1.0, rel=tol)
    assert m["n3_outer_aspect"] == pytest.approx(r_out * (1 - 1 / q) / chord, rel=tol)
    assert m["max_stretch"] == pytest.approx(q, rel=tol)
    assert m["first_step_min"] == pytest.approx(r0 * (q - 1), rel=tol)
    assert m["first_step_max"] == pytest.approx(r0 * (q - 1), rel=tol)


def test_a_sheared_uneven_grid_has_its_closed_form_criteria_and_fails_n2_to_n4() -> None:
    # x = s_i + t_j cos(phi), y = t_j sin(phi): xi-lines along x with steps ds_i, eta-lines at phi
    # with geometric steps dt_j, so N1 to N4, the stretching and the wall angle are closed forms.
    phi, q, dt0 = math.radians(40.0), 1.5, 0.2
    ds = np.array([1.0, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0])
    s = np.concatenate([[0.0], np.cumsum(ds)])
    dt = dt0 * q ** np.arange(6)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    grid = construct2d.CGrid(
        s[None, :] + t[:, None] * math.cos(phi),
        np.broadcast_to(t[:, None] * math.sin(phi), (7, 9)),
        0,
    )
    m = construct2d.criteria(grid)
    tol = 1e3 * np.finfo(np.float64).eps
    extent = max(s[-1] + t[-1] * math.cos(phi), t[-1] * math.sin(phi))
    assert m["min_area"] == pytest.approx(ds.min() * dt0 * math.sin(phi), rel=tol)
    assert m["n1_min_edge_over_extent"] == pytest.approx(ds.min() / extent, rel=tol)
    assert m["n2_outer_min_over_mean"] == pytest.approx(ds.min() / ds.mean(), rel=tol)
    assert m["n3_outer_aspect"] == pytest.approx(dt[-1] / ds.min(), rel=tol)
    assert m["n4_min_angle_deg"] == pytest.approx(40.0, rel=tol)
    assert m["wall_angle_deg"] == pytest.approx(50.0, rel=tol)
    assert m["max_stretch"] == pytest.approx(q, rel=tol)
    checks = construct2d.passes("fine", m)
    keys = ("N1", "N2", "N3", "N4", "areas", "stretch", "wall_angle")
    assert {k: checks[k] for k in keys} == {
        "N1": True,
        "N2": False,
        "N3": False,
        "N4": False,
        "areas": True,
        "stretch": False,
        "wall_angle": False,
    }


def _construct2d_file(path: Path, grid: construct2d.CGrid) -> list[str]:
    # Construct2D 2.1.4 (util.f90, write_srf_grid) writes `imax jmax`, then x and y over j with i
    # from imax down to 1, in ES17.8: a right-handed grid written that way reads left-handed.
    nj, ni = grid.x.shape
    values = [f"{v:17.8E}" for a in (grid.x, grid.y) for v in a[:, ::-1].ravel()]
    path.write_text(f" {ni} {nj}\n" + "\n".join(values) + "\n")
    return values


def test_construct2d_output_is_read_right_handed_and_a_short_file_is_refused(
    tmp_path: Path,
) -> None:
    ni = construct2d.NSRF + 2 * 4
    grid = _polar(ni, 3, 1.0, 1.5, math.pi / ni)
    path = tmp_path / "rae2822.p3d"
    values = _construct2d_file(path, grid)
    read = construct2d.read_grid(path)
    np.testing.assert_allclose(read.x, grid.x, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(read.y, grid.y, rtol=1e-8, atol=1e-12)
    assert read.wake_cells == 4
    for text in (f" {ni} 3\n" + "\n".join(values[:-1]), "", f" {ni}\n", " 3 x\n", f" {ni} 3\n NaN"):
        path.write_text(text + "\n")
        with pytest.raises(VVError) as err:
            construct2d.read_grid(path)
        assert err.value.code == "V0311"


def test_the_inputs_are_the_selected_namelist_and_a_surface_in_construct2d_order(
    tmp_path: Path,
) -> None:
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 11)))
    y = 0.6 * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    y[-1] = 0.0
    upper, lower = np.stack([x, y], axis=1), np.stack([x, -y], axis=1)
    construct2d.write_inputs(upper, lower, tmp_path)
    dat = (tmp_path / "rae2822.dat").read_text().splitlines()
    points = np.array([[float(v) for v in line.split()] for line in dat[1:]])
    # One header line, then 21 points from the upper trailing edge through the leading edge to the
    # lower trailing edge; Construct2D keeps that order when y(2) > y(n - 1) (util.f90).
    assert points.shape == (21, 2)
    np.testing.assert_array_equal(points[0], points[-1])
    np.testing.assert_array_equal(points[10], [0.0, 0.0])
    assert points[1, 1] > points[-2, 1]
    nml = (tmp_path / "grid_options.in").read_text()
    keys = dict(re.findall(r"^\s+(\w+) = (\S+)$", nml, flags=re.MULTILINE))
    # The candidate selected on #449: elliptic, fwki = 700, fwkl = 1, stp1 = 5000.
    assert keys["slvr"] == "'ELLP'"
    assert keys["topo"] == "'CGRD'"
    assert (float(keys["fwki"]), float(keys["fwkl"]), int(keys["stp1"])) == (700.0, 1.0, 5000)
    assert (int(keys["nsrf"]), int(keys["nwke"]), int(keys["jmax"])) == (481, 128, 129)
    # Construct2D's wall_distance (surface_util.f90) turns ypls back into 1e-6 chord.
    re_c, cfrc, ypls = float(keys["recd"]), float(keys["cfrc"]), float(keys["ypls"])
    cf = (2.0 * math.log10(re_c) - 0.65) ** -2.3
    assert ypls * cfrc / (re_c * math.sqrt(0.5 * cf)) == pytest.approx(1.0e-6, rel=1e-14)


def _tape(tmp_path: Path) -> Path:
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
    return tape


def test_the_rae2822_section_is_the_measured_columns_with_the_lower_surface_negative(
    tmp_path: Path,
) -> None:
    upper, lower = construct2d.rae2822(_tape(tmp_path), ())
    np.testing.assert_allclose(upper[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(upper[1], [0.5, 0.06])
    np.testing.assert_allclose(lower[1], [0.5, -0.05])
    # Leading and trailing edges: the mean of the two surfaces' values.
    np.testing.assert_allclose(upper[0], [0.0, -0.5 * 0.05e-3])
    np.testing.assert_array_equal(upper[0], lower[0])
    np.testing.assert_array_equal(upper[-1], lower[-1])


@pytest.mark.network
@pytest.mark.slow
def test_the_tmr_naca0012_grid_passes_n1_to_n4_and_converts_to_a_closed_slab(
    tmp_path: Path,
) -> None:
    # The TMR's coarsest NACA 0012 C-grid, a grid not generated by us, must meet the criteria of
    # #449 (they were bounded with it) and go through plot3d.py with its wake cut interior.
    case = get_case("tmr-2dn00-naca0012")
    cache = Path("~/.cache/autan-vv").expanduser()
    fetch(case, cache, roles=("grid",))
    plot3d = _plot3d()
    x, y = plot3d.read_plot3d_2d(cache / case.name / "n0012_113-33.p2dfmt")
    ni = x.shape[1]
    same = (x[0] == x[0, ::-1]) & (y[0] == y[0, ::-1])
    w = int(np.argmin(same[: ni // 2])) - 1
    grid = construct2d.CGrid(x, y, w)
    checks = construct2d.passes("coarse", construct2d.criteria(grid))
    assert all(checks[k] for k in ("N1", "N2", "N3", "N4", "areas"))
    path = tmp_path / "g.p2dfmt"
    construct2d.write_plot3d(grid, path)
    rx, ry = plot3d.read_plot3d_2d(path)
    np.testing.assert_array_equal(rx, x)
    np.testing.assert_array_equal(ry, y)
    mesh = plot3d.extrude(rx, ry, 1.0, plot3d.ZONES["airfoil"], "xz")
    assert mesh.n_cells == (ni - 1) * (x.shape[0] - 1)
    zones = {z.name: z for z in mesh.zones}
    assert set(zones) == {"airfoil", "farfield", "sides"}
    # Only the airfoil's faces remain on j = 0: the wake's are interior.
    assert zones["airfoil"].count == ni - 1 - 2 * w


def test_a_run_that_writes_no_grid_fails_even_beside_an_earlier_grid(tmp_path: Path) -> None:
    # A Construct2D that exits 0 without writing (the 2.1.4 behaviour when its spline fails): the
    # earlier run's grid in the same directory must not be taken for this run's.
    exe = tmp_path / "construct2d"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    ni = construct2d.NSRF + 2 * 4
    _construct2d_file(out / "rae2822.p3d", _polar(ni, 3, 1.0, 1.5, math.pi / ni))
    with pytest.raises(VVError) as err:
        construct2d.generate(exe, _tape(tmp_path), out)
    assert err.value.code == "V0311"
    assert not (out / "rae2822.p3d").exists()


def test_a_grid_other_than_the_pinned_one_is_refused_and_leaves_no_level(tmp_path: Path) -> None:
    # A Construct2D that writes a complete grid, of other dimensions (V0311) or of the namelist's
    # dimensions but other coordinates (V0312): no level is left for plot3d.py to convert.
    out = tmp_path / "out"
    out.mkdir()
    for nj, ni, code in (
        (3, construct2d.NSRF + 2 * 4, "V0311"),
        (construct2d.JMAX, construct2d.NSRF + 2 * construct2d.NWKE, "V0312"),
    ):
        grid = tmp_path / "grid.p3d"
        _construct2d_file(grid, _polar(ni, nj, 1.0, 1.05, math.pi / ni))
        exe = tmp_path / "construct2d"
        exe.write_text(f"#!/bin/sh\ncp {grid} rae2822.p3d\n")
        exe.chmod(0o755)
        with pytest.raises(VVError) as err:
            construct2d.generate(exe, _tape(tmp_path), out)
        assert err.value.code == code
        assert not list(out.glob("rae2822-*.p2dfmt"))
