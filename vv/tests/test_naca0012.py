"""NACA 0012 of Vassberg and Jameson (2010): geometry and O-meshes against the paper, force gates
against the paper's own grid sequences (J. Aircraft 47(4):1152-1166, doi:10.2514/1.46737)."""

import math
from pathlib import Path

import autan
import numpy as np
import pytest
from autan.units import m
from autan_vv.cases import naca0012 as cases
from autan_vv.errors import VVError
from autan_vv.meshes import naca0012
from autan_vv.runs import mesh_info
from scipy import integrate, optimize

gmsh = pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")

EPS = float(np.finfo(np.float64).eps)


def naca_y(x: np.ndarray) -> np.ndarray:
    """Eq. 1 of the paper (Abbott and von Doenhoff), transcribed independently of the generator."""
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def test_geometry_matches_the_values_the_paper_prints() -> None:
    kt = naca0012.KarmanTrefftz()
    # Eqs. 4-8, printed to 10 decimals. The root of eq. 1 is 1.00893041137; the paper prints
    # 1.0089304115 and evaluates the slope (eq. 5) and angle (eq. 6) there, to every printed digit.
    assert kt.x_te == pytest.approx(1.0089304115, abs=2e-10)
    assert float(naca_y(np.array(kt.x_te))) == pytest.approx(0.0, abs=4 * EPS)
    h = 1e-6
    slope = float(naca_y(np.array(1.0089304115 + h)) - naca_y(np.array(1.0089304115 - h))) / (2 * h)
    assert slope == pytest.approx(-0.1418768821, abs=1e-9)
    tau = 2.0 * math.atan(abs(slope))
    assert math.degrees(tau) == pytest.approx(16.1501066197, abs=1e-7)
    assert (2.0 - kt.power) * math.pi == pytest.approx(0.2818725, abs=1e-7)
    xmax = optimize.minimize_scalar(
        lambda x: -float(naca_y(np.array(x))),
        bounds=(0.2, 0.4),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    assert xmax == pytest.approx(0.2998278780, abs=1e-8)
    assert 2.0 * float(naca_y(np.array(xmax))) == pytest.approx(0.1200345462, abs=1e-10)
    # Eq. 11 in the paper's normalisation (zeta1 = 0.77043505, zeta2 = 0.24642903), which differs
    # from ours by a similarity: the leading edge maps to 0.20139626.
    zeta_le = complex(kt.to_circle(np.array(0j)))
    assert zeta_le.imag == 0.0
    assert 0.24642903 + (0.77043505 - 0.24642903) * zeta_le.real == pytest.approx(
        0.20139626, abs=1e-6
    )
    z = np.array([0.3 + 0.7j, -2.0 - 5.0j, 40.0 + 1.0j])
    np.testing.assert_allclose(kt.to_airfoil(kt.to_circle(z)), z, rtol=1e-13)


def _in_plane(mesh_points: np.ndarray, nc: int) -> np.ndarray:
    layer = nc * (nc + 1)
    p = mesh_points[:layer]
    return (p[:, 0] + 1j * p[:, 2]).reshape(nc, nc + 1)


@pytest.mark.parametrize("nc", [32, 128])
def test_o_mesh_has_the_papers_properties(nc: int) -> None:
    mesh, _ = naca0012.generate_level(nc)
    z = _in_plane(mesh.points, nc)
    np.testing.assert_array_equal(mesh.points[: nc * (nc + 1), 1], 0.0)
    surface = z[:, 0]
    # Surface nodes on the analytic airfoil (eq. 1), the sharp trailing edge and the leading edge.
    x = surface.real
    assert x.min() == 0.0
    assert x.max() == float(f"{naca0012.KarmanTrefftz().x_te:.16g}")
    np.testing.assert_allclose(np.abs(surface.imag), naca_y(x), rtol=0, atol=4 * EPS)
    # Exact symmetry about the chord (eqs. 15-16).
    np.testing.assert_array_equal(z[1:][::-1], np.conj(z[1:]))
    # Far field about 150 chords away (Section IV).
    assert 145.0 < np.abs(z[:, -1] - 0.5).min() < np.abs(z[:, -1] - 0.5).max() < 155.0
    # Cells of aspect ratio one, away from the trailing-edge singularity of the mapping.
    di = np.abs(np.roll(z, -1, axis=0) - z)[:, :-1]
    dj = np.abs(np.diff(z, axis=1))
    aspect = np.maximum(di / dj, dj / di)
    assert np.median(aspect) < 1.0 + 4.0 / nc
    assert aspect.max() < 2.1

    # Volumes: span x (area inside the far-field polygon - airfoil polygon), shoelace formula.
    def shoelace(ring: np.ndarray) -> float:
        return 0.5 * float(
            np.sum(ring.real * np.roll(ring.imag, -1) - np.roll(ring.real, -1) * ring.imag)
        )

    g = mesh.geometry
    assert g.cell_volume.min() > 0.0
    expected = naca0012.SPAN * (shoelace(z[:, -1]) - shoelace(surface))
    assert g.cell_volume.sum() == pytest.approx(expected, rel=1e-12)
    zones = {zone.name: zone for zone in mesh.zones}
    assert list(zones) == list(naca0012.ZONES)
    assert zones["airfoil"].count == zones["farfield"].count == nc
    assert zones["back"].count == zones["front"].count == nc * nc
    # The airfoil polygon converges to the analytic section area at second order (error 0.50/NC^2).
    x_te = naca0012.KarmanTrefftz().x_te
    area = 2.0 * integrate.quad(lambda t: float(naca_y(np.array(t))), 0, x_te)[0]
    assert abs(shoelace(surface) - area) < 0.55 / nc**2


def test_coarser_levels_are_the_finer_ones_with_every_other_line_removed() -> None:
    fine, _ = naca0012.plane_grid(128)
    coarse, _ = naca0012.plane_grid(64)
    np.testing.assert_array_equal(fine[::2, ::2], coarse)


def test_family_is_written_and_read_back_through_gmsh(tmp_path: Path) -> None:
    family, meshes = naca0012.generate_family((8, 16), tmp_path)
    assert [lv.n_cells for lv in family.levels] == [64, 256]
    # sqrt(area / cells): the far-field polygon of 8 sides is 8 % smaller than that of 16.
    assert family.levels[0].h == pytest.approx(2.0 * family.levels[1].h, rel=0.05)
    path = tmp_path / family.levels[0].file
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        names = {gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()}
        assert names == {*naca0012.ZONES, "fluid"}
        types, tags, _ = gmsh.model.mesh.getElements(3)
        assert [int(t) for t in types] == [5]
        assert len(tags[0]) == meshes[0].n_cells
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords).reshape(-1, 3)[np.argsort(node_tags)]
        np.testing.assert_array_equal(xyz, meshes[0].points)
        # Every face of the hexahedra used once is labelled by exactly one zone of the file.
        hexes = np.asarray(gmsh.model.mesh.getElements(3)[2][0], dtype=np.int64).reshape(-1, 8)
        local = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        faces = np.sort(np.concatenate([hexes[:, list(f)] for f in local]), axis=1)
        keys, counts = np.unique(faces, axis=0, return_counts=True)
        outer = {tuple(k) for k in keys[counts == 1].tolist()}
        labelled: list[tuple[int, ...]] = []
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                quads = gmsh.model.mesh.getElements(2, entity)[2][0]
                labelled += [tuple(q) for q in np.sort(np.reshape(quads, (-1, 4)), axis=1).tolist()]
        assert len(labelled) == len(set(labelled))
        assert set(labelled) == outer
    finally:
        gmsh.finalize()


def _levels(nc: list[int], values: dict[str, list[float]]) -> list[cases.LevelForces]:
    n = len(nc)
    get = {k: values.get(k, [0.0] * n) for k in cases.COEFFICIENTS}
    return [
        cases.LevelForces(
            f"nc{c}",
            c * c,
            1.0 / c,
            get["cl"][i],
            get["cd"][i],
            get["cmy"][i],
            "converged",
            "cuda",
            "0" * 40,
            "fp64",
            "",
            "",
        )
        for i, c in enumerate(nc)
    ]


# Three finest grids of the paper's tables and the continuum and order the paper derives from them.
FLO82_TRANSONIC = [1024, 2048, 4096]


@pytest.mark.parametrize(
    ("nc", "coefficient", "values", "estimate", "order"),
    [
        # Table 3, FLO82, M 0.8, alpha 0 deg.
        (FLO82_TRANSONIC, "cd", [0.008338967, 0.008341760, 0.008342211], 0.008342298, 2.631),
        # Table 3, FLO82, M 0.8, alpha 1.25 deg.
        (FLO82_TRANSONIC, "cl", [0.360812844, 0.358281928, 0.357142338], 0.356208937, 1.151),
        (FLO82_TRANSONIC, "cd", [0.022934404, 0.022799839, 0.022737860], 0.022684938, 1.118),
        # Table 2, FLO82, M 0.5, alpha 1.25 deg.
        (FLO82_TRANSONIC, "cl", [0.180382615, 0.180354386, 0.180347832], 0.180345850, 2.107),
        # Table 9, CFL3D with the point vortex, M 0.8, alpha 1.25 deg (finest grid 2048).
        ([512, 1024, 2048], "cd", [0.023082744, 0.022884500, 0.022782607], 0.022674853, 0.960),
    ],
)
def test_continuum_reproduces_the_papers_extrapolation(
    nc: list[int], coefficient: str, values: list[float], estimate: float, order: float
) -> None:
    result = cases.continuum(_levels(nc, {coefficient: values}))[coefficient]
    assert result["estimate"] == pytest.approx(estimate, abs=2e-9)
    assert result["order"] == pytest.approx(order, abs=1e-3)


def test_gates_on_the_papers_flo82_sequences() -> None:
    transonic_lift = _levels(
        FLO82_TRANSONIC,
        {
            "cl": [0.360812844, 0.358281928, 0.357142338],
            "cd": [0.022934404, 0.022799839, 0.022737860],
        },
    )
    estimates = cases.continuum(transonic_lift)
    for name in ("naca0012-m080-a125-cl", "naca0012-m080-a125-cd"):
        assert cases.check(cases.GATES[name], transonic_lift, estimates)["pass"]
    nonlifting = _levels(FLO82_TRANSONIC, {"cd": [0.008338967, 0.008341760, 0.008342211]})
    gate = cases.GATES["naca0012-m080-a000-cd"]
    assert cases.check(gate, nonlifting, cases.continuum(nonlifting))["pass"]
    # A continuum CD two counts off fails.
    shifted = _levels(FLO82_TRANSONIC, {"cd": [0.008538967, 0.008541760, 0.008542211]})
    assert not cases.check(gate, shifted, cases.continuum(shifted))["pass"]
    # Table 2, alpha 1.25 deg: |CD| decreases from 512 to 4096 and extrapolates to +5e-8 from
    # below, but rises from 256 to 512, which the full family must show.
    subcritical_nc = [256, 512, 1024, 2048, 4096]
    subcritical_cd = [-0.000011045, -0.000015534, -0.000005301, -0.000001481, -0.000000388]
    gate = cases.GATES["naca0012-m050-a125-cd"]
    tail = _levels(subcritical_nc[1:], {"cd": subcritical_cd[1:]})
    verdict = cases.check(gate, tail, cases.continuum(tail))
    assert verdict["pass"]
    assert verdict["value"] == pytest.approx(0.000000050, abs=2e-9)
    assert cases.continuum(tail)["cd"]["order_if_zero"] == pytest.approx(1.932, abs=1e-3)
    full = _levels(subcritical_nc, {"cd": subcritical_cd})
    assert not cases.check(gate, full, cases.continuum(full))["pass"]


def test_every_gate_is_stated_in_acceptance_md() -> None:
    text = (Path(__file__).parents[2] / "docs/validation/acceptance.md").read_text()
    for gate in cases.GATES.values():
        row = next((line for line in text.splitlines() if f"`{gate.id}`" in line), None)
        assert row is not None, gate.id
        for value in (gate.target, gate.lower, gate.upper):
            if value:
                assert f"{value:g}" in row, gate.id


def test_study_reads_a_family_of_runs(tmp_path: Path) -> None:
    family, meshes = naca0012.generate_family((8, 16, 32, 64), tmp_path / "meshes")
    runs = []
    for level, mesh in zip(family.levels, meshes, strict=True):
        path = tmp_path / "meshes" / level.file
        case = cases.case("m080-a000", str(path))
        info = mesh_info(mesh, path, "gmsh")
        runs.append(case.run(info, backend=autan.FakeBackend(), runs=tmp_path / "runs").path)
    report = cases.study(runs)
    assert report["condition"] == "m080-a000"
    assert [lv["n_cells"] for lv in report["levels"]] == [64, 256, 1024, 4096]
    assert all(len(lv["case_sha256"]) == len(lv["mesh_sha256"]) == 64 for lv in report["levels"])
    assert not report["evidence"]
    assert [v["gate"] for v in report["verdicts"]] == ["naca0012-m080-a000-cd"]
    with pytest.raises(VVError) as err:
        cases.study(runs[:3])
    assert err.value.code == "V0803"
    # A case that is not the benchmark's (a wall instead of the far field, or a loose
    # convergence criterion) is refused, whatever its forces.
    first = tmp_path / "meshes" / family.levels[0].file
    info = mesh_info(meshes[0], first, "gmsh")
    loose = cases.case("m080-a000", str(first), residual_orders=1.0)
    other_bc = autan.Case("naca0012-m080-a000", mesh=str(first))
    other_bc.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
    other_bc.freestream(mach=0.8, pressure=101325.0 * autan.units.ureg.Pa, ref_length=1.0 * m)
    other_bc.time = autan.Steady(max_iterations=200000, residual_orders=10.0)
    other_bc.boundary("airfoil").slip()
    other_bc.boundary("farfield").slip()
    other_bc.boundary("back").symmetry()
    other_bc.boundary("front").symmetry()
    other_bc.monitor(
        autan.forces("airfoil", ref_area=1.0 * m * m, moment_center=(0.25 * m, 0 * m, 0 * m))
    )
    for bad in (loose, other_bc):
        run = bad.run(info, backend=autan.FakeBackend(), runs=tmp_path / "runs").path
        with pytest.raises(VVError) as err:
            cases.study([*runs[1:], run])
        assert err.value.code == "V0801"
    other = cases.case("m080-a125", str(tmp_path / "meshes" / family.levels[0].file))
    info = mesh_info(meshes[0], tmp_path / "meshes" / family.levels[0].file, "gmsh")
    odd = other.run(info, backend=autan.FakeBackend(), runs=tmp_path / "runs").path
    with pytest.raises(VVError) as err:
        cases.study([*runs[1:], odd])
    assert err.value.code == "V0802"


def test_lifting_cases_carry_the_point_vortex_of_the_flo82_references() -> None:
    for condition, c in cases.CONDITIONS.items():
        rule = next(
            b for b in cases.case(condition, "m.msh").spec.boundaries if b.zones == "farfield"
        )
        vortex = rule.condition.point_vortex  # type: ignore[union-attr]
        if c.alpha_deg == 0.0:
            assert vortex is None
        else:
            assert vortex is not None
            assert vortex.surfaces == "airfoil"
            assert [float(v.to("m").magnitude) for v in vortex.center] == [0.25, 0.0, 0.0]
