"""SU2 as an external reference: the converted mesh against the slab Autan reads, the configuration
against the case, the convergence criterion against case-schema.md, and the study against the
extrapolations of Vassberg and Jameson (2010)."""

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from autan_vv.cases import naca0012 as cases
from autan_vv.cli import main
from autan_vv.errors import VVError
from autan_vv.meshes import naca0012
from autan_vv.su2 import config, history, mesh
from autan_vv.su2 import naca0012 as su2

DATA = Path(__file__).parent / "data"
EPS = float(np.finfo(np.float64).eps)


def _naca_y(x: np.ndarray) -> np.ndarray:
    """Eq. 1 of Vassberg and Jameson, transcribed independently of the generator."""
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)


def _read_su2_mesh(path: Path) -> tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """SU2 native mesh, parsed from its documented layout without the converter's code."""
    tokens = path.read_text().split("\n")
    it = iter(tokens)

    def value(key: str) -> str:
        line = next(it)
        assert line.startswith(key + "="), line
        return line.split("=", 1)[1].strip()

    dim = int(value("NDIME"))
    elements = [list(map(int, next(it).split())) for _ in range(int(value("NELEM")))]
    n = int(value("NPOIN"))
    points = np.array([[float(v) for v in next(it).split()] for _ in range(n)])
    markers = {}
    for _ in range(int(value("NMARK"))):
        tag = value("MARKER_TAG")
        markers[tag] = np.array(
            [list(map(int, next(it).split())) for _ in range(int(value("MARKER_ELEMS")))]
        )
    return dim, np.array(elements), points, markers


def _shoelace(ring: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    )


def test_converted_mesh_is_the_slab_autan_reads(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    nc = 16
    family, meshes = naca0012.generate_family((nc,), tmp_path)
    source = tmp_path / family.levels[0].file
    slab_points = meshes[0].points

    record = mesh.convert(source, tmp_path / "plane.su2", 2)
    dim, elements, points, markers = _read_su2_mesh(tmp_path / "plane.su2")
    assert dim == 2
    assert record.file_sha256 == hashlib.sha256((tmp_path / "plane.su2").read_bytes()).hexdigest()
    assert record.source_sha256 == family.levels[0].file_sha256
    # Points: the slab's y = 0 nodes, (x, z) -> (x, y), bit for bit.
    back = slab_points[slab_points[:, 1] == 0.0][:, [0, 2]]
    assert sorted(map(tuple, points[:, :2].tolist())) == sorted(map(tuple, back.tolist()))
    # Quadrangles (VTK 9), counterclockwise, covering the annulus between the two polygons.
    assert np.all(elements[:, 0] == 9)
    assert len(elements) == nc * nc
    areas = np.array([_shoelace(points[q, :2]) for q in elements[:, 1:5]])
    assert np.all(areas > 0.0)
    assert set(markers) == {"airfoil", "farfield"}
    assert record.dropped_zones == ["back", "front"]

    def ring(edges: np.ndarray) -> np.ndarray:
        nxt = dict(zip(edges[:, 1].tolist(), edges[:, 2].tolist(), strict=True))
        if len(nxt) != len(edges):  # edges listed in the other direction
            nxt = dict(zip(edges[:, 2].tolist(), edges[:, 1].tolist(), strict=True))
        order: list[int] = []
        node = next(iter(nxt))
        for _ in range(len(edges)):
            order.append(node)
            node = nxt[node]
        assert node == order[0]
        return points[order, :2]

    # Boundary edges run with the domain on their left, as in their counterclockwise quadrangle.
    directed = {(int(q[k]), int(q[(k + 1) % 4])) for q in elements[:, 1:5] for k in range(4)}
    for marker in markers.values():
        assert all((int(a), int(b)) in directed for _, a, b in marker)
    airfoil, far = ring(markers["airfoil"]), ring(markers["farfield"])
    assert len(airfoil) == len(far) == nc
    np.testing.assert_allclose(np.abs(airfoil[:, 1]), _naca_y(airfoil[:, 0]), rtol=0, atol=4 * EPS)
    assert np.hypot(far[:, 0] - 0.5, far[:, 1]).min() > 145.0
    expected = abs(_shoelace(far)) - abs(_shoelace(airfoil))
    assert areas.sum() == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(
        meshes[0].geometry.cell_volume.sum() / naca0012.SPAN, rel=1e-12
    )

    record3 = mesh.convert(source, tmp_path / "slab.su2", 3)
    dim3, hexes, points3, markers3 = _read_su2_mesh(tmp_path / "slab.su2")
    assert dim3 == 3
    assert np.all(hexes[:, 0] == 12)
    assert len(hexes) == nc * nc
    np.testing.assert_array_equal(points3[:, :3], slab_points)
    assert {k: len(v) for k, v in markers3.items()} == {
        "airfoil": nc, "farfield": nc, "back": nc * nc, "front": nc * nc
    }  # fmt: skip
    # Both files hold the same plane grid, and the hash says so from either file.
    assert record3.plane_sha256 == record.plane_sha256
    edges = {k: v[:, 1:] for k, v in markers.items()}
    assert mesh.plane_sha256(points[:, :2], elements[:, 1:5], edges) == record.plane_sha256


def test_converter_refuses_what_is_not_a_one_layer_slab(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    family, _ = naca0012.generate_family((8,), tmp_path)
    source = tmp_path / family.levels[0].file
    text = source.read_text()
    broken = tmp_path / "broken.msh"
    broken.write_text(text.replace("$EndNodes\n", "", 1))
    with pytest.raises(VVError) as err:
        mesh.convert(broken, tmp_path / "out.su2", 2)
    assert err.value.code == "V0901"
    # Lift one node of the front plane off it: the nodes no longer lie on two planes.
    lines = text.split("\n")
    start = lines.index("$Nodes")
    for i in range(start + 1, len(lines)):
        parts = lines[i].split()
        if len(parts) == 3 and float(parts[1]) == 1.0:
            lines[i] = f"{parts[0]} 0.999 {parts[2]}"
            break
    warped = tmp_path / "warped.msh"
    warped.write_text("\n".join(lines))
    with pytest.raises(VVError) as err:
        mesh.convert(warped, tmp_path / "out.su2", 2)
    assert err.value.code == "V0902"


def _rewrite_hexes(text: str, order: list[int]) -> str:
    lines = text.split("\n")
    head = lines.index("$Elements")
    for i in range(head + 1, lines.index("$EndElements")):
        parts = lines[i].split()
        if len(parts) == 9:
            lines[i] = " ".join([parts[0], *(parts[1 + k] for k in order)])
    return "\n".join(lines)


def test_converter_reads_any_node_order_of_the_hexahedra(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    family, _ = naca0012.generate_family((8,), tmp_path)
    source = tmp_path / family.levels[0].file
    text = source.read_text()
    reference = mesh.convert(source, tmp_path / "a.su2", 2)
    # The same hexahedra numbered so that the face (0, 1, 5, 4) of Gmsh's ordering lies on y = 0:
    # new (0..7) = old (0, 1, 5, 4, 3, 2, 6, 7).
    rotated = tmp_path / "rotated.msh"
    rotated.write_text(_rewrite_hexes(text, [0, 1, 5, 4, 3, 2, 6, 7]))
    assert mesh.convert(rotated, tmp_path / "b.su2", 2).plane_sha256 == reference.plane_sha256
    # Node tags are strictly positive (Gmsh manual): -1 does not alias the last node.
    negative = tmp_path / "negative.msh"
    lines = text.split("\n")
    i = lines.index("$Elements") + 3
    parts = lines[i].split()
    lines[i] = " ".join([*parts[:-1], "-1"])
    negative.write_text("\n".join(lines))
    with pytest.raises(VVError) as err:
        mesh.convert(negative, tmp_path / "c.su2", 2)
    assert err.value.code == "V0901"


def _cfg(text: str) -> dict[str, str]:
    return dict(line.split("= ", 1) for line in text.strip().split("\n"))


def test_configuration_is_the_autan_case() -> None:
    record = mesh.Record(
        mesh.RECORD_SCHEMA, "s.msh", "", "plane.su2", "", 2, 0, 0,
        {"airfoil": 8, "farfield": 8}, ["back", "front"], "",
    )  # fmt: skip
    case = cases.case("m080-a125", "s.msh")
    cfg = _cfg(config.config(case, record, 0.5, span=2.0))
    # Air of case-schema.md: R = R_u / M with CODATA R_u and M = 28.9647 g/mol.
    assert float(cfg["GAS_CONSTANT"]) == pytest.approx(8.314462618 / 0.0289647, rel=4 * EPS)
    assert float(cfg["GAMMA_VALUE"]) == 1.4
    assert float(cfg["MACH_NUMBER"]) == 0.8
    assert float(cfg["AOA"]) == pytest.approx(1.25, rel=4 * EPS)
    assert float(cfg["FREESTREAM_PRESSURE"]) == 101325.0
    assert float(cfg["FREESTREAM_TEMPERATURE"]) == 288.15
    # Coefficients per unit span in 2D: chord x span / span.
    assert float(cfg["REF_AREA"]) == pytest.approx(1.0 / 2.0)
    assert (float(cfg["REF_ORIGIN_MOMENT_X"]), float(cfg["REF_ORIGIN_MOMENT_Y"])) == (0.25, 0.0)
    assert cfg["MARKER_EULER"] == "( airfoil )"
    assert cfg["MARKER_FAR"] == "( farfield )"
    assert "MARKER_SYM" not in cfg
    assert cfg["CONV_NUM_METHOD_FLOW"] == "ROE"
    assert cfg["SLOPE_LIMITER_FLOW"] == "VENKATAKRISHNAN"
    assert float(cfg["VENKAT_LIMITER_COEFF"]) == 0.5
    assert cfg["CONV_FIELD"].count("REL_RMS_") == 4
    record3 = mesh.Record(
        mesh.RECORD_SCHEMA, "s.msh", "", "slab.su2", "", 3, 0, 0,
        {"airfoil": 8, "farfield": 8, "back": 64, "front": 64}, [], "",
    )  # fmt: skip
    cfg3 = _cfg(config.config(case, record3, 0.5, span=2.0))
    assert cfg3["MARKER_SYM"] == "( back, front )"
    assert float(cfg3["REF_AREA"]) == 1.0
    assert "MOMENTUM-Y" not in cfg3["CONV_FIELD"]
    bad = cases.case("m080-a125", "s.msh")
    bad.boundary("airfoil").wall()
    with pytest.raises(VVError) as err:
        config.config(bad, record, 0.5, span=1.0)
    assert err.value.code == "V0903"


def test_limiter_threshold_matches_autans_on_the_median_cell() -> None:
    # A Cartesian slab of square cells a x a, span s, around the mid-chord: every cell has
    # V^(1/3) = (a^2 s)^(1/3), so SU2's coefficient is K (a^2 s)^(1/3) / L with Autan's K = 10.
    a, s, n = 0.1, 0.5, 8
    xs = 0.5 + a * (np.arange(n + 1) - n / 2)
    zs = a * (np.arange(n + 1) - n / 2)
    grid = np.array([[x, 0.0, z] for x in xs for z in zs])
    pts = np.concatenate([grid, grid + np.array([0.0, s, 0.0])])

    def node(i: int, j: int) -> int:
        return i * (n + 1) + j

    quads = np.array(
        [
            [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            for i in range(n)
            for j in range(n)
        ]
    )
    slab = mesh.Slab(pts, np.hstack([quads, quads + len(grid)]), {}, 0.0, s)
    assert config.venkat_coefficient(slab) == pytest.approx(
        10.0 * (a * a * s) ** (1 / 3), rel=1e-14
    )


def _geometric(first: dict[str, float], ratio: float, n: int) -> history.History:
    it = np.arange(n)
    return history.History(
        iterations=it,
        residuals={k: np.asarray(v * ratio**it, dtype=np.float64) for k, v in first.items()},
        cl=np.zeros(n), cd=np.zeros(n), cmy=np.zeros(n), wall_seconds=None,
    )  # fmt: skip


def test_criterion_is_the_drop_below_the_first_residual() -> None:
    # r_n = r_0 2^-n drops 10 orders at n = ceil(10 / log10 2) = 34, whatever r_0.
    h = _geometric({"density": 3.0, "energy": 7e5}, 0.5, 60)
    assert history.converged_at(h, 10.0) == 34
    assert history.converged_at(h, 20.0) is None
    # An equation that starts below 1e-12 of the largest is exempt while it stays there...
    zero = _geometric({"density": 1.0, "momentum-y": 1e-13}, 0.5, 60)
    zero.residuals["momentum-y"][:] = 1e-13
    assert history.converged_at(zero, 10.0) == 34
    # ...and must drop ten orders below its first value once it has risen above it.
    zero.residuals["momentum-y"][5:] = 1e-11 * 0.5 ** np.arange(55)
    assert history.converged_at(zero, 10.0) == 5 + math.ceil(12 / math.log10(2))


def _watcher(script: str) -> str:
    """The awk program by which a runner script stops SU2."""
    text = (Path(__file__).parents[1] / "su2" / script).read_text()
    match = re.search(r"awk -F, [^']*'\n(.*?)' \"\$", text, flags=re.DOTALL)
    assert match is not None
    return "\n".join(line.strip() for line in match.group(1).split("\n"))


def _history_csv(path: Path, residuals: dict[str, np.ndarray]) -> None:
    header = (DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv").read_text().split("\n")[0]
    names = [h.strip().strip('"') for h in header.split(",")]
    columns = dict(
        zip(("rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]"), residuals.values(), strict=True)
    )
    n = len(next(iter(residuals.values())))
    rows = [header]
    rows += [
        ",".join(
            repr(float(np.log10(columns[c][i]))) if c in columns else ("0.01" if c == "Time(sec)" else "0")
            for c in names
        )
        for i in range(n)
    ]  # fmt: skip
    path.write_text("\n".join(rows) + "\n")


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk is not installed")
def test_runner_stops_su2_where_the_criterion_is_met(tmp_path: Path) -> None:
    program = _watcher("run.sh")
    assert program == _watcher("campaign.sh")
    (tmp_path / "watch.awk").write_text(program)

    def stops(residuals: dict[str, np.ndarray]) -> bool:
        _history_csv(tmp_path / "history.csv", residuals)
        cmd = [
            "awk",
            "-F,",
            "-v",
            "orders=10",
            "-v",
            "stall=2000",
            "-f",
            "watch.awk",
            "history.csv",
        ]
        return subprocess.run(cmd, cwd=tmp_path, check=False).returncode == 0

    it = np.arange(600.0)
    falling = 1e-2 * 0.9**it
    # Met: every equation ten orders below its first value, the exempt one staying at round-off.
    met = {"d": falling, "u": falling, "w": np.full_like(it, 1e-15), "e": falling}
    assert stops(met)
    assert history.converged_at(_geometric_like(met), 10.0) is not None
    # Not met: one equation still above; no stall in 600 rows.
    short = {**met, "e": np.maximum(falling, 1e-11)}
    assert not stops(short)
    assert history.converged_at(_geometric_like(short), 10.0) is None
    # A slowly falling run is not a stall, whatever the exempt equation does.
    slow = 1e-2 * 0.9995 ** np.arange(4000.0)  # 0.43 orders per 2000 rows
    assert not stops({"d": slow, "u": slow, "w": np.full(4000, 1e-15), "e": slow})
    # A residual that stops falling is a stall.
    flat = np.maximum(1e-2 * 0.99 ** np.arange(4000.0), 1e-6)
    assert stops({"d": flat, "u": flat, "w": flat, "e": flat})


def _geometric_like(residuals: dict[str, np.ndarray]) -> history.History:
    n = len(next(iter(residuals.values())))
    return history.History(
        iterations=np.arange(n),
        residuals={k: np.asarray(v, dtype=np.float64) for k, v in residuals.items()},
        cl=np.zeros(n), cd=np.zeros(n), cmy=np.zeros(n), wall_seconds=None,
    )  # fmt: skip


def test_su2_history_columns_and_axes(tmp_path: Path) -> None:
    two = history.read_su2(DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv")
    three = history.read_su2(DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-3d.csv")
    with (DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv").open() as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    last = {k.strip().strip('"'): v for k, v in rows[-1].items()}
    assert two.cl[-1] == float(last["CL"])
    assert two.cd[-1] == float(last["CD"])
    assert two.residuals["density"][-1] == 10.0 ** float(last["rms[Rho]"])
    assert set(two.residuals) == {"density", "momentum-x", "momentum-z", "energy"}
    assert set(three.residuals) == {"density", "momentum-x", "momentum-y", "momentum-z", "energy"}
    # The same plane grid run in 2D and as the slab gives the same forces; the nose-up moment is
    # -CMz in SU2's plane (x, y) and CMy on the slab. Positive incidence lifts.
    assert two.cl[-1] > 0.0
    assert three.cl[-1] == pytest.approx(two.cl[-1], rel=1e-8)
    assert three.cd[-1] == pytest.approx(two.cd[-1], rel=1e-8)
    assert three.cmy[-1] == pytest.approx(two.cmy[-1], rel=1e-8)
    assert two.wall_seconds is not None
    assert np.all(np.diff(two.wall_seconds) > 0.0)
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.csv").write_text(
        (DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv").read_text()
    )
    (run / "case.cfg").write_text("SLOPE_LIMITER_FLOW= NONE\n")
    (run / "provenance.json").write_text(json.dumps({"case_sha256": "a"}))
    assert main(["su2", "history", str(run), "--out", str(tmp_path / "h.json")]) == 0
    row = json.loads((tmp_path / "h.json").read_text())[str(run)]
    assert row["cl"] == two.cl[-1]
    assert row["scheme"] == {"SLOPE_LIMITER_FLOW": "NONE"}
    assert row["provenance"]["case_sha256"] == "a"


# Vassberg and Jameson (2010), Table 3, FLO82 at M 0.8, alpha 1.25 deg, grids 1024, 2048, 4096,
# preceded by a coarser value (outside the three finest, it does not enter the extrapolation).
FLO82_CL = [0.365, 0.360812844, 0.358281928, 0.357142338]
FLO82_CD = [0.0232, 0.022934404, 0.022799839, 0.022737860]


def _write_run(run: Path, cl: float, cd: float, converged: bool) -> None:
    header = (DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv").read_text().split("\n")[0]
    names = [h.strip().strip('"') for h in header.split(",")]
    lines = [header]
    for i in range(40):
        drop = 11.0 * i / 39 if converged else 5.0 * i / 39
        row = {"Inner_Iter": i, "Time(sec)": 0.01, "CL": cl, "CD": cd, "CMz": 0.0}
        values = [row.get(n, -2.0 - drop if n.startswith("rms[") else 0.0) for n in names]
        lines.append(",".join(str(v) for v in values))
    (run / "history.csv").write_text("\n".join(lines) + "\n")
    level = json.loads((run / "level.json").read_text())
    provenance = {"schema": "autan.su2-run/1", "image": "autan-su2:v8.6.0-omp",
                  "image_id": "sha256:0", "su2": "v8.6.0 7048fd35", "threads": 4,
                  "case_sha256": level["case_sha256"], "mesh_sha256": level["mesh"]["file_sha256"],
                  "cpu": "", "wall_seconds": 1.0, "exit_status": 0}  # fmt: skip
    (run / "provenance.json").write_text(json.dumps(provenance))


def test_study_reproduces_the_papers_extrapolation(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    naca0012.generate_family((8, 16, 32, 64), tmp_path / "family")
    runs = su2.prepare(tmp_path / "family", tmp_path / "su2", conditions=("m080-a125",))
    assert [r.name for r in runs] == ["n0008", "n0016", "n0032", "n0064"]
    for run, cl, cd in zip(runs, FLO82_CL, FLO82_CD, strict=True):
        _write_run(run, cl, cd, converged=True)
    report = su2.study(runs)
    assert all(lv["converged"] and lv["iterations"] == 37 for lv in report["levels"])
    assert report["continuum"]["cl"]["estimate"] == pytest.approx(0.356208937, abs=2e-9)
    assert report["continuum"]["cd"]["estimate"] == pytest.approx(0.022684938, abs=2e-9)
    assert all(v["pass"] for v in report["verdicts"])
    assert main(["su2", "study", *map(str, runs), "--out", str(tmp_path / "s.json")]) == 0
    # A level that did not reach the criterion leaves too few levels for an estimate.
    _write_run(runs[0], FLO82_CL[0], FLO82_CD[0], converged=False)
    assert su2.study(runs)["continuum"] is None
    # A run of another configuration, or of an edited one, is not the prepared case.
    _write_run(runs[0], FLO82_CL[0], FLO82_CD[0], converged=True)
    prov = json.loads((runs[0] / "provenance.json").read_text())
    (runs[0] / "provenance.json").write_text(json.dumps({**prov, "mesh_sha256": "0" * 64}))
    with pytest.raises(VVError) as err:
        su2.study(runs)
    assert err.value.code == "V0904"
    _write_run(runs[0], FLO82_CL[0], FLO82_CD[0], converged=True)
    cfg = runs[1] / "case.cfg"
    cfg.write_text(cfg.read_text().replace("CFL_NUMBER= 30.0", "CFL_NUMBER= 60.0"))
    with pytest.raises(VVError) as err:
        su2.study(runs)
    assert err.value.code == "V0904"
