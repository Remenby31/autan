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
from autan_vv.su2 import compare, config, history, mesh
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
    # SU2's reference scheme for this case: JST with the controls of inv_NACA0012.cfg, no limiter.
    jst = _cfg(config.config(case, record, 0.5, span=2.0, scheme="jst"))
    assert jst["CONV_NUM_METHOD_FLOW"] == "JST"
    assert jst["MGCYCLE"] == "FULLMG_CYCLE"
    assert "VENKAT_LIMITER_COEFF" not in jst
    assert "MUSCL_FLOW" not in jst
    with pytest.raises(VVError) as err:
        config.config(case, record, 0.5, span=2.0, scheme="hllc")
    assert err.value.code == "V0903"
    bad = cases.case("m080-a125", "s.msh")
    bad.boundary("airfoil").wall()
    with pytest.raises(VVError) as err:
        config.config(bad, record, 0.5, span=1.0)
    assert err.value.code == "V0903"


def test_published_configuration_is_run_as_it_is() -> None:
    record = mesh.Record(
        mesh.RECORD_SCHEMA, "s.msh", "", "plane.su2", "", 2, 0, 0,
        {"airfoil": 8, "farfield": 8}, ["back", "front"], "",
    )  # fmt: skip
    case = cases.case("m050-a125", "s.msh")
    published = (
        "% a published configuration\nSOLVER= EULER\nMACH_NUMBER= 0.8\nAOA= 1.25\n"
        "CONV_NUM_METHOD_FLOW= JST\nJST_SENSOR_COEFF= ( 0.5, 0.02 )\nMGLEVEL= 3\n"
        "MARKER_EULER= ( airfoil )\nMARKER_FAR= ( farfield )\nREF_LENGTH= 1.0\nREF_AREA= 1.0\n"
        "REF_ORIGIN_MOMENT_X = 0.25\nREF_ORIGIN_MOMENT_Y = 0.00\nREF_ORIGIN_MOMENT_Z = 0.00\n"
        "MARKER_MONITORING= ( airfoil )\nMESH_FILENAME= theirs.su2\nITER= 99\n"
    )
    adopted = config.adopt(published, case, record, span=1.0)
    opts = config.options(adopted)
    before = config.options(published)
    # Only the mesh, the flight condition and the outputs change; every other published option is
    # kept with its value.
    may_change = {"MESH_FILENAME", "MESH_FORMAT", "MACH_NUMBER", "AOA", "FREESTREAM_PRESSURE",
                  "FREESTREAM_TEMPERATURE", "ITER", "CONV_FIELD", "CONV_RESIDUAL_MINVAL",
                  "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME", "OUTPUT_FILES",
                  "OUTPUT_WRT_FREQ"}  # fmt: skip
    assert {k: v for k, v in before.items() if k not in may_change} == {
        k: v for k, v in opts.items() if k in before and k not in may_change
    }
    assert {k for k in opts if opts[k] != before.get(k)} <= may_change
    assert opts["CONV_NUM_METHOD_FLOW"] == "JST"
    assert opts["JST_SENSOR_COEFF"] == "( 0.5, 0.02 )"
    assert float(opts["MACH_NUMBER"]) == 0.5
    assert opts["MESH_FILENAME"] == "plane.su2"
    # A published file whose markers or force reference are not the case's is refused.
    with pytest.raises(VVError) as err:
        config.adopt(published.replace("( farfield )", "( outer )"), case, record, span=1.0)
    assert err.value.code == "V0903"
    with pytest.raises(VVError) as err:
        config.adopt(published.replace("X = 0.25", "X = 0.0"), case, record, span=1.0)
    assert err.value.code == "V0903"
    with pytest.raises(VVError) as err:
        config.adopt(published + "MARKER_MONITORING= ( farfield )\n", case, record, span=1.0)
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
    # A continued run: an equation that woke before the stop (start.woken) is no longer exempt,
    # even if it is back under its floor when the run resumes (case-schema.md, Time).
    later = _geometric({"density": 1.0, "momentum-y": 1.0}, 0.5, 60)
    later.residuals["density"][:] = 1e-12 * 0.5 ** np.arange(60)
    later.residuals["momentum-y"][:] = 1e-14 * 0.5 ** np.arange(60)
    reference = {"density": 1.0, "momentum-y": 1e-13}
    assert history.converged_at(history.continued(later, reference), 10.0) == 1
    woken = history.continued(later, reference, ["momentum-y"])
    # momentum-y must reach 1e-23: 1e-14 * 0.5^k <= 1e-23 at k = 30, line 31 of the history.
    assert history.converged_at(woken, 10.0) == 1 + math.ceil(9 / math.log10(2))


def _watcher(script: str) -> str:
    """The awk program by which a runner script stops SU2."""
    text = (Path(__file__).parents[1] / "su2" / script).read_text()
    match = re.search(r"awk -F, [^']*'\n(.*?)' \"\$", text, flags=re.DOTALL)
    assert match is not None
    return "\n".join(line.strip() for line in match.group(1).split("\n"))


def _history_csv(
    path: Path, residuals: dict[str, np.ndarray], cd: np.ndarray | None = None
) -> None:
    header = (DATA / "su2-v8.6.0-naca0012-m080-a125-n0032-2d.csv").read_text().split("\n")[0]
    names = [h.strip().strip('"') for h in header.split(",")]
    columns = dict(
        zip(("rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]"), residuals.values(), strict=True)
    )
    n = len(next(iter(residuals.values())))
    rows = [header]
    rows += [
        ",".join(
            repr(float(np.log10(columns[c][i]))) if c in columns
            else repr(float(cd[i])) if c == "CD" and cd is not None
            else ("0.01" if c == "Time(sec)" else "0")
            for c in names
        )
        for i in range(n)
    ]  # fmt: skip
    path.write_text("\n".join(rows) + "\n")


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk is not installed")
def test_runner_stops_su2_where_the_criterion_is_met(tmp_path: Path) -> None:
    scripts = Path(__file__).parents[1] / "su2"
    for script in ("run.sh", "campaign.sh"):
        assert subprocess.run(["bash", "-n", str(scripts / script)], check=False).returncode == 0
    program = _watcher("run.sh")
    assert program == _watcher("campaign.sh")
    (tmp_path / "watch.awk").write_text(program)

    def stops(
        residuals: dict[str, np.ndarray], skip_zero: int = 0, cd: np.ndarray | None = None
    ) -> bool:
        _history_csv(tmp_path / "history.csv", residuals, cd)
        cmd = [
            "awk",
            "-F,",
            "-v",
            "orders=10",
            "-v",
            "stall=2000",
            "-v",
            f"skipzero={skip_zero}",
            "-v",
            "stopfile=stop.csv",
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
    # An exempt equation that wakes to 1e-11 and stays there: by default its drop never improves
    # and stops the run; with skipzero=1 the stall is judged on the others, still falling, and the
    # woken equation still has to meet the criterion, so a run whose other equations have met it
    # goes on.
    woken = np.full(4000, 1e-11)
    woken[0] = 1e-15
    assert stops({"d": slow, "u": slow, "w": woken, "e": slow})
    assert not stops({"d": slow, "u": slow, "w": woken, "e": slow}, skip_zero=1)
    assert not stops({**met, "w": woken[:600]}, skip_zero=1)
    # With a stop.csv (#493): every residual at or below its threshold and CD stationary to 1e-7
    # over the window. The verdict does not depend on the first residuals; a woken equation at
    # the floor converges; CD still moving, or one equation above its threshold, does not.
    stop = "\n".join(
        [f"threshold,{c},-8.0" for c in ("rms[Rho]", "rms[RhoU]", "rms[RhoV]", "rms[RhoE]")]
        + ["output,CD", "window,100", "rtol,1e-07"]
    )
    (tmp_path / "stop.csv").write_text(stop + "\n")
    n = 400
    floor = np.full(n, 1e-9)
    settled = np.full(n, 2.8e-3)
    assert stops({"d": floor, "u": floor, "w": floor, "e": floor}, cd=settled)
    shifted = floor.copy()
    shifted[:5] = 1e3
    assert stops({"d": shifted, "u": floor, "w": woken[:n], "e": floor}, cd=settled)
    moving = 2.8e-3 * (1 + 1e-5 * np.cos(np.arange(n)))
    assert not stops({"d": floor, "u": floor, "w": floor, "e": floor}, cd=moving)
    assert not stops({"d": floor, "u": floor, "w": floor, "e": np.full(n, 1e-7)}, cd=settled)
    # SU2 writes its history in buffered blocks: a last line cut short is not a row.
    _history_csv(
        tmp_path / "history.csv", {"d": floor, "u": floor, "w": floor, "e": floor}, settled
    )
    text = (tmp_path / "history.csv").read_text()
    (tmp_path / "history.csv").write_text(text + text.splitlines()[-1][:40])
    cmd = ["awk", "-F,", "-v", "orders=10", "-v", "stall=2000", "-v", "skipzero=0", "-v",
           "stopfile=stop.csv", "-f", "watch.awk", "history.csv"]  # fmt: skip
    assert subprocess.run(cmd, cwd=tmp_path, check=False).returncode == 0
    (tmp_path / "stop.csv").unlink()


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
    assert report["precision"] == "fp64"
    assert all(lv["converged"] and lv["iterations"] == 37 for lv in report["levels"])
    assert report["continuum"]["cl"]["estimate"] == pytest.approx(0.356208937, abs=2e-9)
    assert report["continuum"]["cd"]["estimate"] == pytest.approx(0.022684938, abs=2e-9)
    assert all(v["pass"] for v in report["verdicts"])
    assert main(["su2", "study", *map(str, runs), "--out", str(tmp_path / "s.json")]) == 0
    # A level that did not reach the criterion leaves too few levels for an estimate.
    _write_run(runs[0], FLO82_CL[0], FLO82_CD[0], converged=False)
    assert su2.study(runs)["continuum"] is None
    # Runs of two schemes are not one family.
    level = json.loads((runs[0] / "level.json").read_text())
    (runs[0] / "level.json").write_text(json.dumps({**level, "scheme": "jst"}))
    with pytest.raises(VVError) as err:
        su2.study(runs)
    assert err.value.code == "V0905"
    (runs[0] / "level.json").write_text(json.dumps(level))
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


def test_compare_autan_with_an_su2_study(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    import autan  # noqa: PLC0415
    from autan.units import m  # noqa: PLC0415
    from autan_vv.runs import mesh_info  # noqa: PLC0415

    family, meshes = naca0012.generate_family((8, 16, 32, 64), tmp_path / "family")
    su2_runs = su2.prepare(tmp_path / "family", tmp_path / "su2", conditions=("m080-a125",))
    for run, cl, cd in zip(su2_runs, FLO82_CL, FLO82_CD, strict=True):
        _write_run(run, cl, cd, converged=True)
    (tmp_path / "su2.json").write_text(json.dumps(su2.study(su2_runs)))

    def autan_run(case: autan.Case, level: int, scale: float = 1.0) -> Path:
        path = tmp_path / "family" / family.levels[level].file
        info = mesh_info(meshes[level], path, "gmsh", scale)
        return case.run(info, backend=autan.FakeBackend(), runs=tmp_path / "autan").path

    def mesh(level: int) -> str:
        return str(tmp_path / "family" / family.levels[level].file)

    runs = [autan_run(cases.case("m080-a125", mesh(i)), i) for i in range(4)]
    # A run stopped early and continued: judged against its original run's first residuals, with
    # the iteration count of the uninterrupted run, as the runtime judges it.
    path0 = tmp_path / "family" / family.levels[0].file
    info0 = mesh_info(meshes[0], path0, "gmsh")
    short = cases.case("m080-a125", mesh(0), max_iterations=10)
    first = short.run(info0, backend=autan.FakeBackend(), runs=tmp_path / "autan")
    whole = cases.case("m080-a125", mesh(0)).run(
        info0, backend=autan.FakeBackend(), runs=tmp_path / "autan", start=first, mode="continue"
    )
    level = compare.read_autan(whole.path)
    assert whole.run.status == "converged"
    assert level.converged
    assert level.iterations == whole.run.iterations
    report = compare.compare(runs, tmp_path / "su2.json", tmp_path / "su2")
    assert report["autan"]["point_vortex"]
    for row, run, cl, cd in zip(report["levels"], runs, FLO82_CL, FLO82_CD, strict=True):
        forces = autan.Results.open(run).forces
        assert row["difference"]["cl"] == pytest.approx(float(forces.cl[-1]) - cl, abs=8 * EPS)
        assert row["difference"]["cd"] == pytest.approx(float(forces.cd[-1]) - cd, abs=8 * EPS)
    # SU2's side is extrapolated as the gates do: FLO82's continuum of Table 3.
    assert report["continuum"]["su2"]["cl"]["estimate"] == pytest.approx(0.356208937, abs=2e-9)
    assert report["continuum"]["levels"] == [16, 32, 64]
    assert (
        report["su2"]["report_sha256"]
        == hashlib.sha256((tmp_path / "su2.json").read_bytes()).hexdigest()
    )
    assert all(
        row["autan"]["history_sha256"]
        == hashlib.sha256((run / "history.jsonl").read_bytes()).hexdigest()
        for row, run in zip(report["levels"], runs, strict=True)
    )
    # A family mixing the point-vortex far field with the plain one, or another condition, is refused.
    plain = autan.Case("naca0012-m080-a125", mesh=mesh(0))
    plain.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
    plain.freestream(
        mach=0.8, aoa=1.25 * autan.units.deg, pressure=101325.0 * autan.units.ureg.Pa,
        ref_length=1.0 * m,
    )  # fmt: skip
    plain.time = autan.Steady(max_iterations=200000, residual_orders=10.0)
    plain.boundary("airfoil").slip()
    plain.boundary("farfield").farfield()
    plain.boundary("back").symmetry()
    plain.boundary("front").symmetry()
    plain.monitor(
        autan.forces("airfoil", ref_area=1.0 * m * m, moment_center=(0.25 * m, 0 * m, 0 * m))
    )
    # The same name with another case document (a looser criterion, coordinates read in
    # millimetres) is not the same problem.
    loose = autan_run(cases.case("m080-a125", mesh(0), residual_orders=9.0), 0)
    millimetres = autan.Case("naca0012-m080-a125", mesh=mesh(0), mesh_unit="mm")
    millimetres.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
    millimetres.freestream(
        mach=0.8, aoa=1.25 * autan.units.deg, pressure=101325.0 * autan.units.ureg.Pa,
        ref_length=1.0 * m,
    )  # fmt: skip
    millimetres.time = autan.Steady(max_iterations=200000, residual_orders=10.0)
    millimetres.boundary("airfoil").slip()
    vortex = autan.PointVortex(surfaces="airfoil", center=(0.25 * m, 0.0 * m, 0.0 * m))
    millimetres.boundary("farfield").farfield(point_vortex=vortex)
    millimetres.boundary("back").symmetry()
    millimetres.boundary("front").symmetry()
    millimetres.monitor(
        autan.forces("airfoil", ref_area=1.0 * m * m, moment_center=(0.25 * m, 0 * m, 0 * m))
    )
    scaled = autan_run(millimetres, 0, 1e-3)
    odd_runs = (autan_run(plain, 0), autan_run(cases.case("m050-a125", mesh(0)), 0), loose, scaled)
    for odd in odd_runs:
        with pytest.raises(VVError) as err:
            compare.compare([*runs[1:], odd], tmp_path / "su2.json", tmp_path / "su2")
        assert err.value.code == "V0905"
    # SU2 runs that are not the study's (another scheme) or that read another mesh are refused.
    level_json = tmp_path / "su2" / "m080-a125" / "n0008" / "level.json"
    level = json.loads(level_json.read_text())
    for bad in (
        {**level, "scheme": "jst"},
        {**level, "mesh": {**level["mesh"], "source_sha256": "0" * 64}},
    ):
        level_json.write_text(json.dumps(bad))
        with pytest.raises(VVError) as err:
            compare.compare(runs, tmp_path / "su2.json", tmp_path / "su2")
        assert err.value.code == "V0905"
    level_json.write_text(json.dumps(level))


def test_wall_cp_is_compared_face_by_face(tmp_path: Path) -> None:
    pytest.importorskip("gmsh", reason="gmsh (dev-only 'mesh' group) is not installed")
    import autan  # noqa: PLC0415
    import h5py  # type: ignore[import-untyped]  # noqa: PLC0415
    from autan.units import deg, m  # noqa: PLC0415
    from autan_vv.runs import mesh_info, write_run  # noqa: PLC0415

    nc, mach, gamma, p_inf = 32, 0.8, 1.4, 101325.0
    family, meshes = naca0012.generate_family((nc,), tmp_path / "family")
    path = tmp_path / "family" / family.levels[0].file
    # The run writer refuses monitors; Cp needs the freestream and the solution only.
    case = autan.Case("naca0012-m080-a125", mesh=str(path))
    case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
    pressure = p_inf * autan.units.ureg.Pa
    case.freestream(mach=mach, aoa=1.25 * deg, pressure=pressure, ref_length=1.0 * m)
    case.time = autan.Steady(max_iterations=100, residual_orders=10.0)
    case.boundary("airfoil").slip()
    case.boundary("farfield").farfield()
    case.boundary("back").symmetry()
    case.boundary("front").symmetry()
    n = meshes[0].n_cells
    fields = {"Density": np.ones(n), "VelocityX": np.zeros(n), "VelocityY": np.zeros(n),
              "VelocityZ": np.zeros(n), "Pressure": np.full(n, p_inf),
              "Temperature": np.full(n, 288.15)}  # fmt: skip
    info = mesh_info(meshes[0], path, "gmsh")
    run = write_run(tmp_path / "runs", case, meshes[0], info, fields).path

    def cp_of(x: np.ndarray, shock: float | None) -> np.ndarray:
        # Linear in x, so a face's value is the mean of its end nodes'; a jump of 0.5 at `shock`.
        return -0.4 + 0.3 * x + (0.0 if shock is None else 0.5 * (x > shock))

    # Autan side: the face pressure of p = p_inf + q_inf Cp at each face centre (x along the chord).
    with h5py.File(run / "solution.cgns", "r+") as f:
        zone = f["Base/Zone"]
        xyz = np.stack([zone[f"GridCoordinates/Coordinate{c}/ data"][()] for c in "XYZ"], 1)
        offset = zone["NGonElements/ElementStartOffset/ data"][()]
        conn = zone["NGonElements/ElementConnectivity/ data"][()] - 1
        faces = zone["WallSolution/PointList/ data"][()].ravel() - 1
        centre = np.array([xyz[conn[offset[k] : offset[k + 1]]].mean(axis=0) for k in faces])
        q_inf = 0.5 * gamma * p_inf * mach**2
        zone["WallSolution/Pressure/ data"][...] = p_inf + q_inf * cp_of(centre[:, 0], 0.6)

    # SU2 side: conservative variables at rest, in units where p_inf = rho_inf = 1, at the wall
    # nodes of the converted 2D mesh.
    def surface(name: str, shift: float, shock: float | None) -> Path:
        record = mesh.convert(path, tmp_path / f"{name}.su2", 2)
        _, _, points, markers = _read_su2_mesh(tmp_path / record.file)
        nodes = np.unique(markers["airfoil"][:, 1:])
        p = 1.0 + 0.5 * gamma * mach**2 * (cp_of(points[nodes, 0], shock) + shift)
        lines = ['"PointID","x","y","Density","Momentum_x","Momentum_y","Energy"']
        lines += [f"{k}, {points[k, 0]:.15e}, {points[k, 1]:.15e}, 1.0, 0.0, 0.0, {pk / (gamma - 1.0):.17g}"
                  for k, pk in zip(nodes, p, strict=True)]  # fmt: skip
        out = tmp_path / f"{name}.csv"
        out.write_text("\n".join(lines) + "\n")
        return out

    same = compare.compare_cp(run, surface("same", 0.0, 0.6))
    assert same["faces"] == nc
    # Only the face across the jump differs: its centre is past it, one of its nodes is not.
    assert same["all"]["max_abs"] == pytest.approx(0.25, abs=1e-9)
    assert same["outside_shocks"]["max_abs"] < 1e-9
    assert 0 < same["body"]["faces"] < nc
    # SU2's nodal step reaches the face centres over two faces: the same shock, within one face.
    assert same["shock_upper"]["autan"] == pytest.approx(0.6, abs=0.1)
    assert same["shock_upper"]["su2"] == pytest.approx(0.6, abs=0.1)
    shifted = compare.compare_cp(run, surface("shifted", 0.01, 0.6))
    assert shifted["outside_shocks"]["max_abs"] == pytest.approx(0.01, abs=1e-9)
    # A shock SU2 puts further aft: the faces between and around the two shocks are left out.
    moved = compare.compare_cp(run, surface("moved", 0.0, 0.75))
    assert moved["shock_upper"]["su2"] == pytest.approx(0.75, abs=0.1)
    assert moved["shock_upper"]["su2"] > moved["shock_upper"]["autan"]
    assert moved["all"]["max_abs"] == pytest.approx(0.5, abs=1e-9)
    assert moved["outside_shocks"]["max_abs"] < 1e-9


def _plot3d_2d(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    jd, id_ = x.shape
    values = " ".join(f"{v:.17g}" for v in np.concatenate([x.ravel(), y.ravel()]))
    path.write_text(f"1\n{id_} {jd}\n{values}\n")


def test_flat_plate_grid_converts_with_the_tmr_boundaries(tmp_path: Path) -> None:
    from autan_vv.su2 import flatplate  # noqa: PLC0415

    # A 2DZP-like grid: x from -1/3 to 2 with the plate from x = 0, y from 0 up to 1, stretched.
    xs = np.array([-1.0 / 3.0, -0.1, 0.0, 0.5, 1.0, 1.5, 2.0])
    ys = np.array([0.0, 0.01, 0.1, 0.4, 1.0])
    x, y = np.meshgrid(xs, ys)
    grid = tmp_path / "plate.p2dfmt"
    _plot3d_2d(grid, x, y)
    record = flatplate.convert(grid, tmp_path / "plate.su2")
    dim, elements, points, markers = _read_su2_mesh(tmp_path / "plate.su2")
    assert dim == 2
    assert (record.idim, record.jdim) == (7, 5)
    assert record.file_sha256 == hashlib.sha256((tmp_path / "plate.su2").read_bytes()).hexdigest()
    assert len(elements) == 6 * 4
    areas = np.array([_shoelace(points[q, :2]) for q in elements[:, 1:5]])
    assert np.all(areas > 0.0)
    assert areas.sum() == pytest.approx((2.0 + 1.0 / 3.0) * 1.0, rel=1e-14)

    def nodes(name: str) -> np.ndarray:
        return points[np.unique(markers[name][:, 1:]), :2]

    assert {k: len(v) for k, v in markers.items()} == {
        "inlet": 4, "outlet": 4, "farfield": 6, "symmetry": 2, "wall": 4
    }  # fmt: skip
    assert np.all(nodes("wall")[:, 1] == 0.0)
    assert (nodes("wall")[:, 0].min(), nodes("wall")[:, 0].max()) == (0.0, 2.0)
    assert np.all(nodes("symmetry")[:, 0] <= 0.0)
    assert np.all(nodes("inlet")[:, 0] == -1.0 / 3.0)
    assert np.all(nodes("outlet")[:, 0] == 2.0)
    assert np.all(nodes("farfield")[:, 1] == 1.0)
    # Every boundary edge keeps the domain on its left, as in its counterclockwise quadrangle.
    directed = {(int(q[k]), int(q[(k + 1) % 4])) for q in elements[:, 1:5] for k in range(4)}
    for marker in markers.values():
        assert all((int(a), int(b)) in directed for _, a, b in marker)
    # The same grid written otherwise (points renumbered, farfield edges reversed, as in SU2's own
    # 137x97 file) is the same grid; a moved point is not.
    perm = np.random.default_rng(0).permutation(len(points))
    new = np.empty_like(perm)
    new[perm] = np.arange(len(perm))

    def rewrite(path: Path, pts: np.ndarray) -> None:
        lines = ["NDIME= 2", f"NELEM= {len(elements)}"]
        lines += [f"9 {' '.join(str(new[v]) for v in q[1:5])} {k}" for k, q in enumerate(elements)]
        lines += [f"NPOIN= {len(pts)}"]
        lines += [f"{pts[p, 0]:.17g} {pts[p, 1]:.17g} {k}" for k, p in enumerate(perm)]
        lines += [f"NMARK= {len(markers)}"]
        for name, m in markers.items():
            edges = m[:, [2, 1]] if name == "farfield" else m[:, 1:]
            lines += [f"MARKER_TAG= {name}", f"MARKER_ELEMS= {len(m)}"]
            lines += [f"3 {new[a]} {new[b]}" for a, b in edges]
        path.write_text("\n".join(lines) + "\n")

    pts = points[:, :2].astype(float)
    rewrite(tmp_path / "theirs.su2", pts)
    check = flatplate.same_grid(tmp_path / "plate.su2", tmp_path / "theirs.su2")
    assert check["same"]
    assert check["markers_oriented_otherwise"] == ["farfield"]
    moved = pts.copy()
    moved[markers["wall"][0, 2], 0] += 1e-3
    rewrite(tmp_path / "moved.su2", moved)
    assert not flatplate.same_grid(tmp_path / "plate.su2", tmp_path / "moved.su2")["same"]
    # A grid whose plate does not reach x = 2 is not the 2DZP case.
    _plot3d_2d(grid, x * 0.9, y)
    with pytest.raises(VVError) as err:
        flatplate.convert(grid, tmp_path / "short.su2")
    assert err.value.code == "V0902"


def test_flat_plate_configuration_and_skin_friction(tmp_path: Path) -> None:
    from autan_vv.su2 import flatplate  # noqa: PLC0415

    published = (
        "% published\nSOLVER= RANS\nKIND_TURB_MODEL= SA\nMACH_NUMBER= 0.2\n"
        "MARKER_HEATFLUX= ( wall, 0.0 )\nMARKER_INLET= ( inlet, 302.4, 118309.784, 1.0, 0.0, 0.0 )\n"
        "MARKER_OUTLET= ( outlet, 115056.0, farfield, 115056.0 )\nMARKER_SYM= ( symmetry )\n"
        "CFL_NUMBER= 1000.0\nMESH_FILENAME= theirs.su2\nITER= 1000\n"
    )
    before = config.options(published)
    after = config.options(flatplate.configure(published, "plate.su2"))
    may_change = {"MESH_FILENAME", "MESH_FORMAT", "ITER", "CONV_FIELD", "CONV_RESIDUAL_MINVAL",
                  "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME", "OUTPUT_FILES",
                  "OUTPUT_WRT_FREQ", "SURFACE_FILENAME", "VOLUME_OUTPUT",
                  "WRT_RESTART_COMPACT"}  # fmt: skip
    assert {k: v for k, v in before.items() if k not in may_change} == {
        k: v for k, v in after.items() if k not in may_change
    }
    standard = config.options(flatplate.configure(published, "plate.su2", sa_options="WITHFT2"))
    assert {k: v for k, v in standard.items() if k != "SA_OPTIONS"} == after
    assert standard["SA_OPTIONS"] == "WITHFT2"
    assert after["MESH_FILENAME"] == "plate.su2"
    assert "REL_RMS_NU_TILDE" in after["CONV_FIELD"]
    with pytest.raises(VVError) as err:
        flatplate.configure(published.replace("( symmetry )", "( bottom )"), "plate.su2")
    assert err.value.code == "V0903"
    # cf at the TMR station, linear between the wall nodes around it.
    surface = tmp_path / "surface.csv"
    xs = np.linspace(0.0, 2.0, 9)
    rows = ['"PointID","x","y","Skin_Friction_Coefficient_x"']
    rows += [f"{k}, {xv:.17g}, 0.0, {0.004 - 0.001 * xv:.17g}" for k, xv in enumerate(xs[::-1])]
    surface.write_text("\n".join(rows) + "\n")
    # TMR's station (cf_convergence.dat).
    assert flatplate.CF_STATION == 0.970084071
    assert flatplate.skin_friction(surface) == pytest.approx(0.004 - 0.001 * 0.970084071, rel=1e-14)
    # SU2 v8.6.0's own outputs of the 35x25 grid, configured by `configure`: the surface file
    # carries the skin friction, the history the SA residual.
    real = DATA / "su2-v8.6.0-flatplate-35x25-surface.csv"
    with real.open() as f:
        wall = sorted(
            (float(r["x"]), float(r["Skin_Friction_Coefficient_x"]))
            for r in csv.DictReader(f, skipinitialspace=True)
        )
    left = max(w for w in wall if w[0] <= flatplate.CF_STATION)
    right = min(w for w in wall if w[0] > flatplate.CF_STATION)
    cf = flatplate.skin_friction(real)
    assert min(left[1], right[1]) <= cf <= max(left[1], right[1])
    run = history.read_su2(DATA / "su2-v8.6.0-flatplate-35x25-history.csv")
    assert set(run.residuals) == {"density", "momentum-x", "momentum-z", "energy", "nu-tilde"}
    assert history.converged_at(run, 10.0) == len(run.iterations) - 1


# TMR, FlatPlate/SA/cf_convergence.dat and drag_convergence.dat: CFL3D on the five grids (cells).
CFL3D_CF = {816: 0.279568508e-2, 3264: 0.272834697e-2, 13056: 0.271115173e-2,
            52224: 0.270673749e-2, 208896: 0.270562153e-2}  # fmt: skip
CFL3D_CD = {816: 0.295438152e-2, 3264: 0.288437885e-2, 13056: 0.286620917e-2,
            52224: 0.286130951e-2, 208896: 0.285985288e-2}  # fmt: skip


def _flat_plate_run(
    run: Path, model: str, cells: int, drop: float, first: float = -2.0, nu: float = 0.0
) -> None:
    from autan_vv.su2 import flatplate  # noqa: PLC0415

    run.mkdir(parents=True)
    (run / "m.su2").write_text(f"mesh {cells}\n")
    (run / "case.cfg").write_text("MESH_FILENAME= m.su2\n")
    shutil.copy(DATA / "su2-v8.6.0-flatplate-35x25-log-freestream.txt", run / "log.txt")
    sha = {f: hashlib.sha256((run / f).read_bytes()).hexdigest() for f in ("m.su2", "case.cfg")}
    (run / "level.json").write_text(json.dumps({
        "model": model, "cells": cells, "case_sha256": sha["case.cfg"],
        "published_sha256": flatplate.PUBLISHED_SHA256,
        "mesh": {"file_sha256": sha["m.su2"], "source_sha256": "s"},
    }))  # fmt: skip
    (run / "provenance.json").write_text(json.dumps({
        "case_sha256": sha["case.cfg"], "mesh_sha256": sha["m.su2"], "exit_status": 0,
        "su2": "v8.6.0", "image_id": "sha256:i",
    }))  # fmt: skip
    header = (DATA / "su2-v8.6.0-flatplate-35x25-history.csv").read_text().split("\n")[0]
    names = [h.strip().strip('"') for h in header.split(",")]
    lines = [header]
    for i in range(30):
        rms = first - drop * i / 29
        row = {"Inner_Iter": i, "Time(sec)": 0.01, "CD": CFL3D_CD[cells]}
        lines.append(",".join(
            str(row.get(n, rms + nu * (n == "rms[nu]") if n.startswith("rms[") else 0.0))
            for n in names
        ))  # fmt: skip
    (run / "history.csv").write_text("\n".join(lines) + "\n")
    rows = ['"PointID","x","y","Skin_Friction_Coefficient_x"']
    rows += [f"{k}, {x}, 0.0, {CFL3D_CF[cells]!r}" for k, x in enumerate((0.0, 0.5, 1.5, 2.0))]
    (run / f"{flatplate.SURFACE}.csv").write_text("\n".join(rows) + "\n")


def test_flat_plate_report_against_tmr(tmp_path: Path) -> None:
    from autan_vv.su2 import flatplate  # noqa: PLC0415

    cache = tmp_path / "cache"
    (cache / flatplate.CASE).mkdir(parents=True)
    for name in ("cf_convergence.dat", "drag_convergence.dat"):
        shutil.copy(DATA / f"tmr-flatplate-sa-{name}", cache / flatplate.CASE / name)
    root = tmp_path / "plate"
    for model in flatplate.MODELS:
        for cells in CFL3D_CF:
            _flat_plate_run(root / model / f"n{cells}", model, cells, drop=11.0)
    out = tmp_path / "report.json"
    # SU2 giving CFL3D's values on every grid has CFL3D's continuum (acceptance.md: cf 2.70524e-3,
    # CD 2.85924e-3), no gap to CFL3D, and passes the gate.
    argv = ["su2-flatplate", "results", str(root), "--cache", str(cache), "--out", str(out)]
    assert main(argv) == 0
    model = json.loads(out.read_text())["models"]["sa-noft2"]
    assert model["continuum"]["cf"]["estimate"] == pytest.approx(2.70524e-3, abs=5e-9)
    assert model["continuum"]["cd"]["estimate"] == pytest.approx(2.85924e-3, abs=5e-9)
    assert all(lv["gap"]["CFL3D"] == {"cf": 0.0, "cd": 0.0} for lv in model["levels"])
    assert model["pass"]
    # The absolute criterion of #493 judges every run, from the published runs' own histories
    # and logs; thirty iterations are fewer than the 500 over which CD must stay stationary.
    absolute = flatplate.results(root, cache, ("sa-noft2",), absolute=root)["models"]["sa-noft2"]
    assert [lv["absolute"]["converged"] for lv in absolute["levels"]] == [False] * 5
    assert absolute["absolute"]["continuum"]["cf"] == {"error": "not converged"}
    # A variant whose first residuals are three orders lower stops 8.5 orders down, at or below the
    # levels the published runs had at the criterion: not converged by the protocol's rule, so no
    # estimate, but converged by the amended one (#195), with CFL3D's continuum again. The
    # published runs meet the criterion at row 27 (-2 - 11 * 27/29 <= -12), at -12.24; the variant
    # reaches that at row 25 (-5 - 8.5 * 25/29), its 26th iteration.
    variant = tmp_path / "variant"
    for cells in CFL3D_CF:
        _flat_plate_run(variant / "sa-noft2" / f"n{cells}", "sa-noft2", cells, drop=8.5, first=-5.0)
    with pytest.raises(SystemExit):
        main(["su2-flatplate", "results", str(variant), "--cache", str(cache), "--models",
              "sa-noft2", "--reference", str(root), "--out", str(out)])  # fmt: skip
    model = json.loads(out.read_text())["models"]["sa-noft2"]
    assert model["continuum"]["cf"] == {"error": "not converged"}
    assert [lv["amended"]["iterations"] for lv in model["levels"]] == [26] * 5
    assert model["amended"]["continuum"]["cf"]["estimate"] == pytest.approx(2.70524e-3, abs=5e-9)
    assert model["amended"]["pass"]
    # On the finest grid the SA residual alone stays above its threshold: not converged, no estimate.
    shutil.rmtree(variant / "sa-noft2" / "n208896")
    _flat_plate_run(
        variant / "sa-noft2" / "n208896", "sa-noft2", 208896, drop=8.5, first=-5.0, nu=2.0
    )
    report = flatplate.results(variant, cache, ("sa-noft2",), reference=root)["models"]["sa-noft2"]
    assert not report["levels"][-1]["amended"]["converged"]
    assert report["amended"]["continuum"]["cf"] == {"error": "not converged"}
    # A finest run that did not converge gives no estimate, not one from coarser grids.
    shutil.rmtree(root / "sa" / "n208896")
    _flat_plate_run(root / "sa" / "n208896", "sa", 208896, drop=5.0)
    with pytest.raises(SystemExit):
        main(argv)
    report = flatplate.results(root, cache)["models"]["sa"]
    assert report["continuum"]["cf"] == {"error": "not converged"}
    assert not report["pass"]
    # A configuration changed after the run is refused.
    (root / "sa" / "n816" / "case.cfg").write_text("MESH_FILENAME= m.su2\nCFL_NUMBER= 5\n")
    with pytest.raises(VVError) as err:
        flatplate.results(root, cache)
    assert err.value.code == "V0904"


def test_absolute_criterion_does_not_depend_on_the_initial_state() -> None:
    n = 800
    it = np.arange(n, dtype=np.float64)
    decay = 1e-2 * 0.9**it + 1e-14
    cd = 2.8e-3 * (1.0 + 0.5**it)
    scales = {"d": 1.0, "u": 10.0, "e": 1e5}

    def run(residuals: dict[str, np.ndarray], drag: np.ndarray = cd) -> history.History:
        return history.History(
            iterations=it.astype(np.int64), residuals=residuals, cl=np.zeros(n), cd=drag,
            cmy=np.zeros(n), wall_seconds=None,
        )  # fmt: skip

    base = run({eq: decay * s for eq, s in scales.items()})
    # cd is within 1e-7 of its final value from 0.5^i <= 1e-7 (1 + 0.5^799), i = 24; epsilon is the
    # largest scaled residual there, every equation's being decay[24].
    eps = history.epsilon(base, scales, ("cd",))
    assert eps == pytest.approx(decay[24], rel=1e-12)
    thresholds = {eq: eps * s for eq, s in scales.items()}
    # The residuals reach the thresholds at 24, cd is then stationary over 500 iterations at 524.
    at = history.absolute_converged_at(base, thresholds, ("cd",))
    assert at == 524
    # First residuals three orders higher: the relative criterion moves, the absolute does not.
    shifted = {eq: np.where(it < 5, 1e3, 1.0) * decay * s for eq, s in scales.items()}
    assert history.converged_at(run(shifted), 10.0) != history.converged_at(base, 10.0)
    assert history.absolute_converged_at(run(shifted), thresholds, ("cd",)) == at
    # An equation that starts identically zero and wakes to the floor converges.
    woken = {**base.residuals, "u": np.where(it < 1, 1e-30, 1e-13)}
    assert history.absolute_converged_at(run(woken), thresholds, ("cd",)) == at
    # cd moving by 1e-6 never is stationary to 1e-7.
    wobble = 2.8e-3 * (1.0 + 1e-6 * np.cos(it))
    assert history.absolute_converged_at(run(base.residuals, wobble), thresholds, ("cd",)) is None
    # SU2 v8.6.0's own freestream table (35x25 flat plate log): its values, and the flux scales
    # from them with H = E + p / rho and nu_tilde = 3 mu / rho.
    fs = history.su2_freestream(
        (DATA / "su2-v8.6.0-flatplate-35x25-log-freestream.txt").read_text()
    )
    assert fs == {"pressure": 114455.0, "density": 1.32905, "energy": 217705.0,
                  "speed": 69.4448, "viscosity": 1.84592e-05}  # fmt: skip
    flux = history.flux_scales(fs, 1.0)
    assert flux["density"] == pytest.approx(1.32905 * 69.4448, rel=1e-15)
    assert flux["energy"] == pytest.approx(
        1.32905 * 69.4448 * (217705.0 + 114455.0 / 1.32905), rel=1e-15
    )
    assert flux["nu-tilde"] == pytest.approx(3.0 * 1.84592e-05 / 1.32905 * 69.4448, rel=1e-15)
