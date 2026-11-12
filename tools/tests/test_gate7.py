import json
import pathlib
import subprocess
import sys

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1]
# acceptance.md, D2 gate 7: tau = max(64 * 2^-52, 2 gamma_(d+2)), u = 2^-53.
U = 2.0**-53
TAU_HEX = 64 * 2.0**-52
ZERO = [0.0] * 5
ONE = [1.0] * 5
EQUATIONS = ["Density", "MomentumX", "MomentumY", "MomentumZ", "EnergyStagnationDensity"]
# The frozen case and source solution of the box's controls, which start from no solution.
PINS = ("--case-sha256", "ca", "--solution-sha256", "none")
RUN = {
    "identity": "1d",
    "case": {"name": "box", "sha256": "ca"},
    "mesh": {"path": "/m", "sha256": "4e", "groups_sha256": None},
    "solver": {"commit": "c0", "binary_sha256": "b1"},
    "precision": "fp64",
    "devices": {"kind": "cpu", "count": 1},
    "cell_order": "rcm",
    "start": None,
}


def tool(name: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, TOOLS / name, *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def event(at: str, e: list, d: list, **extra: object) -> dict:
    data = {
        "at": at,
        "iteration": 0,
        "equations": EQUATIONS,
        "E": e,
        "D": d,
        "boundary": ZERO,
        "faces": ZERO,
        "max_faces": 6,
        "max_terms": 6,
        "flux_mismatch": 0,
        "first_mismatch": None,
    }
    return data | extra


def run_dir(base: pathlib.Path, *events: dict, mesh_faces: int = 3, **info: object) -> pathlib.Path:
    run = base / "run"
    run.mkdir(parents=True)
    (run / "mesh-info.json").write_text(json.dumps({"counts": {"faces": mesh_faces}}))
    lines = [{"source": "solver", "message": "every face is planar", "data": {}}]
    lines += [{"source": "solver", "message": "conservation", "data": e} for e in events]
    (run / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (run / "run.json").write_text(json.dumps(RUN | {"status": "completed"} | info))
    return run


@pytest.mark.parametrize(
    ("e", "d", "extra", "verdict", "status"),
    [
        # q = tau exactly passes; the next double above it fails.
        ([TAU_HEX, 0.0, 0.0, 0.0, -TAU_HEX], ONE, {}, "PASS", 0),
        ([0.0, 0.0, 0.0, 0.0, -TAU_HEX * (1 + 2**-52)], ONE, {}, "FAIL", 1),
        # D_k = 0 passes only with E_k = 0.
        (ZERO, ZERO, {}, "PASS", 0),
        ([0.0, 0.0, 1e-300, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0, 1.0], {}, "FAIL", 1),
        # A non-finite sum, written null, fails, as does a q_k that overflows.
        (ZERO, ONE, {"boundary": [0.0, None, 0.0, 0.0, 0.0]}, "FAIL", 1),
        ([1e308, 0.0, 0.0, 0.0, 0.0], [1e-300, 1.0, 1.0, 1.0, 1.0], {}, "FAIL", 1),
        (ZERO, ONE, {"max_terms": 7}, "FAIL", 1),
        # A negative D, whose q would be negative.
        ([1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 1.0, 1.0, 1.0], {}, "FAIL", 1),
        # A mismatch voids the case, whatever else holds.
        (ONE, ONE, {"flux_mismatch": 2, "first_mismatch": 17}, "VOID", 3),
        # Above 62 faces, 2 gamma_(d+2) is the larger term: 200 u passes at d = 100.
        ([0.0, 0.0, 0.0, 0.0, 200 * U], ONE, {"max_faces": 100, "max_terms": 90}, "PASS", 0),
    ],
)
def test_the_verdict_follows_the_gate_7_rules(
    tmp_path: pathlib.Path, e: list, d: list, extra: dict, verdict: str, status: int
) -> None:
    run = run_dir(tmp_path, event("start", e, d, **extra))
    r = tool("gate7.py", run, "--at", "start")
    assert r.returncode == status, r.stderr
    line = json.loads(r.stdout, parse_constant=pytest.fail)
    assert line["verdict"] == verdict
    assert line["at"] == "start"
    assert (line["reasons"] == []) == (verdict == "PASS")


def test_start_reads_the_first_state_and_last_the_converged_solution(
    tmp_path: pathlib.Path,
) -> None:
    start = event("start", ONE, ONE)
    steps = [event("step", ONE, ONE, iteration=1), event("step", ZERO, [2.0] * 5, iteration=2)]
    run = run_dir(tmp_path, start, *steps, status="converged", iterations=2)
    evidence = tmp_path / "evidence.json"
    assert tool("gate7.py", run, "--at", "start", "--evidence", evidence).returncode == 1
    written = json.loads(evidence.read_text())
    assert written["event"] == start
    assert written["q"] == ONE
    assert written["checks"]["q_le_tau"] == [False] * 5
    assert written["run_json"]["status"] == "converged"
    r = tool("gate7.py", run)
    assert r.returncode == 0
    assert json.loads(r.stdout)["q"] == ZERO
    # The last event of a run that did not converge, or not of its last iteration, is no gate.
    for name, info in (("diverged", {"status": "diverged"}), ("short", {"iterations": 3})):
        other = run_dir(tmp_path / name, start, *steps, **({"status": "converged"} | info))
        assert tool("gate7.py", other).returncode == 1


def test_a_run_without_the_event_asked_for_is_unreadable(tmp_path: pathlib.Path) -> None:
    run = run_dir(tmp_path, event("step", ZERO, ONE))
    assert tool("gate7.py", run, "--at", "start").returncode == 2
    assert tool("gate7.py", tmp_path / "missing").returncode == 2


def faces_file(
    path: pathlib.Path,
    e: list,
    d: list,
    rows: list[list[float]],
    boundary: int = 0,
    names: list[str] | None = None,
) -> pathlib.Path:
    names = names or list("abcde")
    head = {"schema": "autan.conservation-faces/1", "equations": names, "E": e, "D": d}
    head |= {"max_faces": 6, "faces": len(rows) - boundary, "boundary_faces": boundary}
    body = "".join(
        f"{i},{int(i >= len(rows) - boundary)},0,0,0," + ",".join(map(repr, r)) + "\n"
        for i, r in enumerate(rows)
    )
    path.write_text(json.dumps(head) + "\nface,boundary,x,y,z," + ",".join(names) + "\n" + body)
    return path


def evidence_pair(
    base: pathlib.Path, sign: float, intact_e: list | None = None
) -> tuple[pathlib.Path, pathlib.Path]:
    # The analytic box's start: F = A (1, 1, 1, 1, 5.5) on a face of area 1/4, D = (12, 12, 12,
    # 12, 66); the mutant's E moved by -sign F.
    flux = [0.25, 0.25, 0.25, 0.25, 1.375]
    d = [12.0, 12.0, 12.0, 12.0, 66.0]
    face = {"matches": 1, "id": 3, "left_cell": 0, "rank": 0, "interface": False, "flux": flux}
    paths = []
    base.mkdir(parents=True, exist_ok=True)
    base_e = intact_e or ZERO
    mutant_e = [b - sign * f for b, f in zip(base_e, flux, strict=True)]
    for name, e in (("intact", base_e), ("mutant", mutant_e)):
        rows = faces_file(base / f"{name}.csv", e, d, [flux, flux, ONE], 1, EQUATIONS)
        selected = {"selected_face": face, "faces_file": str(rows)}
        run = run_dir(base / name, event("start", e, d, **selected))
        paths.append(base / f"{name}.json")
        tool("gate7.py", run, "--at", "start", "--evidence", paths[-1])
    return paths[0], paths[1]


def test_the_control_takes_the_left_incidence_dropped_and_nothing_else(
    tmp_path: pathlib.Path,
) -> None:
    intact, mutant = evidence_pair(tmp_path / "left", 1.0)
    r = tool("gate7_control.py", intact, mutant, *PINS)
    assert r.returncode == 0, r.stdout
    assert json.loads(r.stdout)["pass"] is True
    assert tool("gate7_control.py", intact, mutant, *PINS, "--interface").returncode == 1
    for pins in ((*PINS[:2], "--solution-sha256", "50"), ("--case-sha256", "cb", *PINS[2:])):
        assert tool("gate7_control.py", intact, mutant, *pins).returncode == 1
    assert tool("gate7_control.py", intact, mutant, *PINS[:2]).returncode == 2
    # A record whose tau is not its event's cannot scale the sensitivity checks.
    record = json.loads(mutant.read_text())
    record["tau"] /= 1000
    mutant.write_text(json.dumps(record))
    failed = json.loads(tool("gate7_control.py", intact, mutant, *PINS).stdout)["failed"]
    assert failed == ["records match their events"]
    intact, mutant = evidence_pair(tmp_path / "right", -1.0)
    r = tool("gate7_control.py", intact, mutant, *PINS)
    assert r.returncode == 1
    failed = json.loads(r.stdout)["failed"]
    assert failed
    assert all("signed defect" in name for name in failed)
    # A defect that matches but starts from an intact state the gate rejects.
    intact, mutant = evidence_pair(tmp_path / "failing", 1.0, [0.5, 0.5, 0.5, 0.5, 2.75])
    r = tool("gate7_control.py", intact, mutant, *PINS)
    assert json.loads(r.stdout)["failed"] == ["intact passes gate 7"]


def test_the_control_needs_both_face_files_and_a_run_that_did_not_diverge(
    tmp_path: pathlib.Path,
) -> None:
    intact, mutant = evidence_pair(tmp_path, 1.0)
    record = json.loads(mutant.read_text())
    record["face_rows_sha256"] = None
    mutant.write_text(json.dumps(record))
    assert json.loads(tool("gate7_control.py", intact, mutant, *PINS).stdout)["failed"] == [
        "every face flux unchanged"
    ]
    record = json.loads(intact.read_text())
    record["run_json"]["status"] = "diverged"
    intact.write_text(json.dumps(record))
    failed = json.loads(tool("gate7_control.py", intact, mutant, *PINS).stdout)["failed"]
    assert "intact run ended converged or completed" in failed


def test_the_report_gives_medians_minima_and_the_declared_note(tmp_path: pathlib.Path) -> None:
    # Ratios a: 1/8, 3/8, 5/8, 7/8 and b: 2/8 ... 8/8, so that every median is exact: the mean of
    # the two middle values of an even count. c is zero, d below 100 tau, e N/A (D = E = 0). The
    # last row is a boundary face, outside the population.
    rows = [[1.0, -2.0, 0.0, 1e-20, 0.0], [3.0, 4.0, 0.0, 1e-20, 0.0]]
    rows += [[-5.0, 6.0, 0.0, 1e-20, 0.0], [7.0, -8.0, 0.0, 1e-20, 0.0], [99.0] * 5]
    path = faces_file(tmp_path / "f.csv", ZERO, [8.0, 8.0, 1.0, 1.0, 0.0], rows, boundary=1)
    r = tool("gate7_report.py", path)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["faces"] == 4
    a, b, c, _, e = out["equations"]
    assert (a["median"], a["min"]) == (0.5, 0.125)
    assert (b["median"], b["min"]) == (0.625, 0.25)
    assert (c["median"], c["min"]) == (0.0, 0.0)
    assert e == {"name": "e", "status": "N/A"}
    assert out["median_of_max"] == 0.625
    assert [n.split(":")[0] for n in out["notes"]] == ["c", "d"]
    # D_k = 0 with E_k != 0 invalidates the report, as a non-finite flux or ratio does.
    bad = faces_file(tmp_path / "g.csv", [0.0, 0.0, 0.0, 0.0, 1.0], [*ONE[:4], 0.0], rows, 1)
    assert tool("gate7_report.py", bad).returncode == 1
    tiny = faces_file(tmp_path / "t.csv", ZERO, [1e-320, *ONE[1:]], [[1e308] * 5], 0)
    assert json.loads(tool("gate7_report.py", tiny).stdout)["valid"] is False
    negative = faces_file(tmp_path / "n.csv", ZERO, [-8.0, *ONE[1:]], rows, 1)
    assert tool("gate7_report.py", negative).returncode == 1
    # A non-finite flux invalidates it on a boundary face too, outside the population.
    for name, row in (("interior", 0), ("boundary", 4)):
        broken = [list(r) for r in rows]
        broken[row][0] = float("nan")
        r = tool("gate7_report.py", faces_file(tmp_path / f"{name}.csv", ZERO, ONE, broken, 1))
        assert r.returncode == 1, name
        assert json.loads(r.stdout)["valid"] is False


def test_a_faces_file_off_its_format_is_unreadable(tmp_path: pathlib.Path) -> None:
    good = faces_file(tmp_path / "f.csv", ZERO, ONE, [ONE, ONE, ONE], boundary=1)
    head, header, *rows = good.read_text().splitlines()
    cases = {
        "schema": [head.replace("conservation-faces/1", "conservation-faces/2"), header, *rows],
        "header": [head, header.replace("boundary,", ""), *rows],
        "count": [head, header, *rows[:-2], rows[-1]],
        "repeated id": [head, header, rows[0], rows[0], rows[2]],
        "centroid": [head, header, "0,0,nan,0,0," + rows[0].split(",", 5)[5], *rows[1:]],
        "number": [head, header, rows[0][:-3] + "one", *rows[1:]],
        "no interior face": [head.replace('"faces": 2', '"faces": 0'), header, rows[2]],
    }
    for name, lines in cases.items():
        path = tmp_path / f"{name}.csv"
        path.write_text("\n".join(lines) + "\n")
        assert tool("gate7_report.py", path).returncode == 2, name


def test_evidence_takes_a_faces_file_only_whole_and_of_its_event(tmp_path: pathlib.Path) -> None:
    flux = [0.25, 0.25, 0.25, 0.25, 1.375]
    d = [12.0, 12.0, 12.0, 12.0, 66.0]
    good = faces_file(tmp_path / "good.csv", ZERO, d, [flux, flux, ONE], 1, EQUATIONS)
    head, header, *rows = good.read_text().splitlines()
    # The same truncation on both sides would hash alike: cut rows under the declared counts, the
    # counts cut with them below the mesh's faces, or another event's sums.
    recounted = head.replace('"faces": 2', '"faces": 1')
    other = faces_file(tmp_path / "other.csv", ONE, d, [flux, flux, ONE], 1, EQUATIONS)
    cases = {
        "rows": [head, header, *rows[:-1]],
        "counts": [recounted, header, rows[0], rows[2]],
        "event": other.read_text().splitlines(),
    }
    for name, lines in cases.items():
        path = tmp_path / f"{name}.csv"
        path.write_text("\n".join(lines) + "\n")
        run = run_dir(tmp_path / name, event("start", ZERO, d, faces_file=str(path)))
        evidence = tmp_path / f"{name}.json"
        assert tool("gate7.py", run, "--at", "start", "--evidence", evidence).returncode == 2, name
        assert not evidence.exists()
