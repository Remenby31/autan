import json
import sys
import tomllib
from pathlib import Path

import pytest
from autan_bench import e2e
from autan_bench.errors import BenchError, Code

REFERENCE = json.loads((Path(__file__).parents[1] / "e2e" / "fun3d-dpw4.json").read_text())
MESH = {"source": {"sha256": "m" * 64}, "counts": {"cells": 1_000_000, "nodes": 200_000}}


def run_json(run_id: str, start: dict | None, status: str, wall: float, solve: float,
             iterations: int, total: int, cd: float | None = None) -> dict:  # fmt: skip
    monitors = {"forces.patch_7": {"cd": cd / 2}, "forces.patch_8": {"cd": cd / 2}} if cd else None
    return {
        "run_id": run_id,
        "status": status,
        "mesh": {"sha256": "m" * 64},
        "iterations": total,
        "start": start,
        "timing": {
            "wall_seconds": wall,
            "solve_seconds": solve,
            "iterations": iterations,
            "ns_per_cell_iteration": 0.0,
        },
        "monitors": monitors,
    }


def line(density: float) -> dict:
    names = ("density", "momentum-x", "momentum-y", "momentum-z", "energy")
    return {"residuals": dict.fromkeys(names, density)}


def case(model: str) -> dict:
    return {"physics": {"turbulence": {"model": model}}}


def chain(model: str = "inviscid", cd: float = 0.02) -> list[dict]:
    first = run_json("a", None, "stopped", 15.0, 10.0, 100, 100)
    second = run_json("b", {"run_id": "a", "mode": "continue", "iterations": 100}, "converged",
                      21.0, 20.0, 200, 300, cd=cd)  # fmt: skip
    return [
        {"run": first, "case": case(model), "first": line(1.0), "last": line(1e-3)},
        {"run": second, "case": case(model), "first": line(1e-3), "last": line(1e-8)},
    ]


def test_a_chain_is_reported_per_unit_and_against_fun3d_by_bandwidth():
    doc = e2e.report(chain(), MESH, 200.0, REFERENCE, stages=3)
    r, u, f = doc["result"], doc["per_unit"], doc["fun3d"]
    rel = 1e-12
    # 30 s of solve for 300 iterations of 1e6 cells (2e5 vertices), 36 s of wall time.
    assert (r["iterations"], r["runs"], r["converged"]) == (300, 2, True)
    assert r["time_to_solution_seconds"] == pytest.approx(36.0, rel=rel)
    assert r["solve_seconds"] == pytest.approx(30.0, rel=rel)
    assert r["residual_drop_orders"] == pytest.approx(8.0, rel=rel)
    assert r["cd"] == pytest.approx(0.02, rel=rel)
    assert u["ns_per_cell_iteration"] == pytest.approx(100.0, rel=rel)
    assert u["ns_per_vertex_iteration"] == pytest.approx(500.0, rel=rel)
    assert u["peak_bytes_per_cell_iteration"] == pytest.approx(100.0 * 200.0, rel=rel)
    # Published: FUN3D on the A100, 430 s for 3000 iterations of 3 675 916 vertices.
    a100 = next(p for p in f["published"] if "A100" in p["device"])
    assert a100["ns_per_vertex_iteration"] == pytest.approx(430e9 / (3675916 * 3000), rel=rel)
    # Estimated on GB10 by bandwidth, kept apart from what is published or measured: the
    # best is the V100 (720 s at 900 GB/s), not the A100 (430 s at 1555 GB/s).
    est = f["estimate_on_gb10"]
    assert est["label"].startswith("ESTIMATE")
    assert est["best_gpu"] == "NVIDIA 16 GB SXM V100"
    assert est["autan_time_over_best"] == pytest.approx(36.0 / (720 * 900 / 200.0), rel=rel)
    assert f["label"] == e2e.GRID_LABEL
    assert doc["comparable"] is False
    assert f["autan_grid"]["cells"] == 1_000_000
    assert f["fun3d_grid"]["vertices"] == 3675916
    # Explicit: 1e6 cells x 5 unknowns x 3 stages x 300 iterations in 30 s; PyFR 6.004 GDoF/s
    # at 4 TB/s.
    x = doc["explicit"]
    assert x["gdof_per_second"] == pytest.approx(0.15, rel=rel)
    assert x["mdof_per_second_per_GBps"] == pytest.approx(0.75, rel=rel)
    assert x["pyfr_mdof_per_second_per_GBps"] == pytest.approx(6.004e3 / 4000, rel=rel)
    text = e2e.summary(doc)
    assert "NOT COMPARABLE" in text
    assert "ESTIMATE" in text


def test_c_d_is_judged_only_for_a_converged_sa_run_with_the_declared_tolerance():
    tol = REFERENCE["declared"]["cd_tolerance"]
    near, far = REFERENCE["cd"] + 0.9 * tol, REFERENCE["cd"] + 1.1 * tol
    assert e2e.report(chain(), MESH, 200.0, REFERENCE)["result"]["cd_within_tolerance"] is None
    for cd, verdict in ((near, True), (far, False)):
        doc = e2e.report(chain("sa", cd), MESH, 200.0, REFERENCE)
        assert doc["result"]["cd_within_tolerance"] is verdict


def test_only_a_whole_continued_chain_on_the_meshinfo_s_mesh_is_reported():
    links = chain()
    stopped = links[:1]
    assert e2e.report(stopped, MESH, 200.0, REFERENCE)["result"]["time_to_solution_seconds"] is None
    warm_start = {"run_id": "a", "mode": "warm", "iterations": 100}
    warm = [links[0], {**links[1], "run": {**links[1]["run"], "start": warm_start}}]
    late = links[1:]
    other_mesh = {**MESH, "source": {"sha256": "x" * 64}}
    for bad, mesh in ((warm, MESH), (late, MESH), (links, other_mesh)):
        with pytest.raises(BenchError) as err:
            e2e.report(bad, mesh, 200.0, REFERENCE)
        assert err.value.code == Code.CHAIN_BROKEN


def test_a_residual_that_reaches_zero_is_reported_without_a_drop():
    links = chain()
    links[1]["last"] = line(0.0)
    doc = e2e.report(links, MESH, 200.0, REFERENCE)
    assert doc["result"]["residual_drop_orders"] is None
    assert "unmeasurable" in e2e.summary(doc)


def test_the_dpw4_case_gives_each_zone_its_condition_by_type():
    mesh_info = {"boundary_zones": [
        {"name": "Patch   1", "type": "farfield"}, {"name": "Patch   3", "type": "symmetry"},
        {"name": "Patch   7", "type": "wall"},
    ]}  # fmt: skip
    for turbulence, wall in (("inviscid", "slip"), ("sa", "wall")):
        doc = tomllib.loads(e2e.dpw4_case(mesh_info, "m.cgns", turbulence, max_iterations=10,
                                          residual_orders=8, linear_solver="multigrid"))  # fmt: skip
        assert {b["zones"]: b["condition"]["kind"] for b in doc["boundaries"]} == {
            "Patch   1": "farfield", "Patch   3": "symmetry", "Patch   7": wall,
        }  # fmt: skip
        assert doc["freestream"]["mach"] == 0.85
        assert doc["freestream"]["aoa"] == "2.41 deg"
        # 560 degR
        assert float(doc["freestream"]["temperature"].split()[0]) == pytest.approx(311.1111111)
        assert [m["surfaces"] for m in doc["monitors"]] == ["Patch   7"]
    with pytest.raises(BenchError) as err:
        e2e.dpw4_case({"boundary_zones": [{"name": "i", "type": "inlet"}]}, "m", "sa",
                      max_iterations=1, residual_orders=8, linear_solver="multigrid")  # fmt: skip
    assert err.value.code == Code.UNKNOWN_CONDITION


# The SDK and the solver are the external boundary: fakes do what the real ones do, exit
# codes included (solver-input.md: 130 for a run stopped by its stop file).
FAKE_AUTAN = """#!{python}
import json, sys
args = sys.argv[1:]
out = args[args.index("-o") + 1]
start = args[args.index("--start") + 1] if "--start" in args else None
json.dump({{"start": start}}, open(out, "w"))
"""
FAKE_RUN = """#!{python}
import json, os, pathlib, sys, time
args = sys.argv[1:]
inp = json.load(open(args[args.index("--input") + 1]))
runs, stop = pathlib.Path(args[args.index("--runs") + 1]), args[args.index("--stop-file") + 1]
k = len(list(runs.glob("run-*")))
d = runs / f"run-{{k}}"
d.mkdir()
# The first run goes on until its stop file appears; the second converges on its own.
while k == 0 and not os.path.exists(stop):
    time.sleep(0.05)
status = "stopped" if k == 0 else "converged"
json.dump({{"run_id": d.name, "status": status}}, open(d / "run.json", "w"))
print(d)
sys.exit(130 if status == "stopped" else 0)
"""


def drive(tmp_path: Path, idle: list[str]) -> list[Path]:
    autan = tmp_path / "autan"
    autan.write_text(FAKE_AUTAN.format(python=sys.executable))
    run = tmp_path / "autan run"  # a space: every path is quoted for the shell
    run.write_text(FAKE_RUN.format(python=sys.executable))
    for tool in (autan, run):
        tool.chmod(0o755)
    return e2e.run_chain(case=tmp_path / "case.toml", mesh_info=tmp_path / "mi.json",
                         device_kind="cuda", runs=tmp_path / "my runs", autan=[str(autan)],
                         autan_run=run, lock=["env", "--"], idle=idle, budget_s=1, max_runs=5,
                         cooldown_s=0)  # fmt: skip


def test_a_chain_continues_each_stopped_run_until_one_ends(tmp_path):
    runs = tmp_path / "my runs"
    found = drive(tmp_path, ["true"])
    assert [d.name for d in found] == ["run-0", "run-1"]
    assert json.loads((runs / "input-0.json").read_text())["start"] is None
    assert json.loads((runs / "input-1.json").read_text())["start"] == str(found[0])
    assert (runs / "stop-0").exists()


def test_a_chain_does_not_run_on_a_busy_gpu(tmp_path):
    with pytest.raises(BenchError) as err:
        drive(tmp_path, ["false"])
    assert err.value.code == Code.GPU_BUSY
    assert not list((tmp_path / "my runs").glob("run-*"))
