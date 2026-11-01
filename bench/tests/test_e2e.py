import json
import sys
import tomllib
from pathlib import Path

import pytest
from autan_bench import e2e
from autan_bench.errors import BenchError, Code

REFERENCE = json.loads((Path(__file__).parents[1] / "e2e" / "fun3d-dpw4.json").read_text())


def run_json(run_id: str, start: str | None, status: str, wall: float, solve: float,
             iterations: int, cd: float | None = None) -> dict:  # fmt: skip
    monitors = {"forces.patch_7": {"cd": cd / 2}, "forces.patch_8": {"cd": cd / 2}} if cd else None
    return {
        "run_id": run_id,
        "status": status,
        "start": None if start is None else {"run_id": start},
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


CHAIN = [
    {"run": run_json("a", None, "stopped", 15.0, 10.0, 100), "first": line(1.0),
     "last": line(1e-3)},
    {"run": run_json("b", "a", "converged", 21.0, 20.0, 200, cd=0.02), "first": line(1e-3),
     "last": line(1e-8)},
]  # fmt: skip
MESH = {"cells": 1_000_000, "nodes": 200_000}


def test_a_chain_is_reported_per_unit_and_against_fun3d_by_bandwidth():
    doc = e2e.report(CHAIN, MESH, 200.0, REFERENCE, stages=3)
    r, u, f = doc["result"], doc["per_unit"], doc["fun3d"]
    rel = 1e-12
    # 30 s for 300 iterations of 1e6 cells (2e5 vertices); one setup, the first run's 5 s.
    assert (r["iterations"], r["runs"], r["converged"]) == (300, 2, True)
    assert r["time_to_solution_seconds"] == pytest.approx(35.0, rel=rel)
    assert r["residual_drop_orders"] == pytest.approx(8.0, rel=rel)
    assert r["cd"] == pytest.approx(0.02, rel=rel)
    assert u["ns_per_cell_iteration"] == pytest.approx(100.0, rel=rel)
    assert u["ns_per_vertex_iteration"] == pytest.approx(500.0, rel=rel)
    assert u["peak_bytes_per_cell_iteration"] == pytest.approx(100.0 * 200.0, rel=rel)
    # FUN3D on the A100: 430 s for 3000 iterations of 3 675 916 vertices, at 1555 GB/s.
    a100 = next(row for row in f["rows"] if "A100" in row["device"])
    assert a100["ns_per_vertex_iteration"] == pytest.approx(430e9 / (3675916 * 3000), rel=rel)
    assert a100["gb10_equivalent_seconds"] == pytest.approx(430 * 1555 / 200.0, rel=rel)
    # The reference is the GPU fastest once normalised by bandwidth: the V100 (720 s, 900 GB/s).
    assert f["time_ratio_to_best_gpu"] == pytest.approx(35.0 / (720 * 900 / 200.0), rel=rel)
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
    assert "NOT COMPARABLE" in e2e.summary(doc)


def test_an_unconverged_or_broken_chain_has_no_time_to_solution():
    stopped = [CHAIN[0]]
    assert e2e.report(stopped, MESH, 200.0, REFERENCE)["result"]["time_to_solution_seconds"] is None
    broken = [CHAIN[0], {**CHAIN[1], "run": run_json("b", "x", "converged", 1.0, 1.0, 1)}]
    with pytest.raises(BenchError) as err:
        e2e.report(broken, MESH, 200.0, REFERENCE)
    assert err.value.code == Code.CHAIN_BROKEN


def test_the_dpw4_case_gives_each_zone_its_condition_by_type():
    mesh_info = {"boundary_zones": [
        {"name": "Patch   1", "type": "farfield"}, {"name": "Patch   3", "type": "symmetry"},
        {"name": "Patch   7", "type": "wall"},
    ]}  # fmt: skip
    for turbulence, wall in (("inviscid", "slip"), ("sa", "wall")):
        case = tomllib.loads(e2e.dpw4_case(mesh_info, "m.cgns", turbulence, max_iterations=10,
                                           residual_orders=8, linear_solver="multigrid"))  # fmt: skip
        assert {b["zones"]: b["condition"]["kind"] for b in case["boundaries"]} == {
            "Patch   1": "farfield", "Patch   3": "symmetry", "Patch   7": wall,
        }  # fmt: skip
        assert case["freestream"]["mach"] == 0.85
        assert case["freestream"]["aoa"] == "2.41 deg"
        # 560 degR
        assert float(case["freestream"]["temperature"].split()[0]) == pytest.approx(311.1111111)
        assert [m["surfaces"] for m in case["monitors"]] == ["Patch   7"]
    with pytest.raises(BenchError) as err:
        e2e.dpw4_case({"boundary_zones": [{"name": "i", "type": "inlet"}]}, "m", "sa",
                      max_iterations=1, residual_orders=8, linear_solver="multigrid")  # fmt: skip
    assert err.value.code == Code.UNKNOWN_CONDITION


# The SDK and the solver are the external boundary: fakes write what the real ones would.
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
# The second run converges on its own; the first runs until the stop file appears.
while k == 0 and not os.path.exists(stop):
    time.sleep(0.05)
status = "stopped" if k == 0 else "converged"
start = None if inp["start"] is None else {{"run_id": pathlib.Path(inp["start"]).name}}
json.dump({{"run_id": d.name, "status": status, "start": start}}, open(d / "run.json", "w"))
print(d)
"""


def test_a_chain_continues_each_stopped_run_until_one_ends(tmp_path):
    autan = tmp_path / "autan"
    autan.write_text(FAKE_AUTAN.format(python=sys.executable))
    run = tmp_path / "autan-run"
    run.write_text(FAKE_RUN.format(python=sys.executable))
    for tool in (autan, run):
        tool.chmod(0o755)
    runs = tmp_path / "runs"
    chain = e2e.run_chain(case=tmp_path / "case.toml", mesh_info=tmp_path / "mi.json",
                          device_kind="cuda", runs=runs, autan=[str(autan)], autan_run=run,
                          lock=["env", "--"], budget_s=1, max_runs=5)  # fmt: skip
    assert [d.name for d in chain] == ["run-0", "run-1"]
    assert json.loads((runs / "input-0.json").read_text())["start"] is None
    assert json.loads((runs / "input-1.json").read_text())["start"] == str(chain[0])
    assert (runs / "stop-0").exists()
