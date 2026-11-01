"""End-to-end benchmark: time to solution of a whole case, normalised for comparison with
published solvers (FUN3D on the DPW4 CRM, PyFR's explicit throughput).

A case runs as a chain of `autan-run` runs, each under one hold of the GPU lock: the driver
stops a run with its stop file once the hold's budget is spent and continues it from its
solution (`continue`, which judges convergence as the uninterrupted run would). The report
sums the chain and normalises it by the measured bandwidth; see docs/infra/bench.md.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from autan_bench.errors import BenchError, Code

SCHEMA_ID = "autan.e2e-result/1"

# DPW4 case 1 as FUN3D ran it (AIAA 2023-1226, section IV.A): M 0.85, Re_c 5e6, alpha 2.41 deg
# (C_L 0.5), T 560 degR. The grid is in inches and Autan reads coordinates as metres; the
# coefficients do not depend on the scale, so lengths and areas keep their inch values.
DPW4 = {
    "mach": 0.85,
    "aoa_deg": 2.41,
    "re": 5.0e6,
    "temperature_k": 560.0 * 5.0 / 9.0,
    "ref_length": 275.8,
    # Semispan grid: half the full-span 594 720 in^2.
    "ref_area": 297360.0,
    "moment_center": (1325.9, 0.0, 177.95),
}

CONDITIONS = {
    "inviscid": {"farfield": "farfield", "symmetry": "symmetry", "wall": "slip"},
    "sa": {"farfield": "farfield", "symmetry": "symmetry", "wall": "wall"},
}


def dpw4_case(mesh_info: dict[str, Any], mesh: str, turbulence: str, *, max_iterations: int,
              residual_orders: float, linear_solver: str) -> str:  # fmt: skip
    """The DPW4 case-1 TOML for `mesh`, one boundary rule per zone of its MeshInfo, by type."""
    if turbulence not in CONDITIONS:
        msg = f"turbulence must be one of {sorted(CONDITIONS)}"
        raise BenchError(Code.UNKNOWN_CONDITION, msg)
    model = (
        '{ model = "sa", nu_tilde_ratio = 3.0 }' if turbulence == "sa" else '{ model = "inviscid" }'
    )
    lines = [
        'schema = "autan.case/1"',
        f'name = "dpw4-crm-{turbulence}"',
        f"mesh = {json.dumps(mesh)}",
        "",
        "[physics]",
        'kind = "compressible"',
        'precision = "fp32+acc64"',
        f"turbulence = {model}",
        "",
        "[freestream]",
        f"mach = {DPW4['mach']}",
        f'aoa = "{DPW4["aoa_deg"]} deg"',
        f"re = {DPW4['re']}",
        f'ref_length = "{DPW4["ref_length"]} m"',
        f'temperature = "{DPW4["temperature_k"]!r} K"',
        "",
        "[time]",
        'kind = "steady"',
        f"max_iterations = {max_iterations}",
        f"residual_orders = {float(residual_orders)}",
        f'linear_solver = "{linear_solver}"',
    ]
    walls = []
    for zone in mesh_info["boundary_zones"]:
        kind = CONDITIONS[turbulence].get(zone["type"])
        if kind is None:
            msg = f"zone {zone['name']!r} has type {zone['type']!r}: no DPW4 condition"
            raise BenchError(Code.UNKNOWN_CONDITION, msg)
        if kind in ("wall", "slip"):
            walls.append(zone["name"])
        lines += ["", "[[boundaries]]", f"zones = {json.dumps(zone['name'])}",
                  f'condition = {{ kind = "{kind}" }}']  # fmt: skip
    center = ", ".join(f'"{c} m"' for c in DPW4["moment_center"])
    for zone in walls:
        lines += ["", "[[monitors]]", 'kind = "forces"', f"name = {json.dumps(_monitor(zone))}",
                  f"surfaces = {json.dumps(zone)}", "every = 10",
                  f'ref_area = "{DPW4["ref_area"]} m ** 2"', f"moment_center = [{center}]"]  # fmt: skip
    return "\n".join(lines) + "\n"


def _monitor(zone: str) -> str:
    return "forces." + "_".join(zone.lower().split())


def total_cd(monitors: dict[str, dict[str, float]]) -> float:
    """C_D of the aircraft: the sum over the per-zone force monitors of `dpw4_case`."""
    return math.fsum(m["cd"] for name, m in monitors.items() if name.startswith("forces."))


GB10_VENDOR_GBPS = 273.0  # LPDDR5X-8533 on a 256-bit bus (NVIDIA DGX Spark datasheet)
GRID_LABEL = "different grid (public tetrahedra vs FUN3D's merged prisms)"


def _density_drop(first: dict[str, Any], last: dict[str, Any]) -> float | None:
    r0, r1 = first["residuals"].get("density"), last["residuals"].get("density")
    if not r0 or not r1:
        return None
    return math.log10(r0 / r1)


def report(chain: list[dict[str, Any]], mesh: dict[str, int], stream_gbps: float,
           reference: dict[str, Any], *, stages: int | None = None,
           cpu: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """The end-to-end result of a chain of runs, oldest first.

    Each element is {"run": run.json, "first": its first history line, "last": its last}.
    `mesh` gives the grid's "cells" and "nodes"; `stream_gbps` the GB10 peak measured by
    STREAM. `stages` (right-hand-side evaluations per iteration) adds the explicit-path
    throughput in PyFR's metric; `cpu`, the report of the same case on the CPU, adds the
    speedup per iteration.
    """
    runs = [link["run"] for link in chain]
    for earlier, later in itertools.pairwise(runs):
        if later["start"] is None or later["start"]["run_id"] != earlier["run_id"]:
            msg = f"{later['run_id']} does not continue {earlier['run_id']}"
            raise BenchError(Code.CHAIN_BROKEN, msg)
    if any(r["timing"] is None for r in runs):
        msg = "every run of the chain must record its timing"
        raise BenchError(Code.CHAIN_BROKEN, msg)
    solve = math.fsum(r["timing"]["solve_seconds"] for r in runs)
    iterations = sum(r["timing"]["iterations"] for r in runs)
    setup = runs[0]["timing"]["wall_seconds"] - runs[0]["timing"]["solve_seconds"]
    status = runs[-1]["status"]
    cells, nodes = mesh["cells"], mesh["nodes"]
    ns_cell = solve * 1e9 / (cells * iterations)
    ns_node = solve * 1e9 / (nodes * iterations)
    monitors = runs[-1]["monitors"] or {}
    result: dict[str, Any] = {
        "status": status,
        "converged": status == "converged",
        "runs": len(runs),
        "iterations": iterations,
        "residual_drop_orders": _density_drop(chain[0]["first"], chain[-1]["last"]),
        "solve_seconds": solve,
        "setup_seconds": setup,
        # One setup, as the run would have taken uninterrupted; continuations' reads excluded.
        "time_to_solution_seconds": solve + setup if status == "converged" else None,
        "cd": total_cd(monitors) if monitors else None,
    }
    per_unit = {
        "ns_per_cell_iteration": ns_cell,
        "ns_per_vertex_iteration": ns_node,
        # Bytes the measured peak bandwidth moves in the time of one cell (vertex) iteration.
        "peak_bytes_per_cell_iteration": ns_cell * stream_gbps,
        "peak_bytes_per_vertex_iteration": ns_node * stream_gbps,
        "percent_of_peak": None,
    }
    grid = reference["grid"]
    rows = []
    for ref in reference["runs"]:
        ref_ns = ref["seconds"] * 1e9 / (grid["vertices"] * reference["iterations"])
        scale = ref["peak_GBps"] / stream_gbps
        rows.append({
            "device": ref["device"], "kind": ref["kind"], "seconds": ref["seconds"],
            "peak_GBps": ref["peak_GBps"], "ns_per_vertex_iteration": ref_ns,
            "peak_bytes_per_vertex_iteration": ref_ns * ref["peak_GBps"],
            "gb10_equivalent_seconds": ref["seconds"] * scale,
            "gb10_equivalent_ns_per_vertex_iteration": ref_ns * scale,
        })  # fmt: skip
    best = min((r for r in rows if r["kind"] == "gpu"), key=lambda r: r["gb10_equivalent_seconds"])
    fun3d = {
        "label": GRID_LABEL,
        "source": reference["source"],
        "autan_grid": {"cells": cells, "nodes": nodes, "discretisation": "cell-centred"},
        "fun3d_grid": {
            "vertices": grid["vertices"],
            "elements": grid["elements"],
            "discretisation": grid["discretisation"],
        },
        "conversion": (
            f"Autan solves {cells} cell unknowns, FUN3D {grid['vertices']} vertex "
            f"unknowns ({cells / grid['vertices']:.2f}x); per-vertex costs divide "
            "Autan's time by its own grid's nodes"
        ),
        "fun3d_cd": reference["cd"],
        "fun3d_iterations": reference["iterations"],
        "normalisation": f"time x device peak / GB10 STREAM ({stream_gbps:.1f} GB/s)",
        "rows": rows,
        "time_ratio_to_best_gpu": (
            result["time_to_solution_seconds"] / best["gb10_equivalent_seconds"]
            if result["time_to_solution_seconds"]
            else None
        ),
        "per_vertex_iteration_ratio_to_best_gpu": ns_node
        / best["gb10_equivalent_ns_per_vertex_iteration"],
    }
    doc: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "comparable": False,
        "result": result,
        "per_unit": per_unit,
        "bandwidth": {"gb10_stream_GBps": stream_gbps, "gb10_vendor_GBps": GB10_VENDOR_GBPS},
        "fun3d": fun3d,
        "explicit": None,
        "cpu": None,
        "chain": [{"run_id": r["run_id"], "status": r["status"], **r["timing"]} for r in runs],
    }
    if stages is not None:
        unknowns = len(chain[-1]["last"]["residuals"])
        gdofs = cells * unknowns * stages * iterations / solve / 1e9
        pyfr = reference["pyfr"]
        doc["explicit"] = {
            "stages": stages,
            "unknowns_per_cell": unknowns,
            "gdof_per_second": gdofs,
            "mdof_per_second_per_GBps": gdofs * 1e3 / stream_gbps,
            "pyfr_mdof_per_second_per_GBps": pyfr["gdof_per_s"] * 1e3 / pyfr["peak_GBps"],
            "pyfr": pyfr["source"],
        }
    if cpu is not None:
        doc["cpu"] = {
            "speedup_per_iteration": cpu["per_unit"]["ns_per_cell_iteration"] / ns_cell,
            "cpu_ns_per_cell_iteration": cpu["per_unit"]["ns_per_cell_iteration"],
            "bandwidth_ratio": None,
        }
    return doc


def _history_ends(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """First and last complete lines of a run's history.jsonl."""
    lines = []
    for line in (run_dir / "history.jsonl").read_text().splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if not lines:
        msg = f"{run_dir}: no history"
        raise BenchError(Code.CHAIN_BROKEN, msg)
    return lines[0], lines[-1]


def link(run_dir: Path) -> dict[str, Any]:
    """One run of a chain, as `report` takes it."""
    first, last = _history_ends(run_dir)
    return {"run": json.loads((run_dir / "run.json").read_text()), "first": first, "last": last}


def run_chain(*, case: Path, mesh_info: Path, device_kind: str, runs: Path, autan: list[str],
              autan_run: Path, lock: list[str], budget_s: int, max_runs: int) -> list[Path]:  # fmt: skip
    """Run a case to its end as a chain of runs, each in one hold of the GPU lock.

    Each run is prepared with `autan prepare` (continuing the previous one), then started under
    `lock` (the autan-gpu-lock command and its options, ending with `--`) with a stop file the
    driver creates `budget_s` seconds into the hold. The chain ends when a run converges,
    completes its iteration budget, diverges or fails, or after `max_runs` runs. Returns the
    run directories, oldest first.
    """
    runs.mkdir(parents=True, exist_ok=True)
    chain: list[Path] = []
    for k in range(max_runs):
        solver_input = runs / f"input-{k}.json"
        prepare = [*autan, "prepare", str(case), "--mesh-info", str(mesh_info),
                   "--device-kind", device_kind, "-o", str(solver_input)]  # fmt: skip
        if chain:
            prepare += ["--start", str(chain[-1])]
        subprocess.run(prepare, check=True)
        stop = runs / f"stop-{k}"
        stop.unlink(missing_ok=True)
        # The stop file is created inside the hold, so the budget counts from the lock taken.
        script = (f"(sleep {budget_s}; touch {stop}) & "
                  f"exec {autan_run} --input {solver_input} --runs {runs} --stop-file {stop}")  # fmt: skip
        out = subprocess.run([*lock, "sh", "-c", script], check=False, capture_output=True,
                             text=True)  # fmt: skip
        lines = out.stdout.strip().splitlines()
        if out.returncode != 0 or not lines:
            msg = f"run {k} ended with {out.returncode}: {out.stderr[-2000:]}"
            raise BenchError(Code.RUN_FAILED, msg)
        run_dir = Path(lines[-1])
        chain.append(run_dir)
        status = json.loads((run_dir / "run.json").read_text())["status"]
        if status != "stopped":
            break
    return chain


def summary(doc: dict[str, Any]) -> str:
    """The report in a few lines, the grid caveat first."""
    r, u, f = doc["result"], doc["per_unit"], doc["fun3d"]
    best = min((row for row in f["rows"] if row["kind"] == "gpu"),
               key=lambda row: row["gb10_equivalent_seconds"])  # fmt: skip
    tts = r["time_to_solution_seconds"]
    reached = f"{tts:.0f} s" if tts is not None else "not reached"
    autan_cd = f"{r['cd']:.6f}" if r["cd"] is not None else "-"
    lines = [
        "NOT COMPARABLE" if not doc["comparable"] else "comparable",
        f"{f['label']}; {f['conversion']}",
        (
            f"{r['status']} after {r['iterations']} iterations in {r['runs']} runs, density down "
            f"{r['residual_drop_orders']:.2f} orders; time to solution {reached}"
        ),
        (
            f"{u['ns_per_cell_iteration']:.2f} ns/cell/iter, {u['ns_per_vertex_iteration']:.2f} "
            f"ns/vertex/iter, {u['peak_bytes_per_vertex_iteration'] / 1e3:.1f} kB peak-equivalent "
            "per vertex and iteration"
        ),
        (
            f"FUN3D {best['device']}: {best['seconds']} s, on GB10 by bandwidth "
            f"{best['gb10_equivalent_seconds']:.0f} s, "
            f"{best['gb10_equivalent_ns_per_vertex_iteration']:.1f} ns/vertex/iter; C_D "
            f"{f['fun3d_cd']:.6f}, Autan {autan_cd}"
        ),
    ]
    if doc["explicit"]:
        x = doc["explicit"]
        lines.append(f"explicit: {x['gdof_per_second']:.3f} GDoF/s, "
                     f"{x['mdof_per_second_per_GBps']:.2f} MDoF/s per GB/s "
                     f"(PyFR {x['pyfr_mdof_per_second_per_GBps']:.2f})")  # fmt: skip
    return "\n".join(lines)
