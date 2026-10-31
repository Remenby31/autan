"""Synthetic run directories in the autan.results/1 layout, until the SDK's fake backend lands.

Their numbers are invented, their solver commit is all zeros and `case.sha256` hashes the TOML
bytes instead of the canonical document; never use them as reference
data. `python -m autan_ui.demo RUNS [--live SECONDS]` writes them.
"""

import argparse
import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SOLVER = {"version": "0.0.0", "commit": "0" * 40, "backend": "fake"}
EQUATIONS = ("density", "momentum-x", "momentum-y", "momentum-z", "energy", "nu-tilde")

CASE_TOML = """schema = "autan.case/1"
name = "{name}"
mesh = "crm-hl.cgns"

[physics]
kind = "compressible"
precision = "{precision}"
turbulence = {{ model = "sa", nu_tilde_ratio = 3.0 }}
gas = {{ kind = "ideal-gas", name = "air", molar_mass = "0.0289647 kg / mol", gamma = 1.4, \
prandtl = 0.72, viscosity = {{ law = "sutherland", mu_ref = "1.716e-05 Pa * s", \
t_ref = "273.15 K", s = "110.4 K" }} }}

[freestream]
mach = 0.2
aoa = "{aoa} deg"
sideslip = "0.0 deg"
re = 5490000.0
ref_length = "7.0 m"
temperature = "288.15 K"

[time]
kind = "steady"
max_iterations = {max_iterations}
residual_orders = 8.0

[[boundaries]]
zones = "wing.*"
condition = {{ kind = "wall" }}

[[boundaries]]
zones = "symmetry"
condition = {{ kind = "symmetry" }}

[[monitors]]
kind = "forces"
name = "forces"
surfaces = "wing.*"
every = 10
ref_area = "191.84 m ** 2"
"""

MESH_INFO: dict[str, Any] = {
    "schema": "autan.mesh-info/1",
    "source": {
        "path": "crm-hl.cgns",
        "format": "cgns",
        "sha256": hashlib.sha256(b"demo").hexdigest(),
    },
    "counts": {
        "cells": 48200000,
        "faces": 151000000,
        "nodes": 52000000,
        "cells_by_type": {
            "tetra": 964000,
            "pyramid": 0,
            "prism": 13014000,
            "hexa": 0,
            "polyhedron": 34222000,
        },
    },
    "boundary_zones": [
        {"name": "wing.main", "type": "wall", "native_type": "BCWall", "faces": 812000},
        {"name": "wing.slat", "type": "wall", "native_type": "BCWall", "faces": 214000},
        {"name": "wing.flap", "type": "wall", "native_type": "BCWall", "faces": 301000},
        {"name": "fuselage", "type": "wall", "native_type": "BCWall", "faces": 420000},
        {"name": "symmetry", "type": "symmetry", "native_type": "BCSymmetryPlane", "faces": 90000},
        {"name": "farfield", "type": "farfield", "native_type": "BCFarfield", "faces": 30000},
    ],
    "cell_zones": [{"name": "fluid", "type": None, "native_type": None, "cells": 48200000}],
    "interfaces": [],
    "quality": {"max_non_orthogonality_deg": 62.0, "max_faces_per_cell": 214},
}


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _history_line(iteration: int, aoa: float, rate: float, noise: random.Random) -> dict[str, Any]:
    decay = math.exp(-rate * iteration)
    residuals = {
        eq: 10 ** (-2 - k * 0.3) * decay * (1 + 0.15 * noise.random())
        for k, eq in enumerate(EQUATIONS)
    }
    settle = 1 - math.exp(-iteration / 400)
    cl = 0.11 * aoa * settle + 0.02 * math.sin(iteration / 37) * (1 - settle)
    cd = 0.02 + 0.0007 * aoa**2 * settle
    line: dict[str, Any] = {"iteration": iteration, "time": None, "residuals": residuals}
    if iteration % 10 == 0:
        q_area = 0.5 * 1.2 * 68.0**2 * 191.84
        line["monitors"] = {
            "forces": {
                "fx": cd * q_area,
                "fy": 0.0,
                "fz": cl * q_area,
                "mx": 0.0,
                "my": -0.4 * q_area * 7.0,
                "mz": 0.0,
                "cl": cl,
                "cd": cd,
                "cs": 0.0,
                "cmx": 0.0,
                "cmy": -0.4,
                "cmz": 0.0,
            }
        }
    return line


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_run(
    root: Path,
    *,
    started: datetime,
    aoa: float,
    status: str,
    iterations: int,
    precision: str = "fp32+acc64",
    pace: float = 0.0,
    seed: int = 0,
) -> Path:
    noise = random.Random(seed)
    run_id = f"run-{started.strftime('%Y%m%dT%H%M%SZ')}-{noise.getrandbits(32):08x}"
    directory = root / run_id
    directory.mkdir(parents=True)
    case_text = CASE_TOML.format(
        name="crm-hl", precision=precision, aoa=aoa, max_iterations=max(iterations, 3000)
    )
    (directory / "case.toml").write_text(case_text, encoding="utf-8")
    _write_json(directory / "mesh-info.json", MESH_INFO)
    case_sha = hashlib.sha256(case_text.encode()).hexdigest()
    mesh_sha = MESH_INFO["source"]["sha256"]
    devices = {"kind": "cpu", "count": 1}
    identity = "\n".join(
        [case_sha, mesh_sha, SOLVER["commit"], precision, SOLVER["backend"], "cpu", "1"]
    )
    run: dict[str, Any] = {
        "schema": "autan.results/1",
        "run_id": run_id,
        "identity": hashlib.sha256(identity.encode()).hexdigest(),
        "status": "running",
        "case": {"name": "crm-hl", "sha256": case_sha},
        "mesh": {"path": "/data/crm-hl.cgns", "sha256": mesh_sha},
        "solver": SOLVER,
        "precision": precision,
        "deterministic": True,
        "devices": devices,
        "started": _stamp(started),
        "finished": None,
        "iterations": 0,
        "time": None,
        "averaging": None,
        "diagnostics": [],
    }
    _write_json(directory / "run.json", run)
    rate = 0.0065 if status != "diverged" else 0.002
    with (directory / "history.jsonl").open("a", encoding="utf-8") as history:
        for iteration in range(1, iterations + 1):
            history.write(json.dumps(_history_line(iteration, aoa, rate, noise)) + "\n")
            if pace:
                history.flush()
                time.sleep(pace)
                if iteration % 50 == 0:
                    run["iterations"] = iteration
                    _write_json(directory / "run.json", run)
    run["iterations"] = iterations
    if status != "running":
        run["status"] = status
        run["finished"] = _stamp(started + timedelta(seconds=iterations * 0.28))
    if status == "diverged":
        run["diagnostics"] = [
            {
                "code": "D0107",
                "severity": "divergence",
                "message": f"non-finite density at iteration {iterations}",
                "location": None,
                "fix": "lower the CFL",
                "doc": "docs/sdk/errors.md#d0107",
                "data": {"cell": 3418772, "iteration": iterations},
            }
        ]
    _write_json(directory / "run.json", run)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m autan_ui.demo")
    parser.add_argument("runs", type=Path)
    parser.add_argument("--live", type=float, default=0.0, help="seconds per live iteration")
    args = parser.parse_args()
    now = datetime.now(UTC).replace(microsecond=0)
    polar = (7.05, 11.29, 17.05, 19.57, 21.47)
    for k, aoa in enumerate(polar):
        write_run(
            args.runs,
            started=now - timedelta(hours=6 - k),
            aoa=aoa,
            status="converged",
            iterations=2800,
            seed=k,
        )
    write_run(
        args.runs,
        started=now - timedelta(hours=1),
        aoa=19.57,
        status="completed",
        iterations=3000,
        precision="fp64",
        seed=10,
    )
    write_run(
        args.runs,
        started=now - timedelta(minutes=40),
        aoa=23.0,
        status="diverged",
        iterations=812,
        seed=11,
    )
    if args.live:
        write_run(
            args.runs,
            started=now,
            aoa=19.57,
            status="converged",
            iterations=2800,
            pace=args.live,
            seed=12,
        )


if __name__ == "__main__":
    main()
