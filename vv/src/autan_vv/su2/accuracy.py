"""Phase 2 accuracy of the SU2 comparison (#387): Autan and SU2 on the same NACA 0012 mesh files,
each run judged by the absolute criterion of #493 as its own baseline.

Autan and SU2 solve on different control volumes (cells, median-dual of the nodes), so neither run
is a baseline for the other: a run is converged once the outputs of its condition (OUTPUTS: those
neither zero by symmetry nor below 1e-5) have stayed within RTOL of their final value for WINDOW
iterations, and its residual drop from its first iteration is reported beside it. Forces are read at
the last iteration; continua come from the three finest levels when all three converged.
"""

from __future__ import annotations

import json
import math
import tomllib
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.cases import naca0012 as cases
from autan_vv.errors import VVError
from autan_vv.su2 import compare, history

if TYPE_CHECKING:
    from pathlib import Path

REPORT_SCHEMA = "autan.su2-accuracy/1"
UNIVERSAL_GAS_CONSTANT = 8.314462618
# Outputs each condition's runs must hold steady: CL and CM vanish by symmetry at alpha 0, and CD
# at M 0.5 (about 1e-6) is below the floor a relative tolerance can judge.
OUTPUTS = {"m080-a000": ("cd",), "m080-a125": ("cl", "cd", "cmy"), "m050-a125": ("cl", "cmy")}


def _own_baseline(
    runs: history.History, scales: dict[str, float], outputs: tuple[str, ...]
) -> dict[str, Any]:
    drop = {eq: float(np.log10(r[0] / r[-1])) for eq, r in runs.residuals.items()}
    scales = {eq: scales[eq] for eq in runs.residuals}
    try:
        eps = history.epsilon(runs, scales, outputs)
    except VVError:
        return {"converged": False, "iterations": None, "epsilon": None, "drop": drop}
    at = history.absolute_converged_at(runs, {eq: eps * s for eq, s in scales.items()}, outputs)
    return {
        "converged": at is not None,
        "iterations": None if at is None else int(runs.iterations[at] - runs.iterations[0]) + 1,
        "epsilon": eps,
        "drop": drop,
    }


def _autan_scales(case: dict[str, Any]) -> dict[str, float]:
    """Flux scales of an Autan NACA 0012 case from its freestream (case.toml): ideal gas of the
    case's molar mass and gamma, per unit span of the 1 m chord."""
    gas = case["physics"]["gas"]
    molar_mass = float(str(gas["molar_mass"]).split()[0])
    gamma = float(gas["gamma"])
    pressure = float(str(case["freestream"]["pressure"]).split()[0])
    temperature = float(str(case["freestream"]["temperature"]).split()[0])
    density = pressure / (UNIVERSAL_GAS_CONSTANT / molar_mass * temperature)
    speed = float(case["freestream"]["mach"]) * math.sqrt(gamma * pressure / density)
    energy = pressure / ((gamma - 1.0) * density) + 0.5 * speed**2
    return history.flux_scales(
        {"pressure": pressure, "density": density, "energy": energy, "speed": speed}, 1.0
    )


def autan_level(run: Path) -> dict[str, Any]:
    """One Autan run directory: its condition and level, mesh file SHA-256, solver identity,
    forces at the last iteration and its own-baseline convergence. V0904 if a file is missing."""
    try:
        info = json.loads((run / "run.json").read_text())
        case = tomllib.loads((run / "case.toml").read_text())
        cells = json.loads((run / "mesh-info.json").read_text())["counts"]["cells"]
    except (FileNotFoundError, KeyError) as err:
        msg = f"{run}: {err} is missing"
        raise VVError(msg, code="V0904") from None
    condition = case["name"].removeprefix("naca0012-")
    runs = history.read_autan(run / "history.jsonl")
    n = math.isqrt(cells)
    return {
        "run": info["run_id"], "condition": condition, "n": n, "cells": cells, "h": 1.0 / n,
        "mesh_sha256": info["mesh"]["sha256"], "commit": info["solver"]["commit"],
        "backend": info["solver"]["backend"], "precision": info["precision"],
        "status": info["status"], "last_iteration": int(runs.iterations[-1]),
        "cl": float(runs.cl[-1]), "cd": float(runs.cd[-1]), "cmy": float(runs.cmy[-1]),
        **_own_baseline(runs, _autan_scales(case), OUTPUTS[condition]),
    }  # fmt: skip


def su2_level(run: Path) -> dict[str, Any]:
    """One SU2 case directory (`autan-vv su2 prepare`, run by vv/su2/run.sh): its condition, level,
    source mesh SHA-256, build, forces at the last iteration and its own-baseline convergence.
    V0904 if a file is missing or SU2 failed."""
    level = json.loads((run / "level.json").read_text())
    prov: dict[str, Any] = history.provenance(run)["provenance"]  # type: ignore[assignment]
    if prov["exit_status"] != 0:
        msg = f"{run}: SU2 exited with {prov['exit_status']}"
        raise VVError(msg, code="V0904")
    runs = history.read_su2(run / "history.csv")
    scales = history.flux_scales(history.su2_freestream((run / "log.txt").read_text()), 1.0)
    return {
        "condition": level["condition"], "n": level["n"], "scheme": level["scheme"],
        "mesh_sha256": level["mesh"]["source_sha256"], "su2": prov["su2"],
        "image_id": prov["image_id"], "last_iteration": int(runs.iterations[-1]),
        "cl": float(runs.cl[-1]), "cd": float(runs.cd[-1]), "cmy": float(runs.cmy[-1]),
        **_own_baseline(runs, scales, OUTPUTS[level["condition"]]),
    }  # fmt: skip


def _continuum(levels: list[dict[str, Any]], label: str) -> dict[str, Any]:
    finest = sorted(levels, key=lambda lv: -lv["n"])[:3]
    if len(finest) < 3 or not all(lv["converged"] for lv in finest):
        return {"error": "not converged"}
    return {
        **cases.continuum([compare._forces(lv["n"], lv["cl"], lv["cd"], lv["cmy"], label)
                           for lv in finest]),
        "levels": sorted(lv["n"] for lv in finest),
    }  # fmt: skip


def study(autan_runs: list[Path], su2_cases: list[Path]) -> dict[str, Any]:
    """The report (autan.su2-accuracy/1) of one condition: per level both codes' forces and
    convergence, their difference, and wall Cp face by face (`compare.compare_cp`); per code the
    continuum from its three finest levels. V0905 unless the runs are one condition, one Autan
    build, one SU2 scheme and build, and each level the same mesh file on both sides."""
    autan = {lv["n"]: lv for lv in map(autan_level, autan_runs)}
    su2 = {lv["n"]: lv for lv in map(su2_level, su2_cases)}
    su2_dirs = {json.loads((d / "level.json").read_text())["n"]: d for d in su2_cases}
    autan_dirs = {math.isqrt(json.loads((d / "mesh-info.json").read_text())["counts"]["cells"]): d
                  for d in autan_runs}  # fmt: skip
    keys = (
        {(lv["condition"], lv["commit"], lv["backend"], lv["precision"]) for lv in autan.values()},
        {(lv["condition"], lv["scheme"], lv["su2"], lv["image_id"]) for lv in su2.values()},
        {lv["condition"] for lv in (*autan.values(), *su2.values())},
    )
    if any(len(k) != 1 for k in keys):
        msg = f"the runs mix conditions, builds or schemes: {keys}"
        raise VVError(msg, code="V0905")
    rows = []
    for n in sorted(set(autan) | set(su2)):
        a, s = autan.get(n), su2.get(n)
        row: dict[str, Any] = {"n": n, "autan": a, "su2": s}
        if a is not None and s is not None:
            if a["mesh_sha256"] != s["mesh_sha256"]:
                msg = f"level {n}: Autan and SU2 ran different mesh files"
                raise VVError(msg, code="V0905")
            row["difference"] = {k: a[k] - s[k] for k in ("cl", "cd", "cmy")}
            row["cp"] = compare.compare_cp(autan_dirs[n], su2_dirs[n] / "surface.csv")
        rows.append(row)
    return {
        "schema": REPORT_SCHEMA,
        "condition": next(iter(keys[2])),
        "outputs": list(OUTPUTS[next(iter(keys[2]))]),
        "levels": rows,
        "continuum": {
            "autan": _continuum(list(autan.values()), "autan"),
            "su2": _continuum(list(su2.values()), "su2"),
        },
    }
