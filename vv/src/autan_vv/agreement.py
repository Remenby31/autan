"""Agreement of fp32+acc64 runs with fp64 runs of the same cases (docs/validation/acceptance.md).

Order gates are verdicts in fp64; in fp32+acc64 a verification case is held to agreement with its
fp64 run, level by level, within K times the fp32 epsilon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autan
import numpy as np
from autan.results import field_names

from autan_vv.errors import VVError
from autan_vv.harness import read_geometry

if TYPE_CHECKING:
    from pathlib import Path

EPS32 = 2.0**-23
# Measured up to 182 (Ringleb n = 128, pressure) and 42 (MMS tet n = 32, velocity): 2.7 times above.
K = 500.0
SCHEMA = "autan.vv-agreement/1"


def _problem(results: autan.Results) -> dict[str, Any]:
    """The case apart from its precision and, for a steady case, its stopping targets, which may
    differ between precisions."""
    doc: dict[str, Any] = results.case.model_dump(mode="json")
    doc["physics"].pop("precision", None)
    if doc["time"]["kind"] == "steady":
        doc["time"].pop("residual_orders", None)
        doc["time"].pop("max_iterations", None)
    return doc


def agreement(fp64: list[Path], fp32: list[Path], k: float = K) -> dict[str, Any]:
    """Pair fp64 and fp32+acc64 runs by mesh and give, per pair and field, the volume-weighted L2 of
    their difference over the field's scale (the RMS speed for velocity components, the RMS of the
    fp64 field otherwise) in units of 2^-23; `pass` when every value is at most `k`. Runs of one
    pair solve the same case apart from its precision, with one solver build (V0727)."""
    double = {r.run.mesh.sha256: r for r in (autan.Results.open(p) for p in fp64)}
    single = {r.run.mesh.sha256: r for r in (autan.Results.open(p) for p in fp32)}
    if not double or set(double) != set(single):
        msg = "the fp64 and fp32+acc64 runs must cover the same meshes, one run each"
        raise VVError(msg, code="V0727")
    pairs = []
    for mesh, a in double.items():
        b = single[mesh]
        if (a.run.precision, b.run.precision) != ("fp64", "fp32+acc64"):
            msg = f"runs on mesh {mesh[:12]} are {a.run.precision} and {b.run.precision}"
            raise VVError(msg, code="V0727")
        if a.run.solver.commit != b.run.solver.commit:
            msg = f"runs on mesh {mesh[:12]} come from different solver builds"
            raise VVError(msg, code="V0727")
        if _problem(a) != _problem(b):
            msg = f"runs on mesh {mesh[:12]} solve different cases"
            raise VVError(msg, code="V0727")
        g = read_geometry(a).geometry
        w = g.cell_volume / g.cell_volume.sum()
        names = field_names(a.case)
        f64 = {n: np.asarray(a.field(n)) for n in names}
        f32 = {n: np.asarray(b.field(n)) for n in names}
        velocity = [n for n in names if n.startswith("Velocity")]
        speed = float(np.sqrt(np.sum(w * sum(f64[n] ** 2 for n in velocity)))) if velocity else 0.0
        values = {}
        for n in names:
            scale = speed if n in velocity else float(np.sqrt(np.sum(w * f64[n] ** 2)))
            diff = float(np.sqrt(np.sum(w * (f32[n] - f64[n]) ** 2)))
            values[n] = (
                diff / scale / EPS32 if scale > 0.0 else (0.0 if diff == 0.0 else float("inf"))
            )
        pairs.append({"mesh_sha256": mesh, "n_cells": int(g.cell_volume.size), "fields": values})
    pairs.sort(key=lambda p: p["n_cells"])
    worst = max(v for p in pairs for v in p["fields"].values())
    return {
        "schema": SCHEMA,
        "k": k,
        "eps32": EPS32,
        "pairs": pairs,
        "worst": worst,
        "pass": worst <= k,
    }
