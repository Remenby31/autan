"""Verify solver runs (`autan.results/1`) against an oracle: errors, observed order, gate verdict."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autan
import numpy as np
from autan import cgns
from autan.schema import SA, Compressible, ConstantViscosity, IdealGas, Incompressible, Inviscid, Laminar
from autan.units import si

from autan_vv.errors import VVError
from autan_vv.gates import GATES, Gate
from autan_vv.meshes.facemesh import FaceMesh, FloatArray, Zone
from autan_vv.oracles import Oracle, conserved
from autan_vv.oracles.registry import get_oracle
from autan_vv.order import fit_order

if TYPE_CHECKING:
    from pathlib import Path

VERDICT_SCHEMA = "autan.vv-verdict/1"
UNIVERSAL_GAS_CONSTANT = 8.314462618  # J/(mol K), CODATA 2018 exact
PRECISION_EPS = {"fp64": float(np.finfo(np.float64).eps), "fp32": float(np.finfo(np.float32).eps),
                 "fp32+acc64": float(np.finfo(np.float32).eps)}
FINISHED = ("converged", "completed")


def read_geometry(results: autan.Results) -> FaceMesh:
    """Rebuild the face-based mesh of a run from solution.cgns (NGON_n faces, NFACE_n cells).

    A positive face id in a cell's NFACE list means the face normal points out of that cell
    (results.md); that cell becomes the face's owner. Cell i is NFACE element F + 1 + i.
    """
    zone = cgns.read(results.path / "solution.cgns").child("Base").child("Zone")
    coords = zone.child("GridCoordinates")
    points = np.column_stack([np.asarray(coords.child(f"Coordinate{a}").data, dtype=np.float64) for a in "XYZ"])
    ngon, nface = zone.child("NGonElements"), zone.child("NFaceElements")
    face_offsets = np.asarray(ngon.child("ElementStartOffset").data, dtype=np.int64)
    face_nodes = np.asarray(ngon.child("ElementConnectivity").data, dtype=np.int64) - 1
    cell_offsets = np.asarray(nface.child("ElementStartOffset").data, dtype=np.int64)
    signed = np.asarray(nface.child("ElementConnectivity").data, dtype=np.int64)
    n_faces, n_cells = face_offsets.size - 1, cell_offsets.size - 1
    cell_of = np.repeat(np.arange(n_cells), np.diff(cell_offsets))
    face = np.abs(signed) - 1
    owner = np.full(n_faces, -1, dtype=np.int64)
    neighbour = np.full(n_faces, -1, dtype=np.int64)
    outward = signed > 0
    owner[face[outward]] = cell_of[outward]
    neighbour[face[~outward]] = cell_of[~outward]
    flip = owner < 0  # a face referenced only inward: reverse it so that its only cell owns it
    if np.any(owner < 0) and np.any(neighbour[flip] < 0):
        msg = "a face belongs to no cell"
        raise VVError(msg, code="V0721")
    owner[flip], neighbour[flip] = neighbour[flip], -1
    polys = [face_nodes[face_offsets[f] : face_offsets[f + 1]] for f in range(n_faces)]
    for f in np.flatnonzero(flip):
        polys[f] = polys[f][::-1]
    interior = np.flatnonzero(neighbour >= 0)
    boundary = np.flatnonzero(neighbour < 0)
    order = np.concatenate([interior, boundary])
    sizes = np.array([polys[f].size for f in order])
    mesh = FaceMesh(
        points=points,
        face_offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        face_nodes=np.concatenate([polys[f] for f in order]).astype(np.int64),
        owner=owner[order],
        neighbour=neighbour[interior],
        n_cells=n_cells,
        zones=(Zone("boundary", "boundary", interior.size, boundary.size),),
    )
    mesh.check()
    return mesh


def freestream_oracle(spec: autan.CaseSpec) -> Oracle:
    """The uniform freestream state of a compressible case (case-schema.md, Freestream; T defaults to 288.15 K).

    With `pressure`: rho = p/(R T). With `re`: rho = Re mu(T)/(V L) and p = rho R T. V = M sqrt(gamma R T),
    direction (cos a cos b, -sin b, sin a cos b).
    """
    physics = spec.physics
    if not isinstance(physics, Compressible) or not isinstance(physics.gas, IdealGas):
        msg = "the freestream oracle needs a compressible ideal-gas case"
        raise VVError(msg, code="V0722")
    gas = physics.gas
    fs = spec.freestream
    r = UNIVERSAL_GAS_CONSTANT / si(gas.molar_mass, "kg / mol")
    temperature = si(fs.temperature, "K") if fs.temperature is not None else 288.15
    speed = fs.mach * math.sqrt(gas.gamma * r * temperature)
    if fs.pressure is not None:
        pressure = si(fs.pressure, "Pa")
        density = pressure / (r * temperature)
    else:
        law = gas.viscosity
        if isinstance(law, ConstantViscosity):
            mu = si(law.mu, "Pa * s")
        else:
            t_ref, s = si(law.t_ref, "K"), si(law.s, "K")
            mu = si(law.mu_ref, "Pa * s") * (temperature / t_ref) ** 1.5 * (t_ref + s) / (temperature + s)
        density = fs.re * mu / (speed * si(fs.ref_length, "m"))
        pressure = density * r * temperature
    a, b = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
    velocity = speed * np.array([math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b)])

    def evaluate(points: FloatArray, _t: float | None) -> dict[str, FloatArray]:
        one = np.ones(points.shape[0])
        return {"Density": density * one, "VelocityX": velocity[0] * one, "VelocityY": velocity[1] * one,
                "VelocityZ": velocity[2] * one, "Pressure": pressure * one, "Temperature": temperature * one}

    turbulence = physics.turbulence
    kind = "euler" if isinstance(turbulence, Inviscid) else "navier-stokes" if isinstance(turbulence, Laminar) else "any"
    return Oracle("freestream", "Uniform freestream state of the case.", kind, 3, False, gas.gamma, r, evaluate)


def _check_compatible(results: autan.Results, oracle: Oracle) -> list[str]:
    """Reasons the run cannot be compared with the oracle (empty if it can)."""
    spec = results.case
    reasons = []
    if results.run.status not in FINISHED:
        reasons.append(f"status {results.run.status}, not converged or completed")
    physics = spec.physics
    expected = {"euler": Inviscid, "navier-stokes": Laminar, "rans-sa": SA}
    if oracle.physics in expected:
        if not isinstance(physics, Compressible) or not isinstance(physics.turbulence, expected[oracle.physics]):
            reasons.append(f"physics {physics.kind}/{physics.turbulence.model}, oracle needs {oracle.physics}")
        elif isinstance(physics.gas, IdealGas):
            r = UNIVERSAL_GAS_CONSTANT / si(physics.gas.molar_mass, "kg / mol")
            if not math.isclose(physics.gas.gamma, oracle.gamma, rel_tol=1e-12):
                reasons.append(f"gamma {physics.gas.gamma}, oracle uses {oracle.gamma}")
            if not math.isclose(r, oracle.gas_constant, rel_tol=1e-9):
                reasons.append(f"gas constant {r:.9g} J/(kg K), oracle uses {oracle.gas_constant:.9g}")
    if oracle.physics == "incompressible-navier-stokes" and not isinstance(physics, Incompressible):
        reasons.append("oracle needs an incompressible case")
    if oracle.unsteady and results.run.time is None:
        reasons.append("unsteady oracle but the run has no physical time")
    return reasons


@dataclass(frozen=True)
class RunErrors:
    """Errors of one run against the oracle, with the run's provenance."""

    provenance: dict[str, Any]
    n_cells: int
    volume: float
    h: float
    errors: dict[str, dict[str, float]]


def _h(mesh: FaceMesh, dim: int) -> tuple[float, float]:
    """Total volume and mesh size: (V/N)^(1/3) in 3D; for dim < 3 the mesh must be one cell thick in
    the other directions, and h = (V / (their extent) / N)^(1/dim)."""
    g = mesh.geometry
    volume = float(g.cell_volume.sum())
    if dim == 3:
        return volume, (volume / mesh.n_cells) ** (1.0 / 3.0)
    span = 1.0
    for axis in range(dim, 3):
        centres = g.cell_centroid[:, axis]
        extent = float(np.ptp(mesh.points[:, axis]))
        if np.ptp(centres) > 1e-9 * max(extent, 1.0):
            msg = f"a {dim}D oracle needs a mesh one cell thick along {'xyz'[axis]}"
            raise VVError(msg, code="V0723")
        span *= extent
    return volume, (volume / span / mesh.n_cells) ** (1.0 / dim)


def run_errors(results: autan.Results, oracle: Oracle) -> RunErrors:
    """L1, L2 (volume-weighted) and L-infinity errors of every field the oracle and run share."""
    reasons = _check_compatible(results, oracle)
    if reasons:
        msg = f"run {results.run.run_id} cannot be verified against {oracle.name}: {'; '.join(reasons)}"
        raise VVError(msg, code="V0724")
    mesh = read_geometry(results)
    g = mesh.geometry
    exact = oracle.fields(g.cell_centroid, results.run.time)
    run = {name: results.field(name) for name in exact if name in _field_names(results)}
    if math.isfinite(oracle.gamma) and all(n in run for n in ("Density", "VelocityX", "Pressure")):
        exact |= conserved(exact, oracle.gamma)
        run |= conserved(run, oracle.gamma)
        exact["Entropy"] = exact["Pressure"] / exact["Density"] ** oracle.gamma
        run["Entropy"] = run["Pressure"] / run["Density"] ** oracle.gamma
    weight = g.cell_volume / g.cell_volume.sum()
    errors = {}
    for name, value in run.items():
        diff = value - exact[name]
        errors[name] = {
            "L1": float(np.sum(weight * np.abs(diff))),
            "L2": float(np.sqrt(np.sum(weight * diff**2))),
            "Linf": float(np.max(np.abs(diff))),
            "scale": float(np.max(np.abs(exact[name]))),
        }
    volume, h = _h(mesh, oracle.dim)
    run_info = results.run
    provenance = {
        "path": str(results.path), "run_id": run_info.run_id, "identity": run_info.identity,
        "status": run_info.status, "backend": run_info.solver.backend, "solver_commit": run_info.solver.commit,
        "precision": run_info.precision, "case_sha256": run_info.case.sha256,
        "mesh_sha256": run_info.mesh.sha256, "time": run_info.time,
    }
    return RunErrors(provenance, mesh.n_cells, volume, h, errors)


def _field_names(results: autan.Results) -> set[str]:
    from autan.results import field_names  # noqa: PLC0415

    return set(field_names(results.case))


def _gate_checks(gate: Gate, runs: list[RunErrors], fits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for name in gate.fields:
        if gate.measure == "order":
            fit = fits.get(name, {}).get(gate.norm)
            if fit is None:
                checks.append({"field": name, "pass": False, "reason": "not measured"})
                continue
            fitted, finest = fit["order"], fit["pairwise_orders"][-1]
            ok = all(gate.lower <= v <= gate.upper for v in (fitted, finest))
            checks.append({"field": name, "fitted_order": fitted, "finest_pair_order": finest,
                           "lower": gate.lower, "upper": gate.upper, "pass": ok})
        else:
            worst = max(
                r.errors[name][gate.norm] / max(r.errors[name]["scale"], 1e-300)
                / PRECISION_EPS[r.provenance["precision"]]
                for r in runs
            ) if all(name in r.errors for r in runs) else math.inf
            checks.append({"field": name, "max_relative_error_in_eps": worst, "upper": gate.upper,
                           "pass": worst <= gate.upper})
    return checks


def verify(run_dirs: list[Path], against: str, gate_id: str | None = None) -> dict[str, Any]:
    """Compare runs with an oracle and, with `gate_id`, give a verdict (schema autan.vv-verdict/1).

    `against` is an oracle name (`autan-vv oracles`) or `freestream`, built from each run's case.
    Several runs form a refinement family: distinct meshes, same oracle and precision; the observed
    order of every error norm of every field is fitted over them. `evidence` is false when a run
    comes from the `fake` backend or the oracle is a numerical reference, whatever the gate says.
    """
    opened = [autan.Results.open(p) for p in run_dirs]
    if not opened:
        msg = "no run to verify"
        raise VVError(msg, code="V0725")
    runs = []
    oracle = None
    for results in opened:
        oracle = freestream_oracle(results.case) if against == "freestream" else get_oracle(against)
        runs.append(run_errors(results, oracle))
    assert oracle is not None
    precisions = {r.provenance["precision"] for r in runs}
    if len(precisions) > 1:
        msg = f"runs of one family must share a precision, got {sorted(precisions)}"
        raise VVError(msg, code="V0726")
    fits: dict[str, dict[str, Any]] = {}
    hs = [r.h for r in runs]
    if len(runs) >= 2:
        if len(set(hs)) != len(hs) or len({r.provenance["mesh_sha256"] for r in runs}) != len(runs):
            msg = "a refinement family needs distinct meshes"
            raise VVError(msg, code="V0727")
        for name in runs[0].errors:
            fits[name] = {}
            for norm in ("L1", "L2", "Linf"):
                values = [r.errors[name][norm] for r in runs]
                if min(values) > 0.0:
                    fit = fit_order(hs, values)
                    fits[name][norm] = {"order": fit.order, "pairwise_orders": list(fit.pairwise_orders),
                                        "r_squared": fit.r_squared}
    report: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "oracle": {"name": oracle.name, "physics": oracle.physics, "reference_only": oracle.reference_only,
                   "parameters": dict(oracle.parameters)},
        "runs": [{**r.provenance, "n_cells": r.n_cells, "volume": r.volume, "h": r.h, "errors": r.errors}
                 for r in sorted(runs, key=lambda r: -r.h)],
        "fits": fits,
        "verdict": None,
    }
    if gate_id is not None:
        gate = GATES.get(gate_id)
        if gate is None:
            msg = f"unknown gate {gate_id!r}; known: {', '.join(sorted(GATES))}"
            raise VVError(msg, code="V0728")
        if gate.oracle != oracle.name:
            msg = f"gate {gate_id} is checked against {gate.oracle}, not {oracle.name}"
            raise VVError(msg, code="V0728")
        if gate.measure == "order" and len(runs) < 3:
            msg = "an order gate needs at least three runs"
            raise VVError(msg, code="V0729")
        checks = _gate_checks(gate, runs, fits)
        reasons = []
        if any(r.provenance["backend"] == "fake" for r in runs):
            reasons.append("fake backend: synthetic results are never evidence")
        if oracle.reference_only:
            reasons.append("the oracle is a numerical reference, not an exact solution")
        report["verdict"] = {"gate": gate.id, "milestone": gate.milestone, "checks": checks,
                             "pass": all(c["pass"] for c in checks), "evidence": not reasons,
                             "not_evidence_because": reasons}
    return report
