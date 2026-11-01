"""ONERA M6 inviscid case (run 2308 of Schmitt and Charpin) and its M2 comparators (acceptance.md).

`case` builds the autan.case/1 case on a level of the `onera-m6` mesh family
(`autan_vv.meshes.onera_m6`). `study` reads a run, checks its residual drop (the gate: 10 orders),
cuts the wing at the seven span stations of the experiment, and reports Cp along each section, the
upper-surface shock positions next to the experiment's, and the Cp RMS difference away from shocks.

Shock position: the rearmost point of the upper surface where Cp rises through the critical value
Cp* (sonic flow), by linear interpolation between samples; the same rule is applied to the
experiment's pressure taps and to the computed section, so their positions compare like for like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import autan
import numpy as np
from autan.units import deg, m, si

from autan_vv.cases import get_case, read_tecplot
from autan_vv.errors import VVError
from autan_vv.harness import read_geometry
from autan_vv.meshes import onera_m6 as geometry

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FloatArray

STUDY_SCHEMA = "autan.vv-onera-m6/1"
GAMMA = 1.4
# acceptance.md, M2: residual drop and the shock tolerance at eta = 0.20 to 0.65.
RESIDUAL_ORDERS = 10.0
SHOCK_TOLERANCE = 0.03
SHOCK_STATIONS = (0.20, 0.44, 0.65)
# Taps closer than this to a shock (x/c) are left out of the Cp RMS.
SHOCK_EXCLUSION = 0.05


def conditions() -> dict[str, Any]:
    """Run 2308 as the case catalogue records it (Mach, alpha, section stations)."""
    c = get_case("onera-m6").conditions
    return {
        "mach": c["run_2308"]["mach"],
        "alpha_deg": c["run_2308"]["alpha_deg"],
        "eta": c["section_eta"],
    }


def case(
    mesh: str,
    *,
    precision: Literal["fp64", "fp32+acc64"] = "fp64",
    max_iterations: int = 200000,
    residual_orders: float = RESIDUAL_ORDERS,
) -> autan.Case:
    """The Euler case of run 2308 (M 0.8395, alpha 3.06 deg) on the ONERA M6 mesh `mesh`.

    Slip wing, symmetry plane, characteristic far field; forces on the wing with the reference
    area 0.7532 m2 and mean aerodynamic chord 0.64607 m of Schmitt and Charpin.
    """
    c = conditions()
    result = autan.Case("onera-m6-2308", mesh=mesh)
    result.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision)
    result.freestream(
        mach=c["mach"],
        aoa=c["alpha_deg"] * deg,
        pressure=101325.0 * autan.units.ureg.Pa,
        ref_length=geometry.MEAN_AERODYNAMIC_CHORD * m,
    )
    result.time = autan.Steady(max_iterations=max_iterations, residual_orders=residual_orders)
    result.boundary("wing").slip()
    result.boundary("symmetry").symmetry()
    result.boundary("farfield").farfield()
    result.monitor(autan.forces("wing", ref_area=geometry.REFERENCE_AREA * m * m))
    return result


def critical_cp(mach: float, gamma: float = GAMMA) -> float:
    """Pressure coefficient where the isentropic flow from the freestream turns sonic."""
    ratio = ((2.0 + (gamma - 1.0) * mach**2) / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    return float(2.0 / (gamma * mach**2) * (ratio - 1.0))


def shock_positions(xc: FloatArray, cp: FloatArray, cp_star: float) -> list[float]:
    """x/c of every rise of Cp through cp_star along increasing x/c (supersonic to subsonic)."""
    order = np.argsort(xc, kind="stable")
    x, p = xc[order], cp[order]
    rise = np.flatnonzero((p[:-1] < cp_star) & (p[1:] >= cp_star) & (x[1:] > x[:-1]))
    t = (cp_star - p[rise]) / (p[rise + 1] - p[rise])
    return [float(v) for v in x[rise] + t * (x[rise + 1] - x[rise])]


@dataclass(frozen=True)
class Section:
    """Cp along one span station, upper (z >= 0) and lower surface, sorted by x/c."""

    eta: float
    upper: FloatArray  # (n, 2): x/c, Cp
    lower: FloatArray


def experiment(path: Path) -> dict[float, Section]:
    """The seven sections of `case_2308.dat` (TMR), keyed by eta; upper taps have Z/L > 0."""
    tables = read_tecplot(path)
    eta = conditions()["eta"]
    if len(tables) != len(eta):
        msg = f"{path.name}: {len(tables)} sections, expected {len(eta)}"
        raise VVError(msg, code="V0603")
    out = {}
    for table, station in zip(tables, eta, strict=True):
        cols = {name.lower(): table.data[:, i] for i, name in enumerate(table.variables)}
        xc, zc, cp = cols["x/l"], cols["z/l"], cols["cp"]
        out[station] = _sections(station, xc, zc, cp)
    return out


def _sections(eta: float, xc: FloatArray, zc: FloatArray, cp: FloatArray) -> Section:
    up, lo = zc >= 0.0, zc < 0.0
    upper = np.column_stack([xc[up], cp[up]])
    lower = np.column_stack([xc[lo], cp[lo]])
    return Section(eta, upper[np.argsort(upper[:, 0])], lower[np.argsort(lower[:, 0])])


def node_values(
    points: FloatArray, faces: list[np.ndarray], values: FloatArray
) -> tuple[np.ndarray, FloatArray]:
    """Nodes of `faces` and the area-weighted mean of the values of the faces around each."""
    total = np.zeros(points.shape[0])
    weight = np.zeros(points.shape[0])
    for face, value in zip(faces, values, strict=True):
        p = points[face]
        area = 0.5 * np.linalg.norm(np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0))
        total[face] += area * value
        weight[face] += area
    nodes = np.flatnonzero(weight > 0.0)
    return nodes, total[nodes] / weight[nodes]


def cut(
    points: FloatArray, faces: list[np.ndarray], node_value: FloatArray, y: float
) -> FloatArray:
    """Points (x, z, value) where the edges of `faces` cross the plane at span y.

    `node_value` holds one value per point. Values vary linearly along each edge.
    """
    edges = {
        (min(a, b), max(a, b))
        for face in faces
        for a, b in zip(face.tolist(), np.roll(face, -1).tolist(), strict=True)
    }
    pairs = np.array(sorted(edges), dtype=np.int64)
    ya, yb = points[pairs[:, 0], 1], points[pairs[:, 1], 1]
    crossing = (np.minimum(ya, yb) <= y) & (np.maximum(ya, yb) > y)
    a, b = pairs[crossing, 0], pairs[crossing, 1]
    t = (y - points[a, 1]) / (points[b, 1] - points[a, 1])
    xz = points[a][:, [0, 2]] + t[:, None] * (points[b][:, [0, 2]] - points[a][:, [0, 2]])
    v = node_value[a] + t * (node_value[b] - node_value[a])
    return np.column_stack([xz, v])


def computed_sections(
    points: FloatArray, faces: list[np.ndarray], cp: FloatArray, eta: list[float]
) -> dict[float, Section]:
    """Cp of the wing faces `faces` cut at each station, in x/c of the original local chord."""
    nodes, values = node_values(points, faces, cp)
    per_point = np.full(points.shape[0], np.nan)
    per_point[nodes] = values
    out = {}
    for station in eta:
        y = station * geometry.SEMISPAN
        xzv = cut(points, faces, per_point, y)
        xc = (xzv[:, 0] - geometry.leading_edge(y)) / geometry.chord(y)
        zc = xzv[:, 1] / geometry.chord(y)
        out[station] = _sections(station, xc, zc, xzv[:, 2])
    return out


def compare(
    computed: dict[float, Section], measured: dict[float, Section], mach: float
) -> list[dict[str, Any]]:
    """Per station: upper-surface shocks of both, the rear-shock gap and the Cp RMS off shocks."""
    cp_star = critical_cp(mach)
    rows = []
    for eta, exp in measured.items():
        cfd = computed[eta]
        s_exp = shock_positions(exp.upper[:, 0], exp.upper[:, 1], cp_star)
        s_cfd = shock_positions(cfd.upper[:, 0], cfd.upper[:, 1], cp_star)
        gap = abs(s_cfd[-1] - s_exp[-1]) if s_exp and s_cfd else None
        residuals = []
        for surf_exp, surf_cfd, shocks in (
            (exp.upper, cfd.upper, s_exp + s_cfd),
            (exp.lower, cfd.lower, []),
        ):
            x = surf_exp[:, 0]
            keep = (
                np.all(np.abs(x[:, None] - np.array(shocks)[None, :]) > SHOCK_EXCLUSION, axis=1)
                if shocks
                else np.ones(x.size, bool)
            )
            keep &= (x >= surf_cfd[0, 0]) & (x <= surf_cfd[-1, 0])
            residuals.append(np.interp(x[keep], surf_cfd[:, 0], surf_cfd[:, 1]) - surf_exp[keep, 1])
        diff = np.concatenate(residuals)
        row: dict[str, Any] = {
            "eta": eta,
            "shocks_experiment": s_exp,
            "shocks_computed": s_cfd,
            "rear_shock_gap": gap,
            "cp_rms_off_shocks": float(np.sqrt(np.mean(diff**2))) if diff.size else None,
        }
        if any(math.isclose(eta, s) for s in SHOCK_STATIONS):
            row["shock_within_tolerance"] = gap is not None and gap <= SHOCK_TOLERANCE
        rows.append(row)
    return rows


def residual_drop(results: autan.Results) -> dict[str, float]:
    """Orders of magnitude each residual fell from iteration 1 to the last recorded iteration."""
    history = results.history
    out = {}
    for name in history.equations:
        r = history.residual(name)
        out[name] = float(math.log10(r[0] / r[-1])) if r[-1] > 0.0 else math.inf
    return out


def study(run: Path, experiment_file: Path) -> dict[str, Any]:
    """Residual-drop gate and the section report of one ONERA M6 run against `case_2308.dat`."""
    results = autan.Results.open(run)
    spec = results.case
    c = conditions()
    if spec.freestream.mach is None or not math.isclose(spec.freestream.mach, c["mach"]):
        msg = f"run {run} is not at the Mach number of run 2308 ({c['mach']})"
        raise VVError(msg, code="V0801")
    if spec.freestream.pressure is None:
        msg = f"run {run} gives no freestream pressure; Cp needs it"
        raise VVError(msg, code="V0801")
    p_inf = float(si(spec.freestream.pressure, "Pa"))
    mesh = read_geometry(results)
    faces, pressure = results.wall("Pressure")
    cp = (pressure - p_inf) / (0.5 * GAMMA * p_inf * c["mach"] ** 2)
    polygons = [mesh.face(int(f)) for f in faces]
    computed = computed_sections(mesh.points, polygons, cp, c["eta"])
    measured = experiment(experiment_file)
    drop = residual_drop(results)
    rows = compare(computed, measured, c["mach"])
    return {
        "schema": STUDY_SCHEMA,
        "run": results.run.run_id,
        "status": results.run.status,
        "residual_drop": drop,
        "gate": {
            "id": "onera-m6-residual-drop",
            "pass": results.run.status == "converged" and min(drop.values()) >= RESIDUAL_ORDERS,
        },
        "sections": rows,
        "sections_cp": {
            str(eta): {"upper": sec.upper.tolist(), "lower": sec.lower.tolist()}
            for eta, sec in computed.items()
        },
        # acceptance.md: the ten-order drop is measured in fp64.
        "evidence": results.run.solver.backend != "fake" and results.run.precision == "fp64",
    }
