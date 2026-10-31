"""``cfd.FakeBackend``: writes synthetic runs that conform to ``cfd.results/1``.

The geometry is a unit cube of hexahedra standing in for the real mesh; zone names,
conditions and media come from the resolved case. Fields hold the freestream state,
histories are smooth synthetic curves. Nothing here is a computation: results of
this backend must never be used as reference data.
"""

from __future__ import annotations

import json
import math
import time as clock
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import TypeAdapter

from cfd import cgns
from cfd.errors import CaseError, diagnostic
from cfd.results import (
    Averaging,
    CaseIdentity,
    Devices,
    MeshIdentity,
    Results,
    RunInfo,
    RunWriter,
    Solver,
    identity,
    iso,
    utc_now,
)
from cfd.schema import (
    DDES,
    SA,
    SST,
    CaseSpec,
    Compressible,
    ConstantViscosity,
    Fluid,
    IdealGas,
    Medium,
    SolidZone,
    Steady,
    Sutherland,
)
from cfd.units import si
from cfd.version import COMMIT, __version__

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from cfd.case import Case
    from cfd.meshinfo import MeshInfo
    from cfd.plan import Plan

MEDIUM: TypeAdapter[Fluid | SolidZone] = TypeAdapter(Medium)
GAS_CONSTANT = 8.314462618  # J/(mol K), CODATA 2018 exact value

BC_TYPES = {
    "slip": "BCWallInviscid",
    "symmetry": "BCSymmetryPlane",
    "farfield": "BCFarfield",
    "inlet": "BCInflowSubsonic",
    "velocity-inlet": "BCInflowSubsonic",
    "outlet": "BCOutflowSubsonic",
}
ZONE_CHILDREN = frozenset(
    {
        "ZoneType",
        "GridCoordinates",
        "NGonElements",
        "NFaceElements",
        "ZoneBC",
        "FlowSolution",
        "FlowSolutionMean",
        "WallSolution",
    }
)


def bc_type(condition: dict[str, Any]) -> str:
    if condition["kind"] == "wall":
        return "BCWallViscousIsothermal" if "temperature" in condition else "BCWallViscousHeatFlux"
    return BC_TYPES[condition["kind"]]


def node_names(boundary: list[str], cells: list[str]) -> tuple[list[str], list[str]]:
    """CGNS node names of the zones, by the rule of results.md#solutioncgns."""
    everything = boundary + cells

    def valid(name: str, reserved: frozenset[str]) -> bool:
        return (
            len(name) <= 32
            and "/" not in name
            and name not in (".", "..")
            and everything.count(name) == 1
            and name not in reserved
        )

    return (
        [n if valid(n, frozenset()) else f"boundary-{i:04d}" for i, n in enumerate(boundary)],
        [n if valid(n, ZONE_CHILDREN) else f"cell-zone-{i:04d}" for i, n in enumerate(cells)],
    )


@dataclass(frozen=True)
class Box:
    """A cube of n³ hexahedra written as polyhedra."""

    n: int

    def nodes(self) -> NDArray[np.float64]:
        axis = np.linspace(0.0, 1.0, self.n + 1)
        z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
        return np.stack([x.ravel(), y.ravel(), z.ravel()])

    def _node(self, i: int, j: int, k: int) -> int:
        return i + (self.n + 1) * (j + (self.n + 1) * k)

    def faces(self) -> tuple[list[list[int]], list[int], dict[tuple[str, int, int, int], int]]:
        """Faces (0-based node ids, normal along +axis), boundary face ids, face index."""
        n, at = self.n, self._node
        faces: list[list[int]] = []
        boundary: list[int] = []
        index: dict[tuple[str, int, int, int], int] = {}
        for axis in "xyz":
            for k in range(n + (axis == "z")):
                for j in range(n + (axis == "y")):
                    for i in range(n + (axis == "x")):
                        if axis == "x":
                            quad = [
                                at(i, j, k),
                                at(i, j + 1, k),
                                at(i, j + 1, k + 1),
                                at(i, j, k + 1),
                            ]
                            edge = i in (0, n)
                        elif axis == "y":
                            quad = [
                                at(i, j, k),
                                at(i, j, k + 1),
                                at(i + 1, j, k + 1),
                                at(i + 1, j, k),
                            ]
                            edge = j in (0, n)
                        else:
                            quad = [
                                at(i, j, k),
                                at(i + 1, j, k),
                                at(i + 1, j + 1, k),
                                at(i, j + 1, k),
                            ]
                            edge = k in (0, n)
                        index[axis, i, j, k] = len(faces)
                        if edge:
                            boundary.append(len(faces))
                        faces.append(quad)
        return faces, boundary, index

    def cells(self, index: dict[tuple[str, int, int, int], int]) -> list[list[int]]:
        """Signed 1-based face ids per cell; positive when the normal points out of the cell."""
        n = self.n
        cells = []
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    cells.append(
                        [
                            -(index["x", i, j, k] + 1),
                            index["x", i + 1, j, k] + 1,
                            -(index["y", i, j, k] + 1),
                            index["y", i, j + 1, k] + 1,
                            -(index["z", i, j, k] + 1),
                            index["z", i, j, k + 1] + 1,
                        ]
                    )
        return cells


def viscosity(gas: IdealGas, temperature: float) -> float:
    law = gas.viscosity
    if isinstance(law, ConstantViscosity):
        return si(law.mu, "Pa * s")
    assert isinstance(law, Sutherland)
    t_ref, s = si(law.t_ref, "K"), si(law.s, "K")
    # Sutherland (1893): mu = mu_ref (T / T_ref)^(3/2) (T_ref + S) / (T + S)
    return si(law.mu_ref, "Pa * s") * (temperature / t_ref) ** 1.5 * (t_ref + s) / (temperature + s)


def freestream_state(spec: CaseSpec) -> dict[str, Any]:
    """Density, pressure, temperature, speed, viscosity of the freestream (case-schema.md)."""
    fs, physics = spec.freestream, spec.physics
    alpha, beta = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
    direction = np.array(
        [math.cos(alpha) * math.cos(beta), -math.sin(beta), math.sin(alpha) * math.cos(beta)]
    )
    if isinstance(physics, Compressible):
        gas = physics.gas
        assert fs.temperature is not None and fs.mach is not None
        temperature = si(fs.temperature, "K")
        r = GAS_CONSTANT / si(gas.molar_mass, "kg / mol")
        speed = fs.mach * math.sqrt(gas.gamma * r * temperature)
        mu = viscosity(gas, temperature)
        if fs.re is not None:
            assert fs.ref_length is not None
            density = fs.re * mu / (speed * si(fs.ref_length, "m"))
            pressure = density * r * temperature
        else:
            assert fs.pressure is not None
            pressure = si(fs.pressure, "Pa")
            density = pressure / (r * temperature)
        return {
            "density": density,
            "pressure": pressure,
            "temperature": temperature,
            "speed": speed,
            "viscosity": mu,
            "direction": direction,
        }
    assert fs.velocity is not None
    return {
        "density": si(physics.fluid.density, "kg / m**3"),
        "pressure": 0.0,
        "temperature": None,
        "speed": si(fs.velocity, "m / s"),
        "viscosity": si(physics.fluid.viscosity, "Pa * s"),
        "direction": direction,
    }


def equations(spec: CaseSpec) -> list[str]:
    names = (
        ["density", "momentum-x", "momentum-y", "momentum-z", "energy"]
        if isinstance(spec.physics, Compressible)
        else ["pressure", "momentum-x", "momentum-y", "momentum-z"]
    )
    turbulence = spec.physics.turbulence
    base = turbulence.base if isinstance(turbulence, DDES) else turbulence.model
    return names + {"sa": ["nu-tilde"], "sst": ["k", "omega"]}.get(base, [])


@dataclass(frozen=True)
class FakeBackend:
    """Writes a synthetic run directory that conforms to ``cfd.results/1``.

    ``iterations`` caps the iterations (steady) or steps (unsteady) written; ``pace``
    sleeps that many seconds per iteration to imitate a live run; ``diverge_at`` makes
    the run diverge at that iteration with ``D0107``.
    """

    iterations: int = 200
    pace: float = 0.0
    diverge_at: int | None = None
    cells_per_side: int = 8

    def run(self, case: Case, mesh_info: MeshInfo, runs: Path) -> Results:
        result = case.plan(mesh_info)
        if not result.ok:
            raise CaseError(d for d in result.diagnostics if d.severity == "error")
        spec = case.spec
        started = utc_now()
        solver = Solver(version=__version__, commit=COMMIT, backend="fake")
        devices = Devices(kind="cpu", count=1)
        info = RunInfo(
            run_id=RunWriter.new_run_id(started),
            identity=identity(
                spec.hash(), mesh_info.source.sha256, solver, spec.physics.precision, devices
            ),
            status="running",
            case=CaseIdentity(name=spec.name, sha256=spec.hash()),
            mesh=MeshIdentity(path=str(case.mesh_path), sha256=mesh_info.source.sha256),
            solver=solver,
            precision=spec.physics.precision,
            deterministic=True,
            devices=devices,
            started=iso(started),
            finished=None,
            iterations=0,
            time=None,
            averaging=None,
            diagnostics=(),
        )
        writer = RunWriter(runs, info, spec, mesh_info)
        steady = isinstance(spec.time, Steady)
        budget = spec.time.max_iterations if isinstance(spec.time, Steady) else spec.time.steps
        count = min(self.iterations, budget)
        last = count if self.diverge_at is None else min(count, self.diverge_at - 1)
        state = freestream_state(spec)
        history = _History(spec, state, count)
        for iteration in range(1, last + 1):
            writer.append(history.record(iteration))
            if self.pace:
                clock.sleep(self.pace)

        box = Box(
            max(
                self.cells_per_side,
                len(mesh_info.cell_zones),
                math.ceil(math.sqrt(len(mesh_info.boundary_zones) / 6)),
            )
        )
        layout = _write_solution(
            writer.path / "solution.cgns", spec, result, state, box, writer.info
        )
        changes: dict[str, Any] = {
            "iterations": last,
            "time": None if steady else history.time(last),
        }
        if self.diverge_at is not None and self.diverge_at <= count:
            cell = box.n**3 // 2
            changes["status"] = "diverged"
            changes["diagnostics"] = (
                diagnostic(
                    "D0107",
                    f"divergence at iteration {self.diverge_at} near cell {cell} "
                    f"(zone {layout[cell]!r})",
                    iteration=self.diverge_at,
                    cell=cell,
                    zone=layout[cell],
                ),
            )
        elif steady:
            changes["status"] = "converged"
        else:
            changes["status"] = "completed" if count == budget else "stopped"
            if spec.time.average_from is not None and count == budget:
                start = spec.time.average_from
                changes["averaging"] = Averaging(
                    from_step=start + 1,
                    to_step=count,
                    from_time=history.time(start),
                    to_time=history.time(count),
                )
        writer.finish(**changes)
        return Results.open(writer.path)


class _History:
    """Synthetic residuals decaying by ``residual_orders`` and forces approaching a limit."""

    def __init__(self, spec: CaseSpec, state: dict[str, Any], count: int) -> None:
        self.spec, self.state, self.count = spec, state, max(count, 2)
        self.equations = equations(spec)
        orders = spec.time.residual_orders if isinstance(spec.time, Steady) else 3.0
        self.orders = orders

    def time(self, step: int) -> float:
        dt = self.spec.time.dt if not isinstance(self.spec.time, Steady) else None
        return step * (si(dt, "s") if dt is not None else 1e-3)

    def record(self, iteration: int) -> dict[str, Any]:
        fraction = (iteration - 1) / (self.count - 1)
        residuals = {
            name: 10.0 ** (-self.orders * fraction) * (1 + index)
            for index, name in enumerate(self.equations)
        }
        monitors = {}
        for monitor in self.spec.monitors:
            if iteration % monitor.every == 0:
                monitors[monitor.name] = self._forces(monitor, iteration)
        steady = isinstance(self.spec.time, Steady)
        return {
            "iteration": iteration,
            "time": None if steady else self.time(iteration),
            "residuals": residuals,
            "monitors": monitors,
        }

    def _forces(self, monitor: Any, iteration: int) -> dict[str, float]:
        fs, state = self.spec.freestream, self.state
        alpha, beta = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
        q = 0.5 * state["density"] * state["speed"] ** 2
        area = si(monitor.ref_area, "m**2") if monitor.ref_area is not None else 1.0
        approach = 1.0 - math.exp(-iteration / (0.2 * self.count))
        cl = 2 * math.pi * alpha * approach
        cd = 0.02 + cl**2 / (math.pi * 10.0)
        e_d = np.array(
            [math.cos(alpha) * math.cos(beta), -math.sin(beta), math.sin(alpha) * math.cos(beta)]
        )
        e_l = np.array([-math.sin(alpha), 0.0, math.cos(alpha)])
        e_s = np.cross(e_l, e_d)
        force = q * area * (cd * e_d + cl * e_l)
        length = si(fs.ref_length, "m") if fs.ref_length is not None else 1.0
        center = (
            np.array([si(c, "m") for c in monitor.moment_center])
            if monitor.moment_center is not None
            else np.zeros(3)
        )
        moment = np.cross(np.array([0.25 * length, 0.0, 0.0]) - center, force)
        values = {key: float(v) for key, v in zip(("fx", "fy", "fz"), force, strict=True)}
        values.update({key: float(v) for key, v in zip(("mx", "my", "mz"), moment, strict=True)})
        if monitor.ref_area is not None:
            values.update(
                cd=float(force @ e_d / (q * area)),
                cs=float(force @ e_s / (q * area)),
                cl=float(force @ e_l / (q * area)),
            )
            if fs.ref_length is not None:
                values.update(
                    {
                        f"c{key}": float(v / (q * area * length))
                        for key, v in zip(("mx", "my", "mz"), moment, strict=True)
                    }
                )
        return values


def _write_solution(
    path: Path, spec: CaseSpec, plan: Plan, state: dict[str, Any], box: Box, info: RunInfo
) -> list[str]:
    """Write solution.cgns; returns the cell zone name of every cell."""
    faces, boundary_faces, index = box.faces()
    cells = box.cells(index)
    boundary_zones = [b for b in plan.boundaries if b.faces > 0]
    boundary_names = [b.zone for b in plan.boundaries]
    cell_names = [z.zone for z in plan.cell_zones]
    boundary_nodes, cell_nodes = node_names(boundary_names, cell_names)
    chunks = np.array_split(np.array(boundary_faces), len(boundary_zones))
    owned = dict(zip((b.zone for b in boundary_zones), chunks, strict=True))
    layout = [cell_names[(c % box.n) * len(cell_names) // box.n] for c in range(len(cells))]
    n_faces = len(faces)

    with cgns.create(path) as handle:
        base = cgns.node(handle, "Base", "CGNSBase_t", np.array([3, 3], dtype=np.int32))
        cgns.node(base, "DataClass", "DataClass_t", "Dimensional")
        cgns.node(
            base,
            "DimensionalUnits",
            "DimensionalUnits_t",
            np.frombuffer(
                "".join(
                    u.ljust(32) for u in ("Kilogram", "Meter", "Second", "Kelvin", "Radian")
                ).encode(),
                dtype=np.int8,
            ).reshape(5, 32),
        )
        metadata = {
            k: v
            for k, v in info.document().items()
            if k not in ("status", "finished", "iterations", "diagnostics")
        }
        cgns.node(base, "RunMetadata", "Descriptor_t", json.dumps(metadata, sort_keys=True))
        cgns.node(
            base,
            "ZoneNames",
            "Descriptor_t",
            json.dumps(
                {
                    "boundary": dict(zip(boundary_nodes, boundary_names, strict=True)),
                    "cell": dict(zip(cell_nodes, cell_names, strict=True)),
                },
                sort_keys=True,
            ),
        )
        conditions = {b.zone: b.condition for b in plan.boundaries}
        for zone, name in zip(boundary_names, boundary_nodes, strict=True):
            family = cgns.node(base, name, "Family_t")
            condition = conditions[zone]
            assert condition is not None or zone not in owned
            if condition is not None:
                cgns.node(family, "FamilyBC", "FamilyBC_t", bc_type(condition))
        for name in cell_nodes:
            cgns.node(base, name, "Family_t")

        zone = cgns.node(
            base,
            "Zone",
            "Zone_t",
            np.array([[box.nodes().shape[1], len(cells), 0]], dtype=np.int64),
        )
        cgns.node(zone, "ZoneType", "ZoneType_t", "Unstructured")
        coordinates = cgns.node(zone, "GridCoordinates", "GridCoordinates_t")
        for axis, values in zip("XYZ", box.nodes(), strict=True):
            cgns.data_array(coordinates, f"Coordinate{axis}", values)
        _elements(zone, "NGonElements", 22, 1, [[v + 1 for v in f] for f in faces])
        _elements(zone, "NFaceElements", 23, n_faces + 1, cells)

        zone_bc = cgns.node(zone, "ZoneBC", "ZoneBC_t")
        for zone_name, name in zip(boundary_names, boundary_nodes, strict=True):
            if zone_name not in owned:
                continue
            condition = conditions[zone_name]
            assert condition is not None
            bc = cgns.node(zone_bc, name, "BC_t", bc_type(condition))
            _point_list(bc, owned[zone_name] + 1)
            cgns.node(bc, "GridLocation", "GridLocation_t", "FaceCenter")
            cgns.node(bc, "FamilyName", "FamilyName_t", name)

        for zone_name, name in zip(cell_names, cell_nodes, strict=True):
            members = [c for c, owner in enumerate(layout) if owner == zone_name]
            region = cgns.node(zone, name, "ZoneSubRegion_t", np.array([3], dtype=np.int32))
            _point_list(region, np.array(members) + n_faces + 1)
            cgns.node(region, "GridLocation", "GridLocation_t", "CellCenter")
            cgns.node(region, "FamilyName", "FamilyName_t", name)

        fields = _fields(spec, plan, state, layout)
        for solution_name in ("FlowSolution",) + (
            ("FlowSolutionMean",) if getattr(spec.time, "average_from", None) is not None else ()
        ):
            solution = cgns.node(zone, solution_name, "FlowSolution_t")
            cgns.node(solution, "GridLocation", "GridLocation_t", "CellCenter")
            for name, values in fields.items():
                cgns.data_array(solution, name, values)

        walls = np.sort(
            np.concatenate(
                [
                    owned[b.zone]
                    for b in boundary_zones
                    if b.condition and b.condition["kind"] in ("wall", "slip")
                ]
                or [np.array([], dtype=np.int64)]
            )
        )
        if walls.size:
            region = cgns.node(zone, "WallSolution", "ZoneSubRegion_t", np.array([2], np.int32))
            _point_list(region, walls + 1)
            cgns.node(region, "GridLocation", "GridLocation_t", "FaceCenter")
            cgns.data_array(region, "Pressure", np.full(walls.size, state["pressure"]))
            for axis in "XYZ":
                cgns.data_array(region, f"WallShearStress{axis}", np.zeros(walls.size))
    return layout


def _point_list(parent: Any, ids: NDArray[Any]) -> None:
    cgns.node(parent, "PointList", "IndexArray_t", np.asarray(ids, dtype=np.int64).reshape(1, -1))


def _elements(zone: Any, name: str, kind: int, first: int, items: list[list[int]]) -> None:
    elements = cgns.node(zone, name, "Elements_t", np.array([kind, 0], dtype=np.int32))
    cgns.node(
        elements,
        "ElementRange",
        "IndexRange_t",
        np.array([first, first + len(items) - 1], dtype=np.int64),
    )
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in items])]).astype(np.int64)
    cgns.node(elements, "ElementStartOffset", "DataArray_t", offsets)
    cgns.node(
        elements,
        "ElementConnectivity",
        "DataArray_t",
        np.array([v for i in items for v in i], dtype=np.int64),
    )


def _fields(
    spec: CaseSpec, plan: Plan, state: dict[str, Any], layout: list[str]
) -> dict[str, NDArray[np.float64]]:
    media = {z.zone: MEDIUM.validate_python(z.medium) for z in plan.cell_zones}
    count = len(layout)
    density, velocity = np.empty(count), np.zeros((3, count))
    pressure, temperature = np.empty(count), np.empty(count)
    viscosity_ratio = np.empty(count)
    for cell, zone in enumerate(layout):
        medium = media[zone]
        if isinstance(medium, SolidZone):
            density[cell] = si(medium.material.density, "kg / m**3")
            pressure[cell] = 0.0
        elif isinstance(medium.material, IdealGas):
            r = GAS_CONSTANT / si(medium.material.molar_mass, "kg / mol")
            density[cell] = state["pressure"] / (r * state["temperature"])
            pressure[cell] = state["pressure"]
            velocity[:, cell] = state["speed"] * state["direction"]
        else:
            density[cell] = state["density"]
            pressure[cell] = state["pressure"]
            velocity[:, cell] = state["speed"] * state["direction"]
        temperature[cell] = state["temperature"] or 0.0
        viscosity_ratio[cell] = state["viscosity"] / density[cell]
    fields: dict[str, NDArray[np.float64]] = {
        "Density": density,
        "VelocityX": velocity[0],
        "VelocityY": velocity[1],
        "VelocityZ": velocity[2],
        "Pressure": pressure,
    }
    if isinstance(spec.physics, Compressible):
        fields["Temperature"] = temperature
    turbulence = spec.physics.turbulence
    fluid = np.array([isinstance(media[z], Fluid) for z in layout])
    if isinstance(turbulence, SA) or (isinstance(turbulence, DDES) and turbulence.base == "sa"):
        fields["TurbulentSANuTilde"] = np.where(
            fluid, turbulence.nu_tilde_ratio * viscosity_ratio, 0.0
        )
    if isinstance(turbulence, SST) or (isinstance(turbulence, DDES) and turbulence.base == "sst"):
        k = 1.5 * (turbulence.intensity * state["speed"]) ** 2
        fields["TurbulentEnergyKinetic"] = np.where(fluid, k, 0.0)
        fields["TurbulentDissipationRate"] = np.where(
            fluid, k / (turbulence.viscosity_ratio * viscosity_ratio), 0.0
        )
    return fields
