"""``cfd plan``: resolve a case against its mesh before any GPU is reserved.

Resolution rules: docs/contracts/case-schema.md#boundary-resolution.
Memory model: docs/sdk/plan.md.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from cfd.errors import Diagnostic, diagnostic
from cfd.meshinfo import MeshInfo
from cfd.schema import (
    DDES,
    SA,
    SST,
    CaseSpec,
    Compressible,
    ConstantDensityFluid,
    Farfield,
    Fluid,
    IdealGas,
    Incompressible,
    Inlet,
    Inviscid,
    Slip,
    SolidZone,
    Symmetry,
    VelocityInlet,
    Wall,
)

if TYPE_CHECKING:
    from cfd.case import Case

Provenance = Literal["code", "mesh", "default"]
NON_ORTHOGONALITY_WARNING_DEG = 70.0

_COMPATIBLE_MESH_TYPE = {
    "wall": {"wall", "slip"},
    "slip": {"slip", "wall"},
    "symmetry": {"symmetry"},
    "farfield": {"farfield"},
    "inlet": {"inlet", "velocity-inlet"},
    "outlet": {"outlet"},
}
_MESH_METHOD = {
    "inlet": {"compressible": "inlet(total_pressure=..., total_temperature=...)",
              "incompressible": "velocity_inlet(velocity=...)"},
    "outlet": {"compressible": "outlet(static_pressure=...)",
               "incompressible": "outlet(static_pressure=...)"},
}


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoundaryResolution(_Report):
    zone: str
    faces: int
    mesh_type: str | None
    condition: dict[str, Any] | None
    provenance: Provenance | None
    rule: int | None


class CellZoneResolution(_Report):
    zone: str
    cells: int
    mesh_type: str | None
    medium: dict[str, Any] | None
    provenance: Provenance | None
    rule: int | None


class Memory(_Report):
    model: Literal["cfd.memory/1"] = "cfd.memory/1"
    bytes_total: int
    bytes_per_gpu: int
    gpus: int
    equations: int


class Plan(_Report):
    """Resolution of a case against its mesh: zones with provenance, memory, diagnostics."""

    case: dict[str, str]
    mesh: dict[str, Any]
    boundaries: tuple[BoundaryResolution, ...]
    cell_zones: tuple[CellZoneResolution, ...]
    memory: Memory
    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def document(self) -> dict[str, Any]:
        return {**self.model_dump(mode="json"), "ok": self.ok}


def equations(spec: CaseSpec) -> int:
    """Number of solved equations per cell: flow equations plus turbulence transport."""
    flow = 5 if isinstance(spec.physics, Compressible) else 4
    turbulence = spec.physics.turbulence
    if isinstance(turbulence, DDES):
        extra = 1 if turbulence.base == "sa" else 2
    else:
        extra = {SA: 1, SST: 2}.get(type(turbulence), 0)
    return flow + extra


def memory(spec: CaseSpec, mesh: MeshInfo, gpus: int) -> Memory:
    """Device memory of the arrays listed in docs/sdk/plan.md (model ``cfd.memory/1``)."""
    n = equations(spec)
    fp64 = spec.physics.precision == "fp64"
    compute, accumulate, jacobian = (8, 8, 8) if fp64 else (4, 8, 2)
    viscous = not isinstance(spec.physics.turbulence, Inviscid)
    cells, faces = mesh.counts.cells, mesh.counts.faces
    interior = faces - sum(zone.faces for zone in mesh.boundary_zones)
    per_cell = (2 * n * accumulate + 5 * n * compute + 4 * 8
                + (compute if viscous else 0) + n * n * jacobian)
    per_face = 6 * 8 + 2 * 4
    per_interior_face = 2 * (n * n * jacobian + 4)
    total = cells * per_cell + faces * per_face + interior * per_interior_face
    return Memory(bytes_total=total, bytes_per_gpu=-(-total // gpus), gpus=gpus, equations=n)


def plan(case: Case, mesh: MeshInfo, *, gpus: int = 1) -> Plan:
    spec = case.spec
    found: list[Diagnostic] = []

    if PurePath(mesh.source.path).name != PurePath(spec.mesh).name:
        found.append(diagnostic(
            "E0203", f"MeshInfo describes {mesh.source.path!r}, the case uses {spec.mesh!r}",
            location=case.locate("mesh")))
    worst = mesh.quality.max_non_orthogonality_deg
    if worst > NON_ORTHOGONALITY_WARNING_DEG:
        found.append(diagnostic(
            "W0204", f"maximum non-orthogonality is {worst:g}° (> {NON_ORTHOGONALITY_WARNING_DEG:g}°)",
            location=case.locate("mesh"), max_non_orthogonality_deg=worst))

    boundaries = _resolve_boundaries(case, spec, mesh, found)
    cell_zones = _resolve_cell_zones(case, spec, mesh, found)
    _check_interfaces(case, mesh, cell_zones, found)
    _check_monitors(case, spec, mesh, found)

    return Plan(
        case={"name": spec.name, "sha256": spec.hash()},
        mesh={"path": mesh.source.path, "sha256": mesh.source.sha256,
              "cells": mesh.counts.cells, "faces": mesh.counts.faces,
              "cells_by_type": mesh.counts.cells_by_type.model_dump(),
              "quality": mesh.quality.model_dump()},
        boundaries=tuple(boundaries),
        cell_zones=tuple(cell_zones),
        memory=memory(spec, mesh, gpus),
        diagnostics=tuple(found),
    )


def _matches(pattern: str, names: list[str]) -> list[str]:
    return [name for name in names if fnmatchcase(name, pattern)]


def _resolve_boundaries(
    case: Case, spec: CaseSpec, mesh: MeshInfo, found: list[Diagnostic]
) -> list[BoundaryResolution]:
    names = [zone.name for zone in mesh.boundary_zones]
    owner: dict[str, int] = {}
    for index, rule in enumerate(spec.boundaries):
        where = case.locate(f"boundaries[{index}]")
        matched = _matches(rule.zones, names)
        if not matched:
            found.append(diagnostic(
                "E0410", f"boundary pattern {rule.zones!r} matches no zone "
                f"(zones: {', '.join(names)})", location=where, pattern=rule.zones))
        for name in matched:
            if name in owner:
                found.append(diagnostic(
                    "E0411", f"boundary zone {name!r} is matched by rules "
                    f"{spec.boundaries[owner[name]].zones!r} and {rule.zones!r}",
                    location=where, zone=name))
            else:
                owner[name] = index
        _check_condition(case, spec, index, found)

    inviscid = isinstance(spec.physics.turbulence, Inviscid)
    resolved = []
    for zone in mesh.boundary_zones:
        index = owner.get(zone.name)
        condition: Any = None
        provenance: Provenance | None = None
        if index is not None:
            condition, provenance = spec.boundaries[index].condition, "code"
            if zone.type is not None and condition.kind not in _COMPATIBLE_MESH_TYPE[zone.type]:
                found.append(diagnostic(
                    "W0415", f"boundary zone {zone.name!r} is typed {zone.type!r} in the mesh; "
                    f"the case makes it {condition.kind!r}",
                    location=case.locate(f"boundaries[{index}]"), zone=zone.name))
        elif zone.type in ("wall", "slip", "symmetry", "farfield"):
            provenance = "mesh"
            condition = {"wall": Slip() if inviscid else Wall(), "slip": Slip(),
                         "symmetry": Symmetry(), "farfield": Farfield()}[zone.type]
        elif zone.type in ("inlet", "outlet") and zone.faces > 0:
            method = _MESH_METHOD[zone.type][spec.physics.kind]
            found.append(diagnostic(
                "E0413", f"boundary zone {zone.name!r} is typed {zone.type!r} in the mesh "
                "and needs values from the case", location=case.locate("boundaries"),
                fix=f"case.boundary({zone.name!r}).{method}", zone=zone.name))
        elif zone.faces > 0:
            found.append(diagnostic(
                "E0412", f"boundary zone {zone.name!r} has no condition",
                location=case.locate("boundaries"),
                fix=f"case.boundary({zone.name!r}).wall()  # or slip(), symmetry(), farfield()",
                zone=zone.name))
        resolved.append(BoundaryResolution(
            zone=zone.name, faces=zone.faces, mesh_type=zone.type,
            condition=None if condition is None else condition.model_dump(mode="json",
                                                                         exclude_none=True),
            provenance=provenance, rule=index))
    return resolved


def _check_condition(case: Case, spec: CaseSpec, index: int, found: list[Diagnostic]) -> None:
    condition = spec.boundaries[index].condition
    physics = spec.physics
    problem = None
    if isinstance(condition, Inlet) and isinstance(physics, Incompressible):
        problem = "inlet (total conditions) needs compressible physics; use velocity_inlet"
    elif isinstance(condition, VelocityInlet) and isinstance(physics, Compressible):
        problem = "velocity_inlet needs incompressible physics; use inlet"
    elif isinstance(condition, Wall) and isinstance(physics.turbulence, Inviscid):
        problem = "a no-slip wall needs viscous flow; use slip() with inviscid physics"
    if problem:
        found.append(diagnostic("E0414", problem, location=case.locate(f"boundaries[{index}]")))


def _default_medium(spec: CaseSpec) -> Fluid:
    physics = spec.physics
    material = physics.gas if isinstance(physics, Compressible) else physics.fluid
    return Fluid(material=material)


def _resolve_cell_zones(
    case: Case, spec: CaseSpec, mesh: MeshInfo, found: list[Diagnostic]
) -> list[CellZoneResolution]:
    names = [zone.name for zone in mesh.cell_zones]
    owner: dict[str, int] = {}
    for index, rule in enumerate(spec.cell_zones):
        where = case.locate(f"cell_zones[{index}]")
        matched = _matches(rule.zones, names)
        if not matched:
            found.append(diagnostic(
                "E0501", f"cell zone pattern {rule.zones!r} matches no zone "
                f"(zones: {', '.join(names)})", location=where, pattern=rule.zones))
        for name in matched:
            if name in owner:
                found.append(diagnostic(
                    "E0502", f"cell zone {name!r} is matched by rules "
                    f"{spec.cell_zones[owner[name]].zones!r} and {rule.zones!r}",
                    location=where, zone=name))
            else:
                owner[name] = index
        medium = rule.medium
        if isinstance(medium, Fluid):
            wanted = IdealGas if isinstance(spec.physics, Compressible) else ConstantDensityFluid
            if not isinstance(medium.material, wanted):
                found.append(diagnostic(
                    "E0504", f"{spec.physics.kind} flow needs a {wanted.__name__} in fluid zones, "
                    f"got {type(medium.material).__name__}", location=where))

    resolved = []
    for zone in mesh.cell_zones:
        index = owner.get(zone.name)
        medium: Fluid | SolidZone | None = None
        provenance: Provenance | None = None
        if index is not None:
            medium, provenance = spec.cell_zones[index].medium, "code"
        elif zone.type == "solid":
            found.append(diagnostic(
                "E0503", f"cell zone {zone.name!r} is solid in the mesh and has no material",
                location=case.locate("cell_zones"),
                fix=f"case.cell_zone({zone.name!r}).solid(cfd.materials.COPPER)", zone=zone.name))
        else:
            medium = _default_medium(spec)
            provenance = "mesh" if zone.type == "fluid" else "default"
        resolved.append(CellZoneResolution(
            zone=zone.name, cells=zone.cells, mesh_type=zone.type,
            medium=None if medium is None else medium.model_dump(mode="json", exclude_none=True),
            provenance=provenance, rule=index))
    return resolved


def _check_interfaces(
    case: Case, mesh: MeshInfo, zones: list[CellZoneResolution], found: list[Diagnostic]
) -> None:
    media = {zone.zone: zone.medium for zone in zones}
    for interface in mesh.interfaces:
        a, b = (media[name] for name in interface.zones)
        if a and b and a["kind"] == b["kind"] == "fluid" and a["material"] != b["material"]:
            found.append(diagnostic(
                "E0505", f"fluid zones {interface.zones[0]!r} ({a['material']['name']}) and "
                f"{interface.zones[1]!r} ({b['material']['name']}) share {interface.faces} faces",
                location=case.locate("cell_zones"), zones=list(interface.zones)))


def _check_monitors(
    case: Case, spec: CaseSpec, mesh: MeshInfo, found: list[Diagnostic]
) -> None:
    names = [zone.name for zone in mesh.boundary_zones]
    seen: set[str] = set()
    for index, monitor in enumerate(spec.monitors):
        where = case.locate(f"monitors[{index}]")
        if not _matches(monitor.surfaces, names):
            found.append(diagnostic(
                "E0601", f"monitor {monitor.name!r}: surfaces {monitor.surfaces!r} match no "
                "boundary zone", location=where, monitor=monitor.name))
        if monitor.name in seen:
            found.append(diagnostic(
                "E0602", f"two monitors are named {monitor.name!r}", location=where,
                monitor=monitor.name))
        seen.add(monitor.name)
