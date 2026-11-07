"""``autan plan``: resolve a case against its mesh before any GPU is reserved.

Resolution rules: docs/contracts/case-schema.md#boundary-resolution.
Memory model: docs/sdk/plan.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from autan import face_states, initial_state
from autan.errors import Diagnostic, Location, diagnostic
from autan.manufactured import MMS_BYTES, too_deep
from autan.memory import DeviceKind, Memory, StartMode, StateFileLoad, estimate, extracts
from autan.meshinfo import mesh_location, mesh_sha256
from autan.schema import (
    DDES,
    SA,
    SST,
    CaseSpec,
    Compressible,
    ConstantDensityFluid,
    ConstantViscosity,
    Farfield,
    Fluid,
    IdealGas,
    Incompressible,
    Inlet,
    Inviscid,
    Laminar,
    OracleState,
    PointVortex,
    Slip,
    SolidZone,
    StateFile,
    Steady,
    Symmetry,
    VelocityInlet,
    Wall,
)
from autan.solver_input import gas_constant
from autan.texts import load_json
from autan.units import format_unit, metres_per, parse, si

if TYPE_CHECKING:
    from pathlib import Path

    from autan.case import Case
    from autan.meshinfo import BoundaryZone, MeshInfo

MMS_SCHEMA = "autan.mms/1"
Provenance = Literal["code", "mesh", "default", "verification"]
NON_ORTHOGONALITY_WARNING_DEG = 70.0

_COMPATIBLE_MESH_TYPE = {
    "wall": {"wall"},
    "thermal-wall": {"wall"},
    "slip": {"slip"},
    "symmetry": {"symmetry"},
    "farfield": {"farfield"},
    "inlet": {"inlet", "velocity-inlet"},
    "outlet": {"outlet"},
}
_MESH_METHOD = {
    "thermal-wall": {
        "compressible": "wall(temperature=...)",
        "incompressible": "wall(temperature=...)",
    },
    "inlet": {
        "compressible": "inlet(total_pressure=..., total_temperature=...)",
        "incompressible": "velocity_inlet(velocity=...)",
    },
    "outlet": {
        "compressible": "outlet(static_pressure=...)",
        "incompressible": "outlet(static_pressure=...)",
    },
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
    model = turbulence.base if isinstance(turbulence, DDES) else turbulence
    return flow + {SA: 1, SST: 2}.get(type(model), 0)


def plan(
    case: Case,
    mesh: MeshInfo,
    *,
    gpus: int = 1,
    device_kind: DeviceKind = "cuda",
    start: StartMode | None = None,
) -> Plan:
    spec = case.spec
    found: list[Diagnostic] = []
    files: list[StateFileLoad] = []

    mesh_file = case.mesh_path
    if not mesh_file.exists():
        found.append(
            diagnostic(
                "W0205",
                f"mesh {str(mesh_file)!r} not found; the MeshInfo is taken as given",
                location=case.locate("mesh"),
                mesh=str(mesh_file),
            )
        )
    elif mesh_sha256(mesh_file) != mesh.source.sha256:
        found.append(
            diagnostic(
                "E0203",
                f"the MeshInfo hash differs from the hash of {str(mesh_file)!r}",
                location=case.locate("mesh"),
                mesh=str(mesh_file),
            )
        )
    _check_groups_file(case, mesh, found)
    worst = mesh.quality.max_non_orthogonality_deg
    if worst > NON_ORTHOGONALITY_WARNING_DEG:
        found.append(
            diagnostic(
                "W0204",
                f"maximum non-orthogonality is {worst:g}° (> {NON_ORTHOGONALITY_WARNING_DEG:g}°)",
                location=case.locate("mesh"),
                max_non_orthogonality_deg=worst,
            )
        )

    if spec.verification is not None:
        _check_manufactured(case, spec, found, files)
        boundaries = _manufactured_boundaries(mesh)
    else:
        boundaries = _resolve_boundaries(case, spec, mesh, found, files)
        _check_point_vortices(case, spec, mesh, boundaries, found)
    cell_zones = _resolve_cell_zones(case, spec, mesh, found)
    if spec.initial_state is not None:
        kinds = [(z.medium or {}).get("kind") for z in cell_zones]
        fluid = sum(z.cells for z, k in zip(cell_zones, kinds, strict=True) if k != "solid")
        # A zone left without a medium (E0503) leaves the fluid cells unknown.
        count = fluid if None not in kinds else None
        _check_initial_state(case, spec.initial_state, mesh, count, found, files)
    _check_interfaces(case, mesh, cell_zones, found)
    _check_monitors(case, spec, mesh, found)
    _check_prepared(case, spec, mesh, gpus, device_kind, found)

    scale = metres_per(spec.mesh_unit)
    return Plan(
        case={"name": spec.name, "sha256": spec.hash()},
        mesh={
            "path": mesh.source.path,
            "sha256": mesh.source.sha256,
            "unit": format_unit(spec.mesh_unit),
            "bounds": {
                side: [x * scale for x in getattr(mesh.bounds, side)] for side in ("min", "max")
            },
            "cells": mesh.counts.cells,
            "faces": mesh.counts.faces,
            "cells_by_type": mesh.counts.cells_by_type.model_dump(),
            "quality": mesh.quality.model_dump(),
        },
        boundaries=tuple(boundaries),
        cell_zones=tuple(cell_zones),
        memory=estimate(
            spec,
            mesh,
            equations=equations(spec),
            ranks=gpus,
            device_kind=device_kind,
            vortices=_vortex_conditions(spec, boundaries),
            start=start,
            files=tuple(files),
            input_bytes=_input_bytes(case, mesh, boundaries, cell_zones),
        ),
        diagnostics=tuple(found),
    )


_PATH_MAX = 4096


def _names(data: bytes) -> int:
    """The bytes of every member name of a valid JSON document, in UTF-8."""
    total = 0

    def note(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal total
        total += sum(len(name.encode()) for name, _ in pairs)
        return dict(pairs)

    load_json(data, object_pairs_hook=note)
    return total


# A zone's or a monitor's entry in the solver input besides its texts: keys, a kind, at most
# 20 numbers of at most 24 characters each (Python's repr of a double) and a SHA-256.
_ENTRY = 1024
# A zone name's list formatting around its text as `autan prepare` indents the input: at most 10
# spaces (a point vortex's zones), a comma and a newline.
_ITEM = 16


def _input_bytes(
    case: Case,
    mesh: MeshInfo,
    boundaries: list[BoundaryResolution],
    cell_zones: list[CellZoneResolution],
) -> int:
    """A bound on the solver input's size as ``autan prepare`` writes it (solver-input.md),
    each text escaped as the producers escape it: the case's TOML (at most 6 bytes a byte)
    with its mesh path counted at PATH_MAX (however the case names it), the MeshInfo, each
    zone's entry with its name and its resolved condition or medium (a state file's path at
    PATH_MAX), each monitor with the zones it matches, a point vortex with its walls, the
    files the input names at PATH_MAX, and 64 KiB for the rest of the document."""
    spec = case.spec

    def text(value: object) -> int:
        return len(json.dumps(value))  # as the producers escape it

    toml = len(case.to_toml().encode()) - len(str(spec.mesh).encode()) + _PATH_MAX
    zones = sum(text(z.zone) + text(z.medium) + _ENTRY for z in cell_zones)
    for b in boundaries:
        zones += text(b.zone) + text(b.condition) + _ENTRY
        if (b.condition or {}).get("kind") == "oracle-state":
            zones += 6 * _PATH_MAX
        rule = spec.boundaries[b.rule] if b.rule is not None else None
        if (
            rule is not None
            and isinstance(rule.condition, Farfield)
            and rule.condition.point_vortex
        ):
            walls, _ = vortex_zones(rule.condition.point_vortex, mesh, boundaries)
            zones += sum(text(n) + _ITEM for n in walls)
    monitors = sum(
        text(m.name) + sum(text(n) + _ITEM for n in selected(m.surfaces, m.group, mesh)) + _ENTRY
        for m in spec.monitors
    )
    files = 6 * _PATH_MAX * 3  # the mesh, a start's run and a verification solution
    # The MeshInfo as the producers indent it, nested one level deeper (2 B a line, 4 counted).
    mesh_text = json.dumps(mesh.document(), indent=2)
    mesh_bytes = len(mesh_text) + 4 * (mesh_text.count("\n") + 1)
    return 6 * toml + mesh_bytes + zones + monitors + files + 64 * 2**10


def _vortex_conditions(spec: CaseSpec, boundaries: list[BoundaryResolution]) -> int:
    """The zones whose farfield condition carries a point vortex: one restart field each."""
    rules = [spec.boundaries[b.rule] if b.rule is not None else None for b in boundaries]
    return sum(
        1
        for rule in rules
        if rule is not None
        and isinstance(rule.condition, Farfield)
        and rule.condition.point_vortex is not None
    )


def partition_groups(spec: CaseSpec, device_kind: DeviceKind) -> str:
    """The groups a run's partition keeps whole: the line smoother's lines of a steady CPU run
    with ``linear_solver = "lines"``, none otherwise (docs/contracts/mesh-info.md)."""
    lines = isinstance(spec.time, Steady) and spec.time.linear_solver == "lines"
    return "lines" if lines and device_kind == "cpu" else "none"


def _check_prepared(
    case: Case,
    spec: CaseSpec,
    mesh: MeshInfo,
    gpus: int,
    device_kind: DeviceKind,
    found: list[Diagnostic],
) -> None:
    """The MeshInfo's derived counts are for the run planned: its scale, and on several ranks
    the same partition (mesh-info.md)."""
    problems = []
    scale = metres_per(spec.mesh_unit)
    if mesh.quality.lines.scale != scale:
        problems.append(
            f"quality.lines was computed at scale {mesh.quality.lines.scale:g}, "
            f"the case's mesh_unit is {scale:g} m"
        )
    partition = mesh.partition
    if gpus == 1 and extracts(spec, mesh, 1) and (partition is None or partition.ranks != 1):
        problems.append(
            'a run on one rank with periodic pairs or reproducibility "ranks" needs the '
            "MeshInfo's one-part partition"
        )
    if gpus > 1:
        groups = partition_groups(spec, device_kind)
        if partition is None:
            problems.append(f"a run on {gpus} ranks needs the MeshInfo's partition for {gpus}")
        elif (partition.ranks, partition.groups) != (gpus, groups):
            problems.append(
                f"the partition is for {partition.ranks} ranks with groups {partition.groups!r}, "
                f"the run is on {gpus} with groups {groups!r}"
            )
    found.extend(diagnostic("E0212", problem, location=case.locate("mesh")) for problem in problems)


def groups_file(mesh_file: Path) -> Path:
    """The zone groups file of a mesh: ``<mesh file or polyMesh directory>.zones.json`` next
    to it, symlinks followed, as the runtime finds it from the solver input's path."""
    mesh = mesh_location(mesh_file).resolve()
    return mesh.parent / f"{mesh.name}.zones.json"


def _check_groups_file(case: Case, mesh: MeshInfo, found: list[Diagnostic]) -> None:
    path = groups_file(case.mesh_path)
    source = mesh.groups_source
    if os.path.lexists(path):  # a dangling link is there but unreadable, not absent
        try:
            with path.open("rb") as f:  # hashed in chunks: the file is not held
                digest: str | None = hashlib.file_digest(f, "sha256").hexdigest()
        except OSError:
            digest = None
        if source is None or digest != source.sha256:
            found.append(
                diagnostic(
                    "E0210",
                    f"the MeshInfo's zone groups differ from {str(path)!r}"
                    + ("" if digest else ", which cannot be read"),
                    location=case.locate("mesh"),
                    groups=str(path),
                )
            )
    elif source is not None:
        found.append(
            diagnostic(
                "W0211",
                f"zone groups file {str(path)!r} not found; the MeshInfo's groups are taken "
                "as given",
                location=case.locate("mesh"),
                groups=str(path),
            )
        )


def _matches(pattern: str, names: list[str]) -> list[str]:
    return [name for name in names if fnmatchcase(name, pattern)]


def selected(zones: str | None, group: str | None, mesh: MeshInfo) -> list[str]:
    """The non-periodic boundary zones a selection takes, in MeshInfo order: those matching
    the glob ``zones``, or every zone of the groups whose name matches the glob ``group``."""
    partners = _partners(mesh)
    names = [zone.name for zone in mesh.boundary_zones if zone.name not in partners]
    if zones is not None:
        return _matches(zones, names)
    assert group is not None
    chosen = {z for g in mesh.groups if fnmatchcase(g.name, group) for z in g.zones}
    return [name for name in names if name in chosen]


def _unknown_group(
    group: str | None, mesh: MeshInfo, location: Location, found: list[Diagnostic]
) -> bool:
    """Report E0428 when ``group`` matches no group of the mesh."""
    if group is None or any(fnmatchcase(g.name, group) for g in mesh.groups):
        return False
    known = ", ".join(g.name for g in mesh.groups) or "none: the mesh has no zones.json"
    found.append(
        diagnostic(
            "E0428",
            f"group {group!r} matches no zone group of the mesh",
            location=location,
            fix=f"Use one of the mesh's groups: {known}.",
            group=group,
        )
    )
    return True


def _label(zones: str | None, group: str | None) -> str:
    return repr(zones) if zones is not None else f"group {group!r}"


def _check_manufactured(
    case: Case, spec: CaseSpec, found: list[Diagnostic], files: list[StateFileLoad]
) -> None:
    assert spec.verification is not None
    expected = spec.verification.manufactured.sha256
    source = case.resolve(spec.verification.manufactured.path)
    where = case.locate("verification.manufactured")
    if not source.is_file():
        found.append(
            diagnostic(
                "E0207",
                f"manufactured solution {str(source)!r} not found",
                location=where,
                path=str(source),
            )
        )
    elif (size := source.stat().st_size) > MMS_BYTES:
        found.append(
            diagnostic(
                "E0208",
                f"{str(source)!r} is {size} bytes, above the {MMS_BYTES} of {MMS_SCHEMA}",
                location=where,
                path=str(source),
            )
        )
    elif mesh_sha256(source) != expected:
        found.append(
            diagnostic(
                "E0206",
                f"the SHA-256 of {str(source)!r} differs from the case's sha256",
                location=where,
                path=str(source),
            )
        )
    else:
        files.append(StateFileLoad(size, 0, 0, 0, "manufactured"))
        document = _document(source)
        if document is None or document.get("schema") != MMS_SCHEMA:
            found.append(
                diagnostic(
                    "E0208",
                    f"{str(source)!r} is not an {MMS_SCHEMA} document",
                    location=where,
                    path=str(source),
                )
            )
        elif (reason := too_deep(document)) is not None:
            found.append(diagnostic("E0208", f"{str(source)!r}: {reason}", location=where))
        else:
            found.extend(
                diagnostic("E0209", message, location=where, path=str(source))
                for message in _mismatches(document, spec)
            )


def _document(path: Path) -> dict[str, Any] | None:
    try:
        document = load_json(path.read_text())
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _mismatches(document: dict[str, Any], spec: CaseSpec) -> list[str]:
    """Where the manufactured solution's declared constants differ from the case physics."""
    physics = spec.physics
    if not isinstance(physics, Compressible):
        return ["manufactured solutions are compressible; the case is incompressible"]
    gas, parameters = physics.gas, document.get("parameters", {})
    law = gas.viscosity
    case_values: dict[str, float | None] = {
        "gamma": gas.gamma,
        "gas_constant": gas_constant(gas),
        "prandtl": gas.prandtl if "prandtl" in parameters else None,
        "viscosity": (si(law.mu, "Pa * s") if isinstance(law, ConstantViscosity) else None)
        if "viscosity" in parameters
        else None,
    }
    found = []
    for key, value in parameters.items():
        if key not in case_values:
            continue
        mine = case_values[key]
        if mine is None or not math.isclose(mine, value, rel_tol=1e-12):
            found.append(f"{key} is {value!r} in the solution and {mine!r} in the case")
    model = document.get("model")
    declared = model.get("turbulence") if isinstance(model, dict) else None
    if not isinstance(model, dict) or model.get("physics") != "compressible":
        found.append(f"the solution's model is {model!r}; the case is compressible flow")
    elif declared != physics.turbulence.model:
        found.append(
            f"the solution is {declared!r} turbulence; the case is {physics.turbulence.model!r}"
        )
    if spec.cell_zones:
        found.append("a manufactured solution covers one fluid zone; remove the cell zone rules")
    return found


def _partners(mesh: MeshInfo) -> dict[str, str]:
    """Each zone of a periodic pair, with the other zone of its pair."""
    return {a: b for pair in mesh.periodic for a, b in (pair.zones, pair.zones[::-1])}


def _periodic(zone: BoundaryZone, partner: str) -> BoundaryResolution:
    return BoundaryResolution(
        zone=zone.name,
        faces=zone.faces,
        mesh_type=zone.type,
        condition={"kind": "periodic", "partner": partner},
        provenance="mesh",
        rule=None,
    )


def _manufactured_boundaries(mesh: MeshInfo) -> list[BoundaryResolution]:
    """Every boundary zone but the periodic ones takes the exact data of the manufactured
    solution."""
    partners = _partners(mesh)
    return [
        _periodic(zone, partners[zone.name])
        if zone.name in partners
        else BoundaryResolution(
            zone=zone.name,
            faces=zone.faces,
            mesh_type=zone.type,
            condition={"kind": "manufactured"},
            provenance="verification",
            rule=None,
        )
        for zone in mesh.boundary_zones
    ]


def _resolve_boundaries(
    case: Case,
    spec: CaseSpec,
    mesh: MeshInfo,
    found: list[Diagnostic],
    files: list[StateFileLoad],
) -> list[BoundaryResolution]:
    partners = _partners(mesh)
    names = [zone.name for zone in mesh.boundary_zones if zone.name not in partners]
    owner: dict[str, int] = {}
    for index, rule in enumerate(spec.boundaries):
        where = case.locate(f"boundaries[{index}]")
        if _unknown_group(rule.group, mesh, where, found):
            _check_condition(case, spec, index, found)
            continue
        if (
            rule.zones is not None
            and rule.zones in partners
            and not any(c in rule.zones for c in "*?[")
        ):
            found.append(
                diagnostic(
                    "E0424",
                    f"boundary zone {rule.zones!r} is periodic with {partners[rule.zones]!r} "
                    "in the mesh; it takes no condition",
                    location=where,
                    zone=rule.zones,
                )
            )
            continue
        matched = selected(rule.zones, rule.group, mesh)
        if not matched:
            found.append(
                diagnostic(
                    "E0410",
                    f"boundary selection {_label(rule.zones, rule.group)} matches no zone "
                    f"(zones: {', '.join(names)})",
                    location=where,
                    pattern=rule.zones if rule.zones is not None else rule.group,
                )
            )
        for name in matched:
            if name in owner:
                found.append(
                    diagnostic(
                        "E0411",
                        f"boundary zone {name!r} is matched by rules "
                        f"{_label(spec.boundaries[owner[name]].zones, spec.boundaries[owner[name]].group)}"
                        f" and {_label(rule.zones, rule.group)}",
                        location=where,
                        zone=name,
                    )
                )
            else:
                owner[name] = index
        _check_condition(case, spec, index, found)
        if isinstance(rule.condition, OracleState):
            zones = {z.name: z.faces for z in mesh.boundary_zones if z.name in matched}
            _check_oracle(case, rule.condition, mesh, zones, index, found, files)

    inviscid = isinstance(spec.physics.turbulence, Inviscid)
    resolved = []
    for zone in mesh.boundary_zones:
        if zone.name in partners:
            resolved.append(_periodic(zone, partners[zone.name]))
            continue
        rule_index = owner.get(zone.name)
        condition: Any = None
        provenance: Provenance | None = None
        if rule_index is not None:
            condition, provenance = spec.boundaries[rule_index].condition, "code"
            if zone.type is not None and condition.kind not in _COMPATIBLE_MESH_TYPE[zone.type]:
                found.append(
                    diagnostic(
                        "W0415",
                        f"boundary zone {zone.name!r} is typed {zone.type!r} in the mesh; "
                        f"the case makes it {condition.kind!r}",
                        location=case.locate(f"boundaries[{rule_index}]"),
                        zone=zone.name,
                    )
                )
        elif zone.type in ("wall", "slip", "symmetry", "farfield"):
            provenance = "mesh"
            condition = {
                "wall": Slip() if inviscid else Wall(),
                "slip": Slip(),
                "symmetry": Symmetry(),
                "farfield": Farfield(),
            }[zone.type]
        elif zone.type in _MESH_METHOD and zone.faces > 0:
            method = _MESH_METHOD[zone.type][spec.physics.kind]
            found.append(
                diagnostic(
                    "E0413",
                    f"boundary zone {zone.name!r} is typed {zone.type!r} in the mesh "
                    "and needs values from the case",
                    location=case.locate("boundaries"),
                    fix=f"case.boundary({zone.name!r}).{method}",
                    zone=zone.name,
                )
            )
        elif zone.faces > 0:
            found.append(
                diagnostic(
                    "E0412",
                    f"boundary zone {zone.name!r} has no condition",
                    location=case.locate("boundaries"),
                    fix=f"case.boundary({zone.name!r}).wall()  # or slip(), symmetry(), farfield()",
                    zone=zone.name,
                )
            )
        resolved.append(
            BoundaryResolution(
                zone=zone.name,
                faces=zone.faces,
                mesh_type=zone.type,
                condition=None
                if condition is None
                else condition.model_dump(mode="json", exclude_none=True),
                provenance=provenance,
                rule=rule_index,
            )
        )
    return resolved


def read_regular(path: Path) -> bytes:
    """The bytes of a regular file, read once; ``OSError`` for anything else."""
    # O_NONBLOCK: opening a FIFO must not wait for a writer before the check below.
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            message = f"{path} is not a regular file"
            raise OSError(message)
        return handle.read()


def _check_initial_state(
    case: Case,
    state: StateFile,
    mesh: MeshInfo,
    fluid_cells: int | None,
    found: list[Diagnostic],
    files: list[StateFileLoad],
) -> None:
    """The initial-state file of the case (docs/contracts/initial-state.md), hashed and parsed
    from the same bytes."""
    source = case.resolve(state.path)
    where = case.locate("initial_state")

    def problem(code: str, message: str) -> None:
        found.append(diagnostic(code, message, location=where, path=str(source)))

    try:
        data = read_regular(source)
    except OSError as error:
        problem("E0508", f"initial-state file {str(source)!r} cannot be read: {error}")
        return
    if hashlib.sha256(data).hexdigest() != state.sha256:
        problem("E0508", f"the SHA-256 of {str(source)!r} differs from the case's sha256")
        return
    document = initial_state.parse(data)
    if isinstance(document, str):
        problem(
            "E0508", f"{str(source)!r} is not a valid {initial_state.SCHEMA} document: {document}"
        )
        return
    files.append(StateFileLoad(len(data), document.cells, 0, 0, "initial", _names(data)))
    if document.mesh_sha256 != mesh.source.sha256:
        problem("E0509", "the initial state was written for another mesh")
    if fluid_cells is not None and document.cells != fluid_cells:
        problem(
            "E0510",
            f"the mesh has {fluid_cells} fluid cells; the file gives {document.cells}",
        )


def _check_oracle(
    case: Case,
    oracle: OracleState,
    mesh: MeshInfo,
    zones: dict[str, int],
    index: int,
    found: list[Diagnostic],
    files: list[StateFileLoad],
) -> None:
    """The face-state file of an oracle_state rule (docs/contracts/face-states.md), hashed
    and parsed from the same bytes."""
    source = case.resolve(oracle.path)
    where = case.locate(f"boundaries[{index}].condition")

    def problem(code: str, message: str) -> None:
        found.append(diagnostic(code, message, location=where, path=str(source)))

    try:
        data = read_regular(source)
    except OSError as error:
        problem("E0420", f"face-state file {str(source)!r} cannot be read: {error}")
        return
    if hashlib.sha256(data).hexdigest() != oracle.sha256:
        problem("E0420", f"the SHA-256 of {str(source)!r} differs from the case's sha256")
        return
    document = face_states.parse(data)
    if isinstance(document, str):
        problem(
            "E0420", f"{str(source)!r} is not a valid {face_states.SCHEMA} document: {document}"
        )
        return
    rows = sum(z.faces for z in document.zones.values())
    faces = sum(zones.values())
    files.append(StateFileLoad(len(data), rows, faces, len(document.zones), "faces", _names(data)))
    if document.mesh_sha256 != mesh.source.sha256:
        problem("E0422", "the face states were exported for another mesh")
    for zone, faces in zones.items():
        given = document.zones.get(zone)
        if given is None or given.faces != faces:
            problem(
                "E0423",
                f"zone {zone!r} has {faces} faces; the file gives "
                f"{'none' if given is None else given.faces}",
            )


def _check_condition(case: Case, spec: CaseSpec, index: int, found: list[Diagnostic]) -> None:
    condition = spec.boundaries[index].condition
    physics = spec.physics
    problem = None
    if isinstance(condition, Inlet) and isinstance(physics, Incompressible):
        problem = "inlet (total conditions) needs compressible physics; use velocity_inlet"
    elif isinstance(condition, VelocityInlet) and isinstance(physics, Compressible):
        problem = "velocity_inlet needs incompressible physics; use inlet"
    elif isinstance(condition, OracleState) and (
        isinstance(physics, Incompressible)
        or not isinstance(physics.turbulence, Inviscid | Laminar)
    ):
        problem = (
            "oracle_state imposes compressible states without turbulence variables; "
            "it needs compressible inviscid or laminar physics"
        )
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
            found.append(
                diagnostic(
                    "E0501",
                    f"cell zone pattern {rule.zones!r} matches no zone (zones: {', '.join(names)})",
                    location=where,
                    pattern=rule.zones,
                )
            )
        for name in matched:
            if name in owner:
                found.append(
                    diagnostic(
                        "E0502",
                        f"cell zone {name!r} is matched by rules "
                        f"{spec.cell_zones[owner[name]].zones!r} and {rule.zones!r}",
                        location=where,
                        zone=name,
                    )
                )
            else:
                owner[name] = index
        medium = rule.medium
        if isinstance(medium, SolidZone) and isinstance(spec.physics, Incompressible):
            found.append(
                diagnostic(
                    "E0506", "solid zones need compressible physics in autan.case/1", location=where
                )
            )
        if isinstance(medium, Fluid):
            wanted = IdealGas if isinstance(spec.physics, Compressible) else ConstantDensityFluid
            if not isinstance(medium.material, wanted):
                found.append(
                    diagnostic(
                        "E0504",
                        f"{spec.physics.kind} flow needs a {wanted.__name__} in fluid zones, "
                        f"got {type(medium.material).__name__}",
                        location=where,
                    )
                )

    resolved = []
    for zone in mesh.cell_zones:
        rule_index = owner.get(zone.name)
        resolved_medium: Fluid | SolidZone | None = None
        provenance: Provenance | None = None
        if rule_index is not None:
            resolved_medium, provenance = spec.cell_zones[rule_index].medium, "code"
        elif zone.type == "solid":
            found.append(
                diagnostic(
                    "E0503",
                    f"cell zone {zone.name!r} is solid in the mesh and has no material",
                    location=case.locate("cell_zones"),
                    fix=f"case.cell_zone({zone.name!r}).solid(autan.materials.COPPER)",
                    zone=zone.name,
                )
            )
        else:
            resolved_medium = _default_medium(spec)
            provenance = "mesh" if zone.type == "fluid" else "default"
        resolved.append(
            CellZoneResolution(
                zone=zone.name,
                cells=zone.cells,
                mesh_type=zone.type,
                medium=None
                if resolved_medium is None
                else resolved_medium.model_dump(mode="json", exclude_none=True),
                provenance=provenance,
                rule=rule_index,
            )
        )
    return resolved


def _check_interfaces(
    case: Case, mesh: MeshInfo, zones: list[CellZoneResolution], found: list[Diagnostic]
) -> None:
    media = {zone.zone: zone.medium for zone in zones}
    for interface in mesh.interfaces:
        a, b = (media[name] for name in interface.zones)
        if (
            a
            and b
            and a["kind"] == b["kind"] == "fluid"
            and _properties(a["material"]) != _properties(b["material"])
        ):
            found.append(
                diagnostic(
                    "E0505",
                    f"fluid zones {interface.zones[0]!r} ({a['material']['name']}) and "
                    f"{interface.zones[1]!r} ({b['material']['name']}) share {interface.faces} faces",
                    location=case.locate("cell_zones"),
                    zones=list(interface.zones),
                )
            )


def _properties(material: dict[str, Any]) -> dict[str, Any]:
    """Physical properties in SI base units, rounded to 12 significant digits; no name."""

    def canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: canonical(v) for k, v in value.items() if k != "name"}
        if isinstance(value, str) and " " in value:
            magnitude = parse(value, "").to_base_units().magnitude
            return float(f"{magnitude:.12g}")
        return value

    result: dict[str, Any] = canonical(material)
    return result


def vortex_zones(
    vortex: PointVortex, mesh: MeshInfo, boundaries: tuple[BoundaryResolution, ...] | list[Any]
) -> tuple[list[str], list[str]]:
    """The zones a point vortex selects (periodic zones skipped), and those of them that are
    not wall or slip zones."""
    kinds = {b.zone: (b.condition or {}).get("kind") for b in boundaries}
    matched = selected(vortex.surfaces, vortex.group, mesh)
    return matched, [z for z in matched if kinds.get(z) not in ("wall", "slip")]


def _check_point_vortices(
    case: Case,
    spec: CaseSpec,
    mesh: MeshInfo,
    boundaries: list[BoundaryResolution],
    found: list[Diagnostic],
) -> None:
    partners = _partners(mesh)
    for index, rule in enumerate(spec.boundaries):
        vortex = rule.condition.point_vortex if isinstance(rule.condition, Farfield) else None
        if vortex is None:
            continue
        where = case.locate(f"boundaries[{index}].condition.point_vortex")
        if _unknown_group(vortex.group, mesh, where, found):
            continue
        matched, others = vortex_zones(vortex, mesh, boundaries)
        surfaces = vortex.surfaces
        if surfaces is not None and surfaces in partners and not any(c in surfaces for c in "*?["):
            problem = f"zone {surfaces!r} is periodic"
        elif not matched:
            problem = f"{_label(surfaces, vortex.group)} matches no boundary zone"
        elif others:
            problem = f"it matches zones that are not wall or slip: {', '.join(others)}"
        else:
            continue
        found.append(
            diagnostic(
                "E0425",
                f"point vortex of boundaries[{index}]: {problem}",
                location=where,
                surfaces=surfaces if surfaces is not None else vortex.group,
            )
        )


def _check_monitors(case: Case, spec: CaseSpec, mesh: MeshInfo, found: list[Diagnostic]) -> None:
    partners = _partners(mesh)
    seen: set[str] = set()
    for index, monitor in enumerate(spec.monitors):
        where = case.locate(f"monitors[{index}]")
        surfaces = monitor.surfaces
        if _unknown_group(monitor.group, mesh, where, found):
            pass
        elif (
            surfaces is not None and surfaces in partners and not any(c in surfaces for c in "*?[")
        ):
            found.append(
                diagnostic(
                    "E0603",
                    f"monitor {monitor.name!r}: boundary zone {monitor.surfaces!r} is periodic "
                    "and carries no load",
                    location=where,
                    monitor=monitor.name,
                    zone=monitor.surfaces,
                )
            )
        elif not selected(surfaces, monitor.group, mesh):
            found.append(
                diagnostic(
                    "E0601",
                    f"monitor {monitor.name!r}: {_label(surfaces, monitor.group)} matches no "
                    "boundary zone",
                    location=where,
                    monitor=monitor.name,
                )
            )
        if monitor.name in seen:
            found.append(
                diagnostic(
                    "E0602",
                    f"two monitors are named {monitor.name!r}",
                    location=where,
                    monitor=monitor.name,
                )
            )
        seen.add(monitor.name)
