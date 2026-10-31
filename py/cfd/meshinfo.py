"""MeshInfo ``cfd.mesh-info/1``: the mesh summary emitted by the core (docs/contracts/mesh-info.md)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cfd.errors import CaseError, Location, diagnostic
from cfd.units import fail

SCHEMA_ID = "cfd.mesh-info/1"

BoundaryType = Literal["wall", "thermal-wall", "slip", "symmetry", "farfield", "inlet", "outlet"]
Name = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_Strict):
    path: Name
    format: Literal["cgns", "gmsh", "fluent", "polymesh"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CellsByType(_Strict):
    tetra: Annotated[int, Field(ge=0)]
    pyramid: Annotated[int, Field(ge=0)]
    prism: Annotated[int, Field(ge=0)]
    hexa: Annotated[int, Field(ge=0)]
    polyhedron: Annotated[int, Field(ge=0)]


class Counts(_Strict):
    cells: Annotated[int, Field(ge=1)]
    faces: Annotated[int, Field(ge=1)]
    nodes: Annotated[int, Field(ge=1)]
    cells_by_type: CellsByType


class BoundaryZone(_Strict):
    name: Name
    type: BoundaryType | None
    native_type: str | None
    faces: Annotated[int, Field(ge=0)]


class CellZone(_Strict):
    name: Name
    type: Literal["fluid", "solid"] | None
    native_type: str | None
    cells: Annotated[int, Field(ge=1)]


class Interface(_Strict):
    zones: tuple[Name, Name]
    faces: Annotated[int, Field(ge=1)]


class Quality(_Strict):
    max_non_orthogonality_deg: Annotated[float, Field(ge=0, le=180)]
    max_faces_per_cell: Annotated[int, Field(ge=4)]


class MeshInfo(_Strict):
    """Summary of a mesh: counts, typed boundary and cell zones, interfaces, quality, hash."""

    schema_: Literal["cfd.mesh-info/1"] = Field(alias="schema")
    source: Source
    counts: Counts
    boundary_zones: tuple[BoundaryZone, ...]
    cell_zones: Annotated[tuple[CellZone, ...], Field(min_length=1)]
    interfaces: tuple[Interface, ...]
    quality: Quality

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        if sum(self.counts.cells_by_type.model_dump().values()) != self.counts.cells:
            problems.append("counts.cells_by_type does not sum to counts.cells")
        if sum(zone.cells for zone in self.cell_zones) != self.counts.cells:
            problems.append("cell_zones[].cells does not sum to counts.cells")
        if sum(zone.faces for zone in self.boundary_zones) > self.counts.faces:
            problems.append("boundary_zones[].faces exceeds counts.faces")
        for label, zones in (
            ("boundary_zones", self.boundary_zones),
            ("cell_zones", self.cell_zones),
        ):
            names = [zone.name for zone in zones]
            if len(set(names)) != len(names):
                problems.append(f"{label} has duplicate names")
        cell_names = {zone.name for zone in self.cell_zones}
        pairs = [interface.zones for interface in self.interfaces]
        for a, b in pairs:
            if a >= b:
                problems.append(f"interface {[a, b]} is not two distinct sorted names")
            if a not in cell_names or b not in cell_names:
                problems.append(f"interface {[a, b]} names an unknown cell zone")
        if len(set(pairs)) != len(pairs):
            problems.append("interfaces lists a pair twice")
        if problems:
            fail("E0202", "{problems}", problems="; ".join(problems))
        return self

    @classmethod
    def load(cls, path: str | Path) -> MeshInfo:
        """Read a MeshInfo JSON file; any defect raises ``CaseError`` with ``E0202``."""
        location = Location(file=str(path))
        try:
            document = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CaseError(
                [diagnostic("E0202", f"cannot read {path}: {error}", location=location)]
            ) from None
        return cls.from_document(document, location)

    @classmethod
    def from_document(cls, document: object, location: Location) -> MeshInfo:
        if isinstance(document, dict) and document.get("schema") != SCHEMA_ID:
            raise CaseError(
                [
                    diagnostic(
                        "E0202",
                        f"schema is {document.get('schema')!r}, expected {SCHEMA_ID!r}",
                        location=location,
                    )
                ]
            )
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            raise CaseError(
                diagnostic(
                    "E0202",
                    f"{'.'.join(map(str, e['loc'])) or 'mesh-info'}: {e['msg']}",
                    location=location,
                )
                for e in error.errors(include_url=False)
            ) from None

    def document(self) -> dict[str, object]:
        return self.model_dump(mode="json", by_alias=True)
