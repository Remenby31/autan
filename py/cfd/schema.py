"""The case document ``cfd.case/1`` as pydantic models (docs/contracts/case-schema.md)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pint
import rfc8785
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from cfd.errors import CaseError, Diagnostic, Location, diagnostic
from cfd.units import Dimension, K, deg, fail, validate_number

SCHEMA_ID = "cfd.case/1"

Length = Annotated[pint.Quantity, Dimension("[length]", positive=True)]
Area = Annotated[pint.Quantity, Dimension("[length] ** 2", positive=True)]
Speed = Annotated[pint.Quantity, Dimension("[length] / [time]", positive=True)]
Duration = Annotated[pint.Quantity, Dimension("[time]", positive=True)]
Temperature = Annotated[pint.Quantity, Dimension("[temperature]", positive=True)]
Pressure = Annotated[pint.Quantity, Dimension("[pressure]", positive=True)]
Density = Annotated[pint.Quantity, Dimension("[mass] / [length] ** 3", positive=True)]
DynamicViscosity = Annotated[pint.Quantity, Dimension("[mass] / [length] / [time]", positive=True)]
MolarMass = Annotated[pint.Quantity, Dimension("[mass] / [substance]", positive=True)]
SpecificHeat = Annotated[
    pint.Quantity, Dimension("[length] ** 2 / [time] ** 2 / [temperature]", positive=True)
]
Conductivity = Annotated[
    pint.Quantity, Dimension("[mass] * [length] / [time] ** 3 / [temperature]", positive=True)
]
Angle = Annotated[pint.Quantity, Dimension("[angle]")]
Coordinate = Annotated[pint.Quantity, Dimension("[length]")]
PositiveNumber = Annotated[float, BeforeValidator(lambda v: validate_number(v, positive=True))]
Pattern = Annotated[str, Field(min_length=1)]


def _caller_location() -> Location:
    frame = sys._getframe(1)
    package = str(Path(__file__).parent)
    while frame is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back  # type: ignore[assignment]
    if frame is None:
        return Location()
    return Location(file=frame.f_code.co_filename, line=frame.f_lineno)


def diagnostics_from(error: ValidationError, location: Location) -> list[Diagnostic]:
    """Convert pydantic errors to coded diagnostics; ``cfd.E####`` error types keep their code."""
    found = []
    for item in error.errors(include_url=False):
        path = document_path(item["loc"])
        code = item["type"][4:] if item["type"].startswith("cfd.") else "E0003"
        where = location.model_copy(
            update={"path": ".".join(p for p in (location.path, path) if p) or None}
        )
        found.append(diagnostic(code, f"{path or 'case'}: {item['msg']}", location=where))
    return found


def document_path(parts: tuple[int | str, ...]) -> str:
    """``("boundaries", 0, "condition")`` -> ``"boundaries[0].condition"``."""
    text = ""
    for part in parts:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else part)
    return text


class Spec(BaseModel):
    """Base of every case model: immutable, strict about unknown keys, coded errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    def __init__(self, /, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            raise CaseError(diagnostics_from(error, _caller_location())) from None

    # Nested validation must not run this conversion, only direct construction does:
    # pydantic calls a model's __init__ during validation unless it is marked as the base one.
    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]


# --- materials ---------------------------------------------------------------------------


class Sutherland(Spec):
    law: Literal["sutherland"] = "sutherland"
    mu_ref: DynamicViscosity
    t_ref: Temperature
    s: Temperature


class ConstantViscosity(Spec):
    law: Literal["constant"] = "constant"
    mu: DynamicViscosity


Viscosity = Annotated[Sutherland | ConstantViscosity, Field(discriminator="law")]


class IdealGas(Spec):
    """A calorically perfect gas: molar mass, ratio of specific heats, Prandtl number, viscosity."""

    kind: Literal["ideal-gas"] = "ideal-gas"
    name: str
    molar_mass: MolarMass
    gamma: PositiveNumber
    prandtl: PositiveNumber
    viscosity: Viscosity

    @model_validator(mode="after")
    def _gamma_above_one(self) -> Self:
        if self.gamma <= 1:
            fail("E0304", "gamma must be > 1, got {g}", g=self.gamma)
        return self


class ConstantDensityFluid(Spec):
    """A fluid of constant density and viscosity, for incompressible flow."""

    kind: Literal["constant-density-fluid"] = "constant-density-fluid"
    name: str
    density: Density
    viscosity: DynamicViscosity


class Solid(Spec):
    """A solid for conjugate heat transfer: density, specific heat, thermal conductivity."""

    kind: Literal["solid"] = "solid"
    name: str
    density: Density
    specific_heat: SpecificHeat
    conductivity: Conductivity


FluidMaterial = Annotated[IdealGas | ConstantDensityFluid, Field(discriminator="kind")]


# --- physics -----------------------------------------------------------------------------


class Inviscid(Spec):
    """Euler equations."""

    model: Literal["inviscid"] = "inviscid"


class Laminar(Spec):
    """Navier-Stokes equations without a turbulence model."""

    model: Literal["laminar"] = "laminar"


class SA(Spec):
    """Spalart-Allmaras, negative variant (SA-neg); freestream and inlets get nu~ = ratio * nu."""

    model: Literal["sa"] = "sa"
    nu_tilde_ratio: PositiveNumber = 3.0


class SST(Spec):
    """Menter k-omega SST; freestream and inlets get k from ``intensity``, omega from mu_t/mu."""

    model: Literal["sst"] = "sst"
    intensity: PositiveNumber = 0.001
    viscosity_ratio: PositiveNumber = 0.009


class DDES(Spec):
    """Delayed detached-eddy simulation on an SA or SST base model; needs an unsteady run."""

    model: Literal["ddes"] = "ddes"
    base: Literal["sa", "sst"] = "sst"
    nu_tilde_ratio: PositiveNumber = 3.0
    intensity: PositiveNumber = 0.001
    viscosity_ratio: PositiveNumber = 0.009


class WMLES(Spec):
    """Wall-modelled large-eddy simulation; needs an unsteady run."""

    model: Literal["wmles"] = "wmles"
    wall_model: Literal["equilibrium"] = "equilibrium"


Turbulence = Annotated[Inviscid | Laminar | SA | SST | DDES | WMLES, Field(discriminator="model")]
Precision = Literal["fp32+acc64", "fp64"]


def _air() -> IdealGas:
    from cfd.gases import AIR  # noqa: PLC0415 -- cfd.gases imports this module

    return AIR


class Compressible(Spec):
    """Density-based compressible flow; ``gas`` is the default medium of fluid zones."""

    kind: Literal["compressible"] = "compressible"
    turbulence: Turbulence
    precision: Precision = "fp32+acc64"
    gas: IdealGas = Field(default_factory=_air)


class Incompressible(Spec):
    """Pressure-based incompressible flow; ``fluid`` is the default medium of fluid zones."""

    kind: Literal["incompressible"] = "incompressible"
    turbulence: Turbulence
    precision: Precision = "fp32+acc64"
    fluid: ConstantDensityFluid


Physics = Annotated[Compressible | Incompressible, Field(discriminator="kind")]


class Freestream(Spec):
    mach: PositiveNumber | None = None
    velocity: Speed | None = None
    aoa: Angle = Field(default_factory=lambda: 0.0 * deg)
    sideslip: Angle = Field(default_factory=lambda: 0.0 * deg)
    temperature: Temperature | None = None
    pressure: Pressure | None = None
    re: PositiveNumber | None = None
    ref_length: Length | None = None


# --- time --------------------------------------------------------------------------------


class Steady(Spec):
    """Local time stepping until every residual drops ``residual_orders`` orders, at most
    ``max_iterations`` iterations."""

    kind: Literal["steady"] = "steady"
    max_iterations: Annotated[int, Field(ge=1)] = 10000
    residual_orders: PositiveNumber = 8.0


class Unsteady(Spec):
    """Time-accurate run of ``steps`` steps, with either a fixed ``dt`` or a ``cfl`` target."""

    kind: Literal["unsteady"] = "unsteady"
    steps: Annotated[int, Field(ge=1)]
    dt: Duration | None = None
    cfl: PositiveNumber | None = None
    average_from: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.dt is None) == (self.cfl is None):
            fail("E0003", "give exactly one of dt or cfl")
        if self.average_from is not None and self.average_from >= self.steps:
            fail("E0304", "average_from must be < steps")
        return self


Time = Annotated[Steady | Unsteady, Field(discriminator="kind")]


# --- boundaries, cell zones, monitors ----------------------------------------------------


class Wall(Spec):
    kind: Literal["wall"] = "wall"
    temperature: Temperature | None = None


class Slip(Spec):
    kind: Literal["slip"] = "slip"


class Symmetry(Spec):
    kind: Literal["symmetry"] = "symmetry"


class Farfield(Spec):
    kind: Literal["farfield"] = "farfield"


class Inlet(Spec):
    kind: Literal["inlet"] = "inlet"
    total_pressure: Pressure
    total_temperature: Temperature


class VelocityInlet(Spec):
    kind: Literal["velocity-inlet"] = "velocity-inlet"
    velocity: Speed


class Outlet(Spec):
    kind: Literal["outlet"] = "outlet"
    static_pressure: Pressure


Condition = Annotated[
    Wall | Slip | Symmetry | Farfield | Inlet | VelocityInlet | Outlet,
    Field(discriminator="kind"),
]


class BoundaryRule(Spec):
    zones: Pattern
    condition: Condition


class Fluid(Spec):
    kind: Literal["fluid"] = "fluid"
    material: FluidMaterial


class SolidZone(Spec):
    kind: Literal["solid"] = "solid"
    material: Solid


Medium = Annotated[Fluid | SolidZone, Field(discriminator="kind")]


class CellZoneRule(Spec):
    zones: Pattern
    medium: Medium


class Forces(Spec):
    """Integrated forces and moments over the boundary zones matching ``surfaces``."""

    kind: Literal["forces"] = "forces"
    name: Annotated[str, Field(min_length=1)] = "forces"
    surfaces: Pattern
    every: Annotated[int, Field(ge=1)] = 1
    ref_area: Area | None = None
    moment_center: tuple[Coordinate, Coordinate, Coordinate] | None = None


Monitor = Annotated[Forces, Field(discriminator="kind")]


# --- case --------------------------------------------------------------------------------


class CaseSpec(Spec):
    """A complete case document, ``cfd.case/1``."""

    schema_: Literal["cfd.case/1"] = Field(default="cfd.case/1", alias="schema")
    name: Annotated[str, Field(min_length=1)]
    mesh: Annotated[str, Field(min_length=1)]
    physics: Physics
    freestream: Freestream
    time: Time = Field(default_factory=Steady)
    boundaries: tuple[BoundaryRule, ...] = ()
    cell_zones: tuple[CellZoneRule, ...] = ()
    monitors: tuple[Monitor, ...] = ()

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        _raise_inconsistencies(self, _caller_location())

    __init__.__pydantic_base_init__ = True  # type: ignore[attr-defined]

    @model_validator(mode="after")
    def _fill_defaults(self) -> Self:
        if isinstance(self.physics, Compressible) and self.freestream.temperature is None:
            filled = self.freestream.model_copy(update={"temperature": 288.15 * K})
            object.__setattr__(self, "freestream", filled)
        return self

    @classmethod
    def from_document(cls, document: dict[str, Any], location: Location) -> CaseSpec:
        """Validate a JSON/TOML tree; every problem is reported as a coded diagnostic."""
        version = document.get("schema")
        if version != SCHEMA_ID:
            raise CaseError(
                [
                    diagnostic(
                        "E0004",
                        f"schema is {version!r}, expected {SCHEMA_ID!r}",
                        location=location.model_copy(update={"path": "schema"}),
                    )
                ]
            )
        try:
            spec = cls.model_validate(document)
        except ValidationError as error:
            raise CaseError(diagnostics_from(error, location)) from None
        _raise_inconsistencies(spec, location)
        return spec

    def document(self) -> dict[str, Any]:
        """The case as a JSON-compatible tree, defaults filled in."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def hash(self) -> str:
        """SHA-256 of the RFC 8785 canonical JSON (docs/contracts/case-schema.md#case-hash)."""
        return hashlib.sha256(rfc8785.dumps(self.document())).hexdigest()


def _raise_inconsistencies(spec: CaseSpec, location: Location) -> None:
    found = consistency(spec)
    if found:
        raise CaseError(
            diagnostic(code, message, location=location.model_copy(update={"path": path}))
            for code, path, message in found
        )


def consistency(spec: CaseSpec) -> list[tuple[str, str, str]]:
    """Rules spanning several fields, as (code, document path, message)."""
    found: list[tuple[str, str, str]] = []
    fs, physics = spec.freestream, spec.physics
    if isinstance(physics, Compressible):
        if fs.mach is None:
            found.append(("E0306", "freestream.mach", "compressible freestream needs mach"))
        if fs.velocity is not None:
            found.append(
                ("E0306", "freestream.velocity", "compressible freestream takes mach, not velocity")
            )
        if (fs.pressure is None) == (fs.re is None):
            found.append(
                (
                    "E0306",
                    "freestream",
                    "compressible freestream needs exactly one of pressure or re",
                )
            )
        if fs.re is not None and fs.ref_length is None:
            found.append(("E0306", "freestream.ref_length", "re needs ref_length"))
    else:
        if fs.velocity is None:
            found.append(
                ("E0306", "freestream.velocity", "incompressible freestream needs velocity")
            )
        found.extend(
            ("E0306", f"freestream.{key}", f"incompressible freestream takes no {key}")
            for key in ("mach", "re", "pressure", "temperature")
            if getattr(fs, key) is not None
        )
    if isinstance(physics.turbulence, DDES | WMLES) and not isinstance(spec.time, Unsteady):
        found.append(
            ("E0305", "time", f"{physics.turbulence.model} needs case.time = cfd.Unsteady(...)")
        )
    return found
