"""``autan.solver-input/1``: the resolved, SI-only input of the solver runtime ``autan-run``.

The SDK builds it after a plan without errors (docs/contracts/solver-input.md): every
boundary and cell zone resolved, every quantity converted to SI, the freestream state
closed. The runtime applies no unit, default or resolution rule of its own.
"""

from __future__ import annotations

import math
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from autan.errors import CaseError
from autan.results import Devices  # noqa: TC001 -- pydantic resolves field types at runtime
from autan.schema import (
    DDES,
    SA,
    SST,
    WMLES,
    CaseSpec,
    Compressible,
    ConstantDensityFluid,
    ConstantViscosity,
    Fluid,
    IdealGas,
    Medium,
    Solid,
    SolidZone,
    Steady,
    Sutherland,
)
from autan.units import parse, si

if TYPE_CHECKING:
    from autan.case import Case
    from autan.meshinfo import MeshInfo

SCHEMA_ID = "autan.solver-input/1"
GAS_CONSTANT = 8.314462618  # J/(mol K), CODATA 2018 exact value

Number = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Vector = tuple[Number, Number, Number]
_MEDIUM: TypeAdapter[Fluid | SolidZone] = TypeAdapter(Medium)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseRef(_Strict):
    name: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    toml: str


class MeshRef(_Strict):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SutherlandLaw(_Strict):
    law: Literal["sutherland"] = "sutherland"
    mu_ref: Positive
    t_ref: Positive
    s: Positive


class ConstantLaw(_Strict):
    law: Literal["constant"] = "constant"
    mu: Positive


class Gas(_Strict):
    kind: Literal["ideal-gas"] = "ideal-gas"
    name: str
    gas_constant: Positive
    gamma: Annotated[float, Field(gt=1)]
    prandtl: Positive
    viscosity: Annotated[SutherlandLaw | ConstantLaw, Field(discriminator="law")]


class Liquid(_Strict):
    kind: Literal["constant-density-fluid"] = "constant-density-fluid"
    name: str
    density: Positive
    viscosity: Positive


class SolidMedium(_Strict):
    kind: Literal["solid"] = "solid"
    name: str
    density: Positive
    specific_heat: Positive
    conductivity: Positive


MediumInput = Annotated[Gas | Liquid | SolidMedium, Field(discriminator="kind")]


class TurbulenceInput(_Strict):
    model: Literal["inviscid", "laminar", "sa", "sst", "ddes", "wmles"]
    base: Literal["sa", "sst"] | None
    wall_model: Literal["equilibrium"] | None
    nu_tilde_ratio: Positive | None
    intensity: Positive | None
    viscosity_ratio: Positive | None


class PhysicsInput(_Strict):
    kind: Literal["compressible", "incompressible"]
    turbulence: TurbulenceInput


class FreestreamInput(_Strict):
    density: Positive
    pressure: Number
    temperature: Positive | None
    velocity: Vector
    speed: Positive
    viscosity: Positive
    mach: Positive | None
    re: Positive | None
    ref_length: Positive | None
    aoa: Number
    sideslip: Number
    nu_tilde: Positive | None
    k: Positive | None
    omega: Positive | None


class BoundaryInput(_Strict):
    zone: str
    kind: Literal["wall", "slip", "symmetry", "farfield", "inlet", "velocity-inlet", "outlet"]
    temperature: Positive | None
    total_pressure: Positive | None
    total_temperature: Positive | None
    velocity: Positive | None
    static_pressure: Positive | None


class CellZoneInput(_Strict):
    zone: str
    medium: MediumInput


class SteadyInput(_Strict):
    kind: Literal["steady"] = "steady"
    max_iterations: Annotated[int, Field(ge=1)]
    residual_orders: Positive


class UnsteadyInput(_Strict):
    kind: Literal["unsteady"] = "unsteady"
    steps: Annotated[int, Field(ge=1)]
    dt: Positive | None
    cfl: Positive | None
    average_from: Annotated[int, Field(ge=0)] | None


class ForcesInput(_Strict):
    kind: Literal["forces"] = "forces"
    name: str
    every: Annotated[int, Field(ge=1)]
    zones: tuple[str, ...]
    ref_area: Positive | None
    ref_length: Positive | None
    moment_center: Vector


class SolverInput(_Strict):
    """The input document of ``autan-run`` (docs/contracts/solver-input.md); SI throughout."""

    schema_: Literal["autan.solver-input/1"] = Field(default="autan.solver-input/1", alias="schema")
    case: CaseRef
    mesh: MeshRef
    precision: Literal["fp32+acc64", "fp64"]
    deterministic: bool
    devices: Devices
    physics: PhysicsInput
    freestream: FreestreamInput
    boundaries: tuple[BoundaryInput, ...]
    cell_zones: tuple[CellZoneInput, ...]
    time: Annotated[SteadyInput | UnsteadyInput, Field(discriminator="kind")]
    monitors: tuple[ForcesInput, ...]

    model_config = ConfigDict(populate_by_name=True)

    def document(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def viscosity(gas: IdealGas, temperature: float) -> float:
    """Dynamic viscosity of ``gas`` at ``temperature`` (K), in Pa s."""
    law = gas.viscosity
    if isinstance(law, ConstantViscosity):
        return si(law.mu, "Pa * s")
    assert isinstance(law, Sutherland)
    t_ref, s = si(law.t_ref, "K"), si(law.s, "K")
    # Sutherland (1893): mu = mu_ref (T / T_ref)^(3/2) (T_ref + S) / (T + S)
    mu_ref = si(law.mu_ref, "Pa * s")
    return mu_ref * math.pow(temperature / t_ref, 1.5) * (t_ref + s) / (temperature + s)


def gas_constant(gas: IdealGas) -> float:
    """Specific gas constant R = R_u / M, in J/(kg K)."""
    return GAS_CONSTANT / si(gas.molar_mass, "kg / mol")


def freestream_state(spec: CaseSpec) -> FreestreamInput:
    """Close the freestream state of case-schema.md#freestream in SI."""
    fs, physics = spec.freestream, spec.physics
    alpha, beta = si(fs.aoa, "rad"), si(fs.sideslip, "rad")
    direction = (
        math.cos(alpha) * math.cos(beta),
        0.0 - math.sin(beta),
        math.sin(alpha) * math.cos(beta),
    )
    temperature: float | None
    if isinstance(physics, Compressible):
        assert fs.temperature is not None
        assert fs.mach is not None
        gas = physics.gas
        temperature = si(fs.temperature, "K")
        r = gas_constant(gas)
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
    else:
        assert fs.velocity is not None
        temperature, pressure = None, 0.0
        density = si(physics.fluid.density, "kg / m**3")
        mu = si(physics.fluid.viscosity, "Pa * s")
        speed = si(fs.velocity, "m / s")
    turbulence = _turbulence(spec)
    nu_tilde = k = omega = None
    if turbulence.nu_tilde_ratio is not None:
        nu_tilde = turbulence.nu_tilde_ratio * mu / density
    if turbulence.intensity is not None and turbulence.viscosity_ratio is not None:
        k = 1.5 * (turbulence.intensity * speed) ** 2
        omega = density * k / (turbulence.viscosity_ratio * mu)
    return FreestreamInput(
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=(speed * direction[0], speed * direction[1], speed * direction[2]),
        speed=speed,
        viscosity=mu,
        mach=fs.mach,
        re=fs.re,
        ref_length=si(fs.ref_length, "m") if fs.ref_length is not None else None,
        aoa=alpha,
        sideslip=beta,
        nu_tilde=nu_tilde,
        k=k,
        omega=omega,
    )


def _turbulence(spec: CaseSpec) -> TurbulenceInput:
    turbulence = spec.physics.turbulence
    base = turbulence.base if isinstance(turbulence, DDES) else turbulence
    return TurbulenceInput(
        model=turbulence.model,
        base=turbulence.base.model if isinstance(turbulence, DDES) else None,
        wall_model=turbulence.wall_model if isinstance(turbulence, WMLES) else None,
        nu_tilde_ratio=base.nu_tilde_ratio if isinstance(base, SA) else None,
        intensity=base.intensity if isinstance(base, SST) else None,
        viscosity_ratio=base.viscosity_ratio if isinstance(base, SST) else None,
    )


def _medium(medium: Fluid | SolidZone) -> MediumInput:
    material = medium.material
    if isinstance(material, Solid):
        return SolidMedium(
            name=material.name,
            density=si(material.density, "kg / m**3"),
            specific_heat=si(material.specific_heat, "J / kg / K"),
            conductivity=si(material.conductivity, "W / m / K"),
        )
    if isinstance(material, ConstantDensityFluid):
        return Liquid(
            name=material.name,
            density=si(material.density, "kg / m**3"),
            viscosity=si(material.viscosity, "Pa * s"),
        )
    law = material.viscosity
    viscosity_law: SutherlandLaw | ConstantLaw = (
        SutherlandLaw(mu_ref=si(law.mu_ref, "Pa * s"), t_ref=si(law.t_ref, "K"), s=si(law.s, "K"))
        if isinstance(law, Sutherland)
        else ConstantLaw(mu=si(law.mu, "Pa * s"))
    )
    return Gas(
        name=material.name,
        gas_constant=gas_constant(material),
        gamma=material.gamma,
        prandtl=material.prandtl,
        viscosity=viscosity_law,
    )


def _si(condition: dict[str, Any], key: str, unit: str) -> float | None:
    return float(parse(condition[key], unit).to(unit).magnitude) if key in condition else None


def build(case: Case, mesh_info: MeshInfo, *, devices: Devices, deterministic: bool) -> SolverInput:
    """The solver input of ``case`` on ``mesh_info``; plan errors raise ``CaseError``."""
    plan = case.plan(mesh_info)
    if not plan.ok:
        raise CaseError(d for d in plan.diagnostics if d.severity == "error")
    spec = case.spec
    boundaries = []
    for resolution in plan.boundaries:
        condition = resolution.condition
        if condition is None:
            continue
        boundaries.append(
            BoundaryInput(
                zone=resolution.zone,
                kind=condition["kind"],
                temperature=_si(condition, "temperature", "K"),
                total_pressure=_si(condition, "total_pressure", "Pa"),
                total_temperature=_si(condition, "total_temperature", "K"),
                velocity=_si(condition, "velocity", "m / s"),
                static_pressure=_si(condition, "static_pressure", "Pa"),
            )
        )
    cell_zones = tuple(
        CellZoneInput(zone=z.zone, medium=_medium(_MEDIUM.validate_python(z.medium)))
        for z in plan.cell_zones
    )
    zone_names = [b.zone for b in plan.boundaries]
    monitors = tuple(
        ForcesInput(
            name=m.name,
            every=m.every,
            zones=tuple(n for n in zone_names if fnmatchcase(n, m.surfaces)),
            ref_area=si(m.ref_area, "m**2") if m.ref_area is not None else None,
            ref_length=(
                si(spec.freestream.ref_length, "m")
                if spec.freestream.ref_length is not None
                else None
            ),
            moment_center=(
                si(m.moment_center[0], "m"),
                si(m.moment_center[1], "m"),
                si(m.moment_center[2], "m"),
            ),
        )
        for m in spec.monitors
    )
    time: SteadyInput | UnsteadyInput = (
        SteadyInput(
            max_iterations=spec.time.max_iterations, residual_orders=spec.time.residual_orders
        )
        if isinstance(spec.time, Steady)
        else UnsteadyInput(
            steps=spec.time.steps,
            dt=si(spec.time.dt, "s") if spec.time.dt is not None else None,
            cfl=spec.time.cfl,
            average_from=spec.time.average_from,
        )
    )
    return SolverInput(
        case=CaseRef(name=spec.name, sha256=spec.hash(), toml=case.to_toml()),
        mesh=MeshRef(path=str(case.mesh_path), sha256=mesh_info.source.sha256),
        precision=spec.physics.precision,
        deterministic=deterministic,
        devices=devices,
        physics=PhysicsInput(kind=spec.physics.kind, turbulence=_turbulence(spec)),
        freestream=freestream_state(spec),
        boundaries=tuple(boundaries),
        cell_zones=cell_zones,
        time=time,
        monitors=monitors,
    )
