# Contract: case schema

A case is a typed document that says what to compute on which mesh. Python code builds it (`cfd.Case`), TOML and JSON store it, the UI and agents edit it, the solver consumes it. Schema identifier: `cfd.case/1`. The JSON Schema is generated from the SDK (`cfd schema case`) and is the machine-readable form of this page.

## Example

The CRM-HL case of `docs/design/dx.html`, as Python and as the TOML it serializes to:

```python
import cfd
from cfd.units import deg, m

case = cfd.Case("crm-hl", mesh="crm-hl.cgns")
case.physics = cfd.Compressible(turbulence=cfd.SA(), precision="fp32+acc64")
case.freestream(mach=0.20, aoa=19.57 * deg, re=5.49e6, ref_length=7.0 * m)
case.boundary("wing.*").wall()
case.boundary("symmetry").symmetry()
case.monitor(cfd.forces("wing.*"), every=10)
```

```toml
schema = "cfd.case/1"
name = "crm-hl"
mesh = "crm-hl.cgns"

[physics]
kind = "compressible"
precision = "fp32+acc64"
turbulence = { model = "sa" }
gas = { kind = "ideal-gas", name = "air", molar_mass = "0.0289647 kg / mol", gamma = 1.4, prandtl = 0.72, viscosity = { law = "sutherland", mu_ref = "1.716e-05 Pa * s", t_ref = "273.15 K", s = "110.4 K" } }

[freestream]
mach = 0.2
aoa = "19.57 deg"
sideslip = "0.0 deg"
re = 5490000.0
ref_length = "7.0 m"
temperature = "288.15 K"

[time]
kind = "steady"

[[boundaries]]
zones = "wing.*"
condition = { kind = "wall" }

[[boundaries]]
zones = "symmetry"
condition = { kind = "symmetry" }

[[monitors]]
kind = "forces"
name = "forces"
surfaces = "wing.*"
every = 10
```

## Quantities

A dimensional value is written as a string `"<number> <unit>"`, for example `"7.0 m"`, `"101325.0 Pa"`, `"1.716e-05 Pa * s"`. Units are parsed by [Pint](https://pint.readthedocs.io) with its default registry. Each field declares a dimension; a value of another dimension is rejected (`E0301`), a bare number in a dimensional field is rejected (`E0302`), an unknown unit is rejected (`E0303`). Angles need an angle unit (`deg`, `rad`): a bare `19.57` is ambiguous and rejected. Dimensionless fields (`mach`, `re`, `gamma`, `prandtl`) are plain numbers.

Values keep the unit they were written in; consumers convert to SI. Fields marked *> 0* reject zero and negative values (`E0304`); temperatures are checked after conversion to kelvin.

## Tree

Every table rejects unknown keys. Discriminated unions are tagged by `kind` (or `model`, `law`).

### Case

| Key | Type | Default |
|---|---|---|
| `schema` | `"cfd.case/1"` | required (`E0004` otherwise) |
| `name` | non-empty string | required |
| `mesh` | path, relative to the case file | required |
| `physics` | [Physics](#physics) | required |
| `freestream` | [Freestream](#freestream) | required |
| `time` | [Time](#time) | `{kind = "steady"}` |
| `boundaries` | list of [BoundaryRule](#boundaries) | `[]` |
| `cell_zones` | list of [CellZoneRule](#cell-zones) | `[]` |
| `monitors` | list of [Monitor](#monitors) | `[]` |

### Physics

`kind = "compressible"`: density-based solver.

| Key | Type | Default |
|---|---|---|
| `turbulence` | [Turbulence](#turbulence) | required |
| `precision` | `"fp32+acc64"` · `"fp64"` | `"fp32+acc64"` |
| `gas` | [IdealGas](#materials) | air |

`kind = "incompressible"`: pressure-based solver.

| Key | Type | Default |
|---|---|---|
| `turbulence` | [Turbulence](#turbulence) | required |
| `precision` | `"fp32+acc64"` · `"fp64"` | `"fp32+acc64"` |
| `fluid` | [ConstantDensityFluid](#materials) | required |

`precision = "fp32+acc64"` computes in FP32 and accumulates solution updates, norms and geometry in FP64 (see `docs/PLAN.md`). `"fp64"` computes everything in FP64.

`gas` and `fluid` are the default medium of every fluid cell zone.

### Turbulence

| `model` | Options | Notes |
|---|---|---|
| `"inviscid"` | — | Euler equations. |
| `"laminar"` | — | Navier-Stokes without a turbulence model. |
| `"sa"` | — | Spalart-Allmaras, negative variant (SA-neg). |
| `"sst"` | — | Menter k-ω SST. |
| `"ddes"` | `base = "sa"` · `"sst"` (default `"sst"`) | Delayed detached-eddy simulation. Needs `time.kind = "unsteady"` (`E0305`). |
| `"wmles"` | `wall_model = "equilibrium"` (default) | Wall-modelled LES. Needs `time.kind = "unsteady"` (`E0305`). |

### Freestream

The freestream state defines farfield conditions and the initial state. It is expressed with the physics' default medium.

| Key | Type | Compressible | Incompressible |
|---|---|---|---|
| `mach` | number > 0 | required | forbidden |
| `velocity` | speed > 0 | forbidden | required |
| `aoa` | angle | default `"0.0 deg"` | default `"0.0 deg"` |
| `sideslip` | angle | default `"0.0 deg"` | default `"0.0 deg"` |
| `temperature` | temperature > 0 | default `"288.15 K"` | forbidden |
| `pressure` | pressure > 0 | exactly one of `pressure` or `re` | forbidden |
| `re` | number > 0, Reynolds number based on `ref_length` | exactly one of `pressure` or `re` | forbidden |
| `ref_length` | length > 0 | required with `re` | optional |

A violation of this table is `E0306`. With `re`, the static pressure follows from the ideal-gas law and the gas viscosity at `temperature`.

The flow direction in mesh axes is (cos α cos β, −sin β, sin α cos β) with α = `aoa`, β = `sideslip`.

### Time

| `kind` | Keys |
|---|---|
| `"steady"` | — (local time stepping) |
| `"unsteady"` | `steps` integer ≥ 1; exactly one of `dt` (time > 0) or `cfl` (number > 0); `average_from` integer in [0, `steps`), optional |

### Boundaries

A rule applies a condition to every boundary zone whose name matches `zones`, a glob pattern (`*`, `?`, `[…]`, case-sensitive, matched against the whole name).

| `condition.kind` | Keys | Physics |
|---|---|---|
| `"wall"` | `temperature` (temperature > 0), optional: isothermal if set, adiabatic otherwise | viscous; with `inviscid` turbulence use `slip` (`E0414`) |
| `"slip"` | — | all |
| `"symmetry"` | — | all |
| `"farfield"` | — (uses the freestream) | all |
| `"inlet"` | `total_pressure` (pressure > 0), `total_temperature` (temperature > 0) | compressible (`E0414` otherwise) |
| `"velocity-inlet"` | `velocity` (speed > 0) | incompressible (`E0414` otherwise) |
| `"outlet"` | `static_pressure` (pressure > 0) | all |

Inlet flow is normal to the boundary face.

### Cell zones

A rule assigns a medium to every cell zone whose name matches `zones` (same glob rules).

| `medium.kind` | Keys |
|---|---|
| `"fluid"` | `material`: an [IdealGas](#materials) (compressible) or a [ConstantDensityFluid](#materials) (incompressible); `E0504` otherwise |
| `"solid"` | `material`: a [Solid](#materials) |

### Materials

Materials are stored in full, never by reference to a library entry, so the case hash covers every property. `cfd.gases` and `cfd.materials` provide presets with cited sources.

| `kind` | Keys |
|---|---|
| `"ideal-gas"` | `name`; `molar_mass` (kg/mol-dimension > 0); `gamma` > 1; `prandtl` > 0; `viscosity`: `{law = "sutherland", mu_ref, t_ref, s}` or `{law = "constant", mu}` (dynamic viscosity > 0, temperatures > 0) |
| `"constant-density-fluid"` | `name`; `density` > 0; `viscosity` (dynamic viscosity > 0) |
| `"solid"` | `name`; `density` > 0; `specific_heat` > 0; `conductivity` > 0 |

### Monitors

| `kind` | Keys |
|---|---|
| `"forces"` | `name` (unique among monitors, default `"forces"`, `E0602`); `surfaces` glob over boundary zones; `every` integer ≥ 1, default 1; `ref_area` area > 0, optional; `moment_center` three lengths, default origin |

Coefficients are reported only when `ref_area` is set; moment coefficients use `freestream.ref_length`. The recorded quantities are listed in [results.md](results.md#history).

## Boundary resolution

The solver needs one condition per boundary zone of the mesh. The SDK resolves them from the case and the mesh's [MeshInfo](mesh-info.md), and records where each comes from (its **provenance**):

1. Each rule must match at least one zone (`E0410`). A zone matched by two rules is ambiguous (`E0411`): narrow the patterns.
2. A zone matched by a rule takes the rule's condition. Provenance `code`. If the mesh typed the zone with a different kind, the override is reported (`W0415`).
3. Otherwise the zone's mesh type applies. Provenance `mesh`. `wall` becomes an adiabatic wall, `slip`, `symmetry` and `farfield` map one to one. `inlet` and `outlet` need values that only the case can give (`E0413`).
4. Otherwise the zone has no condition (`E0412`), unless it has no faces.

Cell zones resolve the same way: rules (`E0501` no match, `E0502` ambiguous), then the mesh type (`fluid` takes the physics' default medium; `solid` needs a material, `E0503`), then, for a zone the mesh leaves untyped, a fluid with the physics' default medium (provenance `default`). Two fluid zones with different materials that share an interface would need species transport, which `cfd.case/1` does not provide (`E0505`).

Monitor surfaces must match at least one boundary zone (`E0601`).

`cfd plan` prints this resolution; `cfd plan --json` returns it.

## Case hash

The case hash is the SHA-256 of the case's canonical JSON: the document with defaults filled in, keys sorted, no insignificant whitespace (`separators=(",", ":")`), UTF-8, numbers in shortest round-trip form. Two documents that differ only in formatting or key order have the same hash; writing the same value in another unit changes it.

## Diagnostics

Every problem is a diagnostic with a code, a location, a suggested fix and a documentation page. The registry and its generated reference are in `docs/sdk/errors.md`, generated from `cfd.errors`.
