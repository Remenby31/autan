# Verification harness

`autan-vv verify` reads solver runs in the [results contract](../contracts/results.md), compares them with an [oracle](oracles.md), fits the observed order over a refinement family and, with `--gate`, gives the verdict of a gate of [acceptance.md](acceptance.md).

```sh
autan-vv verify runs/run-A runs/run-B runs/run-C runs/run-D --against mms-euler --gate euler-mms-order --out verdict.json
autan-vv verify runs/run-X --against freestream --gate freestream-preservation
```

The command prints the worst error of each run, the L2 order of each field and the gate result, and exits with status 1 when the gate fails and 2 on a coded error. `--out` writes the full report.

## What it does

1. Opens each run with `autan.Results.open`, which checks the directory against the contract.
2. Refuses a run that cannot be compared with the oracle (`V0724`): status other than `converged` or `completed`; physics other than the oracle's (Euler ↔ `Inviscid`, laminar NS ↔ `Laminar`, RANS-SA ↔ `SA`, incompressible oracles ↔ incompressible cases); γ or gas constant different from the oracle's; for viscous oracles, a viscosity law or Prandtl number different from the oracle's, in the default gas and in every cell-zone material; solid zones; for Poiseuille, anything but a laminar incompressible case with the oracle's density and viscosity; an unsteady oracle and a run without physical time. Several runs must form one refinement family (`V0726`, `V0727`): the same case apart from its mesh and name, the same solver commit and backend, the same precision, the same physical time, the same domain, distinct meshes.
3. Reads the fields through `autan.Results` (which checks `solution.cgns` against the contract), rebuilds the cell geometry from its `NGON_n` faces and `NFACE_n` cells with signed face ids, with the core's conventions and a closure check, refuses fields that do not hold one value per cell (`V0721`), and evaluates the oracle at the cell centroids, where results.md locates the cell values, at the run's physical time.
4. Computes, for every field the run and the oracle share, plus the conserved variables ρ, ρu, ρv, ρw, ρE and the entropy p/ρ^γ for gases, the volume-weighted L1 and L2 errors and the L∞ error.
5. Fits the order of every norm of every field over h, using only the levels whose relative error is at least 100 times the rounding floor of the precision. h is the cell measure along the oracle's axes to the power 1/(number of axes); along the other directions every cell must span the whole mesh, one layer (`V0723`). `autan_vv.meshes.families.cartesian` builds such one-layer meshes.
6. With `--gate`, checks the gate (`V0728` for an unknown gate or one of another oracle, `V0729` for an order gate with fewer than four runs). Prerequisites first: some gates fix the physical time and the domain (the vortex: t = 40, [−10, 10]²). An `order` gate then requires, for every listed field, four levels above the rounding floor and both the fitted and the finest-pair order finite and inside its interval — for `euler-mms-order`, at least its lower bound, an order above the upper bound being a warning (above 2.6, the verdict asks for investigation), with errors decreasing strictly from level to level; a field the oracle makes identically zero (ρw of a planar flow) must instead stay zero to 64 ε of its vector's magnitude. A `max-error` gate bounds the relative error by a multiple of the precision's epsilon on every run, a vector component being measured against the whole vector; `freestream-preservation` checks every field the run writes, turbulence included. Errors are relative to each field's magnitude.

`freestream` is built from the case (case-schema.md, Freestream and Physics): ρ from the ideal-gas law, or from Re and the viscosity law at the freestream temperature; speed M√(γRT); direction (cos α cos β, −sin β, sin α cos β); SA ν̃ = ratio·ν; SST k = 1.5 (I V)², ω = ρk/(ratio·μ).

## Report (`autan.vv-verdict/1`)

`oracle` (name, physics, `reference_only`, parameters); `runs` (path, run id, identity, status, backend, solver commit, precision, case and mesh hashes, physical time, cell count, volume, h, bounds, and per field `L1`, `L2`, `Linf`, `scale`); `fits` (per field and norm: levels used and excluded, order, pairwise orders, R²); `verdict` (gate, milestone, prerequisite and field checks with measured values and bounds, `pass`, `warnings`, `investigate`, `evidence`, `not_evidence_because`). The report is strict JSON: a measurement that is not finite is written `null`; it is never dropped from a fit and fails its check.

**Evidence.** `evidence` is false whenever a run comes from the `fake` backend or the oracle is a numerical reference (`reference_only`; none is registered today): the verdict is then computed but proves nothing. A PR quotes a verdict as evidence only when `evidence` is true.

## How it is tested

`vv/tests/test_harness.py`:

- runs written by `autan.FakeBackend` (the SDK's producer), checked against the freestream computed independently from case-schema.md: the gate passes and is marked not evidence;
- runs written through `autan`'s own writers (`autan_vv.runs`, non-deterministic fixtures that refuse monitors and averaging) on hex, polyhedral and hybrid families, read back by `autan.Results.open`: the geometry rebuilt from `solution.cgns` equals the mesh's to 10⁻¹³; fields equal to an oracle times (1 + C h^p g(x)) give the order p to ±0.02, and the order gate passes for p = 2 and fails for p = 1;
- the vortex gate on one-layer meshes: the identically zero ρw passes as preserved, and a run at the wrong time fails its prerequisite; FP32 levels under the rounding floor do not count; a wrong freestream temperature fails; a Sutherland case is refused by the constant-viscosity Couette oracle; a multi-layer thin mesh is refused by a 2D oracle;
- every coded refusal, mixed families (another Mach number, the same run twice), the CLI exit statuses (1 for a failed gate, 2 for a coded error, including the SDK's), the strict-JSON report, and that every gate in the code is stated, with its numbers, in acceptance.md.

## Limits and requests

- Manufactured cases on the real solver use the case schema's `[verification]` table (path and SHA-256 of an `autan.mms/1` file), being added by the SDK & contracts owner.
- The Couette gate checks the temperature; the wall heat flux joins it once results.md carries it.
- The harness compares every cell. Oracles defined on part of a mesh (Blasius upstream of the leading edge) need meshes restricted to their domain; the oracle refuses other points (`V0708`).
- Time orders (BDF2, Runge–Kutta) need a family refined in time at fixed mesh; the report fits over h only.
