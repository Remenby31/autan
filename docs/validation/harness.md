# Verification harness

`autan-vv verify` reads solver runs in the [results contract](../contracts/results.md), compares them with an [oracle](oracles.md), fits the observed order over a refinement family and, with `--gate`, gives the verdict of a gate of [acceptance.md](acceptance.md).

```sh
autan-vv verify runs/run-A runs/run-B runs/run-C --against mms-euler --gate euler-mms-order --out verdict.json
autan-vv verify runs/run-X --against freestream --gate freestream-preservation
```

The command prints the worst error of each run, the L2 order of each field and the gate result, and exits with status 1 when the gate fails and 2 on a coded error. `--out` writes the full report.

## What it does

1. Opens each run with `autan.Results.open`, which checks the directory against the contract.
2. Refuses a run that cannot be compared with the oracle (`V0724`): status other than `converged` or `completed`; physics other than the oracle's (Euler ↔ `Inviscid`, laminar NS ↔ `Laminar`, RANS-SA ↔ `SA`, incompressible oracles ↔ incompressible cases); γ or gas constant different from the oracle's; an unsteady oracle and a run without physical time.
3. Rebuilds the cell geometry from `solution.cgns` (`NGON_n` faces, `NFACE_n` cells with signed face ids) with the same conventions as the core — closure is checked — and evaluates the oracle at the cell centroids, at the run's physical time.
4. Computes, for every field the run and the oracle share, plus the conserved variables ρ, ρu, ρv, ρw, ρE and the entropy p/ρ^γ for gases, the volume-weighted L1 and L2 errors and the L∞ error.
5. With several runs (distinct meshes, one precision: `V0726`, `V0727`), fits the order of every norm of every field over h. h is the cell measure along the oracle's axes to the power 1/(number of axes); a mesh must be one cell thick along the other directions (`V0723`).
6. With `--gate`, checks the gate (`V0728` for a gate of another oracle, `V0729` for an order gate with fewer than three runs): an `order` gate requires both the fitted order and the finest-pair order of every listed field inside its interval; a `max-error` gate bounds the relative L∞ error by a multiple of the run precision's epsilon on every run.

`freestream` is built from each run's case (case-schema.md, Freestream): ρ from the ideal-gas law, or from Re and the viscosity law at the freestream temperature; speed M√(γRT); direction (cos α cos β, −sin β, sin α cos β).

## Report (`autan.vv-verdict/1`)

`oracle` (name, physics, `reference_only`, parameters); `runs` (path, run id, identity, status, backend, solver commit, precision, case and mesh hashes, physical time, cell count, volume, h, and per field `L1`, `L2`, `Linf`, `scale`); `fits` (per field and norm: order, pairwise orders, R²); `verdict` (gate, milestone, checks with measured values and bounds, `pass`, `evidence`, `not_evidence_because`).

**Evidence.** `evidence` is false whenever a run comes from the `fake` backend or the oracle is a numerical reference (`reference_only`; none is registered today): the verdict is then computed but proves nothing. A PR quotes a verdict as evidence only when `evidence` is true.

## How it is tested

`vv/tests/test_harness.py`:

- runs written by `autan.FakeBackend` (the SDK's producer), checked against the freestream computed independently from case-schema.md: the gate passes and is marked not evidence;
- runs written through `autan`'s own writers (`autan_vv.runs`) on hex, polyhedral and hybrid families, read back by `autan.Results.open`: the geometry rebuilt from `solution.cgns` equals the mesh's to 10⁻¹³; fields equal to an oracle times (1 + C h^p g(x)) give the order p to ±0.02, and the order gate passes for p = 2 and fails for p = 1;
- every coded refusal, the CLI exit status, the strict-JSON report, and that every gate in the code is stated, with its numbers, in acceptance.md.

## Limits and requests

- The real solver cannot yet be told to add a manufactured source or to impose exact boundary data: the case schema has no such field. A `verification` table naming the manufactured solution (and its `autan.mms/1` hash) is requested from the case-schema owner.
- The harness compares every cell. Oracles defined on part of a mesh (Blasius upstream of the leading edge) need meshes restricted to their domain; the oracle refuses other points (`V0708`).
- Time orders (BDF2, Runge–Kutta) need a family refined in time at fixed mesh; the report fits over h only.
