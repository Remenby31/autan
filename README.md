# cfd-solver

A GPU-native finite-volume solver for compressible flow on unstructured meshes of arbitrary polyhedra, with a discrete adjoint on the GPU and a typed Python SDK.

> **Status: pre-alpha. Nothing runs yet.** The repository holds the plan, the research behind it, the contracts between components and the first foundations (milestone M0). There is no release, no installable package and no stable API. Do not use it for engineering work.

## Why

No open, portable code does unstructured, compressible CFD on GPUs together with a discrete adjoint. Commercial GPU solvers leave out the density-based solver or the adjoint, or both. The open codes keep their adjoint on the CPU. New NVIDIA GPUs also make FP64 rare. This project sets out to:

- run the whole solver on the GPU, NVIDIA and AMD, from code generated once per kernel;
- compute in FP32 with FP64 accumulation, and prove in CI that the answer does not depend on it;
- differentiate the generated kernels to get the adjoint on the GPU;
- publish its verification and benchmark numbers with every change.

The full argument, with sources, is in [the state of the art](docs/research/state-of-the-art-2026-10.md). The design documents are in French; code, contracts and reference documentation are in English.

## Done and planned

| Milestone | Content | State |
|---|---|---|
| M0 Foundations | Face-based polyhedral mesh, Gmsh, Fluent and CGNS readers, partitioning, CI, benchmark harness, V&V framework | In progress |
| M1 Generator spike | Kernel DSL to CUDA, HIP and CPU; gradients; scalar advection-diffusion; adjoint of one kernel | Planned |
| M2 Compressible Euler | Roe, HLLC and AUSM+-up fluxes; explicit then implicit solver | Planned |
| M3 Navier-Stokes + SA | Viscous fluxes, Spalart-Allmaras, agglomeration multigrid | Planned |
| M4 SDK + CLI v0 | Typed case schema with units, `cfd plan`, `run`, `logs`, `diff`, coded errors | Planned |
| M5 Multi-GPU and precision | GPU-aware MPI, FP32 versus FP64 study | Planned |
| M6 GPU discrete adjoint | Adjoint of the generated kernels, shape optimisation | Planned |
| M7–M11 | SST, DDES, WMLES; pressure-based solver; multi-gas and conjugate heat transfer; web UI; AI-native workflows | Planned |

Each milestone closes on numbered exit criteria. They are in [the plan](docs/PLAN.md), with the risks and the critical path.

## Architecture

```
core/      face-based mesh, zones, partitioning, halos, FP64 geometry
io/        Gmsh, Fluent .msh, CGNS and OpenFOAM polyMesh readers; CGNS and VTK writers
kernels/   DSL sources of the face and cell kernels (fluxes, gradients, limiters, boundary conditions)
codegen/   generator to CUDA, HIP and CPU, with tangent and adjoint versions
solvers/   time stepping, Newton-Krylov, block-sparse 5×5 systems, agglomeration multigrid
physics/   flux schemes and turbulence models
py/        Python SDK, case schema and `cfd` CLI
vv/        manufactured solutions, observed-order tools, reference cases
bench/     bandwidth and time-per-cell benchmarks on DGX Spark
```

Most of these directories do not exist yet. Components talk through the [contracts](docs/contracts/): the [case schema](docs/contracts/case-schema.md), the [MeshInfo summary](docs/contracts/mesh-info.md) and the [results layout](docs/contracts/results.md).

## How correctness and speed are proven

Every change carries its own evidence. The rules are in [AGENTS.md](AGENTS.md#non-negotiables); in short:

- **Correctness.** A change to the numerics shows the observed order of accuracy from manufactured solutions, on tetrahedral, hexahedral, polyhedral and hybrid meshes, in FP32 and FP64. Expected values come from analytic solutions, conservation laws or published reference data, never from the solver itself.
- **Performance.** A kernel change reports the achieved bandwidth as a percentage of the measured peak of the GPU, and the time per cell per iteration, from the benchmark harness on a DGX Spark. A regression above 5 % blocks the merge.
- **Reproducibility.** Every run records the mesh hash, the case hash, the solver commit and the precision. A bitwise-deterministic mode stays available.

Development hardware is two NVIDIA DGX Spark (GB10). Production targets are B200/GB200 and MI300X/MI355X.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the pull request flow, and [AGENTS.md](AGENTS.md) for the rules every contributor follows. Report a wrong numerical result with the [numerical discrepancy form](https://github.com/Remenby31/cfd-solver/issues/new?template=2-numerical-discrepancy.yml), and a vulnerability as described in [SECURITY.md](SECURITY.md). Everyone follows the [code of conduct](CODE_OF_CONDUCT.md).

## Citing

Use GitHub's "Cite this repository" button, which reads [CITATION.cff](CITATION.cff). There is no paper or archived release yet.

## License

[Apache-2.0](LICENSE).
