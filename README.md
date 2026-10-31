<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo-light.svg" alt="" height="56">
  </picture>
  &nbsp;
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/wordmark-dark.svg">
    <img src="docs/assets/wordmark-light.svg" alt="Autan" height="40">
  </picture>
</p>

# Autan

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![HIP/ROCm](https://img.shields.io/badge/HIP-ROCm-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/projects/HIP/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CMake 3.28+](https://img.shields.io/badge/CMake-3.28%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Made in France](https://img.shields.io/badge/made%20in-France-0055A4)](#about)
<br>
[![CI](https://github.com/Remenby31/autan/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Remenby31/autan/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Remenby31/autan/badge)](https://scorecard.dev/viewer/?uri=github.com/Remenby31/autan)

Autan is a GPU-native finite-volume solver for compressible flow on unstructured meshes of arbitrary polyhedra, with a discrete adjoint on the GPU and a typed Python SDK.

> **Status: pre-alpha. The solver does not compute a flow yet.** The repository holds the plan, the research behind it, the contracts between components and the first foundations of milestone M0: the polyhedral mesh core, the Gmsh, Fluent and OpenFOAM readers, and the SDK's case schema and CLI skeleton. There is no release and no stable API. Do not use it for engineering work.

**Documentation: [remenby31.github.io/autan](https://remenby31.github.io/autan/)**

## Why

No open, portable code runs unstructured, compressible CFD on GPUs together with a discrete adjoint ([state of the art](docs/research/state-of-the-art-2026-10.md#2-où-battre-létat-de-lart)):

- Fluent's GPU solver is pressure-based only, without an adjoint. STAR-CCM+ runs its coupled compressible solver on the GPU, but it is closed and has no GPU adjoint.
- SU2 has a discrete adjoint, on the CPU only. NASA FUN3D runs on GPUs but is export-controlled (ITAR).
- New NVIDIA GPUs make FP64 rare: on GB10 or an RTX PRO 6000, an FP64 kernel becomes compute-bound.

This project sets out to:

- run the whole solver on the GPU, NVIDIA and AMD, from code generated once per kernel;
- compute in FP32 with FP64 accumulation, and prove in CI that the answer does not depend on it;
- differentiate the generated kernels to get the adjoint on the GPU;
- publish its verification and benchmark numbers with every change.

## Done and planned

| Milestone | Content | State |
|---|---|---|
| M0 Foundations | Face-based polyhedral mesh, Gmsh, Fluent and CGNS readers, partitioning, CI, benchmark harness, V&V framework | In progress |
| M1 Generator spike | Kernel DSL to CUDA, HIP and CPU; gradients; scalar advection-diffusion; adjoint of one kernel | Planned |
| M2 Compressible Euler | Roe, HLLC and AUSM+-up fluxes; explicit then implicit solver | Planned |
| M3 Navier-Stokes + SA | Viscous fluxes, Spalart-Allmaras, agglomeration multigrid | Planned |
| M4 SDK + CLI v0 | Typed case schema with units, `autan plan`, `run`, `logs`, `diff`, coded errors | Planned |
| M5 Multi-GPU and precision | GPU-aware MPI, FP32 versus FP64 study | Planned |
| M6 GPU discrete adjoint | Adjoint of the generated kernels, shape optimisation | Planned |
| M7–M11 | SST, DDES, WMLES; pressure-based solver; multi-gas and conjugate heat transfer; web UI; AI-native workflows | Planned |

Each milestone closes on numbered exit criteria. They are in [the plan](docs/PLAN.md), with the risks and the critical path.

## Architecture

Three parts: a C++ core that reads and partitions meshes, a Python SDK that checks cases and reads results, and, from M1, a kernel generator that emits CUDA, HIP and CPU code with its tangent and adjoint. They exchange three documents: the [case](docs/contracts/case-schema.md), the [MeshInfo summary](docs/contracts/mesh-info.md) and the [run directory](docs/contracts/results.md). [Architecture](docs/concepts/architecture.md) shows the components and the milestone of each.

## How correctness and speed are proven

Every change carries its own evidence. The rules are in [AGENTS.md](AGENTS.md#non-negotiables); in short:

- **Correctness.** A change to the numerics shows the observed order of accuracy from manufactured solutions, on tetrahedral, hexahedral, polyhedral and hybrid meshes, in FP32 and FP64. Expected values come from analytic solutions, conservation laws or published reference data, never from the solver itself.
- **Performance.** A kernel change reports the achieved bandwidth as a percentage of the measured peak of the GPU, and the time per cell per iteration, from the benchmark harness on a DGX Spark. A regression above 5 % blocks the merge.
- **Reproducibility.** Every run records the mesh hash, the case hash, the solver commit and the precision. A bitwise-deterministic mode stays available.

Development hardware is two NVIDIA DGX Spark (GB10). Production targets are B200/GB200 and MI300X/MI355X.

## Contributing

The [documentation site](https://remenby31.github.io/autan/) renders `docs/`. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the pull request flow, and [AGENTS.md](AGENTS.md) for the rules every contributor follows. Report a wrong numerical result with the [numerical discrepancy form](https://github.com/Remenby31/autan/issues/new?template=2-numerical-discrepancy.yml), and a vulnerability as described in [SECURITY.md](SECURITY.md). Everyone follows the [code of conduct](CODE_OF_CONDUCT.md).

## Citing

Use GitHub's "Cite this repository" button, which reads [CITATION.cff](CITATION.cff). There is no paper or archived release yet.

## About

Designed and developed in France by [Baptiste Cruvellier](https://github.com/Remenby31). The autan is the wind that blows over Toulouse, the capital of French aeronautics. The design documents are in French; code, contracts and reference documentation are in English.

## License

[Apache-2.0](LICENSE).
