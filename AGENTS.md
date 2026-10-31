# AGENTS.md

Rules for every contributor, human or agent. Read this before touching the code.

## What this project is

A GPU-native, unstructured finite-volume CFD solver: arbitrary polyhedral cells, compressible first, discrete adjoint on GPU, typed Python SDK. Plan and gates: `docs/PLAN.md`. Research behind the choices: `docs/research/`.

## Non-negotiables

1. **Correctness is proven, not claimed.** Any change to numerics lands with:
   - unit tests for the new code;
   - a method-of-manufactured-solutions (MMS) run showing the expected observed order on tetrahedral, hexahedral, polyhedral and hybrid meshes, in FP32 and FP64;
   - conservation checks where they apply.
   Put the observed orders in the PR description. A PR that changes numerics without this evidence is not ready.
2. **Performance is measured, not estimated.** Any kernel change reports, from the `bench/` harness on a DGX Spark:
   - achieved bandwidth as a % of the measured GB10 peak (target ≥ 70 % on hot kernels);
   - time per cell per iteration.
   A regression above 5 % on a tracked benchmark blocks the merge.
3. **Results are reproducible.** Every run records mesh hash, case hash, solver commit and precision. A bitwise-deterministic mode must keep working.
4. **No silent fallbacks.** Never fall back to CPU, FP64 or a slower path without logging it.

## Code

- C++20 for the core, CUDA and HIP from the kernel generator. Python 3.12+ for the SDK, CLI and validation.
- Code, comments, docs and commit messages in English.
- Small, focused diffs. No opportunistic refactors of files you do not own.
- No redundant tests: delete a test that guards nothing another test does not.

## Ownership (wave 1)

| Area | Owner |
|---|---|
| `CMakeLists.txt`, `core/`, `io/`, `tests/core/`, `docs/contracts/mesh.md` | CFD - Noyau maillage |
| `vv/`, `docs/validation/` | CFD - V&V |
| `py/`, `pyproject.toml`, `docs/contracts/case-schema.md`, `docs/contracts/results.md` | CFD - SDK & contrats |
| `docker/`, `bench/`, `.github/workflows/gpu-*.yml`, `docs/infra/` | CFD - HPC Spark |
| `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, other root meta files, `.github/` (except GPU workflows), lint and format configs | CFD - Standards du dépôt |

Need a change in someone else's area? Say so in your PR description instead of editing it.

## Workflow

- One branch and one worktree per task. Before every push: `git fetch origin && git merge origin/main`.
- Open a PR and stop. Do not merge; the lead reviews and merges.
- PR description ≤ ~250 words, plain language; details in a `<details>` block.
- Before marking a PR ready, get an adversarial review from Codex (skill `ask-codex`) and address or answer each finding.

## Hardware

- Two DGX Spark (GB10, 128 GB unified memory, CUDA 13.0): `ssh spark` and `ssh spark-5359` over Tailscale. No direct ConnectX link between them: multi-node runs are for correctness only.
- They are shared machines running other Docker stacks. Never stop, modify or inspect other containers. Build and run inside our own containers (`docker/`). No sudo.
- Serialize GPU benchmarks with `flock /tmp/cfd-gpu.lock <cmd>` and check that the GPU is idle before timing. Never wait for a time slot; queue on the lock.
- GB10 FP64 throughput is very low: design for FP32 compute with FP64 accumulation where needed.
