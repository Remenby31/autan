# AGENTS.md

Rules for every contributor, human or agent. Read this before touching the code. Setup, hooks and the pull request mechanics are in `CONTRIBUTING.md`.

## What this project is

A GPU-native, unstructured finite-volume CFD solver: arbitrary polyhedral cells, compressible first, discrete adjoint on GPU, typed Python SDK. Plan and gates: `docs/PLAN.md`. Research behind the choices: `docs/research/`.

## Names

The project is **Autan**, after the wind that blows over Toulouse. Write "Autan" in prose and `autan` in identifiers. No other name, and no compatibility aliases for the old one.

| Thing | Name |
|---|---|
| Repository, docs site | `Remenby31/autan`, `https://remenby31.github.io/autan/` |
| Python distribution, import, CLI | `autan` (`import autan`, `autan plan`) |
| V&V package | `autan-vv` (`import autan_vv`) |
| Schema identifiers | `autan.case/1`, `autan.mesh-info/1`, `autan.results/1` |
| C++ | namespace `autan`, CMake project `autan`, library `autan_core`, tool `autan-mesh-info` |
| Environment variables, images | `AUTAN_*`, `ghcr.io/remenby31/autan-*` |

Error codes do not change.

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

## Source of truth

The code, this file, `docs/PLAN.md`, the contracts in `docs/contracts/` and the open issue you work on describe the target. A closed issue, a past pull request or a transcript records a past intent, not a constraint.

`AGENTS.md` and the files in `docs/contracts/` change only through a pull request the lead approves for that change.

## Design

- No backward compatibility: nothing is released. Remove obsolete paths; add no compatibility layers, fallbacks or lenient readers for our own formats. Readers of external formats (Gmsh, Fluent, CGNS) follow the format specification and reject what it forbids, with a coded error.
- Choose the simplest durable implementation that meets the current milestone. Prefer an existing dependency; read its documentation and types before adding a package or reimplementing one.
- Licences: everything we ship carries a permissive or weak-copyleft licence (MIT, BSD, Apache-2.0, MPL, LGPL, CeCILL-C). AGPL, SSPL, BUSL, non-commercial and other source-available licences are refused (ParMETIS is non-commercial: use METIS or Scotch). GPL needs the lead's approval. Check the licence before adding a dependency, image or dataset. One exception, decided by the lead: the NVIDIA CUDA runtime wheels from PyPI (`nvidia-*`) may be declared as dependencies, never vendored or linked statically, as JAX and PyTorch do; the GPU driver is proprietary anyway.
- Before changing a shared API, schema or contract, find and update every consumer in the same change: tests, benchmarks, scripts, examples, docs. Verify a contract by sending the real producer's output through the real consumer (for example, the C++ core's `MeshInfo` JSON through the Python schema), never through a handwritten fixture alone.

## Code

- C++20 for the core, CUDA and HIP from the kernel generator. Python 3.12+ for the SDK, CLI and validation.
- Code, comments, docs and commit messages in English.
- Small, focused diffs. No opportunistic refactors of files you do not own.
- Comments only where the code cannot say it: the equation, paper and equation number a kernel implements (`// Roe (1981), eq. 12; entropy fix: Harten (1983)`), a non-obvious numerical choice and its reason, and tool directives that name their rule. No narrating comments, no change history in code, names or docstrings.
- Docstrings only on public SDK exports, CLI commands and published schemas: one paragraph of caller contract.
- Python dependencies come only from a committed `uv.lock` (`uv sync --locked`). Never resolve requirements at test time.

## Tests

- A test protects behaviour at a public seam: a mesh read through the public reader, a kernel through the solver API, a case through the SDK, a CLI command through its output.
- **Expected values come from the specification, never from the code under test**: analytic solutions, manufactured solutions, conservation laws, published reference data. A golden file produced by our own code is not an oracle unless a second, independent source confirms it.
- One behaviour, one primary test; extend it rather than adding another. No parametrisation over equivalent cases; do parametrise over cell types, precisions and backends, which are not equivalent.
- Floating-point tolerances scale with the precision under test (a multiple of its machine epsilon), so one test covers FP32 and FP64. Make a test robust to round-off rather than adding a second reference value per platform.
- A regression test fails on the code before the fix, for the reason the bug gives.
- No mocks of our own classes; fakes only at external boundaries. Assert outcomes and error codes, never exact message text.
- No redundant tests: delete a test that guards nothing another test does not.

## Docs

- Update the affected documentation in the same change. Revise the page that owns the fact; do not append change notes.
- State each fact once and link to it elsewhere. Leave each page you touch shorter or clearer than you found it.

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
- Stage named files only. Commit messages follow Conventional Commits.
- Run `pre-commit run --all-files` before every push; CI runs the same hooks. Fix what a hook reports, never bypass it with `--no-verify`.
- A failing check is yours until you show it fails on `main` too.
- A PR title is a Conventional Commit that reads on its own: the squash keeps it as the only commit on `main`. The body opens with the impact, one bullet per topic, then the verification evidence. ≤ ~250 words; details in a `<details>` block. Fill in `.github/pull_request_template.md`. If an agent wrote the change, the body names the tool and the model. After a review, rewrite the body to describe the change as it stands.
- Code, comments and PR text describe the change itself; prompts, session history and discarded attempts stay out of them.
- An issue states the present, the expected result, how to verify it and what is out of scope. When the target changes, rewrite the body instead of commenting.
- Open a PR and stop. Do not merge; the lead reviews and merges. Never enable auto-merge.
- Ask before anything destructive: force-push, rewriting pushed history, deleting a branch you did not create, discarding uncommitted work.
- In GitHub prose, keep each paragraph on one source line (single newlines render as hard breaks).
- Before marking a PR ready, get an adversarial review from Codex (skill `ask-codex`) and address or answer each finding.

## Hardware

- Two DGX Spark (GB10, 128 GB unified memory, CUDA 13.0): `ssh spark` and `ssh spark-5359` over Tailscale. No direct ConnectX link between them: multi-node runs are for correctness only.
- They are shared machines running other Docker stacks. Never stop, modify or inspect other containers. Build and run inside our own containers (`docker/`). No sudo.
- Serialize GPU benchmarks with `flock /tmp/cfd-gpu.lock <cmd>` and check that the GPU is idle before timing. Never wait for a time slot; queue on the lock.
- GB10 FP64 throughput is very low: design for FP32 compute with FP64 accumulation where needed.
