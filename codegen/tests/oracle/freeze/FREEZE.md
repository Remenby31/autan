# ADR 0003 held-out validation: stage 1 freeze

This directory freezes stage 1 of the held-out validation of ADR 0003 (linear exactness against
certified per-cell radii, #43). The commit that adds it is the **freeze commit**. Codex reviews it,
and the orchestrator approves it in writing, **before any held-out instance runs**.

## What the freeze commit pins

- **The ADR**, as amended up to the stage split (`docs/adr/0003-derived-linear-exactness-bound.md`).
- **The oracle and its tests,** by SHA-256 (`manifest-stage1.json`, `oracle_sha256`).
- **The declared precision contract** of the gradient loops (`precision_contract`). It is written by hand from `runtime.hpp`, the ADR's roles-and-casts table and `kernels/gradients.py`'s roles, and checked against the generator on every instance.
- **The compiled generated gradient headers and `runtime.hpp`,** by SHA-256 (`frozen_code_sha256`, digested without `#line`).
- **The case manifest:**
  - 36 stage 1 instances: hex, tet, hybrid, poly × N = 6, 12, 24 × clean (seed 11) and perturbed 0.15 with seeds 11 and 13. The clean meshes do not depend on the seed (`codegen/spike/meshes.py` draws only when the perturbation is nonzero), so one clean instance per family and N;
  - per instance, fields a ∈ {−7.5, 250} with G = (−0.9, 0.4, 1.3), policies fp64 / fp32+acc64 / fp32, methods LSQ and Green-Gauss;
  - the positive fixtures: every candidate tried, with its full specification (kind, N, perturbation, seed, a, G), outcome and the reason it does not certify, and the approved set. `make_manifest.py` refuses to write the manifest unless every declared candidate was evaluated on all 6 policy × method pairs, and unless the approved set equals the one the tests assert;
  - the placement of every instance (`machine`, and `placement` for the rule and why a record does not depend on the machine);
  - the boundary cases and the controls.
- **The build:** this commit, the `autan-dev:local` image, Release, `-ffp-contract=off` (root `CMakeLists.txt`), with the compiler and its version recorded in the freeze record.

## The freeze record (after the freeze commit)

The spike library is built from the freeze commit, in the frozen image. Its SHA-256 then goes into `freeze-record.json` (a follow-up commit that changes no code), together with:
- the image ID;
- the compiler and its version;
- the build's CMake cache flags.

On every held-out instance, the runner checks the loaded library against that digest, and the instance is invalid on a mismatch. The freeze record is reviewed with the freeze commit's approval, before any held-out run.

## How stage 1 runs

- **One queue job per instance on its declared machine:** PRIO 08, one thread, `RESERVE_GIB` ≤ 5, `EST_MIN` from the manifest. The command is `python -m oracle.heldout run --manifest … --instance ID --library-sha256 … --out ATTEMPT_FILE`, with one record file per attempt; an existing file is never overwritten.
- **Placement:** every stage 1 instance on spark-5512 (no deadline). spark-5359 is not used for the held-out run.
- **The library:** built once, its SHA-256 checked on arrival and by the runner on every instance.
- **Order inside a job:** library digest, then check 1, the certified geometry built once, check 2, then D1 and check 4 for every field, policy and method.
- **Equivalence of building once** (`freeze/equivalence.py`, on B only): the certificates projected from the single construction equal those check 2 rebuilds, Cert by Cert, and the exact geometry is deterministic. Check 2 gives the same non-empty issues on corrupted stored geometry with the geometry reused or rebuilt. The whole records are identical either way.
- **Records:** every attempt file is created exclusively, from its first step and with its provenance, then checkpointed after every step through a unique temporary file. Each check's outcome is persisted before its digest. An exception adds an invalid entry naming its stage, without discarding anything.
- **Attempts:** an attempt that reported an issue (an invalidity, a check issue, a check not covering every entry), or that completed without passing, counts against stage 1 whatever comes after. Only an attempt interrupted without any of these may be superseded by a later one. Every attempt is kept.
- **Auxiliary suite:** one job runs the oracle suite on the frozen build through `python -m oracle.heldout aux --manifest … --library-sha256 … --out ATTEMPT_FILE -- <pytest arguments>`. The attempt file is created before pytest starts. The wrapper captures pytest's exit status itself, and records each test's exact node id and worst outcome through the `oracle.freeze.nodeid_report` plugin. Every attempt must finish, exit 0 and pass every frozen node id.
- **Alongside:** one job runs the positive fixtures, the boundary cases and the controls on the same frozen build.
- **Records:** every run is kept, failed ones included, with its raw JSON record.

## Verdict

- **Fail-closed** (`heldout.passed`, `heldout.collect`): an instance passes only if it is valid, complete (exactly its declared field × policy × method runs, each covering every cell and component) and checks 1–4 report nothing. Stage 1 also needs every instance of the manifest to have such a record (a missing record is a failure), verified against the approved manifest bytes and library digest, and every auxiliary attempt to have exited 0 with every frozen node id passed.
- **Stage 1 results are recorded and frozen, not called a verdict.** There is no PASS on the gate until stage 2 (vv's agglomerated family, frozen and approved separately) is done.
- **A stage 1 failure stays a failure.** Any invalid or failing stage 1 instance makes M1 FAIL, whatever stage 2 gives.
- **Reporting:** the per-source-class radii (ADR decision 4) are a separate pass after the stage results. They are reporting only and never read by the gate.
- **Amendments:** any change to an equation, the code, the build or the manifest after the freeze needs a new written approval and a new disjoint case set.
