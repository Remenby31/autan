<!-- Title: a Conventional Commit that reads on its own, e.g. "feat(core): read Fluent polyhedral cells". The squash keeps only the title on main. -->

Closes #<!-- issue; use "Refs #N" for partial or stacked work -->

## Impact

<!-- One bullet per topic, most important first: what is different for a user, a caller or a contributor once this merges. Add ", breaking" when they must change something. -->

## Change

<!-- At most five bullets: what the code does once this merges, as its API, CLI, schema or output shows it. For a fix, the behaviour before and after. -->

## Verification

<!-- Tick what applies; delete a line only when the rule cannot apply, and say why under "Not applicable". Rules: AGENTS.md#non-negotiables. -->

- [ ] Tests cover the new behaviour at a public seam; expected values come from the specification, not from this code.
- [ ] `pre-commit run --all-files` passes.
- [ ] Numerics changed: MMS observed orders on tetrahedral, hexahedral, polyhedral and hybrid meshes, in FP32 and FP64, are in the table below; conservation is checked where it applies.
- [ ] Kernel changed: `bench/` results from a DGX Spark are below: % of measured GB10 peak bandwidth and time per cell per iteration, against `main`.
- [ ] Runs still record mesh hash, case hash, solver commit and precision; the bitwise-deterministic mode still passes.
- [ ] Any new fallback (CPU, FP64, slower path) is logged.
- [ ] Every consumer of a changed API, schema or contract is updated in this PR.
- [ ] Documentation that owns an affected fact is updated.
- [ ] Codex adversarial review done; each finding is fixed or answered below.

<!-- Numerics evidence (delete if none):

| Mesh | Precision | Variable | Observed order (L2) | Observed order (L∞) | Expected |
|---|---|---|---|---|---|
| tet | FP32 | | | | |

Kernel evidence (delete if none):

| Kernel | Cells | % GB10 peak BW | ns / cell / iteration | main | Δ |
|---|---|---|---|---|---|
| | | | | | |
-->

**Not applicable:**

<!-- Which boxes do not apply and why. -->

<details>
<summary>Review findings and agent details</summary>

<!-- Codex findings and how each was addressed. If an agent wrote this change, name the tool and model. -->

</details>
