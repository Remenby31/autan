# 0001: FP32 under an order gate is judged by its agreement with FP64

**Status:** accepted, 2026-11-06. **Decided by:** the lead's delegate for CFD, relayed by the orchestrator, applying the lead's rule of 2026-11-01. **Issues:** #524, #137.

## Context

`docs/validation/acceptance.md` has stated since 2026-11-01, as the lead's decision, that from M2 on the order gates (the Euler and Navier-Stokes manufactured solutions, Ringleb, Sod) are verdicts in `fp64`. In `fp32+acc64`, a verification case passes when it agrees with its `fp64` run on every level (`autan-vv agree`, at most 500 × 2⁻²³ per field), and its orders are reported, not gated.

The rows of `euler-mms-order` and `ns-mms-order` said "FP32 and FP64", which reads as a gate on FP32 orders, and contradicted that statement. #137 applied `ns-mms-order` to the FP32 runs as an order gate. Clean FP32 hex then failed (#524): only three of its levels sat above 100 times the FP32 rounding floor, although its errors equal the FP64 run's to four significant digits.

## Decision

The general statement holds, and the rows say so: a verdict in FP64; in `fp32+acc64`, agreement with the FP64 run on every level, by `autan-vv agree`, with its orders reported, not gated. There is one FP32 rule, the one stated since 2026-11-01. No threshold changes: K stays 500 × 2⁻²³, with the measurements that justify it.

**This correction comes after results.** #524's failure was known when the decision was taken, and the re-evaluation of #137's FP32 runs under the stated rule followed it. All eight families (four cell types, clean and perturbed) pass `agree`, at most 15.9 × 2⁻²³ (#524). What changed is not the threshold but which rule applies to FP32: the one the document already stated, against the rows that contradicted it.

`agree` also compares a study's files by content: the mesh by its sha256, and every other file the case records with its sha256 (manufactured solution, initial state, oracle states) by that sha256, so that studies keeping their own copies of them pair (#524).

## Consequences

- FP32 runs of an order gate are paired with the FP64 runs of the same scheme, build and meshes, and judged by `agree`. The FP32 orders are reported next to the verdict, never as one.
- A run does not record which numerical scheme produced it (#538). Until it does, the FP64 partner of an FP32 study is named by the study, not inferred.
