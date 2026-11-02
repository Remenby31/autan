# The steady CFL number from a line search on the residual

Design for #395, written before any code, by CFD - Multigrille and CFD - Euler. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃ ([euler.md](euler.md#time-stepping)); its CFL number follows the switched evolution relaxation (SER, #277). This note adds to that law a line search on the true residual, with step rejection and growth only after a full update (pseudo-transient continuation as Fidkowski and Ceze apply it to RANS), on the assembled J̃ (no matrix-free product).

## Problem, measured

TMR flat plate (#395, `physics/ns-sa`): with the SER free to raise the CFL number (#277 removed a round-off ratio that pinned it at 1), the 69×49 plate settles on a period-2 cycle at CFL ≈ 8 × 10³: every step full (α = 1), its linear solve within tolerance, the state alternating between two states for thousands of steps, the residual at −3 to −4 (SA and laminar). At a state of the cycle J̃ misses 38–70 % of J v on the wall cells' density and energy rows (central differences of R, #395). Pinned at CFL 1 the same plate converged (2777–3040 steps); capped at 10 it crawled, at 100 it reached −8.5 in 6000 steps. #397 (true Jacobian, matrix-free, limiter frozen at U) removed the cycle on the laminar plate (1655 steps) and SA 35×25 (2017) but left SA 69×49 at 7–8.5 orders in 6000 steps, and lost steps or convergence on the limited Euler cases (bump 384, NACA 256²): closed as measured.

## Why the CFL number must follow the model's quality

The step is a Newton step on a model of the residual: R(U + ΔU) ≈ R(U) + J̃ ΔU, damped by V/Δt. The SER (Mulder and van Leer 1985) raises the CFL number as the residual falls, so it removes the damping on the strength of the past residual, whether or not the model still describes the next step. Three things make the model leave the residual over a full step:
- **J̃ itself**: it holds the reconstruction's gradients fixed, and misses 38–70 % of J v on the plate's wall rows (#395);
- **the limiter**: Venkatakrishnan's limiter is smooth in the differences, but its minimum and maximum over the neighbours switch neighbour, so φ is only piecewise smooth in U;
- **the reconstruction's positivity fallback** (`reconstruct`): a face side switches from the linear value to the cell value.

#397 removed the first by applying the true Jacobian with φ and the fallback frozen at U, and lost on the limited Euler cases (bump 384 not converged, NACA M 0.8 α 1.25° +193 % steps): with the limiter frozen, the model is the derivative of a residual the step then leaves. Whatever the source, the symptom is the same: the residual after the step is not what the model predicted, and at the CFL numbers the SER reaches nothing in the step responds to it (a period-2 cycle on the plate, #395).

Pseudo-transient continuation with a test on the residual after the step responds to it directly. Ceze and Fidkowski (2013, AIAA 2013-2686, read in full) compare CFL strategies on RANS-SA, DG. The strategies driven by the residual's evolution alone (SER, eq. 12; RDM, eq. 14) fail on their hardest cases, because they lower and raise the CFL number by transients. The ones that grow the CFL number only after a full update and cut it after a strongly limited one converge (EXP, eq. 13; their line search, Algorithm 4, which tests the unsteady residual M/Δt (U_trial − U) + R(U_trial) equation by equation; their hardest wing case converges only with their constrained solver). That paper is the evidence for the strategy, not the source of the rule below. Fidkowski and Ceze (2016, AIAA SciTech, "High-order output-based adaptive simulations of turbulent flow over a three-dimensional bump", §IV.C, read in full) give the complete algorithm with its parameters, used unchanged "for many other problems": the rule below is theirs, its test on the steady residual. Autan's adaptations are named where they occur: the norm (the SER's, over the non-exempt equations), the admissible fraction in place of their physicality fraction, and the combination with the SER. The sources therefore prescribe a residual test on the trial step, with a rejection, and not a ratio of actual to predicted reduction, which neither paper uses: the Change section follows the line-search form (see Parameters).

**Does the published test fire on #395's symptom? Checked before any code** (the orchestrator's condition; reading declared before the measurement: if ‖R_{n+1}‖ / ‖R_n‖ exceeds 2 on at least one step of the cycle, the rule acts there; if it stays at or below 2, the rule does not address the symptom, and nothing is coded). Measured on the existing histories of the 69×49 plate (#395's runs on `physics/ns-sa` c3966068), in the declared norm: the non-exempt equations, each scaled by its value at the first second-order step, combined by their maximum (the SER's) and, as a check, by their root mean square:

| Run (69×49) | Steps | ‖R_{n+1}‖ / ‖R_n‖, max norm | rms norm | Steps above 2 |
|---|---|---|---|---|
| SA, assembled J̃ (the 2-cycle) | 301–600 | 0.13 – 179 | 0.16 – 115 | 58 / 53 |
| | 601–1500 | 0.949 – 1.054 | 0.956 – 1.046 | **0 / 0** |
| | 1501–6000 | 0.954 – 1.048 | 0.960 – 1.041 | **0 / 0** |
| SA, limiter off | 601–6000 | 0.955 – 1.048 | 0.961 – 1.041 | **0 / 0** |
| laminar, assembled J̃ | 601–1500 | 0.12 – 412 | 0.17 – 327 | 224 / 187 |
| | 1501–6000 | 0.10 – 441 | 0.16 – 417 | 1094 / 938 |

By the declared reading: **on the SA plate's established 2-cycle the published test never fires** (its two states' residuals differ by less than 5 %), so the rule, as published and with its values, does not address the SA symptom of #395; the laminar plate, which wanders rather than cycles, exceeds 2 on a fifth of its steps, and so does the SA plate before the cycle sets in (steps 301–600, 58 steps). Whether cutting the CFL number during that approach would keep the SA cycle from forming cannot be read from these histories; it is not claimed. β_residual is not lowered to make the test fire on the cycle: that would fit the rule to the case.

## Change

From the second order on, the steady step tests its trial update on the residual, as Fidkowski and Ceze's algorithm does (the Parameters section states it and its values): the update is cut back while the residual after it is more than twice the residual before, rejected when that takes it below a quarter of the step, and the CFL number grows by 1.2 only after a full update. The test uses the true residual R(U + ω ΔU), so it responds to whatever makes the model leave it: J̃'s missing gradients, the limiter's switches, the reconstruction's fallback. The operator stays the assembled J̃ and its multigrid (or the line sweeps): nothing of #397's matrix-free product.

**The trial and what a rejection restores.** The trial is the flow update only: U + ω ΔU, its residual evaluated with everything else held at the current state (ν̃ and the eddy viscosity, a point vortex's circulation), as the linear model holds them. What the step does after an accepted flow update is unchanged and comes after the test: the loosely coupled SA update and the eddy viscosity (`physics/ns-sa`), the circulation measured on the next residual. A backtrack or a rejection therefore only discards trial flow states; nothing else has moved, and a rejection keeps U as it is.

**A rejection is a runtime step, as today's.** The steady step already rejects a step when its linear solve fails (residual drop above 0.9 or a singular block) or no admissible fraction exists: the state is kept, the step ends, and the next CFL number is half this one; at the smallest CFL number the run diverges. A line-search rejection (ω < ω_min) does the same with the source's factor: the state is kept, the step ends (its history line repeats the residual of the kept state), the cap for the next step is max(β_CFL,decrease · CFL, CFL_min) and the penalty is set as for a failed solve (`penalty_ = min(penalty_, CFL · drop / CFL_start)`, the rejection flag set); a rejection at the smallest CFL number (1) is a divergence, as now. So the source's "reduce the CFL number and solve again" is the next runtime step's solve at the reduced CFL number, and every rejection path (linear failure, no admissible step, line search) updates the cap and penalty once per runtime step.

**No side effect from the trials.** The trial norms use the exemption mask and the per-equation references fixed at the start of the step; they do not update the round-off exemption (`relative_drop`, `note_exemptions`), whose state changes only from the reported residuals, as now. A rejected or backtracked trial therefore leaves the SER, the exemption and the runtime's convergence test where they were.

**Cost.** Each trial costs one residual evaluation and its norms (exact sums over the ranks): one per accepted full step, plus one per backtrack. It is not the next step's residual: that one is evaluated with its Jacobian (`residual_and_jacobian`) after the SA update, so the trial's is extra. A rejection costs its trials and the step. The counts (residual evaluations, backtracks, rejections) are logged per run and reported in the cost gate.

**Restart and ranks.** Between steps the rule carries its growth factor and the CFL number and cap (restart state, as the SER's); the trial norms are exact sums reduced over the ranks, ω the same on every rank, so that a continued run repeats the uninterrupted one bit for bit and `reproducibility = "ranks"` keeps its bits.

The SER stays the law the CFL number may grow by: the line search's factor only limits it, and a rejection overrides it. One rule, with #356's if it is merged first (see Parameters): CFL_{n+1} = min(SER_{n+1}, c_n CFL_n, the shortened, under-solved or rejected cap). The first-order phase is unchanged.

Not done: a ratio of actual to predicted reduction (a trust-region control). Neither source uses one, and none read in full prescribes its thresholds for this setting.

## Parameters

From Fidkowski and Ceze 2016, §IV.C, the parameters as published, none tuned on #395's or the gate's cases. Their algorithm, with Autan's existing admissibility in place of their physicality fraction:

1. Given U and ΔU (the linear solve, unchanged), ω = the admissible fraction of the step (`admissible_step`: no cell loses more than half its density or pressure, as now; it plays their ω_phys).
2. While ω ≥ ω_min and ‖R(U + ω ΔU)‖ > β_residual ‖R(U)‖: ω ← β_line ω.
3. If ω < ω_min: do not take the step; CFL ← β_CFL,decrease · CFL, and solve again from U (in Autan: the next runtime step, see Change).
4. Otherwise take U ← U + ω ΔU; if ω = 1, CFL ← β_CFL,increase · CFL (capped by the SER's value, so that the two laws are one, see below); if ω < 1 the CFL number stays.

| Parameter | Value (Fidkowski and Ceze 2016) |
|---|---|
| β_residual | 2.0 |
| β_line | 0.5 |
| ω_min | 0.24 |
| β_CFL,increase | 1.2 |
| β_CFL,decrease | 0.1 |
| starting CFL number | 1.0 (Autan's first order keeps its start, unchanged) |

The norm ‖·‖ is the one the SER uses: over the equations it does not exempt (the round-off exemption, #277), each scaled by its value at the start of the order, so that no equation dominates by its units (Ceze and Fidkowski 2013 require a drop in each equation's norm, Algorithm 4, for the same reason). Each trial costs one residual evaluation (Change, Cost). The rule applies from the second order on; the first-order phase is unchanged.

Combined with #356 (an under-solved step lowers the next CFL number as a shortened one) and the SER as one rule: CFL_{n+1} = min(SER_{n+1}, c_n CFL_n, the shortened or under-solved cap), with c_n = β_CFL,increase after a full step, 1 after a partial one; a rejected step (3) sets CFL ← β_CFL,decrease CFL and solves again. These factors are declared here and not changed on the results; if the gate fails, the change closes with the measurement.

## Gate, declared before any run

All runs fp64 unless stated, against `main` at the merge base (the reference), the case files identical across the two.

**TMR flat plate** (`physics/tmr/case.sh flatplate`, 6000 steps; #395's measure: the first step at which every equation but the identically zero z-momentum is ten orders below its peak, and every such equation at most 10⁻¹⁰ of its peak at the last step), unchanged from #395:

- laminar 69×49: at most 3332 steps;
- SA 35×25: at most 2260 steps;
- SA 69×49: at most 3332 steps;
- SA 137×97: within the 6000 steps.

**Transonic SA with an active limiter**: RAE 2822, AGARD AR-138 case 9 (M 0.73, α 2.79° corrected, Re 6.5 × 10⁶ per chord; `vv` catalog `rae2822`), on a C-grid generated by CFD - Navier-Stokes SA (generator in `vv/meshes/`) from the TMR ordinates `f8621.txt` (Cook, McDonald and Firmin, CC0-1.0, pinned), three levels declared before their generation (the NPARC grid and `geom.txt` are NPARC's, whose licence is ambiguous, #408), 10 000 steps on the level NS declares as the reference; the limiter is active by construction (no manufactured solution). **The grid is to be generated (NS); ΨTC cannot be validated before it exists.** Pass:

- ten orders below each equation's peak within the 10 000 steps (the plate's measure), and steps at most `main`'s when `main` converges;
- CL, CD and CM at the last step within 10⁻⁸ |c| + 10⁻¹⁰ of `main`'s when `main` converges; when it does not, of a reference run of `main` with the CFL number held at 1 to ten orders (the iteration converged on the plate at CFL 1, #395), so that both runs compare one discrete solution;
- reported, not gated: CN, CD, CM against the case 9 experiment (CN 0.803, CM −0.099, CD 0.0168).

**Accuracy** (AGENTS.md): the discretisation does not change; the Euler MMS observed orders on hex, tet, poly and hybrid in fp64 (`euler-mms-order` PASS) and fp32+acc64 agreeing with fp64 level by level (`autan-vv agree`), unchanged from `main`'s.

**Cost**: CPU time per step and to convergence on the plate and the Euler cases, measured alone under the GPU lock, reported against `main` with the counts of residual evaluations, backtracks and rejections (each trial one residual and its exact-sum norms; a rejected step costs its trials and its solve).

**Euler, no regression** (fp64, the runtime's convergence: ten orders; the cases of #356 and #397, case and mesh hashes recorded):

- first order unchanged: the history of every case below is identical to `main`'s, line for line (bit for bit), up to the step that switches to second order;
- limited cases: NACA 0012 of #183 at 256² (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°) and Ni's bump 384: converged to ten orders, in at most `main`'s steps (tolerance 0: the runs are deterministic, the count is an integer);
- smooth cases: Euler MMS hex, tet and poly (levels 16–48) and Ni's bump 96: steps at most 1.02 × `main`'s, the gain or loss recorded in the PR for every case;
- forces at convergence on every case: within 10⁻⁸ |c| + 10⁻¹⁰ of `main`'s (the absolute term for the zero lift at α 0°);
- `fp32+acc64`: NACA M 0.8 α 0° 256² ends with `main`'s status (its round-off floor) in at most `main`'s steps, its surface Cp within 0.001 of its `fp64` run outside the shocks, and its CL, CD and CM within 0.1 % of the `fp64` run's on the same mesh ([acceptance.md](../validation/acceptance.md); 10⁻⁶ absolute for the zero lift and pitching moment at α 0°).

**Tests** (public seams): a steady run whose second-order phase backtracks and rejects (logged), continued right after a backtracked step and right after a rejected one, repeats the uninterrupted run bit for bit (histories and solution arrays); the same case with `"lines"` under `reproducibility = "ranks"` on 1, 2 and 4 ranks gives one history, byte for byte; a rejection at the smallest CFL number ends the run diverged.
