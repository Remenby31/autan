# The steady CFL number from the step's actual and predicted reductions

Design for #395, written before any code, by CFD - Multigrille and CFD - Euler. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃ ([euler.md](euler.md#time-stepping)); its CFL number follows the switched evolution relaxation (SER, #277). This note adds to that law a factor from the ratio of the reduction the step achieved to the reduction the linear model predicted, the trust-region form of pseudo-transient continuation, on the assembled J̃ (no matrix-free product).

## Problem, measured

TMR flat plate (#395, `physics/ns-sa`): with the SER free to raise the CFL number (#277 removed a round-off ratio that pinned it at 1), the 69×49 plate settles on a period-2 cycle at CFL ≈ 8 × 10³: every step full (α = 1), its linear solve within tolerance, the state alternating between two states for thousands of steps, the residual at −3 to −4 (SA and laminar). At a state of the cycle J̃ misses 38–70 % of J v on the wall cells' density and energy rows (central differences of R, #395). Pinned at CFL 1 the same plate converged (2777–3040 steps); capped at 10 it crawled, at 100 it reached −8.5 in 6000 steps. #397 (true Jacobian, matrix-free, limiter frozen at U) removed the cycle on the laminar plate (1655 steps) and SA 35×25 (2017) but left SA 69×49 at 7–8.5 orders in 6000 steps, and lost steps or convergence on the limited Euler cases (bump 384, NACA 256²): closed as measured.

## Why the CFL number must follow the model's quality

The step is a Newton step on a model of the residual: R(U + ΔU) ≈ R(U) + J̃ ΔU, damped by V/Δt. The SER (Mulder and van Leer 1985) raises the CFL number as the residual falls, so it removes the damping on the strength of the past residual, whether or not the model still describes the next step. Three things make the model leave the residual over a full step:
- **J̃ itself**: it holds the reconstruction's gradients fixed, and misses 38–70 % of J v on the plate's wall rows (#395);
- **the limiter**: Venkatakrishnan's limiter is smooth in the differences, but its minimum and maximum over the neighbours switch neighbour, so φ is only piecewise smooth in U;
- **the reconstruction's positivity fallback** (`reconstruct`): a face side switches from the linear value to the cell value.

#397 removed the first by applying the true Jacobian with φ and the fallback frozen at U, and lost on the limited Euler cases (bump 384 not converged, NACA M 0.8 α 1.25° +193 % steps): with the limiter frozen, the model is the derivative of a residual the step then leaves. Whatever the source, the symptom is the same: the residual after the step is not what the model predicted, and at the CFL numbers the SER reaches nothing in the step responds to it (a period-2 cycle on the plate, #395).

Pseudo-transient continuation with a test on the residual after the step responds to it directly. Ceze and Fidkowski (2013, AIAA 2013-2686, read in full) compare CFL strategies on RANS-SA, DG. The strategies driven by the residual's evolution alone (SER, eq. 12; RDM, eq. 14) fail on their hardest cases, because they lower and raise the CFL number by transients. The ones that grow the CFL number only after a full update and cut it after a strongly limited one converge (EXP, eq. 13; their line search, Algorithm 4, backtracks until the residual falls). Fidkowski and Ceze (2016, AIAA SciTech, "High-order output-based adaptive simulations of turbulent flow over a three-dimensional bump", §IV.C, read in full) give the complete algorithm with its parameters, used unchanged "for many other problems". The sources therefore prescribe a residual test on the trial step, with a rejection, and not a ratio of actual to predicted reduction, which neither paper uses: the Change section below should follow the line-search form (see Parameters).

## Change

*(CFD - Multigrille.)* After each accepted step, from the second order on:

- predicted reduction: pred = ‖R(U)‖ − ‖R(U) + J̃ ΔU‖, the linear model of the residual without the pseudo-time term;
- actual reduction: act = ‖R(U)‖ − ‖R(U + α ΔU)‖, the next step's residual (already evaluated: no extra residual);
- ρ = act / pred; the norm is the SER's, over the equations it does not exempt, each scaled by its value at the start of the order (the same scaling as the SER ratio, so that no equation dominates by its units).

The next CFL number is the SER's value, capped by a factor of the current one: CFL_{n+1} = min(SER_{n+1}, c(ρ_n) CFL_n, the shortened-step cap), c(ρ) as the parameters section sets it. A step is never rejected for its ratio (the state moves as now): the ratio only governs how far the CFL number may go next. One product J̃ ΔU per step (the assembled matrix, no residual) and three norms; the state the rule needs (the last ρ, or the factor it set) is restart state.

Relation to the SER and to #356: the SER stays the law the CFL number grows by; c(ρ) only limits it. #356's under-solved rule, if merged first, stays as its own trigger (a solve above 0.1 lowers the next CFL number as a shortened step): the two caps combine by their minimum, declared together.

## Parameters

From Fidkowski and Ceze 2016, §IV.C, the parameters as published, none tuned on #395's or the gate's cases. Their algorithm, with Autan's existing admissibility in place of their physicality fraction:

1. Given U and ΔU (the linear solve, unchanged), ω = the admissible fraction of the step (`admissible_step`: no cell loses more than half its density or pressure, as now; it plays their ω_phys).
2. While ω ≥ ω_min and ‖R(U + ω ΔU)‖ > β_residual ‖R(U)‖: ω ← β_line ω.
3. If ω < ω_min: do not take the step; CFL ← β_CFL,decrease · CFL, and solve again from U.
4. Otherwise take U ← U + ω ΔU; if ω = 1, CFL ← β_CFL,increase · CFL (capped by the SER's value, so that the two laws are one, see below); if ω < 1 the CFL number stays.

| Parameter | Value (Fidkowski and Ceze 2016) |
|---|---|
| β_residual | 2.0 |
| β_line | 0.5 |
| ω_min | 0.24 |
| β_CFL,increase | 1.2 |
| β_CFL,decrease | 0.1 |
| starting CFL number | 1.0 (Autan's first order keeps its start, unchanged) |

The norm ‖·‖ is the one the SER uses: over the equations it does not exempt (the round-off exemption, #277), each scaled by its value at the start of the order, so that no equation dominates by its units (Ceze and Fidkowski 2013 require a drop in each equation's norm, Algorithm 4, for the same reason). R(U + ω ΔU) at ω = 1 is the next step's residual, so a step accepted at once costs no extra residual; each backtrack costs one. The rule applies from the second order on; the first-order phase is unchanged.

Combined with #356 (an under-solved step lowers the next CFL number as a shortened one) and the SER as one rule: CFL_{n+1} = min(SER_{n+1}, c_n CFL_n, the shortened or under-solved cap), with c_n = β_CFL,increase after a full step, 1 after a partial one; a rejected step (3) sets CFL ← β_CFL,decrease CFL and solves again. These factors are declared here and not changed on the results; if the gate fails, the change closes with the measurement.

## Gate, declared before any run

All runs fp64 unless stated, against `main` at the merge base (the reference), the case files identical across the two.

**TMR flat plate** (`physics/tmr/case.sh flatplate`, 6000 steps; #395's measure: the first step at which every equation but the identically zero z-momentum is ten orders below its peak, and every such equation at most 10⁻¹⁰ of its peak at the last step), unchanged from #395:

- laminar 69×49: at most 3332 steps;
- SA 35×25: at most 2260 steps;
- SA 69×49: at most 3332 steps;
- SA 137×97: within the 6000 steps.

**Transonic SA with an active limiter**: RAE 2822, AGARD AR-138 case 9 (M 0.73, α 2.79° corrected, Re 6.5 × 10⁶ per chord; `vv` catalog `rae2822`), on the NPARC study #4 C-grid (369 × 65, one layer, `physics/tmr/plot3d.py`), 10 000 steps. To be set up by CFD - Navier-Stokes SA before any run (grid licence checked; the case added to `case.sh`); the limiter is active by construction (no manufactured solution). Pass:

- ten orders below each equation's peak within the 10 000 steps (the plate's measure), and steps at most `main`'s when `main` converges;
- CL, CD and CM at the last step within 10⁻⁸ |c| + 10⁻¹⁰ of `main`'s when `main` converges; when it does not, of a reference run of `main` with the CFL number held at 1 to ten orders (the iteration converged on the plate at CFL 1, #395), so that both runs compare one discrete solution;
- reported, not gated: CN, CD, CM against the case 9 experiment (CN 0.803, CM −0.099, CD 0.0168).

**Accuracy** (AGENTS.md): the discretisation does not change; the Euler MMS observed orders on hex, tet, poly and hybrid in fp64 (`euler-mms-order` PASS) and fp32+acc64 agreeing with fp64 level by level (`autan-vv agree`), unchanged from `main`'s.

**Cost**: CPU time per step and to convergence on the plate and the Euler cases, measured alone under the GPU lock, reported against `main` (one product J̃ ΔU and three norms per step: expected below 2 % per step).

**Euler, no regression** (fp64, the runtime's convergence: ten orders; the cases of #356 and #397, case and mesh hashes recorded):

- first order unchanged: the history of every case below is identical to `main`'s, line for line (bit for bit), up to the step that switches to second order;
- limited cases: NACA 0012 of #183 at 256² (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°) and Ni's bump 384: converged to ten orders, in at most `main`'s steps (tolerance 0: the runs are deterministic, the count is an integer);
- smooth cases: Euler MMS hex, tet and poly (levels 16–48) and Ni's bump 96: steps at most 1.02 × `main`'s, the gain or loss recorded in the PR for every case;
- forces at convergence on every case: within 10⁻⁸ |c| + 10⁻¹⁰ of `main`'s (the absolute term for the zero lift at α 0°);
- `fp32+acc64`: NACA M 0.8 α 0° 256² ends with `main`'s status (its round-off floor) in at most `main`'s steps, its surface Cp within 0.001 of its `fp64` run outside the shocks.
