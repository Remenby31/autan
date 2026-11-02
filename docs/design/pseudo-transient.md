# The steady CFL number from the step's actual and predicted reductions

Design for #395, written before any code, by CFD - Multigrille and CFD - Euler. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃ ([euler.md](euler.md#time-stepping)); its CFL number follows the switched evolution relaxation (SER, #277). This note adds to that law a factor from the ratio of the reduction the step achieved to the reduction the linear model predicted, the trust-region form of pseudo-transient continuation, on the assembled J̃ (no matrix-free product).

## Problem, measured

TMR flat plate (#395, `physics/ns-sa`): with the SER free to raise the CFL number (#277 removed a round-off ratio that pinned it at 1), the 69×49 plate settles on a period-2 cycle at CFL ≈ 8 × 10³: every step full (α = 1), its linear solve within tolerance, the state alternating between two states for thousands of steps, the residual at −3 to −4 (SA and laminar). At a state of the cycle J̃ misses 38–70 % of J v on the wall cells' density and energy rows (central differences of R, #395). Pinned at CFL 1 the same plate converged (2777–3040 steps); capped at 10 it crawled, at 100 it reached −8.5 in 6000 steps. #397 (true Jacobian, matrix-free, limiter frozen at U) removed the cycle on the laminar plate (1655 steps) and SA 35×25 (2017) but left SA 69×49 at 7–8.5 orders in 6000 steps, and lost steps or convergence on the limited Euler cases (bump 384, NACA 256²): closed as measured.

## Why the CFL number must follow the model's quality

*(CFD - Euler: the a priori argument — model mismatch from the limiter's switches, the reconstruction's fallback and the frozen gradients; what the SER does at large CFL numbers; why a ratio of actual to predicted reduction detects it whatever its source; sources read in full.)*

## Change

*(CFD - Multigrille.)* After each accepted step, from the second order on:

- predicted reduction: pred = ‖R(U)‖ − ‖R(U) + J̃ ΔU‖, the linear model of the residual without the pseudo-time term;
- actual reduction: act = ‖R(U)‖ − ‖R(U + α ΔU)‖, the next step's residual (already evaluated: no extra residual);
- ρ = act / pred; the norm is the SER's, over the equations it does not exempt, each scaled by its value at the start of the order (the same scaling as the SER ratio, so that no equation dominates by its units).

The next CFL number is the SER's value, capped by a factor of the current one: CFL_{n+1} = min(SER_{n+1}, c(ρ_n) CFL_n, the shortened-step cap), c(ρ) as the parameters section sets it. A step is never rejected for its ratio (the state moves as now): the ratio only governs how far the CFL number may go next. One product J̃ ΔU per step (the assembled matrix, no residual) and three norms; the state the rule needs (the last ρ, or the factor it set) is restart state.

Relation to the SER and to #356: the SER stays the law the CFL number grows by; c(ρ) only limits it. #356's under-solved rule, if merged first, stays as its own trigger (a solve above 0.1 lowers the next CFL number as a shortened step): the two caps combine by their minimum, declared together.

## Parameters

*(CFD - Euler: c(ρ) and its thresholds from Kelley and Keyes 1998, SIAM J. Numer. Anal. 35(2), and Ceze and Fidkowski; none tuned on #395's cases.)*

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

*(CFD - Euler: the Euler no-regression gate, tolerances in numbers.)*
