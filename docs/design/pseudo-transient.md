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

*(CFD - Multigrille: the plate, unchanged from #395 — laminar 69×49 ≤ 3332 steps, SA 35×25 ≤ 2260, SA 69×49 ≤ 3332, SA 137×97 < 6000, ten orders below each equation's peak and every such equation ≤ 10⁻¹⁰ of its peak at the last step; the transonic SA case with an active limiter (case, mesh and reference from CFD - Navier-Stokes SA).)*

*(CFD - Euler: the Euler no-regression gate, tolerances in numbers.)*
