# The flow and ν̃ solved together in the steady step

Design for #395, note 2 (the SA coupling), written before any code by CFD - Multigrille with CFD - Navier-Stokes SA (co-signing). The steady implicit step of `physics/ns-sa` solves the flow (5 equations) by GCR on (V/Δt + J̃) ΔU = −R(U), then updates ρν̃ by its own implicit step with ν̃'s approximate Jacobian, clipped to −0.5 … +2 of max(|ρν̃|, ρν̃∞) per step, the flow's eddy viscosity μ_t following ν̃ after the update (loose coupling). This note solves both together: one GCR on the six equations, its product the derivative of the coupled residual, preconditioned by the two existing solves in turn.

## Problem, measured (#395)

| TMR flat plate, fp64, 6000 steps | Assembled J̃ (`main`'s flow step) | Flow product matrix-free (#397) |
|---|---|---|
| laminar 69×49 | wanders at −3 to −4, never ten orders | ten orders at step 1655 |
| SA 35×25 | — | ten orders at step 2017 |
| SA 69×49 | period-2 cycle at −3 to −4 | not converged: 7.1–8.5 orders (ν̃ 6.6) |
| SA 137×97 | — | not converged: 7.2–8.3 orders (ν̃ 6.7) |

- With the flow's true Jacobian the laminar plate converges and the SA plate does not: what remains is the coupling.
- Inside the SA cycle every equation enters the alternation within 12 steps of the others, ν̃ with the largest amplitude (5.8 % a step); freezing ν̃ and μ_t inside the cycle leaves the flow wandering as the laminar flow does on J̃ (#395, isolation diagnostic).
- In the loose step the flow's linear model has no ∂R_flow/∂ν̃ (μ_t is held) and ν̃'s has no ∂R_ν̃/∂U (the flow is held): at the CFL numbers the SER reaches, each solve moves its unknowns against a partner that the next solve moves back.

## Source

Wang, Diskin, Nielsen and Liu 2021, "Improvements in Iterative Convergence of FUN3D Solutions", AIAA 2021-0857 (NTRS 20205010283; read in full, §II, the method), FUN3D's HANIM with SA-neg:

- §II.D.2, eq. 5–6: GCR solves the exact linearisation of the **coupled** mean-flow and turbulence-model equations, the product a Fréchet (finite-difference) derivative of the coupled residual, ε = max(1, ‖Qⁿ‖) ε₀ (eq. 7), ε₀ ≈ 10⁻⁷ for turbulent flows at moderate or high speed and ≈ 10⁻⁴ for low-speed turbulent or laminar flows.
- §II.D.1: the preconditioner is the defect correction with the approximate Jacobians, the mean flow and the turbulence model "decoupled and solved in consecutive order".
- §II.B: the legacy solver (decoupled linear iterations, as Autan's loose step) is what HANIM replaces.

## Change

*(CFD - Multigrille, to complete with NS: from the second order on, on the host's steady step of `NavierStokes` with SA: the unknowns (U, ρν̃), six a cell; GCR (≤ 10 iterations, tolerance 0.1, as now) on (V/Δt + J) Δ = −R(U, ρν̃) with J the coupled Jacobian by finite differences of the coupled residual — the flow residual with μ_t from the perturbed ν̃, ν̃'s residual from the perturbed flow — ε by #397's rule on the scaled unknowns (ν̃'s scale ρ∞ν̃∞) and its admissible and representable probes, the limiter and fallback frozen at U (#402, NS's `residual_frozen`, and the same for ν̃'s residual if it has a limiter); the preconditioner: the flow's multigrid cycle on V/Δt + J̃_flow, then ν̃'s implicit solve on V/Δt + J̃_ν̃ with the flow's correction (HANIM's consecutive order); the ν̃ clip applied to the accepted update's ν̃ component as today, and μ_t from the updated ν̃; restart state; ranks.)*

## Gate, declared before any run

*(To complete: the TMR flat plate — SA 69×49 ≤ 3332 steps, SA 35×25 ≤ 2260, SA 137×97 within 6000 (#395's measure); the laminar plate unchanged (no ν̃); RAE 2822 case 9 when its grid exists; the SA MMS orders (`rans-sa-mms-order`) unchanged, fp32+acc64 agreeing with fp64; Euler bit for bit (no SA); cost per step against the loose step with the matrix-free flow, measured alone under the GPU lock.)*
