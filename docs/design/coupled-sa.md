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

From the step after the switch to second order, on the host's steady step of `NavierStokes` with SA (laminar runs, Euler and the device unchanged), the step's unknowns are (U, ρν̃), six a cell, and one GCR solves

  (V/Δt + J) Δ = −R(U, ρν̃),   J the coupled Jacobian,

with the solve's tolerance (0.1), iteration limit (10) and failure rule (0.9: the step rejected) unchanged.

**The product** is the finite-difference derivative of the coupled residual, its non-differentiable switches frozen at the current state (HANIM eq. 6, with #397's step rule; NS's API, CFD - Navier-Stokes SA codes it):
1. μ_t from the perturbed ν̃, into a buffer of the caller's (`eddy_viscosity_into`), not into the flow's cache;
2. the flow's residual at U + ε v_U with that μ_t (`NavierStokes::residual_frozen`, #402's frozen limiter and fallback, NS's viscous workspace);
3. ν̃'s residual at ρν̃ + ε v_ν̃ reading the perturbed flow (`SpalartAllmaras::freeze` and `residual_frozen`, a workspace of its own). Its switches are held at the current state, as #402's `keep`: the upwind side of its second-order convection (the mass flux's sign), the branches of its source (Allmaras, Johnson and Spalart 2012, Note 1(c)'s S̃ limit, r = min(·, 10), S̃'s sign), the SA-neg branch on ν̃'s sign, and the flow face values' positivity fallback.

ε is #397's: √ε_R (1 + ‖S⁻¹(U, ρν̃)‖) / ‖S⁻¹ v‖, with the flow's freestream scales and, for ρν̃, the root mean square of the current ρν̃ over the owned cells (it reaches 10³–10⁴ ρ∞ν̃∞ in the boundary layer; the freestream value would weigh it wrongly); the probes' admissibility (density and pressure within 10 %, ρν̃ within the bounds below) and representability as in #397. The rows of equations at the round-off floor take the assembled operator's rows, as in #397.

**The preconditioner** is the two existing solves in consecutive order (HANIM §II.D.1): one multigrid cycle (or the line sweeps) on the flow's V/Δt + J̃, then ν̃'s implicit solve on its V/Δt + J̃_ν̃ with the flow's correction applied to ν̃'s right-hand side through its existing Jacobian blocks if NS's J̃_ν̃ has them, and without them otherwise (stated in the code and the tests). It is not a linear operator of fixed form across iterations unless both are run to a fixed count, which they are (one cycle; ν̃'s fixed sweeps), so GCR's flexible form is unchanged.

**The update and its bounds.** The loose step's clip of ρν̃'s update to −0.5 … +2 × max(|ρν̃|, ρν̃∞) moves into the step's admissibility check: the accepted fraction α is the largest that keeps every cell's density and pressure above (1 − 0.5) of their values, as now, and ρν̃'s change within those bounds; the step is shortened as a whole (α on all six unknowns), not clipped component by component, which would leave the Newton direction. SA-neg allows ρν̃ < 0, so the bound is relative to max(|ρν̃|, ρν̃∞), not ρν̃ ≥ 0. μ_t follows the accepted ν̃.

**State, ranks, cost.** No new restart state beyond the loose step's (ρν̃ is already a field). Every reduction is an exact sum over the owned unknowns, ε the same on every rank. Per GCR iteration: one coupled frozen residual (the flow's and ν̃'s), the preconditioner's two solves; per step, the two `freeze` and the frozen residuals at the current state.

## Gate, declared before any run

On `physics/ns-sa` with this change and the flow product of #397 (the matrix-free flow model, positive on the laminar plate), fp64 unless stated, against the same branch with the loose step and the same flow product (the case files identical):

- **TMR flat plate, SA** (`physics/tmr/case.sh flatplate`, 6000 steps; #395's measure: ten orders below each equation's peak, the identically zero z-momentum excepted, every such equation at most 10⁻¹⁰ of its peak at the last step, no W0710): 69×49 in at most 3332 steps, 35×25 in at most 2260, 137×97 within the 6000.
- **The converged solution is the discretisation's**: on 69×49, the skin friction along the plate and the velocity and ν̃ profiles at x = 0.97 within 10⁻⁶ of their magnitude of an independent run converged with the CFL number held at 1 (the loose step converged there, #395).
- **Unchanged**: the laminar plate (no ν̃: bit for bit with the flow product alone); the Euler cases (no SA: bit for bit with `main`).
- **RAE 2822 case 9** (transonic SA, limiter active), once its grid exists: reported, and this change cannot be accepted for transonic SA before it runs.
- **Accuracy** (AGENTS.md): the discretisation does not change; the SA manufactured solution's observed orders (`rans-sa-mms-order`) unchanged in fp64, fp32+acc64 agreeing with fp64 level by level (`autan-vv agree`).
- **Cost**: CPU time per step against the loose step with the same flow product, measured alone under the GPU lock, reported with the time to convergence.
- **Tests** (public seams): the coupled product agrees with central differences of the coupled residual at a fixed state (SA MMS state, limiter off) to O(ε); a run continued at a coupled step repeats the uninterrupted one bit for bit; a step whose ν̃ update would leave the bounds is shortened (α < 1, logged), not clipped.
