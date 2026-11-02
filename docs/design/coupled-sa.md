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
1. the perturbed flow's primitives, cell and boundary, in the frozen workspace (U + ε v_U);
2. μ_t, cell and boundary, from the perturbed ν̃ and those perturbed primitives (μ_t = ρν̃ f_v1(ρν̃/μ(T)) depends on the flow through T and μ(T) under Sutherland's law), into a buffer of the caller's (`eddy_viscosity_into`), not into the flow's cache;
3. the flow's residual at U + ε v_U with that μ_t (`NavierStokes::residual_frozen`, #402's frozen limiter and fallback, NS's viscous workspace);
4. ν̃'s residual at ρν̃ + ε v_ν̃ reading the perturbed flow (`SpalartAllmaras::freeze` and `residual_frozen`, a workspace of its own).

The baseline R(U, ρν̃) of the product is evaluated by the same sequence at the current state, so that both ends of the difference take the same path. Its switches are held at the current state, as #402's `keep`: the upwind side of its second-order convection (the mass flux's sign), the branches of its source (Allmaras, Johnson and Spalart 2012, Note 1(c)'s S̃ limit, r = min(·, 10), S̃'s sign), the SA-neg branch on ν̃'s sign (also selecting f_n), and ν̃'s sign at a face. Not held: the positivity fallback of the reconstructed face states (Euler's limited reconstruction and the unlimited `face_value` the viscous terms and ν̃'s residual share). A probe admissible in the cells can still switch it at a face (#429 review: a reconstructed face density crossing zero under a relative change of 2·10⁻⁶), so a probe whose evaluation switches any of them (`NavierStokes::frozen_switches`) or whose residual is not finite is inadmissible: ε is halved, as for the cell bounds. Roe's eigenvalue |v·n| is neither held nor smooth (#402); a probe across v·n = 0 at a face is not detected.

ε is #397's: √ε_R (1 + ‖S⁻¹(U, ρν̃)‖) / ‖S⁻¹ v‖, with the flow's freestream scales and, for ρν̃, the root mean square of the current ρν̃ over the owned cells (it reaches 10³–10⁴ ρ∞ν̃∞ in the boundary layer; the freestream value would weigh it wrongly); the probes' admissibility (density and pressure within 10 %, ρν̃ within the bounds below) and representability as in #397. Equations at the round-off floor are left out of the solve (amendment 2 below).

**The preconditioner** is the two existing solves in consecutive order (HANIM §II.D.1): one multigrid cycle (or the line sweeps) on the flow's V/Δt + J̃, then ν̃'s implicit solve on its own V/Δt + J̃_ν̃ and its own right-hand side. J̃_ν̃ holds only the scalar blocks ∂R_ν̃/∂ν̃ (`SpalartAllmaras::assemble`: diagonal, upper and lower) and no ∂R_ν̃/∂U block, so the flow's correction does not enter ν̃'s preconditioning solve; the coupling is the product's alone. It is not a linear operator: the multigrid's K-cycle has coarse Krylov coefficients and a correction scaling that depend on the right-hand side, so the preconditioner may change from one GCR iteration to the next; GCR's flexible form (Eisenstat, Elman and Schultz 1983, with a variable right preconditioner, as `solvers::Gcr` is written) is kept for that reason.

**The GCR's norm** (amendment 1, after the first gate run): the GCR's inner products, and with them its tolerance and failure tests, are on the S⁻¹-scaled residual, S the scales of ε above (the flow's freestream scales, the rms of ρν̃), so that the iteration minimises ‖S⁻¹ r‖. This is the left scaling S₁ = D_F of the scaled iterative solvers in SUNDIALS (Linear Algebraic Solvers, eq. 10.1–10.2, minimising ‖S₁ P₁⁻¹(b − A x)‖₂; KINSOL §7.2.4 chooses D_F so that D_F F has components of roughly one magnitude, and §7.2.9 tests ‖J δ + F‖_{D_F}; read in full). With right preconditioning only, as here, S₂ does not change GCR's minimisation. Measured without it (#395, comment 6584507926): the unscaled norm is dominated by the energy rows, so on SA 69×49 an accepted step (unscaled drop 0.80) multiplied momentum-z's scaled residual by 2.6·10⁹; momentum-z left its round-off floor (W0710), its relative residual then held the SER's CFL number at 1 for the remaining 5433 steps, and the run stalled.

**Equations at the round-off floor** (amendment 2, declared before its run): an equation the runtime exempts as at its round-off floor (the span momentum of a one-cell-thick mesh, identically zero by symmetry) is left out of the coupled solve: its rows of the right-hand side and of each preconditioned direction are zero, and its update is zero. Its residual is round-off, so the finite difference resolves no derivative for it and solving for it only puts round-off into its unknowns; with the scaled norm it still has almost no weight in the GCR, which can raise its residual (measured with amendment 1 alone, #395 comment 6584798608, steps 1235–1257 of SA 69×49: ×162, ×21, ×1.7, then W0710 at step 1257, after the ten orders). An equation that leaves its floor re-enters the solve from the next step, as the exemption decides. The flow's own matrix-free step (laminar) keeps the assembled rows of matrix-free-step.md.

**μ_t at the current state.** The coupled step evaluates the flow's residual with the μ_t of the current (U, ρν̃) (one more flow residual a step), not with the one the previous step wrote from its own primitives: the product's base is then the step's residual, and a continued run repeats the uninterrupted one.

**The update and its bounds.** The loose step's clip of ρν̃'s update to −0.5 … +2 × max(|ρν̃|, ρν̃∞) moves into the step's admissibility check: the accepted fraction α is the largest that keeps every cell's density and pressure above (1 − 0.5) of their values, as now, and ρν̃'s change within those bounds; the step is shortened as a whole (α on all six unknowns), not clipped component by component, which would leave the Newton direction. SA-neg allows ρν̃ < 0, so the bound is relative to max(|ρν̃|, ρν̃∞), not ρν̃ ≥ 0. μ_t follows the accepted ν̃.

**State, ranks, cost.** No new restart state beyond the loose step's (ρν̃ is already a field). Every reduction is an exact sum over the owned unknowns, ε the same on every rank. Per GCR iteration: one coupled frozen residual (the flow's and ν̃'s), the preconditioner's two solves; per step, the two `freeze` and the frozen residuals at the current state.

## Gate, declared before any run

On `physics/ns-sa` with this change and the flow product of #397 (the matrix-free flow model, positive on the laminar plate), fp64 unless stated, against the same branch with the loose step and the same flow product (the case files identical):

- **TMR flat plate, SA** (`physics/tmr/case.sh flatplate`, 6000 steps; #395's measure: ten orders below each equation's peak, the identically zero z-momentum excepted, every such equation at most 10⁻¹⁰ of its peak at the last step, no W0710): 69×49 in at most 3332 steps, 35×25 in at most 2260, 137×97 within the 6000.
- **The converged solution is the discretisation's**: on 69×49, the skin friction along the plate and the velocity and ν̃ profiles at x = 0.97 within 10⁻⁶ of their magnitude of an independent run converged with the CFL number held at 1 (the loose step converged there, #395).
- **Unchanged**: the laminar plate (no ν̃: bit for bit with the flow product alone); the Euler cases (no SA: bit for bit with `main`).
- **RAE 2822 case 9** (transonic SA, limiter active): this change is accepted for the TMR plate cases only. Its activation on a transonic case waits for a separate gate on RAE 2822, declared before that run once the grid exists (convergence within a declared budget and forces against a reference run of the same discretisation), and fails if that gate does.
- **Accuracy** (AGENTS.md): the discretisation does not change; the SA manufactured solution's observed orders the same as the loose step's on the same commit, the errors within 10⁻⁶ relative on every level (`rans-sa-mms-order` does not pass today with the loose step either: ρE 1.84 fitted, 1.91 on the finest pair, Cartesian family 16–48, measured before 46c2e04; #137's status, verification incomplete), fp32+acc64 agreeing with fp64 level by level (`autan-vv agree`).
- **Time to ten orders against the loose step at CFL 1** (the real competitor: the loose step with the CFL number held at 1 converges on the plate, 2777–3040 steps, #395): on SA 69×49 and SA 35×25, the CPU time of autan-run to ten orders (#395's measure) with this change is at most that of the loose step held at CFL 1, both measured alone under the GPU lock on the same machine (spark-5512 if its STREAM bandwidth, recorded with the runs, is within 3 % of its baseline; otherwise both interleaved in the same window). Converging in fewer steps but more time fails this gate; it is not relaxed afterwards.
- **Cost per step**: CPU time per step against the loose step with the same flow product, measured alone under the GPU lock, reported.
- **Tests** (public seams): the coupled product agrees with central differences of the coupled residual at a fixed state (SA MMS state, limiter off) to O(ε); at states built to exercise every frozen branch of ν̃'s residual — negative ν̃ (SA-neg), the S̃ limit active, r at its cap of 10, S̃ < 0, a mass flux changing sign across a face — the frozen product equals central differences of the frozen residual (the base state's selections held), and a probe that crosses a branch boundary keeps the base state's selection; a run continued at a coupled step repeats the uninterrupted one bit for bit; a step whose ν̃ update would leave the bounds is shortened (α < 1, logged), not clipped.

## Gate, re-declared after amendment 1 (before any run)

First run of the gate above, without amendment 1 (#429, a82dca8c): SA 35×25 passed (ten orders at step 447); SA 69×49 failed (6.2–8.1 orders at 6000 steps, W0710 at step 567). The same gate applies to the change with amendment 1, its thresholds unchanged (69×49 ≤ 3332 steps, 35×25 ≤ 2260, 137×97 within 6000, no W0710; profiles, unchanged runs, RAE 2822, accuracy, cost and tests as above), with one competitor redefined:

- **Time to ten orders**: the competitor is the configuration that converged the plate in #395, c3966068 with #277 reverted (out `r-880dcce4`: ten orders on 69×49 at step 3042), run as is with its own binary. The loose step with the CFL number held at 1 on this branch does not converge (cfl_max = 1 on a82dca8c: 1.9 orders on 35×25, 2.2 on 69×49 in 6000 steps), so it cannot be the competitor. On SA 35×25 and 69×49, the CPU time (`solve_seconds`) of autan-run to the step of ten orders with this change is at most the competitor's, both run to their own step of ten orders, interleaved in one hold under the GPU lock on spark-5512, its STREAM bandwidth recorded. Converging in fewer steps but more time fails; it is not relaxed afterwards.

## Amendment 2: the gate, unchanged

With amendment 1 (8bc2c05f), SA 69×49 reached ten orders at step 1015 and failed "no W0710" (W0710 at step 1257, after the ten orders); SA 35×25 passed at step 291; 137×97 ran. Amendment 2 runs the gate above unchanged, "no W0710" over the whole run included, on the same three plates, with the time comparison of the re-declared gate. If it fails, the floor's calibration (#354) is the question, not this note.
