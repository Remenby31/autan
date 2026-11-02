# The true Jacobian where the limiter is inactive

Design for #395 (the flow's model; the coupled SA solve is note 2), written before any code by CFD - Euler and CFD - Multigrille. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃. This note applies the true Jacobian matrix-free in the step's linear solve, as #397 did, but only on the rows whose cells the limiter leaves untouched (φ = 1 in every component, the reconstruction's fallback unchanged, at the step's two base states), where the frozen product is the true Jacobian, and keeps J̃ elsewhere.

## Problem and what was measured

- With the SER free to raise the CFL number, the TMR flat plate 69×49 does not converge on the assembled J̃: the laminar plate wanders at −3 to −4, the SA plate locks into a period-2 cycle (#395).
- J̃'s gap to J at second order is 15–25 % over the domain and 38–70 % on the wall cells' continuity and energy rows, and it is convective (99.5–100 %): the reconstruction's gradients held fixed (#395).
- Filling that gap inside J̃'s face-neighbour pattern fails (note 1, `design/reconstruction-jacobian`, both candidates above 0.5 × on the wall ring's continuity rows).
- The true Jacobian applied matrix-free with the limiter and fallback frozen at U (#397) converged the laminar plate in 1655 steps and SA 35×25 in 2017, but lost steps or convergence wherever the limiter is active (Ni's bump 384 not converged; NACA 0012 256²: +19 %, +21 %, +193 % steps). The residual does not follow a model frozen across the limiter's switches at the SER's CFL numbers.

## The activity criterion

**What switches.** Between its switches the residual is C∞ in U: the fluxes of the selected states are rational in W, and Venkatakrishnan's ψ(d) = (Δ² + ε² + 2dΔ) / (Δ² + 2d² + dΔ + ε²) has a denominator (Δ + d/2)² + 7d²/4 + ε² > 0 ([euler.md](euler.md), Venkatakrishnan 1995, eq. 5; `venkat` in `kernels/euler.py`). The reconstruction's discrete selections, per cell c and component k, all read from `kernels/euler.py`:
- s1, `at_most_one`: whether φ_ck = min(min_f ψ_f, 1) takes the 1. The criterion reads the strict condition before the clamp, min_f ψ_f > 1. A tie, min_f ψ_f = 1, also gives φ = 1, but with one-sided derivatives (Δ = ε² = 1, d = ½: d(φd)/dd is 1 on one side and 5/6 on the other, where the frozen product gives 1); a tie is therefore not eligible;
- s2, the limiter's `Min` over the cell's faces: the face attaining min_f ψ_f;
- s3, the bounds' `Min` and `Max` over the gradient stencil, boundary states included: the entries attaining W_min and W_max;
- s4, `venkat`'s `dp = dmax if d > 0.0 else dmin`: the bound each face's increment is measured against (ψ is continuous at d = 0, its slope is not);
- s5, `reconstruct`'s positivity select on each face side: the reconstructed value or the cell value.

**Where the frozen product is the true Jacobian.** #402's frozen residual holds φ and the s5 branches at U and recomputes everything else. Where min_f ψ_f > 1 for every component, φ ≡ 1 in a neighbourhood of U (an open condition): there R does not depend on φ, so s2–s4 and the bounds play no part, and holding φ costs nothing. Where φ < 1, φ varies smoothly with U and the frozen product drops its derivative: d(φ d) = φ δd + d δφ, of which the product keeps φ δd (with d = Δ = ε² = 1, φ = 0.8 and d(φd)/dd = 0.4 against 0.8 frozen). Such cells are therefore never matrix-free, switched or not: the matrix-free rows apply the true Jacobian, not #397's frozen one.

**The criterion.** A cell is eligible at step n when, at both of the last two base states U_{n−1} and U_n, min_f ψ_f > 1 for all five components (or the limiter is off), and its face sides' s5 branches are the same at both. A row of the step's operator is matrix-free when its cell and every face neighbour are eligible, and assembled (J̃) otherwise: R_i reads only the reconstructions of cell i and of its face neighbours. Each `residual_and_jacobian` at a base state records, per owned cell, one flag (min_f ψ_f > 1 for all components) and a 64-bit hash of its s5 branches, filled into the ghosts; a rejected step keeps U, its record and therefore the mask. The first second-order step has no previous record, so all its rows are assembled, as on `main`; a continuation carries the base state's record (Change). The mask is fixed for a whole linear solve, so the Krylov operator is one linear operator.

**Why zero is the threshold.** The step's model, R(U_n) + J ΔU for whatever J, is the first-order Taylor expansion of R along ΔU; its remainder is bounded by the second derivative along the segment only where R is C² there (Taylor's theorem). An s5 branch that changed between U_{n−1} and U_n, or a φ below 1, means that R is not smooth or not frozen-exact on the segment just travelled, at the step's own scale ‖ΔU‖, and no finite margin would be justified a priori: the tests are properties (a flag, an equality), not values fitted to any case.

**Declared limits.**
- The criterion is a predictor: it reads the step just taken, not the next. A switch first crossed by step n + 1 is seen at step n + 2, and a switch crossed and crossed back between two base states is not seen.
- The flux functions' own switches (|λ| and Harten's entropy fix in Roe, the wave-speed selections of HLLC, AUSM+-up's split functions, the characteristic boundary states' sub- or supersonic and in- or outflow branches) are not in the criterion. They are the first-order scheme's: J̃ linearises across them the same way, so they do not separate the two operators. The finite-difference quotient (scale h ‖v‖, h ≈ √ε relative, Knoll and Keyes 2004) crosses one only where its argument is within that distance of the switch, such as |v·n| at a stagnation point or a stationary contact; there the quotient is a secant across the kink, as #397's was.
- Cost: one flag and one hash per cell per base state, one ghost exchange of them, one comparison per step; no extra residual evaluation. Neither is stored by the residual: the clamped φ cannot tell a tie from min_f ψ_f > 1, the fused limiter keeps the bounds in registers (#414), and the s5 branches are not written. The record is therefore one more pass over the gradient stencil and the faces: it recomputes the bounds and min_f ψ_f, and re-evaluates the s5 branches from the primitives, gradients and φ the residual left (as `Euler::boundary_pressure` re-evaluates the face state). Test (public seam): a cell built with min_f ψ_f = 1 exactly in one component is not eligible; with min_f ψ_f > 1 it is.

## Coverage check, declared before measuring

Before any code, on states the current solver already produces, the criterion is applied to the laminar TMR flat plate 69×49 (#395's case and protocol, `physics/ns-sa`, no solver change). The count goes into the run's diagnostic output only.

**States.** The base-state pairs (999, 1000), (1999, 2000) and (2999, 3000) of the run that wanders at −3 to −4 (#395).

**What is measured.** At each pair:
- Rows: a row is matrix-free when its cell and every face neighbour are eligible at both base states of the pair (the criterion above).
- Gap: g = J v − J̃ v on the wall ring's continuity rows (the cells with a face on the plate, #395's ring), for #395's three random v. J v is taken by central differences with ε = 10⁻⁵, and J̃ v by the assembled operator.
- Remaining gap: ‖g restricted to the rows that are not matrix-free‖ / ‖g‖. It is what the mixed operator leaves on the ring if its matrix-free rows are exact.
- Shares of matrix-free rows over the ring and over the domain: reported.

**Threshold, set now.** The remaining gap must be at most 0.5 for each of the three v at all three pairs. That is note 1's threshold, ‖(J − J̃) v‖ ≤ 0.5 × its value on the ring's continuity rows, applied to the mixed operator. The gap is measured, not modelled: no assumption on how it is spread over the rows.

**Reading.**
- If the threshold is met, the note proceeds to its PR.
- If any value is above 0.5, the matrix-free rows do not reach enough of the gap where J̃ misses 38–70 %. The note then does not address #395's symptom: that is recorded in #395 and nothing is coded.

**Declared limit of this check.** The diagnostic reads the stored φ, so it counts a tie, min_f ψ_f = 1, as eligible. In FP64 an exact tie is not expected on these states. Its effect could only raise the share of matrix-free rows, which the reading does not gate.

## Change

From the step after the switch to second order, on the host's steady step with `"multigrid"` or `"lines"` (the first-order phase, `"gauss-seidel"`, unsteady runs and the device unchanged), the linear solve's operator is mixed, row by row, with the mask the criterion section defines (fixed for the whole solve, so the Krylov operator is one linear operator):

  y_i = (V/Δt)_i v_i + [R_frozen(U + ε v) − R_frozen(U)]_i / ε   on a matrix-free row i;
  y_i = (A v)_i, A = V/Δt + J̃ the assembled matrix                 on any other row.

- R_frozen is #402's `residual_frozen` (φ and the reconstruction's fallback held at U, its own workspace; `NavierStokes` has the same API on `physics/ns-frozen`); ε, the admissible and representable probes and their failure rule are #397's (Knoll and Keyes 2004, §2.2, on the freestream-scaled unknowns, ε_R of the storage precision). Those parts of `docs/design/matrix-free-step.md` (#397, closed) are reused unchanged.
- The rows of equations at the round-off floor (the runtime's exemption) take the assembled operator, as in #397: their difference would be round-off over ε.
- The preconditioner is unchanged (one multigrid cycle, or the line sweeps, on V/Δt + J̃); tolerance, iteration limit, failure rule and SER unchanged.
- The difference quotient is a linear operator only to O(ε) (Knoll and Keyes 2004, §2.2: the forward difference's truncation error, and a direction-dependent ε); GCR treats it as linear, as every Jacobian-free Newton-Krylov method does. This is a declared approximation of the inexact Newton step, not checked per product; its effect is what the gate measures.
- A step with no matrix-free row computes no difference: it is `main`'s step, bit for bit (the first second-order step always is: the criterion has no record before it).
- State between steps: each owned cell's record of the last base state (its eligibility flag and its fallback hash, criterion section), restart state (1 flag and 8 bytes a cell), so that a continued run rebuilds the same mask and repeats the uninterrupted one bit for bit; the hashes are exchanged to the ghosts and compared per cell, so the mask is the same on any number of ranks, and every reduction of the product is #397's (exact sums over the owned unknowns).
- Logged: a `linear` info event at the first step with matrix-free rows; per step, the share of matrix-free rows in the `linear` events' data, and per run its smallest, mean and largest over the second-order steps in the run's end event, so that the gate reports how much of each case ran matrix-free.

A risk stated before any measurement: where the limiter is active over much of the domain (φ < 1 in many cells, as it may be in the plate's boundary layer), most rows stay on J̃, and this change can do much less than #397's 1655 steps on the laminar plate. The gate decides; the share of matrix-free rows per step is reported with it.

Cost: on a step with matrix-free rows, `freeze` at U (primitives, boundary states, gradients and bounds, limiter, ghost exchanges, fallback flags), one frozen residual at U and one per GCR iteration (at most 10), the probes' checks and their reductions (#397's), one assembled-matrix product per GCR iteration for the assembled rows, and the criterion's record pass (Euler's section); on a step without, the record pass only. All of it is in the cost gate.

## Gate, declared before any run

Against `main` at the merge base unless stated (the case files identical); fp64 unless stated.

- **Laminar TMR flat plate 69×49** on `physics/ns-frozen` with this change, against the same branch without it: ten orders below each equation's peak (the identically zero z-momentum excepted) in at most 3332 steps, every such equation at most 10⁻¹⁰ of its peak at the last step, and no W0710 (#395's measure, its reason stated there: the runtime's rule counts from iteration 1, where the plate's density sits at 10⁻⁸). The converged solution is the discretisation's, checked against an independent run of the same branch without the change with the CFL number held at 1 to the same ten orders (it converged on the plate, #395): skin friction along the plate and the velocity profile at x = 0.97 within 10⁻⁶ of their magnitude, so that a wrong product cannot pass by converging somewhere else.
- **Euler, no regression** (the cases of #356 and #397, case and mesh hashes recorded):
  - first order bit for bit: every history identical to `main`'s, line for line, up to the switch to second order;
  - limited cases, NACA 0012 at 256² (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°) and Ni's bump 384: converged to ten orders in at most `main`'s steps (tolerance 0);
  - smooth cases, Euler MMS hex, tet and poly (levels 16–32) and Ni's bump 96: at most 1.02 × `main`'s steps, the gain or loss recorded per case;
  - forces at convergence within 10⁻⁸ |c| + 10⁻¹⁰ of `main`'s;
  - fp32+acc64, NACA 0012 M 0.8 α 0° and M 0.8 α 1.25° at 256²: ends at its round-off floor in at most `main`'s steps, its CL, CD and CM within 0.1 % of its fp64 run on the same mesh (10⁻⁶ absolute for the zero lift and moment at α 0°) and its surface Cp within 0.001 of it outside the shocks ([acceptance.md](../validation/acceptance.md));
  - the fraction of matrix-free rows reported per case.
- **RAE 2822 case 9** (transonic SA, limiter active), once its grid exists (CFD - Navier-Stokes SA, from the TMR ordinates): this change cannot be accepted for SA before it runs; its pass rule is the pseudo-transient note's (`docs/design/pseudo-transient.md`).
- **Cost**: CPU time per step at most 1.30 × without the change on the laminar plate and on the Euler cases, measured alone under the GPU lock; time to convergence reported.
- **Accuracy** (AGENTS.md): the discretisation does not change; the Euler MMS observed orders on hex, tet, poly and hybrid in fp64 unchanged (`euler-mms-order` PASS), fp32+acc64 agreeing with fp64 level by level (`autan-vv agree`).
- **Tests** (public seams): a run continued at a step whose mask holds both kinds of rows repeats the uninterrupted one bit for bit (histories and solution arrays); `"lines"` under `reproducibility = "ranks"` on 1, 2 and 4 ranks gives one history and one set of solution arrays, byte for byte, with matrix-free rows present, in fp64 and fp32+acc64; on an unlimited verification case past the switch every row becomes matrix-free once no selection changes, and on a case with an active limiter (a shock) the rows of the switching cells stay assembled.

SA 69×49 is note 2's (the coupled SA solve), with this note's flow model.
