# The true Jacobian where the limiter does not switch

Design for #395 (the flow's model; the coupled SA solve is note 2), written before any code by CFD - Euler and CFD - Multigrille. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃. This note applies the true Jacobian matrix-free in the step's linear solve, as #397 did, but only while an objective test finds the limiter and the reconstruction's fallback inactive over the step, and keeps J̃ elsewhere.

## Problem and what was measured

- With the SER free to raise the CFL number, the TMR flat plate 69×49 does not converge on the assembled J̃: the laminar plate wanders at −3 to −4, the SA plate locks into a period-2 cycle (#395).
- J̃'s gap to J at second order is 15–25 % over the domain and 38–70 % on the wall cells' continuity and energy rows, and it is convective (99.5–100 %): the reconstruction's gradients held fixed (#395).
- Filling that gap inside J̃'s face-neighbour pattern fails (note 1, `design/reconstruction-jacobian`, both candidates above 0.5 × on the wall ring's continuity rows).
- The true Jacobian applied matrix-free with the limiter and fallback frozen at U (#397) converged the laminar plate in 1655 steps and SA 35×25 in 2017, but lost steps or convergence wherever the limiter is active (Ni's bump 384 not converged; NACA 0012 256²: +19 %, +21 %, +193 % steps). The residual does not follow a model frozen across the limiter's switches at the SER's CFL numbers.

## The activity criterion

**What switches.** Between its switches the residual is C∞ in U: the fluxes of the selected states are rational in W, and Venkatakrishnan's ψ(d) = (Δ² + ε² + 2dΔ) / (Δ² + 2d² + dΔ + ε²) has a denominator (Δ + d/2)² + 7d²/4 + ε² > 0 ([euler.md](euler.md), Venkatakrishnan 1995, eq. 5; `venkat` in `kernels/euler.py`). The reconstruction makes five discrete selections per cell c and component k, all read from `kernels/euler.py`:
- s1, `at_most_one`: whether φ_ck = min(min_f ψ_f, 1) takes the 1;
- s2, the limiter's `Min` over the cell's faces: the face f attaining min_f ψ_f;
- s3, the bounds' `Min` and `Max` over the gradient stencil: the entries attaining W_min and W_max;
- s4, `venkat`'s `dp = dmax if d > 0.0 else dmin`: which bound each face's increment d_f is measured against (ψ is continuous at d = 0, its slope is not);
- s5, `reconstruct`'s positivity select on each face side: the reconstructed value or the cell value.
The flux functions' own switches (|λ|, Harten's entropy fix, the upwind side) are the first-order scheme's, which J̃ already differentiates exactly; they are outside the criterion.

**The criterion.** A cell is active for step n when any of its selections s1–s5 differs between the step's base state U_{n−1} and a state the step evaluated (its accepted state U_n, or a rejected trial). Each residual evaluation records, per owned cell, a 64-bit hash of its selections, filled into the ghosts; a step compares its base state's hashes with those of each state it evaluates. A row of the step n + 1 operator is matrix-free when its cell and every face neighbour are inactive, and assembled (J̃) otherwise: R_i reads only the reconstructions of cell i and of its face neighbours. The first step has no previous step, so all its rows are assembled, as on `main`. The mask is fixed for a whole linear solve, so the Krylov operator is one linear operator.

**Why zero is the threshold.** The step's model, R(U_n) + J ΔU for whatever J, is the first-order Taylor expansion of R along ΔU; its remainder is bounded by the second derivative along the segment only where R is C² on it (Taylor's theorem). A selection that changed between U_{n−1} and U_n means that R is not smooth on the segment just travelled at the step's own scale, ‖ΔU‖, and no finite threshold would be justified a priori: any change is the property tested, and none means no kink was crossed at the states seen. The threshold is a property, not a value fitted to any case.

**Declared limits.**
- The criterion is a predictor: it reads the step just taken, not the next. A switch first crossed by step n + 1 is seen at step n + 2.
- A selection that changes and changes back within one step is not seen.
- The finite-difference quotient of the product, at scale h ‖v‖ ≪ ‖ΔU‖, is #397's: the frozen residual of #402 holds φ and the fallback, and is C∞ on [U, U + hv].
- Cost: one hash per cell per residual evaluation, one ghost exchange of it, one comparison per step; no extra residual evaluation. The generated reductions do not output which term they selected, so the record is one more cell pass over the stencil that re-evaluates s1–s5 from the primitives, gradients and bounds the residual left (as `Euler::boundary_pressure` re-evaluates the face state), unless the generator adds an argmin output.

## Change

From the step after the switch to second order, on the host's steady step with `"multigrid"` or `"lines"` (the first-order phase, `"gauss-seidel"`, unsteady runs and the device unchanged), the linear solve's operator is mixed, row by row, with the mask the criterion section defines (fixed for the whole solve, so the Krylov operator is one linear operator):

  y_i = (V/Δt)_i v_i + [R_frozen(U + ε v) − R_frozen(U)]_i / ε   on a matrix-free row i;
  y_i = (A v)_i, A = V/Δt + J̃ the assembled matrix                 on any other row.

- R_frozen is #402's `residual_frozen` (φ and the reconstruction's fallback held at U, its own workspace; `NavierStokes` has the same API on `physics/ns-frozen`); ε, the admissible and representable probes and their failure rule are #397's (Knoll and Keyes 2004, §2.2, on the freestream-scaled unknowns, ε_R of the storage precision). Those parts of `docs/design/matrix-free-step.md` (#397, closed) are reused unchanged.
- The rows of equations at the round-off floor (the runtime's exemption) take the assembled operator, as in #397: their difference would be round-off over ε.
- The preconditioner is unchanged (one multigrid cycle, or the line sweeps, on V/Δt + J̃); tolerance, iteration limit, failure rule and SER unchanged.
- The difference quotient is a linear operator only to O(ε) (Knoll and Keyes 2004, §2.2: the forward difference's truncation error, and a direction-dependent ε); GCR treats it as linear, as every Jacobian-free Newton-Krylov method does. This is a declared approximation of the inexact Newton step, not checked per product; its effect is what the gate measures.
- A step with no matrix-free row computes no difference: it is `main`'s step, bit for bit (the first second-order step always is: the criterion has no record before it).
- State between steps: each owned cell's selection hash of the last step, restart state (8 bytes a cell), so that a continued run rebuilds the same mask and repeats the uninterrupted one bit for bit; the hashes are exchanged to the ghosts and compared per cell, so the mask is the same on any number of ranks, and every reduction of the product is #397's (exact sums over the owned unknowns).
- Logged: a `linear` info event at the first step with matrix-free rows; per run, the fraction of matrix-free rows (smallest, mean, largest over the second-order steps) in the run's end event, so that the gate can report how much of each case ran matrix-free.

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
