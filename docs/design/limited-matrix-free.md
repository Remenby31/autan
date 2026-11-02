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

*(CFD - Multigrille: #397's product (Knoll and Keyes ε on the scaled unknowns, admissible and representable probes, the frozen residual of #402, the floor rows on the assembled operator), used when the criterion holds, the assembled J̃ otherwise; the preconditioner unchanged; restart state; ranks.)*

## Gate, declared before any run

*(To be completed: the laminar TMR flat plate 69×49 ≤ 3332 steps on `physics/ns-sa` (#395's measure); Euler no-regression unchanged — limited cases in at most `main`'s steps (tolerance 0), smooth cases ≤ 1.02 ×, forces within 10⁻⁸ |c| + 10⁻¹⁰, first order bit for bit; RAE 2822 case 9 once its grid exists; cost ≤ 1.30 × per step, measured alone under the GPU lock. SA 69×49 stays note 2's.)*
