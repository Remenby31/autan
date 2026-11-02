# The true Jacobian where the limiter does not switch

Design for #395 (the flow's model; the coupled SA solve is note 2), written before any code by CFD - Euler and CFD - Multigrille. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) with the assembled approximate Jacobian J̃. This note applies the true Jacobian matrix-free in the step's linear solve, as #397 did, but only while an objective test finds the limiter and the reconstruction's fallback inactive over the step, and keeps J̃ elsewhere.

## Problem and what was measured

- With the SER free to raise the CFL number, the TMR flat plate 69×49 does not converge on the assembled J̃: the laminar plate wanders at −3 to −4, the SA plate locks into a period-2 cycle (#395).
- J̃'s gap to J at second order is 15–25 % over the domain and 38–70 % on the wall cells' continuity and energy rows, and it is convective (99.5–100 %): the reconstruction's gradients held fixed (#395).
- Filling that gap inside J̃'s face-neighbour pattern fails (note 1, `design/reconstruction-jacobian`, both candidates above 0.5 × on the wall ring's continuity rows).
- The true Jacobian applied matrix-free with the limiter and fallback frozen at U (#397) converged the laminar plate in 1655 steps and SA 35×25 in 2017, but lost steps or convergence wherever the limiter is active (Ni's bump 384 not converged; NACA 0012 256²: +19 %, +21 %, +193 % steps). The residual does not follow a model frozen across the limiter's switches at the SER's CFL numbers.

## The activity criterion

*(CFD - Euler: the objective test of the limiter's and the fallback's activity over a step — what switches (φ, the neighbour attaining the minimum or maximum, the fallback flag), where it is evaluated (per cell or globally), the threshold and its a priori justification from a property or a source read in full, never from the gate's cases.)*

## Change

*(CFD - Multigrille: #397's product (Knoll and Keyes ε on the scaled unknowns, admissible and representable probes, the frozen residual of #402, the floor rows on the assembled operator), used when the criterion holds, the assembled J̃ otherwise; the preconditioner unchanged; restart state; ranks.)*

## Gate, declared before any run

*(To be completed: the laminar TMR flat plate 69×49 ≤ 3332 steps on `physics/ns-sa` (#395's measure); Euler no-regression unchanged — limited cases in at most `main`'s steps (tolerance 0), smooth cases ≤ 1.02 ×, forces within 10⁻⁸ |c| + 10⁻¹⁰, first order bit for bit; RAE 2822 case 9 once its grid exists; cost ≤ 1.30 × per step, measured alone under the GPU lock. SA 69×49 stays note 2's.)*
