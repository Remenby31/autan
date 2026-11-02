# The steady step's operator from the second order on

Design for #395, written before any code. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U), J̃ the approximate Jacobian assembled in 5 × 5 blocks ([euler.md](euler.md#time-stepping)): the faces' dF/dW with the reconstruction's gradients and limiter held fixed. It is solved by GCR preconditioned by one multigrid cycle on the same matrix ([multigrid.md](multigrid.md)), or by the line sweeps (`"lines"`), to a relative residual of 0.1 in at most 10 iterations.

## Problem, measured

On the TMR flat plate 69×49 (#395; `physics/ns-sa` at c3966068, SA and laminar), once the SER raises the CFL number past ~10³ at second order, the iteration settles on a period-2 cycle: every step is taken in full (α = 1), its linear solve meets the tolerance, and the state flips between two states for thousands of steps. Before #277 the plate converged only because a round-off residual pinned the CFL number at 1. At a state of the cycle, central differences of the residual give ‖(J − J̃) v‖ / ‖J v‖ of 10⁻⁹–10⁻² at first order, 0.15–0.25 at second order away from the wall, and 0.38–0.70 on the density and energy rows of the wall cells (aspect ratio 10³–10⁴). The limiter (off: the same cycle), the SA coupling (laminar: the same stall), the boundary fit (off: no better, #395) and the linear solve (it meets its tolerance) are ruled out. The missing part is the convective reconstruction's dependence on the neighbours' states through the frozen gradients: a defect correction with an operator that far from J overshoots by a factor 2 once V/Δt no longer dominates.

## Change

**From the step after the switch to second order, the linear solve's operator is the true Jacobian, applied matrix-free; J̃ stays the preconditioner.** GCR solves (V/Δt + J) ΔU = −R(U), the product being

  (V/Δt) v + [R(U + ε v) − R(U)] / ε,

R the same residual the step reports (second order, the limiter as evaluated, the boundary conditions, `ν̃` held: see SA below), preconditioned exactly as now: one multigrid cycle on V/Δt + J̃ (`"multigrid"`), or the line sweeps on it (`"lines"`). Tolerance (0.1), iteration limit (10), the failure rule (0.9: step rejected) and the SER are unchanged. This is the inexact Newton-Krylov step with an approximate-Jacobian preconditioner of FUN3D's HANIM (Wang, Diskin, Nielsen and Liu 2021, AIAA SciTech) and of ADflow's ANK at its matrix-free level (Yildirim, Kenway, Mader and Martins 2019, J. Comput. Phys. 397:108741), kept inside main's SER rather than their CFL controls (#207 measured those slower here).

**Activation**: the step after the one that switches to second order (`switch_order`, logged as the `solver` info event of #356, or the first step of a run that starts at second order: unsteady runs, continued runs past the switch), and only with `"multigrid"` or `"lines"`. The first-order phase keeps the assembled operator: there J̃ is the Jacobian up to the viscous blocks (measured above: 10⁻⁹ away from the wall), so a residual per Krylov iteration would buy nothing. `"gauss-seidel"` (fixed sweeps, no Krylov method) and the device steady step are unchanged. A `linear` info event marks the first matrix-free step. The state that decides it (`second_order_`) is already in the restart record, so a continued run repeats the uninterrupted one bit for bit.

**ε** (Knoll and Keyes 2004, J. Comput. Phys. 193:357–397, §2.2): ε = √((1 + ‖U‖) ε_R) / ‖v‖, ‖·‖ the Euclidean norm over the owned unknowns (an exact sum over the ranks, so every rank uses the same ε), ε_R the machine epsilon of the precision the residual is evaluated in: 2⁻⁵² in `fp64`, 2⁻²³ in `fp32+acc64`. The stored state in `fp32+acc64` is FP32 differences from the reference ([euler.md](euler.md)), so a perturbation built for FP64 (ε ‖v‖ ≈ 10⁻⁸ ‖U‖) would round away; with ε_R = 2⁻²³ the perturbation is ≈ 3 × 10⁻⁴ ‖U‖, the truncation error of the one-sided difference O(ε) relative, and the residual's own FP32 rounding (≤ ε₃₂ Φ_k per equation, [euler.md](euler.md)) contributes at most √ε₃₂ ≈ 3 × 10⁻⁴ relative to the product: both well inside the 0.1 the solve asks for. The difference R(U + εv) − R(U) is taken in FP64 from the two residual arrays (FP64 in both precisions: `fp32_acc64::residual` is `double`). No false convergence: the history and the convergence test keep using R(U), not the linear residual; the round-off floor rule of `fp32+acc64` is unchanged.

**SA**: the mean-flow product holds ν̃, and so μ_t, at the current state; the SA equation keeps its loosely coupled step after the flow's. The SA rows are not in the matrix-free operator: the coupling cost (ν̃ lagging the flow) is what the laminar/SA comparison of the gate measures. Coupling ν̃ into the product (6 equations, μ_t from the perturbed ν̃ before the flow residual) is #235's scope.

**Cost**: one residual evaluation per GCR iteration (at most 10 per step) in place of a block matrix product; the preconditioner, the assembly and the hierarchy are unchanged. Memory: two arrays of 5 n doubles (U + εv and its residual) and the state of the perturbed U.

## Gate, declared before any run

Flat plate (`physics/tmr/case.sh flatplate`, fp64, 6000 steps), with this change merged into `physics/ns-sa`; measure: the first step at which every equation but the identically zero z-momentum is ten orders below its peak (#395):

- laminar 69×49, then SA 69×49: at most 3332 steps (2777 × 1.2); SA 35×25 and 137×97 reported.

No regression, against `main` at the merge base, steps to ten orders (fp64, the runtime's convergence):

- Euler MMS hex/tet/poly (16–48), Ni's bump 96/384, NACA 0012 (M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°) at 128² and 256²: within 5 % or fewer steps; forces at convergence within 10⁻⁸ |c| + 10⁻¹⁰.
- Cost: CPU time to convergence of autan-run on the same cases and the plates, reported against `main` with the time per step; on the Euler cases at most 1.30 × `main`'s (the residual per Krylov iteration is the price; above, the gate fails, the activation is not retuned on the results).
- `fp32+acc64`: the laminar plate 69×49 and NACA M 0.8 α 0° 256² end at their round-off floor (status as `main`'s), the NACA's surface Cp within 0.001 of its `fp64` run outside the shocks ([acceptance.md](../validation/acceptance.md)); the flat plate's skin friction at x = 0.97 within 1 % of its `fp64` run.
- Accuracy (AGENTS.md): Euler MMS observed orders on hex, tet, poly and hybrid in `fp64` and `fp32+acc64`, `euler-mms-order` PASS, unchanged from `main`'s.

Tests: the matrix-free product agrees with the assembled one where J̃ is the Jacobian (a first-order state of the verification case: within √ε_R of its magnitude, in both precisions); a run logs the first matrix-free step at the step after the switch to second order and not before; a run continued across the switch repeats the uninterrupted one bit for bit.

## Ownership

The steady step is CFD - Euler's (`physics/src/euler_physics.cpp`, reviewed by them); the GCR and the multigrid's outer operator are `solvers/` (a `Multigrid::solve` taking the outer product). [multigrid.md](multigrid.md) and [euler.md](euler.md) state the operator and link here.
