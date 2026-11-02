# Linear solve forcing and the CFL number of steady runs

Design for #332, written before any code. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) by GCR preconditioned by the multigrid, to a fixed relative residual of 0.1 in at most 10 iterations, and its CFL number follows the switched evolution relaxation of [euler.md](euler.md) (SER, #277). On NACA 0012 M 0.8 α 0° the residual rises and falls by tenths of a decade after the switch to second order: at the CFL the SER reaches, the solve stops above its tolerance and the step departs from the Newton step (#332: tightening the solve to 10⁻³ in 50 iterations removes the oscillation and converges in 263 steps instead of 430; #231's hierarchy reduces it to 6 rises and 307 steps, not enough by the rule declared there; the limiter is not involved).

## Change

Two rules from the literature, applied to the steady step of the Euler physics; the multigrid itself does not change.

**1. The CFL number grows only after a converged linear solve.** FUN3D's HANIM (Wang, Diskin, Nielsen and Liu 2021, AIAA SciTech, "Improvements in Iterative Convergence of FUN3D Solutions", §II.D) raises the CFL number only after a step whose linear solve reached its tolerance and lowers it after a failed one; ADflow's ANK does the same through its linear-solve success test (Yildirim, Kenway, Mader and Martins 2019, J. Comput. Phys. 397:108741, §3.6). Here, with the SER kept as the law:

- the solve reached its tolerance η_k: the next CFL number is the SER's, as today;
- it stopped above η_k but below the failure threshold 0.9: the next CFL number is at most this step's (`cfl_cap_ = cfl_`), no growth;
- it failed (above 0.9): unchanged from today, the step is rejected and the next CFL number is at most half this one.

**2. The tolerance η_k follows the convergence (inexact Newton forcing).** Eisenstat and Walker (1996, SIAM J. Sci. Comput. 17(1):16–32), Choice 2 with its safeguard:

- η_k = γ (‖F_k‖ / ‖F_{k−1}‖)^α with γ = 0.9, α = 2 (their experiments' values, §3);
- safeguard (§2.1): η_k ← max{η_k, γ η_{k−1}^α} whenever γ η_{k−1}^α > 0.1;
- bounds: η_k ≤ η_max = 0.1 (today's tolerance: the forcing term only ever tightens the solve), and η_k ≥ 0.5 τ / ‖F_k‖ (Kelley 1995, Iterative Methods for Linear and Nonlinear Equations, SIAM, eq. 6.20: no oversolving once the residual nears the target τ);
- ‖F_k‖ is the SER's relative residual (the largest residual over the equations that set the CFL number, relative to its largest value in the order, #277), and τ = 10^−residual_orders; η_0 = η_max at the first step of each order.

The iteration cap stays 10. With η_max = 0.1, rule 2 changes nothing while the residual ratio exceeds 1/3, so in the oscillating transient of #332 (ratios near 1) rule 1 does the work: the CFL number stops growing where the solve can no longer follow it. Rule 2 tightens the solve once the residual falls fast, where #332's tightened variant gained its steps.

## State and determinism

η_{k−1} and ‖F_{k−1}‖ join the restart state (`SolverForcing`, `SolverPreviousResidual`), so a continued run repeats the uninterrupted one bit for bit; both are reductions already exact over the ranks, so `reproducibility = "ranks"` is unaffected.

## Gate, declared before any run

NACA 0012 of #183, M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°, at 128², 256², 512² and 1024², fp64, against main at the commit the PR merges on:

- rises of log10 |R_ρ| above 0.2 decade between consecutive steps, over the second-order phase until ten orders: **at most 2 per run**;
- steps to ten orders **≤ main's** on every run;
- force coefficients at convergence within **10⁻⁸ relative** of main's (both converged to ten orders);
- no regression elsewhere: Ni's bump 96 / 384 and the Euler MMS hex 16–48 within 5 % of main's steps; CPU time reported.

If a case fails, the rules are not retuned on the results; the PR closes with the measurement, as #305 did.

## Ownership

The steady step and its SER are CFD - Euler's (#277): the change is theirs to review and lands in `physics/src/euler_physics.cpp`, `docs/design/euler.md` and the Euler tests; `solvers/` does not change (`Multigrid::solve` already takes the tolerance).
