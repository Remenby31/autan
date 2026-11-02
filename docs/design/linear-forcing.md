# The CFL number of steady runs after an under-solved step

Design for #332, written before any code. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) by GCR preconditioned by the multigrid, to a relative residual of 0.1 in at most 10 iterations; its CFL number follows the switched evolution relaxation (SER) of [euler.md](euler.md) (#277), capped by `cfl_cap_` and scaled by `penalty_`. On NACA 0012 M 0.8 α 0° the residual rises and falls by tenths of a decade after the switch to second order: at the CFL number the SER reaches, the solve stops above its tolerance and the step departs from the Newton step (#332: tightening the solve to 10⁻³ in 50 iterations removes the oscillation and converges in 263 steps instead of 430; #231's hierarchy cuts it to 6 rises and 307 steps, not enough by the rule declared there; the limiter is not involved).

## Change

**An under-solved step lowers the CFL number the way a shortened step does.** A step whose linear solve stops above its tolerance (0.1) but below the failure threshold (0.9) is taken in full, as today, and the SER treats it as it treats a shortened step: the next CFL number is capped at half this step's (`cfl_cap_ = min(cfl_cap_, cfl_ / 2)`, the smallest of the caps applying), and `penalty_` drops to the value that gives that CFL number from the current residual (`penalty_ = min(penalty_, cfl_ · drop / cfl_start)`, then halved by the next step's update), so that it recovers by ×1.5 per step, as after a shortened step. Without the penalty, the cap alone would last one step: a solve that succeeds at the halved CFL number would let the SER return at once to the CFL number that under-solved (`10⁶ → 5·10⁵ → 10⁶ …`). `shortened_` itself is not set, since the step is not shortened. A step that reaches its tolerance leaves the CFL number to the SER, as today. A failed solve (above 0.9, singular or not finite) is unchanged: rejected, half cap. At the smallest CFL number (1), where no halving is possible, an under-solved step is taken as today and logged (`linear` warning, as now); the run is not rejected for it.

This is an Autan rule, not a published one. The nearest published rules differ:

- FUN3D's HANIM (Wang, Diskin, Nielsen and Liu 2021, AIAA SciTech, "Improvements in Iterative Convergence of FUN3D Solutions", §II.D) treats a GCR that misses its target within its budget as a failure: the step is rejected and the CFL number divided by 10. Rejecting at 0.1 was measured too strict with the approximate Jacobian of the defect correction (#207: 229–468 steps on the Euler MMS where the SER takes 41–60).
- ADflow's ANK (Yildirim, Kenway, Mader and Martins 2019, J. Comput. Phys. 397:108741, §§3.4–3.7) accepts a linear solve that misses its target and adapts the CFL number from the accepted step length (eq. 20), not from the linear solve.

The rule keeps HANIM's principle (the CFL number must not outrun the linear solve) and ANK's acceptance of a missed solve, and reuses the SER's existing halving rather than a new parameter. Unlike freezing the CFL number, it corrects a CFL number that is already too high: each under-solved step halves it until the solve reaches its tolerance again, and the penalty then lets it grow back by ×1.5 per step, the SER's own recovery after a shortened step.

An Eisenstat–Walker forcing term (1996, SIAM J. Sci. Comput. 17(1):16–32, Choice 2) was considered and left out: bounded by today's tolerance 0.1 its safeguard (γ η²_{k−1} > 0.1) can never act, and in the transient of #332 (residual ratios near 1) it asks for nothing tighter than 0.1. It may return as a separate change with its own evidence.

## Scope, state and determinism

Only the host implicit step with `linear_solver = "multigrid"` (and `"lines"` once #282 adds it) measures its linear residual; `"gauss-seidel"` runs a fixed number of sweeps and the device steady step is explicit, so neither changes. The state the rule uses is the cap and the penalty, both already in the restart state (`SolverCflCap`, `SolverPenalty`): a continued run repeats the uninterrupted one bit for bit, as today. The steady Euler physics does not yet run under `reproducibility = "ranks"` (E1109); the rule adds no sum.

A `solver` info event marks the first step solved at second order (the step after the switch), so that the gate below finds the phase from the log.

## Gate, declared before any run

Cases: NACA 0012 of #183, M 0.5 α 1.25°, M 0.8 α 0°, M 0.8 α 1.25°, at 128², 256², 512² and 1024², fp64, budget 3000 steps. Two controls, both recorded with case and mesh hashes: d04f7cf1 (the commit #332's step 1 measured) and main at the commit the PR merges on.

Over the second-order phase (from the event above) until ten orders, on log10 |R_ρ|:

- **rises**: increases above 0.2 decade between consecutive steps, at most 2 per run;
- **excursion**: the largest rise of the residual above its running minimum, at most 0.5 decade per run (so that several small rises cannot add up unnoticed).

Against each control, per run:

- steps to ten orders no more than the control's;
- CPU time of autan-run no more than 1.10 times the control's, and the total number of GCR iterations reported;
- each force coefficient at convergence within 10⁻⁸ |c_control| + 10⁻¹⁰ (absolute floor for the zero lift at α 0°).

A control that does not reach ten orders within the budget is reported as such; the run must then reach ten orders. No regression elsewhere: Ni's bump 96 / 384 and the Euler MMS on hex, tet and poly families (16–48) within 5 % of the control's steps; and one fp32+acc64 run, NACA M 0.8 α 0° 256², ending at its round-off floor (status as main's) in no more steps than main's. Tests: a steady run whose solve stops above its tolerance logs the reduced CFL number at the next step and its recovery by ×1.5 per step after, and a run continued right after such a step repeats the uninterrupted one bit for bit. If the gate fails, θ is not retuned on the results; the change closes with the measurement, as #305 did.

## Ownership

The steady step and its SER are CFD - Euler's (#277): the change lands in `physics/src/euler_physics.cpp`, `docs/design/euler.md` and the Euler tests, reviewed by them; [multigrid.md](multigrid.md) states the tolerance and links here. `solvers/` does not change.
