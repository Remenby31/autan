# The CFL number of steady runs after an under-solved step

Design for #332, written before any code. The steady implicit step solves (V/Δt + J̃) ΔU = −R(U) by GCR preconditioned by the multigrid, to a relative residual of 0.1 in at most 10 iterations; its CFL number follows the switched evolution relaxation (SER) of [euler.md](euler.md) (#277), capped by `cfl_cap_` and scaled by `penalty_`. On NACA 0012 M 0.8 α 0° the residual rises and falls by tenths of a decade after the switch to second order: at the CFL number the SER reaches, the solve stops above its tolerance and the step departs from the Newton step (#332: tightening the solve to 10⁻³ in 50 iterations removes the oscillation and converges in 263 steps instead of 430; #231's hierarchy cuts it to 6 rises and 307 steps, not enough by the rule declared there; the limiter is not involved).

## Change

**An under-solved step lowers the CFL number exactly as a shortened step does.** A step whose linear solve stops above its tolerance (0.1) but below the failure threshold (0.9) is taken in full, as today, and goes through the SER's existing shortened-step path (`steady_step`, unchanged): `penalty_ = min(penalty_, cfl_ · drop / cfl_start)` (the penalty that gives this step's CFL number from its residual), `cfl_cap_ = cfl_ / 2`, and a flag `under_solved_` that makes the next step halve the penalty as `shortened_` does (`penalty_ = (shortened_ || under_solved_) ? ½ penalty_ : min(1, 1.5 penalty_)`). The step itself is not shortened: `shortened_` and its "step shortened" event keep meaning an actual shortening, and the under-solved step keeps today's `linear` warning. A step that reaches its tolerance leaves the CFL number to the SER, as today. A failed solve (above 0.9, singular or not finite) is unchanged: rejected, half cap.

The recovery is the SER's, explicit: after an under-solved step at CFL number C, and while the residual stays near its level (the SER ratio unchanged), the next CFL numbers are C/2, then ×1.5 per step (`penalty_ = min(1, 1.5 · penalty_)`): 0.75 C, 1.125 C, 1.69 C …, back at the law's own value once the penalty reaches 1. If the solve misses again on the way up, the same path halves it again. So, with full steps and an unchanged SER ratio, the CFL number settles within a factor 2 below the largest value the solve follows in 10 iterations: never more than ½ below it after a miss, and 3 steps (½ · 1.5³ > 1) from a value it already missed. Those assumptions do not always hold: a shortened step halves the penalty too (25 instead of 75 after a miss at 100), and at the switch to second order the SER's reference is reset, so the next value follows the new ratio; the claim is a property of the stationary case, not a bound on every sequence. A falling residual raises the law's value as today; the penalty only limits it.

Why ½ and ×1.5: they are the SER's existing factors for a shortened step (#185), kept so that one mechanism, already measured on every case of [multigrid.md](multigrid.md#measurements), governs both events; no new parameter is introduced. At the smallest CFL number (1), where no halving is possible, an under-solved step is taken and logged (`linear` warning, as now), and it neither sets `under_solved_` nor lowers the penalty: repeated misses at the floor would otherwise halve the penalty without bound and delay the recovery indefinitely. The run is not rejected for it.

This is an Autan rule, not a published one. The nearest published rules differ:

- FUN3D's HANIM (Wang, Diskin, Nielsen and Liu 2021, AIAA SciTech, "Improvements in Iterative Convergence of FUN3D Solutions", §II.D) treats a GCR that misses its target within its budget as a failure: the step is rejected and the CFL number divided by 10. Rejecting at 0.1 was measured too strict with the approximate Jacobian of the defect correction (#207: 229–468 steps on the Euler MMS where the SER takes 41–60).
- ADflow's ANK (Yildirim, Kenway, Mader and Martins 2019, J. Comput. Phys. 397:108741, §§3.4–3.7) accepts a linear solve that misses its target and adapts the CFL number from the accepted step length (eq. 20), not from the linear solve.

The rule keeps HANIM's principle (the CFL number must not outrun the linear solve: it grows only while the solve succeeds and drops when it does not) and ANK's acceptance of a missed solve, within the SER's law (Mulder and van Leer 1985, J. Comput. Phys. 59:232–246) and its existing penalty. HANIM recovers by ×2 per successful step after ÷10; the SER's ÷2 then ×1.5 is gentler and already in place.

An Eisenstat–Walker forcing term (1996, SIAM J. Sci. Comput. 17(1):16–32, Choice 2) was considered and left out: bounded by today's tolerance 0.1 its safeguard (γ η²_{k−1} > 0.1) can never act, and in the transient of #332 (residual ratios near 1) it asks for nothing tighter than 0.1. It may return as a separate change with its own evidence.

## Scope, state and determinism

Only the host implicit step with `linear_solver = "multigrid"` (and `"lines"` once #282 adds it) measures its linear residual; `"gauss-seidel"` runs a fixed number of sweeps and the device steady step is explicit, so neither changes. The state the rule uses is the cap and the penalty, already in the restart state (`SolverCflCap`, `SolverPenalty`), and `under_solved_`, a new restart field `SolverUnderSolved`: a continued run repeats the uninterrupted one bit for bit, as today. The steady Euler physics does not yet run under `reproducibility = "ranks"` (E1109); the rule adds no sum.

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

A control that does not reach ten orders within the budget is reported as such; the run must then reach ten orders, and the force comparison with that control does not apply (no converged value to compare with). No regression elsewhere: Ni's bump 96 / 384 and the Euler MMS on hex, tet and poly families (16–48) within 5 % of the control's steps; and one fp32+acc64 run, NACA M 0.8 α 0° 256², ending at its round-off floor (status as main's) in no more steps than main's, with its surface Cp within 0.001 of the fp64 run's on the same mesh outside the shocks ([acceptance.md](../validation/acceptance.md)). The discretisation does not change, so the converged solutions should not either; the AGENTS.md evidence is run anyway: the Euler MMS observed orders on the hex, tet, poly and hybrid families in fp64 and fp32+acc64, unchanged from main's. Tests: a steady run whose solve stops above its tolerance logs the reduced CFL number at the next step and its recovery by ×1.5 per step after, and a run continued right after such a step repeats the uninterrupted one bit for bit. If the gate fails, θ is not retuned on the results; the change closes with the measurement, as #305 did.

## Ownership

The steady step and its SER are CFD - Euler's (#277): the change lands in `physics/src/euler_physics.cpp`, `docs/design/euler.md` and the Euler tests, reviewed by them; [multigrid.md](multigrid.md) states the tolerance and links here. `solvers/` does not change.
