# 0005: For the alpha, the GPU carries the Navier-Stokes and SA operators and laminar runs; RANS runs end to end on the GPU in the beta

**Status:** accepted, 2026-11-07. **Decided by:** the lead's delegate for CFD (Main meta travail), relayed by the orchestrator. **Issues:** #586, #163, #137.

## Context

#586 brings the Navier-Stokes and Spalart-Allmaras (SA-neg) residuals to the device. Its criteria hold them bit for bit with the host where no `exp` or `log` is evaluated, and to within a tolerance declared beforehand where one is. The residual is then the same, but a run on the device does not step the way a run on the host does (`physics/src/euler_physics.cpp`):

- **A steady case on the host** takes implicit steps: the flow through its assembled Jacobian and the multigrid, line or Gauss-Seidel solver (`steady_step`). SA is loosely coupled to the flow: its scalar system is assembled and solved by symmetric Gauss-Seidel sweeps in cell order, which are sequential, and its update is clamped. SA runs steady cases only.
- **An unsteady case on the host** takes SSP-RK3 steps with one global time step (`unsteady_step`).
- **On a device,** an unsteady case takes the same SSP-RK3 steps. A steady case takes them in pseudo-time with a local time step (`device_step`), which the host does not have. No implicit step runs on the device today.

No parallel SA step on the device reproduces the host's: a bit-for-bit copy of sequential cell-order sweeps would run on one thread. Three options were weighed:

- **(A)** An explicit SA step, on the host and on the device. Rejected: the SA equation is stiff near the wall (destruction ∝ (ν̃/d)² over wall-resolved cells of high aspect ratio), and we expect that an explicit step bounded by the local CFL number would not converge the TMR cases in useful time. That is an expectation, not a measurement.
- **(B)** The implicit SA step with Gauss-Seidel sweeps by colour (the runtime's colouring, as the flow's `gauss-seidel` solver already sweeps), on the host and on the device, with the same order on both.
- **(C)** The operators on the device (NS and SA residuals, eddy viscosity, wall distance), proven equal to the host under #586's criteria, and laminar runs end to end on the device. RANS runs stay on the host.

## Decision

**Alpha: (C).** The GPU deliverable of M3 is:
- the Navier-Stokes and SA residuals and the eddy viscosity on the device, equal to the host under #586's criteria (bit for bit, or within the tolerance declared beforehand where `exp` or `log` is evaluated);
- the wall distance on the device (#245);
- **laminar** Navier-Stokes runs end to end on the device. An unsteady run takes the host's scheme and gives its results under the same criteria. A steady run takes the device's explicit pseudo-time steps, which have no host counterpart: its converged solution is compared with the host's within a tolerance that #586's PR (d) declares before it measures.

The alpha's TMR cases (flat plate, bump, NACA 0012) are run and converged **on the host**. Their results are host results and are never presented as GPU results.

**Beta: (B), after the device's implicit flow step.** RANS runs end to end on the device. SA takes the colour-ordered implicit step on both sides. The flow's implicit step on the device needs a device Jacobian assembly (a prerequisite that #163 assumes), then #163's multigrid and line smoother. We expect the flow to need them: an explicit flow step should not converge wall-resolved grids in useful time. We expect SA's single-level sweeps to suffice, as they do on the host today. Both are expectations, to be measured.

## Consequences

- `docs/PLAN.md`, M3: the GPU line says what the alpha carries and what waits for the beta.
- #586's PR (d) wires `laminar` into `autan-run --backend cuda`. `sa` on a device stays refused with a coded error, with no host fallback. #586 is revised to match.
- **Estimate for (B):** 4–6 working days. That covers moving the host SA solve to colour order (this changes the host's SA bits, so the host TMR results are re-verified), the device SA assembly from the generated Jacobian loops, the scalar colour sweep and clamp kernels, and the tests. It is less if a scalar colour sweep already exists by then. The GPU line smoother of `solvers/` works on 5×5 blocks.
- An end-to-end RANS run on the device needs the device Jacobian and #163 first, then (B). Their estimates govern the beta's date.
