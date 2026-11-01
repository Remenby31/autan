# Contract: solver runtime

`autan-run` runs a [solver input](solver-input.md) and writes the [run directory](results.md). Its invocation, exit statuses and inputs are in solver-input.md; its outputs in results.md. This page fixes the seam between the runtime and the physics: what a discretisation implements so that the runtime can drive it. Header: `runtime/include/autan/runtime/physics.hpp`, library `autan::runtime`.

## Division of work

The runtime owns the iteration loop, the convergence test, stop signals, the run directory, the history (residual names, monitor cadence, force coefficients) and `solution.cgns`. A physics owns its state arrays (host or device), its kernels, their schedule and their precision. The runtime never touches device memory.

## Physics

A physics is constructed from a `PhysicsContext` (the assembled mesh, its FP64 geometry and the parsed solver input), with its state set to each cell zone's `initial` values. For a verification case, `input.verification->text` holds the `autan.mms/1` document the runtime read and checked: the physics takes the boundary data, source terms and auxiliary fields from it (solver-input.md, `verification`). It implements:

| Member | Contract |
|---|---|
| `step(iteration)` | Advances one pseudo-time iteration (steady) or one time step (unsteady); returns a `StepReport`: one residual norm per equation of `residual_names(input)`, as results.md#history defines it; for unsteady runs the physical time advanced; and, if the state or a residual became non-finite or non-physical, the first offending cell. |
| `accumulate_average(weight)` | Adds the current state to the time average with that weight (the step's time step); called after each step past `average_from`. |
| `fields(mean)` | The cell fields of `field_names(input)`, in that order, current state or time average. |
| `wall_fields(faces)` | `Pressure`, `WallShearStressX`, `WallShearStressY`, `WallShearStressZ` on the given wall and slip faces. |
| `loads(zones, moment_center)` | Force and moment the fluid exerts on the boundary faces of the zones (results.md, forces monitor), in N and N·m; the runtime derives the coefficients. |

An exception from any member ends the run as `failed` with `E0704`. A report with a diverged cell or a non-finite residual ends it as `diverged` with `D0107`; the solution is then written only if every field is finite.

## Registration

A build of `autan-run` lists its physics in a `PhysicsRegistry`: each entry matches `physics.kind`, `physics.turbulence.model`, `precision` and `devices.kind` (or `*`). The first match runs the case; none is `E0706` before any directory exists. Implementations templated on precision register one entry per precision they instantiate.
