# Contract: solver runtime

`autan-run` runs a [solver input](solver-input.md) and writes the [run directory](results.md). Its invocation, exit statuses and inputs are in solver-input.md; its outputs in results.md. This page fixes the seam between the runtime and the physics: what a discretisation implements so that the runtime can drive it. Header: `runtime/include/autan/runtime/physics.hpp`, library `autan::runtime`.

## Division of work

The runtime owns the iteration loop, the convergence test, stop signals, the run directory, the history (residual names, monitor cadence, force coefficients) and `solution.cgns`. A physics owns its state arrays (host or device), its kernels, their schedule and their precision. The runtime never touches device memory.

## Physics

A physics is constructed from a `PhysicsContext` (the assembled mesh, its FP64 geometry and the parsed solver input), with its state set to each cell zone's `initial` values, or to `start` when the input has one (below). The context's references stay valid until the run ends; a physics may keep a copy. Its `log` writes events to the run's `log.jsonl` (results.md): level, source, message, and optionally a registry code and data; the runtime stamps the time and the iteration. For a verification case, `input.verification->text` holds the `autan.mms/1` document the runtime read and checked: the physics takes the boundary data, source terms and auxiliary fields from it (solver-input.md, `verification`). For each `oracle-state` condition the runtime reads the face-state file before any directory exists, checks its hash (`E0420`), document (`E0420`), mesh (`E0422`) and zone coverage (`E0423`), and pairs every face of the zone with one row by centroid ([face-states.md](face-states.md), `E0421`); `boundary_states` then holds, per face zone index, the zone's states in mesh face order (5 × its faces, component-major), and the run directory archives the file under `states/`. It implements:

On a mesh with periodic pairs, `mesh` and `geometry` are those of its one-part partition ([partitioning.md](../design/partitioning.md#periodic-images)): the owned cells are the mesh's cells in order, the image ghosts follow with their geometry filled, each periodic face is an interior face, and `ghosts.forward(components, space, stream)` fills the ghosts from their owners (host memory; device fields are `E0145` for now). A physics calls it before any loop that reads ghost values and loops over owned faces and cells only. Without pairs there are no ghosts and `forward` does nothing. Fields, norms and scales cover the owned cells; faces passed to `wall_fields` and `loads` are faces of `mesh`.

| Member | Contract |
|---|---|
| `step(iteration)` | Advances one pseudo-time iteration (steady) or one time step (unsteady); returns a `StepReport`: one residual norm per equation of `residual_names(input)`, as results.md#history defines it; for unsteady runs the physical time advanced; and, if the state or a residual became non-finite or non-physical, the first offending cell. |
| `accumulate_average(weight)` | Adds the current state to the time average with that weight (the step's time step); called after each step past `average_from`. |
| `fields(mean)` | The cell fields of `field_names(input)`, in that order, current state or time average. |
| `restart_fields()` | The cell-centred state beyond `fields(false)` that a `continue` needs to go on bit for bit (earlier time levels, averaging sums), written as `RestartState`; empty when there is none. |
| `wall_fields(faces)` | `Pressure`, `WallShearStressX`, `WallShearStressY`, `WallShearStressZ` on the given wall and slip faces. |
| `loads(zones, moment_center)` | Force and moment the fluid exerts on the boundary faces of the zones (results.md, forces monitor), in N and N·m; the runtime derives the coefficients. |

An exception from any member ends the run as `failed` with `E0704`. A report with a diverged cell or a non-finite residual ends it as `diverged` with `D0107`; the solution is then written only if every field is finite.

## Start

For an input with a `start` ([solver-input.md](solver-input.md)), the runtime checks the previous `solution.cgns` against `start.solution_sha256` (`E0709`), reads it (`E0707` if it cannot), and checks its mesh (`E0203`), precision and fields (`E0708`). All of this happens before any directory exists. `PhysicsContext::start` then holds its `FlowSolution` and, for `continue`, its `FlowSolutionMean` and `RestartState`. The physics starts from those instead of `initial`. For `warm` it takes the fields only. A `continue` calls `step` from `start.iterations + 1`. The runtime carries on the physical time and the averaging interval, and measures steady convergence against `start.residual_reference`. A physics whose `restart_fields` round-trips its whole state makes the continued run reproduce the uninterrupted one: same history lines and same solution arrays.

## Registration

A build of `autan-run` lists its physics in a `PhysicsRegistry`: each entry matches `physics.kind`, `physics.turbulence.model`, `precision` and `devices.kind` (or `*`). An entry may also declare a scope, what it supports beyond its key (media, zones, conditions). The first match runs the case; no match, or a match whose scope refuses the input, is `E0706` before any directory exists, with `data.path` naming the offending key. The Euler physics refuses any cell zone that is not an ideal gas, cell zones of different gases (any property differs; `data.zone`, `data.medium`), and boundary conditions other than slip, symmetry, farfield, inlet, outlet and manufactured (`data.zone`, `data.condition`). The input's `periodic` pairs must be the mesh's, and a periodic zone takes no condition (`E0705` otherwise). The Euler physics refuses periodic pairs until it exchanges ghost cells. Implementations templated on precision register one entry per precision they instantiate.
