# `autan run`

`autan run CASE --mesh-info MESHINFO [--backend fake|cpu|cuda|hip] [--device-count N] [--deterministic] [--runs DIR] [--json]` plans the case, refuses it on any error, then writes a run directory as specified in [results.md](../contracts/results.md) under `DIR` (default `./runs`). In Python: `case.run(mesh_info, backend=..., runs="runs")`, which returns `autan.Results`.

## Solver backend

`--backend cpu|cuda|hip` (Python: `autan.SolverBackend(devices=autan.Devices(kind="cuda", count=8))`) writes the case's [solver input](../contracts/solver-input.md) and runs the solver runtime `autan-run`, found through `$AUTAN_RUN` or the PATH (`E0804` if neither). The runtime's input errors come back as their own coded diagnostics: today every case gets `E0706`, since `autan-run` registers no physics yet. Ctrl-C or SIGTERM is passed on to the runtime, which finalizes the run as `stopped`.

## Fake backend

`--backend fake` (the default; Python: `autan.FakeBackend()`) writes contract-conforming, deterministic, **synthetic** results so the UI and V&V can be built against the real formats; `run.json` records `"backend": "fake"`, and these results must never be used as reference data.

| What | Fake backend |
|---|---|
| Geometry | A unit cube of n³ hexahedra written as polyhedra, n ≥ 8 and set by the MeshInfo zone counts; boundary zones are consecutive groups of its boundary faces, cell zones are slabs along x. Zone names, conditions and media are the case's. |
| Fields | The freestream state of [case-schema.md](../contracts/case-schema.md#freestream) in every fluid cell (each zone with its own gas), the solid density in solid cells, zero wall shear. |
| History | Residuals that reach `residual_orders` orders of decay at iteration 100, where a steady run converges (`completed` if `max_iterations` ends it first); forces approaching a thin-airfoil lift, projected as the contract defines. One line per iteration, flushed as written. The trajectory depends only on the case and the mesh, so runs with the same identity agree line for line. Unsteady runs without `dt` use 1 ms steps. |
| Options | `--iterations N` stops the run after N iterations (or steps), status `stopped`; `--pace S` sleeps S seconds per iteration to imitate a live run; `--diverge-at N` stops at iteration N with status `diverged` and `D0107`. |

SIGINT or SIGTERM stops the run: it is finalized as `stopped`, with its solution, and the command exits with 130. Exit status otherwise: 0, or 1 when the plan has errors or the run diverged.

To browse runs, `autan ui [--runs DIR] [--port N] [--no-open]` serves the web UI for the run directories under `DIR` (default `./runs`) on `127.0.0.1:8765`. It needs the `ui` extra (`uv add 'autan[ui]'`); without it the command exits with `E0805`.
