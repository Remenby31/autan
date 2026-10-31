# `autan run` and the fake backend

`autan run CASE --mesh-info MESHINFO [--runs DIR] [--json]` plans the case, refuses it on any error, then writes a run directory as specified in [results.md](../contracts/results.md) under `DIR` (default `./runs`). In Python: `case.run(mesh_info, backend=autan.FakeBackend(), runs="runs")`, which returns `autan.Results`.

Until the C++ core is built, the only backend is `fake` (`autan.FakeBackend`). It writes contract-conforming, deterministic, **synthetic** results so the UI and V&V can be built against the real formats; `run.json` records `"backend": "fake"`, and these results must never be used as reference data.

| What | Fake backend |
|---|---|
| Geometry | A unit cube of n³ hexahedra written as polyhedra, n ≥ 8 and set by the MeshInfo zone counts; boundary zones are consecutive groups of its boundary faces, cell zones are slabs along x. Zone names, conditions and media are the case's. |
| Fields | The freestream state of [case-schema.md](../contracts/case-schema.md#freestream) in every fluid cell (each zone with its own gas), the solid density in solid cells, zero wall shear. |
| History | Residuals that reach `residual_orders` orders of decay at iteration 100, where a steady run converges (`completed` if `max_iterations` ends it first); forces approaching a thin-airfoil lift, projected as the contract defines. One line per iteration, flushed as written. The trajectory depends only on the case and the mesh, so runs with the same identity agree line for line. Unsteady runs without `dt` use 1 ms steps. |
| Options | `--iterations N` stops the run after N iterations (or steps), status `stopped`; `--pace S` sleeps S seconds per iteration to imitate a live run; `--diverge-at N` stops at iteration N with status `diverged` and `D0107`. |

SIGINT or SIGTERM stops the run: it is finalized as `stopped`, with its solution, and the command exits with 130. Exit status otherwise: 0, or 1 when the plan has errors or the run diverged.
