# `cfd run` and the fake backend

`cfd run CASE --mesh-info MESHINFO [--runs DIR] [--json]` plans the case, refuses it on any error, then writes a run directory as specified in [results.md](../contracts/results.md) under `DIR` (default `./runs`). In Python: `case.run(mesh_info, backend=cfd.FakeBackend(), runs="runs")`, which returns `cfd.Results`.

Until the C++ core is built, the only backend is `fake` (`cfd.FakeBackend`). It writes contract-conforming, deterministic, **synthetic** results so the UI and V&V can be built against the real formats; `run.json` records `"backend": "fake"`, and these results must never be used as reference data.

| What | Fake backend |
|---|---|
| Geometry | A unit cube of n³ hexahedra written as polyhedra, n ≥ 8; boundary zones are consecutive groups of its boundary faces, cell zones are slabs along x. Zone names, conditions and media are the case's. |
| Fields | The freestream state of [case-schema.md](../contracts/case-schema.md#freestream) in every fluid cell (each zone with its own gas), the solid density in solid cells, zero wall shear. |
| History | Residuals decaying by `residual_orders` orders over the iterations written; forces approaching a thin-airfoil lift, projected as the contract defines. One line per iteration, flushed as written. |
| Options | `--iterations N` caps the iterations (or steps) written; `--pace S` sleeps S seconds per iteration to imitate a live run; `--diverge-at N` stops at iteration N with status `diverged` and `D0107`. |

Exit status: 0, or 1 when the plan has errors or the run diverged.
