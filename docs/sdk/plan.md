# `autan plan`

`autan plan` checks a case against its mesh before you reserve a GPU. It shows the condition of every boundary and cell zone and where it comes from ([boundary resolution](../contracts/case-schema.md#boundary-resolution)), reports coded [diagnostics](errors.md) and estimates device memory.

```text
autan plan CASE --mesh-info MESHINFO [--gpus N] [--device-kind cpu|cuda|hip] [--json]
```

It exits with 1 when it finds an error. The SDK does not call the mesh readers yet: write the MeshInfo with `autan-mesh-info --json MESH` and pass it with `--mesh-info`. `plan` compares its `source.sha256` with the hash of the mesh file (`E0203`) and warns when the file is absent (`W0205`). [Check a case before you run it](../guides/check-a-case.md) walks through the output.

## Memory model `autan.memory/2`

The estimate is what one rank allocates at its peak, on the host and on its device, by component; `--gpus` is the number of ranks and `--device-kind` (`cpu`, `cuda`, `hip`, default `cuda`) their device. `bytes_per_gpu` is a rank's host plus device bytes, which share one pool on a unified-memory device such as the GB10, and `bytes_total` that times the ranks. `autan prepare` writes a rank's figures into the solver input (`memory`), and `autan-run` refuses to start a run that does not fit (`E0711`).

Each component counts what the code allocates, listed by the lane that owns it and summed here (#325). Notation, per rank: C cells, F faces, F_i interior faces, F_b boundary faces, K = 2F_i + F_b, S gradient stencil entries (bounded per cell: tetrahedron 16, pyramid and prism 25, hexahedron 36, polyhedron 110), n solved equations, s = 4 bytes in `fp32+acc64` and 8 in `fp64`; N nodes, P face nodes (`counts.face_nodes`) and the cell and face counts of the whole mesh where marked *global*.

| Component | Where | Bytes |
|---|---|---|
| `mesh` | host | the global mesh 24N + 16F + 4P + 12F_i + 4F_b + 20C, its FP64 geometry 56F + 32C, colourings 12C, all *global*: every rank keeps them |
| `mesh setup` | host | the reader's form and the MeshInfo pass, about the mesh and its geometry again, which the allocator keeps resident |
| `partition` | host | several ranks: the rank's share of the mesh and geometry, and 9F face keys |
| `solution output` | host | rank 0 gathers 2n + 2 fields of 8C *global* bytes, and the writer copies them |
| `physics` | host | packed geometry 8(16F + 4C + 3S), stencil 8C + 36S, state, primitives, limiters, gradients and boundary data in s, solution, stages, residual in 8 bytes |
| `physics setup` | host | 8(16F + 4C + 3K) + 24S during setup |
| `physics device`, `device context` | device, host | on `cuda` and `hip`: the same state, topology, stencil and geometry on the device; 512 MiB for the device context |
| `jacobian` | host | steady runs on `cpu` (a device run solves explicitly): face and cell Jacobians in s, the block matrix 8n²(C + 2F_i) |
| `multigrid`, `lines`, `gauss-seidel` | host | the case's `linear_solver`: levels below the finest at most L + 0.15(C − L), L the cells of lines of at least 4 cells (which pair along their line) and the rest agglomerating by up to 8, with 7 neighbours per coarse cell; line smoothing for the MeshInfo's `quality.lines.cells` on the finest level (the multigrid's own count, for the cell order it states) and for every coarse cell when there are lines; the lines built on the whole mesh at setup; the inverse diagonal blocks |

The blocks of the linear solvers are counted for n = 5 and scaled by n²/25. The figures were calibrated on the CPU against measured host peaks (`memory` event of the log, results.md): the multigrid and lines terms on NACA 0012 256², a 48³ hexahedral manufactured solution and the TMR flat plate 137×97 (within ×1.04 to ×1.20 of the measured solver memory), the mesh terms on a 1.17 M-cell tetrahedral mesh. The estimate is for the cell order the MeshInfo states (`quality.lines.order`); a run with another order (`AUTAN_CELL_ORDER`) may form slightly different lines. The model is checked on other cases with two criteria fixed before measuring: **safety**, the estimate at least the measured peak (a refusal must never let through a run that does not fit), and **precision**, at most 1.25 times it. Round 2, declared before its measurements: tet n024, polyhedral n024, NACA 0012 1024² and the DPW4 CRM grid, on spark-5359. The estimate is conservative: it can exceed the peak by more than the precision bound on unstructured meshes, so `autan-run` may refuse a run that would fit; `--allow-oversubscription` starts it, explicitly and logged (`W0712`).
