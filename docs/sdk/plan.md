# `cfd plan`

`cfd plan CASE --mesh-info MESHINFO [--gpus N] [--json]` checks a case against its mesh before any GPU is reserved: it resolves every boundary and cell zone and shows where each condition comes from ([case-schema.md](../contracts/case-schema.md#boundary-resolution)), reports coded diagnostics ([errors.md](errors.md)) and estimates device memory. It exits with 1 when an error is found. Until the C++ mesh readers exist, the MeshInfo is passed explicitly.

## Memory model `cfd.memory/1`

The estimate counts the arrays below; the solver workspace (Krylov vectors, multigrid levels, communication buffers) is not counted yet, so the figure is a lower bound. It will be calibrated against the measured footprint once the implicit solver exists (M2).

Notation: C cells, F faces, F_i interior faces (F minus the boundary faces of the MeshInfo), n solved equations (5 compressible or 4 incompressible, plus 1 for SA or DDES on SA, 2 for SST or DDES on SST). Byte sizes per precision:

| | `fp32+acc64` | `fp64` |
|---|---|---|
| b_c, compute | 4 | 8 |
| b_a, accumulation | 8 | 8 |
| b_j, Jacobian storage | 2 (FP16/BF16) | 8 |

| Array | Bytes |
|---|---|
| Solution and residual | 2 n b_a per cell |
| Update, limiter, gradients (3 components) | 5 n b_c per cell |
| Cell volume and centroid, FP64 | 32 per cell |
| Wall distance (viscous flow only) | b_c per cell |
| Jacobian diagonal block | n² b_j per cell |
| Face area vector and centroid, FP64; owner and neighbour, int32 | 56 per face |
| Jacobian off-diagonal blocks and their column index | 2 (n² b_j + 4) per interior face |

Per-GPU memory is the total divided by `--gpus`, rounded up; halo copies are not counted.
