# The cost of a J̃ on a wider pattern

Cost note for #395 option (i), written before any decision by CFD - Euler and CFD - Multigrille, with CFD - Générateur kernels and CFD - Multigrille GPU consulted for the device side. Nothing here is a proposal to code; the orchestrator decides after reading.

## What is measured so far

The steady implicit step's J̃ holds the face blocks of the first-order linearisation: dF/dW at the reconstructed states times dW/dU, with the reconstruction's gradients frozen. Its gap to J on the laminar TMR flat plate 69×49 is convective (99.5–100 %) and largest on the wall ring's continuity and energy rows (38–70 %) (#395). Already measured:

- #397's matrix-free product of the true residual, limiter frozen, converged the laminar plate in 1655 steps.
- A fit to the face neighbours only, inside today's pattern, does not close the gap (note 1, both candidates above 0.5 × on the ring's continuity rows).
- Without the limiter the plate does not converge either (#395), and the limiter barely changes ‖(J − J̃) v‖ (to three digits).

What is left is the gradient's chain, which reaches beyond the face neighbours.

## Which pattern the true Jacobian needs (CFD - Euler)

R_i sums the fluxes of cell i's faces. A face's flux reads the reconstructed states of both sides, W_s + φ_s ∇W_s · r_sf for s = i and for the neighbour j. ∇W_s is the least-squares fit over s's gradient stencil. On `face2` (the default, [euler.md](euler.md)) that stencil is s's face neighbours and their face neighbours, so:

- the side s = i reaches the cells within two face steps of i;
- the side s = j reaches the cells within two face steps of j, hence within three of i.

The pattern of the true J at second order is therefore the third ring of face neighbours. The second ring holds every term of the cell's own side but only part of its neighbours' sides.

Measured on the stencil's own weights, as the share of Σ |W_sk · r_sf| (over faces, both sides s and both rows) whose column k lies outside a row's pattern. Meshes: `autan-vv meshes`, n = 24; the plate is TMR's 69×49.

| Mesh | Outside ring 1 (today) | Outside ring 2 |
|---|---|---|
| hex | 59.7 % | 16.8 % |
| tet | 59.5 % | 16.6 % |
| prism | 56.9 % | 15.7 % |
| hybrid | 58.4 % | 17.1 % |
| poly | 66.1 % | 14.7 % |
| plate, all cells | 52.0 % | 13.4 % |
| plate, wall ring | 49.9 % | 11.8 % |

The third ring holds every term: 0 % outside.

**Expected gain, argued from what is measured.**
- With the third ring and the limiter and fallback frozen, J̃'s convective part is the frozen-limiter J, entry for entry. #397 applied the same J matrix-free on the plate (its build there did not freeze the limiter, which plays no part on this case) and converged the laminar plate in 1655 steps. That is the gain this option can claim, under two conditions: the viscous terms reach no further than ring 3 (CFD - Navier-Stokes SA to confirm), and the linear solver converges on the wider matrix (Multigrille's section).
- The second ring is not argued to be enough. It leaves 11.8 % of the weights out on the wall ring, and note 1 left 18.5 % out on the same ring and still failed its threshold, at 0.9–1.8 × the gap without the change. Whether 11.8 % is small enough is not known before measuring.
- If the second ring is costed at all, it needs the same declared check as note 1 before it is coded: the remaining ring continuity gap at most 0.5 × the gap without the change, at the same states. Its lumping must keep a uniform δW mapped to zero (note 1's finding).

## Memory (CFD - Euler: J̃'s blocks; CFD - Multigrille: the hierarchy)

Off-diagonal 5 × 5 blocks per row, measured on the same meshes: the mean over all cells, and over the cells at least three face steps from the boundary.

| Mesh | Ring 1 (today) | Ring 2 | Ring 3 | Growth, ring 2 / ring 3 (diagonal included) |
|---|---|---|---|---|
| hex | 5.75 (interior 6) | 22.3 (24) | 55.6 (62) | 3.4 × / 8.4 × |
| tet | 3.87 (4) | 13.7 (14.5) | 33.5 (36.3) | 3.0 × / 7.1 × |
| prism | 4.85 (5) | 18.0 (19) | 44.5 (48.0) | 3.2 × / 7.8 × |
| hybrid | 4.32 (4.25) | 15.4 (15.2) | 37.7 (37.7) | 3.1 × / 7.3 × |
| poly | 13.0 (14.6) | 61.5 (73.1) | 171 (217) | 4.5 × / 12.3 × |

`BlockMatrix` stores FP64 blocks: a diagonal and its inverse per cell, and one upper and one lower block per interior face. That is 200 B a block, (2 + n₁) × 200 B a cell today with n₁ the ring-1 count. On a general block-CSR pattern it is (2 + n_k) × 200 B, plus the column indices.

**DPW4 CRM** (9.79 M mixed cells, [su2.md](../validation/su2.md); hybrid row as the mix, tet and prism rows as its bounds):

| | Ring 1 (today) | Ring 2 | Ring 3 |
|---|---|---|---|
| J̃, finest level (hybrid) | 12.4 GB | 34.1 GB | 77.7 GB |
| range, tet to prism | 11.5–13.4 GB | 30.8–39.2 GB | 69.5–91.0 GB |

- None of these is in #358's DPW4 estimate (89.5 GiB, at most 104.2 GiB). That run is CUDA and explicit, with no J̃.
- An implicit DPW4 run, CPU today or on the device once #218 exists, adds J̃ and its hierarchy (Multigrille's section) to the physics' share of that estimate.
- Ring 3 alone, at about 78 GB, does not fit beside the physics on one GB10. Ring 2 might, depending on the hierarchy.
- Multigrille's section gives the totals with the multigrid levels, against 104.2 GiB.

## Cost per step (CFD - Euler: assembly; CFD - Multigrille: smoothing)

Assembly, per cell, from the operation counts of `Euler::assemble` (the chain in primitive variables, then one dW/dU per destination block, as note 1's design, with its consistency kept):

- Today: n₁ + 1 blocks, each a 5 × 5 by 5 × 5 product (125 multiply-adds): about 0.7 k for hybrid.
- Chain: per face side and stencil entry, 25 multiply-adds into a primitive accumulator. That is about 2 × 4.3 × 15.4 × 25 ≈ 3.3 k for hybrid. Then one dW/dU product per destination block: 16.4 × 125 ≈ 2.1 k on ring 2, or 38.7 × 125 ≈ 4.8 k on ring 3.
- The blocks written grow with the pattern: 3.1 × on ring 2 and 7.3 × on ring 3 (hybrid).
- On the host the assembly is bandwidth-bound like the rest of the step. Its time grows with the bytes written, so 3–7 × today's assembly, before Multigrille's smoothing costs.

## Effort and PR breakdown

CFD - Euler, host:
1. The chain in `Euler::assemble` on the stencil's own weights (not a fit), behind a `Scheme` parameter set only by `NavierStokes`, Euler bit for bit. About 1 week.
   - On ring 3 it can be tested exactly: second order, limiter off, A v against central differences of R on the four meshes, in FP64 and FP32+acc64, which today's J̃ fails.
   - It needs Multigrille's general block pattern first.

CFD - Générateur kernels and CFD - Multigrille GPU, consulted, to be filled by them:
- The device assembly (generated face loops writing stencil-entry blocks).
- `DeviceMesh` holding the wider pattern.
- The device multigrid on it.

*(CFD - Multigrille: the general block-CSR `BlockMatrix`; the agglomeration and Galerkin coarse operators with edges between non-adjacent agglomerates; the line smoother and Gauss-Seidel on the wider pattern, with a distance-2 colouring; memory of the hierarchy and totals against DPW4; smoothing cost per step; weeks and PRs.)*
