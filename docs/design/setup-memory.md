# Setup memory

What each setup phase holds after the mesh is read and before the first iteration, and the bound that covers it, so that the memory estimate ([plan.md](../sdk/plan.md)) holds for any input the contracts accept. Every container a phase keeps or grows with its input is listed with it. [reader-memory.md](reader-memory.md) does the same for the readers.

## Classes

| Class | Bound | Charged in (plan.md component) |
|---|---|---|
| **M** mesh counts | grows with what MeshInfo counts: nodes `N`, cells `C`, faces `F`, interior faces `F_i`, boundary faces `F_b`, face nodes `P`, zones, periodic pairs | `mesh`, `mesh setup`, `renumbered mesh`, `physics`, `physics setup` |
| **E** stencil entries | grows with `counts.stencil_entries`, the final stencil after deduplication | `physics`, `physics setup` |
| **L** multigrid levels | grows with `quality.multigrid` (`cells_by_level`, `edges_by_level`) | the case's `linear_solver` component |
| **Q** partition | grows with `partition` (`cells_by_rank`, `ghosts_by_rank`, `faces_by_rank`, `interior_faces_by_rank`, `stencil_entries_by_rank`) | `partition`, the per-rank counts of every component |
| **V** device | the packed arrays a device run copies, sized by M, E and L | `device setup`, `physics device` |
| **F** fixed | independent of the input | `runtime` |

A temporary that grows with an intermediate count no class bounds, such as walks before deduplication, pairs before they merge, or candidates before selection, is a gap. Either the phase removes the duplicates before expanding, or its multiplicity gets a bound, a class and a term.

## Geometry and stencils (`codegen/cpp/geometry.cpp`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|

## Partition and halos (`partition/`, `runtime/src/comm.cpp`; « CFD - Noyau maillage »)

| Held | Grows with | Class |
|---|---|---|

## Renumbering (`core/`; « CFD - Noyau maillage »)

| Held | Grows with | Class |
|---|---|---|

## Multigrid hierarchy (`solvers/`; « CFD - Multigrille »)

| Held | Grows with | Class |
|---|---|---|
| `agglomerate`'s fine graph `g` (`agglomeration.cpp:126-180`): owner, neighbour, weight, CSR adjacency, zone, volume, centroid, per-cell boundary zones, normals and areas, weakest/strongest, lines | `F_i` (24 B each), `C` (140 B each with push_back capacity), (cell, boundary zone) pairs ≤ `F_b` (72 B) | M |
| Level 0's copy (`as_level`, `:677`): owner, neighbour, lines | `F_i` (8 B), `C` (4 B + 8 B per line; lines ≤ `C`) | M |
| `coarsen`'s per-level work arrays (`:285-302`, `:480`, `:599`): line_of, position, agg, members, coarse lines, grouped, point order, renumber | `C` on level 0 (≤ 124 B per cell with capacity), then `cells_by_level[k]` | M, L |
| The agglomeration front, a `std::set` (`:487`) | one entry (64 B) per edge from a new agglomerate to a free node, so ≤ `F_i` live; level-0 edges are faces, so a cell pair sharing several faces counts several times (none in any measured mesh) | M |
| The line-grouping candidates and their `votes` `std::map` (`:346-350`) | **gap:** the candidates of one group step, ≤ 2 · max_lines · line length · degree map nodes (about 64 B); line length has no cap (732 on RAE fine, 223 on DPW4), bounded only by about 4 `edges_by_level[k]` | gap (to close: a flat vector bounded by the group's edges) |
| Coarse edge `pairs` before they merge (`:650`) | one per cut fine edge, ≤ `edges_by_level[k]` (16 B, up to 2× without reserve) | L |
| The coarse graph (`:608-670`) | `edges_by_level[k+1]` (48 B with capacity), `cells_by_level[k+1]` (156 B), boundary pairs (72 B) | L |
| Each level's maps to the coarser one, kept (`coarse_node`, `coarse_edge`, `coarse_flipped`) | `cells_by_level[k]` (4 B), `edges_by_level[k]` (5 B) | L |
| `line_colours` (`:706-734`): line_of, adjacent lines, colour; the colour lists kept | `cells_by_level[k]` (8 B), 2 `edges_by_level[k]` (8 B), lines (4 B + 8 B kept) | L |
| The multigrid's operators and smoother (`multigrid.cpp:36-150`, `:273-279`): adjacency, line factors (`lower`, `gain`, `dinv` 200 B per slot or cell), work vectors (5 doubles × 5, plus 25 per cell on the levels between) | `cells_by_level[k]`, `edges_by_level[k]` | L |
| The Galerkin coarse operators (`multigrid.cpp:296-298`, first `set_operator`) | 25 doubles per `cells_by_level[k]` and 2 × 25 per `edges_by_level[k]`, k ≥ 1 | L |
| The coarsest dense LU (`multigrid.cpp:239-253`) | (5 `cells_by_level[L−1]`)², only if 5 n ≤ 1000: ≤ 8 MB | F |

**Peak.** The setup transient is the largest of these three moments:
- P1, inside `coarsen` on level 0, during the isotropic loop: ≤ 96 `F_i` + 284 `C` + 72 `F_b` bytes;
- P2, at the coarse-edge build: ≤ 69 `F_i` + 48 `edges_by_level[1]` + 296 `C` + 156 `cells_by_level[1]` + 144 `F_b`;
- after `agglomerate` returns, `line_colours` on level 0, ≤ 32 `C` + 32 `F_i`, beside the multigrid's kept arrays.

This assumes every cell is in some line (lines ≤ `C`) and counts push_back capacity at 2×. Measured with a counting `operator new` around `agglomerate` (main bd3285e0), the heap peak is 0.54–0.80 of max(P1, P2) on 43 meshes: the TMR plates, RAE ×3, NACA, the bumps and every MMS family level up to hybrid n048. It is 0.60 on NACA 0012 1024² (1.05 M cells, 479 MB) and 0.74 on DPW4 (21.6 M cells, 8.97 GB).

The coarsening ratio is enforced for cells only (`cells_by_level[k+1]` ≤ 0.8 `cells_by_level[k]`, `agglomeration.cpp:753`). For edges, the largest measured `edges_by_level[k+1]` / `edges_by_level[k]` is 0.62 (DPW4 level 0 to 1). The L class charges MeshInfo's measured counts, so it does not depend on that ratio.

## Face colouring (`codegen/cpp/colouring.cpp`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|

## Device packing (`codegen/`, `physics/src/euler_device.cu`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|
