# Setup memory

What each setup phase holds after the mesh is read and before the first iteration, and the bound that covers it, so that the memory estimate ([plan.md](../sdk/plan.md)) holds for any input the contracts accept. Every container a phase keeps or grows with its input is listed with it. [reader-memory.md](reader-memory.md) does the same for the readers.

## Classes

| Class | Bound | Charged in (plan.md component) |
|---|---|---|
| **M** mesh counts | grows with what MeshInfo counts: nodes `N`, cells `C`, faces `F`, interior faces `F_i`, boundary faces `F_b`, face nodes `P`, zones, periodic pairs | `mesh`, `mesh setup`, `renumbered mesh`, `physics`, `physics setup` |
| **E** stencil entries | grows with `counts.stencil_entries`, the final stencil after deduplication | `physics`, `physics setup` |
| **L** multigrid levels | grows with `quality.multigrid` (`cells_by_level`, `edges_by_level`) | the case's `linear_solver` component |
| **Q** partition | grows with `partition` (`cells_by_rank`, `ghosts_by_rank`, `faces_by_rank`, `interior_faces_by_rank`, `stencil_entries_by_rank`, `nodes_by_rank`, `face_nodes_by_rank`, `sent_cells_by_rank`) | `partition`, the per-rank counts of every component |
| **V** device | the packed arrays a device run copies, sized by M, E and L | `device setup`, `physics device` |
| **F** fixed | independent of the input | `runtime` |

A temporary that grows with an intermediate count no class bounds, such as walks before deduplication, pairs before they merge, or candidates before selection, is a gap. Either the phase removes the duplicates before expanding, or its multiplicity gets a bound, a class and a term.

## Geometry and stencils (`codegen/cpp/geometry.cpp`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|
| `pack_geometry`'s `PackedGeometry`: 25 doubles a face (area, centroid, normal, both centroid offsets, moment, magnitude), 4 a cell, 4 an incidence (LSQ weights, side) | faces, cells, incidences K | M: `physics setup` 8(25F + 4C + 4K); the physics keeps converted copies (`physics`) |
| its offsets `d`, one a face of the cell being weighted | the faces of one cell | M |
| under reproducibility `"ranks"` (`GlobalKeys`): `global_order`'s incidences 4K and twins 8K, built in `pack_geometry` and freed when it returns, then built again and kept by the physics with 1 byte a face of orientation | incidences, faces | M: `physics setup` 12K, `physics` 12K + F |
| `incidence_twins`, 8K | incidences | M: `device setup`, construction 8K |
| `cell_stencil`'s marks `listed` 4C and `bound_listed` 4F_b | cells, boundary faces | M: `physics setup` 4C + 4F_b |
| a vertex stencil's node lists: offsets and cursors 16N + 8, and 2 entries of 4 bytes a face node | nodes, face nodes | M: `physics setup` 16N + 8P (the runtime's gradient is face2 today) |
| `cells`, `bfaces`: the row being built, each cell and boundary face once (the marks) however many faces or nodes reach it | one row of the final stencil | E |
| the stencil: offsets 8(C + 1), entries 4E, twins 8E, FP64 weights 24E, and one row of offsets while weighting | cells, stencil entries | E: `physics` 8C + 4S, `physics setup` 32S |
| `sliced`, `sliced_incidences`, `sliced_values`: every slice of rows padded to its longest | up to `slice` times the entries | none: the runtime builds no sliced layout (`DeviceMesh` options `stencil_slice` and `incidence_slice` 0 in `euler_device.cu`); benchmarks and the spike only, and a term before the runtime uses one |

## Assembly (`core/src/mesh.cpp`; « CFD - Noyau maillage »)

`assemble` runs after every reader, in the renumbering and in each rank's extraction; its other temporaries are those rows'.

| Held | Grows with | Class |
|---|---|---|
| `check_cells_closed`'s edge records, 16 B each, reserved once for the owned cell with the most face nodes (`mesh.cpp:183`) | that cell's face nodes, at most P | M |

## Partition and halos (`partition/`, `runtime/src/comm.cpp`; « CFD - Noyau maillage »)

Every rank reads the whole mesh and runs these steps on it, so every row is held on every rank, beside the global mesh and its geometry. `index_t` is 4 bytes, `offset_t` 8; a `std::map` or `std::set` node about 64. A = the dual graph's arcs, at most 2 `F_i`; per rank, `L_r` = `C_r` + `G_r`. `N_r`, `P_r` and `S_r` are `partition.nodes_by_rank`, `face_nodes_by_rank` and `sent_cells_by_rank`.

| Held | Grows with | Class |
|---|---|---|
| cell weights, 4 B a cell, and a copy of them made even without groups (`partition.cpp:248`, `:273`) | C | M (the copy a gap) |
| the dual graph, `offsets` and `adjacency`, 8 B each, and its counting arrays (`partition.cpp:39`–`51`) | C, A | M |
| with groups: `node_of`, 8 B a cell; the contracted graph, its arcs gathered per node by `push_back` before they are deduplicated (`partition.cpp:71`–`84`) | C; arcs before deduplication | M; gap |
| METIS's `xadj`, `adjncy`, `vwgt`, `part` at `idx_t` width, its workspace, the part of each cell returned as 4 B (`partition.cpp:135`–`152`) | C, A | M |
| `Covering`, built even without periodic pairs (`extract.cpp:92`–`97`) | F, N: 6F + 16N | M |
| `cells_of`, every part's cells on every rank, by `push_back` (`extract.cpp:315`) | C | M |
| `local`, `ghosts_of`'s sets and frontiers, `first_layer`, `faces` (`extract.cpp:278`–`344`) | `L_r`, `G_r`, `F_r` | Q |
| the `nodes` map (`extract.cpp:380`) and the local `MeshInput`'s `x`, `y`, `z`, `face_nodes` by `push_back` (`extract.cpp:389`–`404`) | `N_r`, `P_r` | Q |
| the local `MeshInput`'s face and cell arrays, `assemble`'s temporaries, the local mesh, geometry, `face_global_id`, `face_reversed` (`extract.cpp:364`–`447`, `mesh.cpp:273`–`377`) | `F_r`, `F_ir`, `L_r`, `G_r` | Q |
| one neighbour's ghosts, recomputed for each neighbour in turn (`extract.cpp:472`) | that neighbour's `C_q` + `G_q` | Q |
| send lists, slots and interface faces (`extract.cpp:473`–`494`), `HaloExchange`'s index copies and messages (`halo_exchange.cpp:322`–`336`) | `S_r`, `G_r` | Q |
| `host_slots`, `host_ghosts`: 16 components of 8 B each, whatever the run moves (`halo_exchange.cpp:366`) | `S_r`, `G_r` | Q |
| `MpiGhosts`, one exchange per memory space for the run; `fill_ghosts`' temporary exchange (`comm.cpp:108`, `:211`) | as `HaloExchange` | as above |
| the per-rank counts `send_count`, `expected`, `received` (`halo_exchange.cpp:295`) | ranks | F |
| `colour_cells` over the global mesh: 44 B a cell, and its cell pairs gathered by `push_back` with a cell's pairs with itself before they are filtered (`colouring.cpp:33`–`96`) | C; `F_i` + periodic faces | M; gap |

Peaks: partitioning, 20C + 8A + `idx_t` × (3C + A) + METIS's workspace beside the global mesh and geometry; `extract_partition`, at `assemble` of the local mesh, 6F + 16N + 4C beside them plus the Q and gap rows above.

## Renumbering (`core/`; « CFD - Noyau maillage »)

The cell order (`core/src/renumber.cpp`) runs on the global mesh, its geometry alive; `mesh_counts.cpp` takes the same path for MeshInfo.

| Held | Grows with | Class |
|---|---|---|
| `rcm_order`'s graph: counts, arcs before deduplication, `offsets`, `neighbours` (reserved at the size before deduplication) (`renumber.cpp:33`–`53`) | C, `F_i` | M |
| `level`, `placed`, `by_degree`, `order`, the stable sort's buffer, `old_to_new` (`renumber.cpp:112`–`153`) | C | M |
| the breadth-first search's `touched` by `push_back`, frontiers, `last` (`renumber.cpp:68`–`102`) | a component's cells | M (the growth's spare a gap) |
| `renumber_cells`' `MeshInput` copy of the mesh, `assemble`'s temporaries, the renumbered mesh and its `AssemblyMap` (`point_source` by `push_back`) (`renumber.cpp:165`–`194`, `mesh.cpp:273`–`377`) | N, F, `F_i`, P, C | M |
| the renumbered geometry, built while the read one is alive (`renumber.cpp:209`–`231`, `run.cpp:1838`) | F, C: 56F + 32C | M |
| `Ordered`, the renumbered mesh, `new_to_old` and the map, kept for the run beside the read mesh, which the output is written on (`run.cpp:1816`–`1832`) | N, F, P, C | M |

Peaks: `rcm_order`, 37C + 8`F_i`; `renumber_cells`, 40C + 56N + 50F + 8`F_i` + 8P; the geometry's copy, a second mesh and geometry until the read geometry is released (`run.cpp:1840`); each beside the global mesh and geometry.

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
| `colour`, one a face of the range | faces | M: `device setup`, construction 8F |
| `stamp`, one a colour: a face's colour is at most the faces of its two cells, already coloured | 2 × `quality.max_faces_per_cell` | M |
| `offsets` and `next`, 8 bytes a colour each | colours | M |
| `faces`, kept by `DeviceMesh` | faces | M: `device mesh host` 4F |

## Device packing (`codegen/`, `physics/src/euler_device.cu`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|
| `geometry_`, the packed geometry moved in, until `release_host` after bind | faces, cells, incidences | M: `device setup` 8(25F + 4C + 4K) |
| the twins of `incidence_twins` and the colourings above | incidences, faces | M: `device setup`, construction 8K + 8F |
| the topology, colourings, stencil (no twins: Euler drops them) and geometry on the device | faces, cells, incidences, stencil entries | V: `physics device` |
| the device face scratch, 5 values of 8 bytes a face (`face_scratch_width`) | faces | V: `device face scratch` 40F |
| the weights' upload: the physics' FP64 weights and `DeviceMesh`'s copy, 24E each | stencil entries | E: `device setup` 48S |
| `bind`'s staged chunk, at most `upload_chunk` values (2^20) in the device type | fixed | F: `device setup` 8 MiB |
| a second-order run without limiter: the limiter's 5C ones, uploaded from a host vector in `s` | cells | M: `device setup`, weights phase n·C·s |

## Per step (`runtime/src/run.cpp`, `physics/src/euler_physics.cpp`)

What an iteration adds stays bounded by the case or the mesh, or leaves memory as it is written. No container grows with the number of steps.

| Held | Grows with | Class |
|---|---|---|
| the history line, `record` with its residuals and the monitors due this step, appended to `history.jsonl` then dropped | equations, monitors | streamed |
| log lines, appended to `log.jsonl` as they occur | — | streamed |
| `info_["diagnostics"]`: W0150 once at setup; W0710 once per equation leaving its floor; D0107, E0704 once, ending the run | equations | F |
| `info_["diagnostics"]`: W0714, one for every failed checkpoint write, replaced in place with their count, and one if removing the checkpoint fails; the RunMetadata copies no diagnostics (#462) | — | F |
| `info_["monitors"]`, the final record of each monitor; `info_["averaging"]` and `info_["checkpoint"]`, replaced each step | monitors | F: `input` |
| `woken`, `first`, the exemption's floors | equations | F |
| the physics' solver state (`first_norms_`, penalty, CFL cap) and timing scalars (`solve_seconds_`, `log_seconds_`) | — | F |
| the running averages, `accumulate_average` into arrays sized at setup | cells | M: `physics` |
