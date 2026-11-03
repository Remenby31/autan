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

Every rank reads the whole mesh and runs these steps on it, so every row is held on every rank, beside the global mesh and its geometry. `index_t` is 4 bytes, `offset_t` 8; a `std::map` or `std::set` node about 64. A = the dual graph's arcs, at most 2 `F_i`; per rank, `L_r` = `C_r` + `G_r`. `N_r`, `P_r` (the local nodes and face nodes) and `S_r` (the cells a rank sends) are not MeshInfo counts: their rows are gaps.

| Held | Grows with | Class |
|---|---|---|
| cell weights, 4 B a cell, and a copy of them made even without groups (`partition.cpp:248`, `:273`) | C | M (the copy a gap) |
| the dual graph, `offsets` and `adjacency`, 8 B each, and its counting arrays (`partition.cpp:39`–`51`) | C, A | M |
| with groups: `node_of`, 8 B a cell; the contracted graph, its arcs gathered per node by `push_back` before they are deduplicated (`partition.cpp:71`–`84`) | C; arcs before deduplication | M; gap |
| METIS's `xadj`, `adjncy`, `vwgt`, `part` at `idx_t` width, its workspace, the part of each cell returned as 4 B (`partition.cpp:135`–`152`) | C, A | M |
| `Covering`, built even without periodic pairs (`extract.cpp:92`–`97`) | F, N: 6F + 16N | M |
| `cells_of`, every part's cells on every rank, by `push_back` (`extract.cpp:315`) | C | M |
| `local`, `ghosts_of`'s sets and frontiers, `first_layer`, `faces` (`extract.cpp:278`–`344`) | `L_r`, `G_r`, `F_r` | Q |
| the `nodes` map (`extract.cpp:380`) and the local `MeshInput`'s `x`, `y`, `z`, `face_nodes` by `push_back` (`extract.cpp:389`–`404`) | `N_r`, `P_r` | gap |
| the local `MeshInput`'s face and cell arrays, `assemble`'s temporaries, the local mesh, geometry, `face_global_id`, `face_reversed` (`extract.cpp:364`–`447`, `mesh.cpp:273`–`377`) | `F_r`, `F_ir`, `L_r`, `G_r` | Q |
| one neighbour's ghosts, recomputed for each neighbour in turn (`extract.cpp:472`) | that neighbour's `C_q` + `G_q` | Q |
| send lists, slots and interface faces (`extract.cpp:473`–`494`), `HaloExchange`'s index copies and messages (`halo_exchange.cpp:322`–`336`) | `S_r`, `G_r` | gap (`S_r`), Q |
| `host_slots`, `host_ghosts`: 16 components of 8 B each, whatever the run moves (`halo_exchange.cpp:366`) | `S_r`, `G_r` | gap (`S_r`), Q |
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

## Face colouring (`codegen/cpp/colouring.cpp`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|

## Device packing (`codegen/`, `physics/src/euler_device.cu`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|
