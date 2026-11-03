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

## Face colouring (`codegen/cpp/colouring.cpp`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|

## Device packing (`codegen/`, `physics/src/euler_device.cu`; « CFD - Générateur kernels »)

| Held | Grows with | Class |
|---|---|---|
