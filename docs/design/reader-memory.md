# Reader memory

What each reader holds while it reads, and the bound that covers it, so that the memory estimate ([plan.md](../sdk/plan.md), `mesh reading`) holds for any file a reader accepts. Every container a reader keeps or grows with its input is listed with its class; a container that fits no class is a defect, fixed where it is found.

## Classes

| Class | Bound | Where it is stated |
|---|---|---|
| **S** text | a text file is read once into memory at its size `S`, and the reader parses views of it | plan.md, `S = source.bytes` |
| **M** mesh counts | grows with what MeshInfo counts: nodes `N_s` and `N`, cells `C`, faces `F`, boundary faces `F_b`, face nodes, zones, periodic pairs `P` | mesh-info.md, `counts`; plan.md's per-format terms |
| **K** CGNS tree | at most 16 384 nodes in at most 16 levels, 1 MiB of other data read at open; reading costs at most 1 MiB + 16 KiB a node | mesh.md, Memory of a CGNS read |
| **D** CGNS descriptors, filters and point lists | `source.descriptor_bytes`, `source.filtered_bytes` (largest chunk), HDF5's caches, 12 MiB 64 KiB, and 8 × `source.point_list_ids` + 1 MiB of listed ids | mesh.md, Memory of a CGNS read |
| **C** chunks | element sections and point lists read 4096 offsets and 2^17 entries at a time: 1 MiB 32 KiB | mesh.md, Memory of a CGNS read |
| **T** texts | a name, key, header entry or other free text at most 4096 bytes (E0122), `ZoneNames` (families + 1) × 25 600 bytes, `RunMetadata` 64 KiB, a state file's strings 4096 (written in at most 6 each) and numbers 64 characters | mesh.md, E0122; results.md; face-states.md, initial-state.md |
| **R** refused counts | Fluent fluid, solid and interior zone records ≤ 16 384; a polyMesh uniform list ≤ the faces or cells it labels; a Fluent section header ≤ 8 fields; a state file ≤ 16 384 zones and nesting ≤ its structure; a start's FlowSolution and FlowSolutionMean exactly the run's fields, for a continue its RestartState exactly the physics' arrays and its SolverState exactly the physics' names and lengths | mesh.md, Readers; results.md |
| **F** fixed | independent of the input | — |

A temporary that lives for one item of the input (one element, one face, one record) is listed only when that item's size is not already bounded by its class.

## CGNS mesh (`io/src/cgns_read.cpp`, `cgns_util.hpp`, `self_contained.cpp`)

| Held | Grows with | Class |
|---|---|---|
| the library's tree and eager data (`cg_open`) | nodes, data read at open | K |
| `CgioTree::opened_` and the HDF5 objects of the ids a walk holds | nodes listed and not released; `child()` releases the siblings it passes, the tree the rest when it ends | K |
| the link scan's `visited` and `pending` frames; HDF5's link visit | nodes, links, stopping past 3(K+1) + 64 links or 18 path components | K |
| the node count's `pending` and `labels` | nodes still to visit, ids released as walked | K |
| `largest_filtered_bytes`'s object visit | HDF5 objects, after the link scan's bound | K |
| `x_, y_, z_` and their copies into the mesh | nodes `N_s` | M |
| `families_` | `Family_t` nodes | K |
| `sections_`, one `Section` each | `Elements_t` nodes | K |
| `Section::elements`, one `Element` and its connectivity per kept element: cells, and faces a point set names, in any section | cells and labelled faces | M |
| `Section::skipped` runs | at most one per kept element of the section, plus one | M |
| `referenced_sections_` | sections | K |
| `referenced_ranges_`, one interval per `IndexRange_t` naming a face or `MIXED` section | point-set nodes | K |
| `referenced_ids_`, the listed ids of face and `MIXED` sections, sorted and distinct at the end, in storage reserved once for every list entry; the chunk they are read through | 8 × `source.point_list_ids` + 1 MiB | D |
| section chunk buffers `bounds`, `data`; an element larger than 2^17 entries read into its own storage | 1 MiB 32 KiB; that element is kept | C, M |
| BC and subregion point sets: ranges judged id by id; lists in chunks, those of faces or cells kept distinct as they grow, at most twice their distinct ids plus a chunk, 24 B each with the vector's spare | labelled faces and cells | C, M |
| `face_label_`, `cell_label_` | labelled faces and cells | M |
| `face_zones_`, `cell_zones_` and their index maps | zones | M |
| `periodic_`, `declared_`, `sides`, `partner` | periodic pairs and their faces | M |
| `ZoneNames` text and its DOM | `source.zone_names_bytes`, at most (families + 1) × 25 600 | T, D |
| `PeriodicTranslation` text and DOM | 4096 bytes each | T |
| `read_integers<N>` of a node's whole data (`Elements_t` headers, `Zone_t` sizes) | data read at open | K |
| `MeshInput` or `CellwiseMesh` handed to assembly | cells, faces, face nodes, nodes, zones | M |

## CGNS start file (`io/src/cgns_solution.cpp`, `runtime/src/run.cpp` `load_start`)

| Held | Grows with | Class |
|---|---|---|
| the walks of `solution_cells`, `check_start_solution`, `solution_field_names` | nodes, after the count | K |
| `RunMetadata` text and DOM | 64 KiB | T |
| the fields of `FlowSolution`, `FlowSolutionMean`, `RestartState`, `cells` doubles each | the run's fields, and the physics' restart arrays, checked by name before any array is read | M, R |
| `ZoneNames`, `PeriodicTranslation` | the writer's sizes for the run's mesh | T |

## Gmsh (`io/src/gmsh.cpp`)

| Held | Grows with | Class |
|---|---|---|
| `data_`, the file | its size | S |
| section names | 4096 bytes, checked before they are copied | T |
| `$PhysicalNames`, read again for the zones the elements make | names the zones take | M, T |
| `entity_physical_`, 12 bytes per entity of a group that a block of surface or volume elements names, once per 65 536 distinct blocks | blocks of kept elements | M |
| `block_entities`' pairs, of blocks of surface or volume elements | at most 65 536 of 8 bytes: 512 KiB | F |
| `tags_`, `order_`, `xyz_` and the mesh's copy | nodes `N_s` | M |
| `periodic_links_`, one per distinct group pair | `$Periodic` records, each longer than its entry | S |
| a periodic link's node pairs | that link's records | S |
| `mesh_` (`CellwiseMesh`) | cells, labelled faces, zones | M |

## Fluent (`io/src/fluent.cpp`)

| Held | Grows with | Class |
|---|---|---|
| `data_`, the file | its size | S |
| a section header's fields | at most 8 | R |
| `xyz_`, `node_seen_`, `cell_fluent_zone_`, `cell_element_` | nodes and cells, checked against the file's size | M |
| `face_offsets_`, `face_nodes_`, `face_cell0_`, `face_cell1_`, `face_block_` and their copies | faces and face nodes | M |
| `face_blocks_`, `cell_zone_order_` | face and cell sections, bounded by faces and cells | M |
| `records_`, the last record's offset per zone id (a map node, 64 bytes), read again for the zones cells and faces use and the boundary zones declared without faces | distinct zone ids: fluid, solid and interior ones at most 16 384, the others each a face block's zone or a zone of MeshInfo | R, M |
| a section's raw node or cell-type buffer | that section | M |
| the mesh's face zones and their names, copied once more into assembly's uniqueness check (`core/src/mesh.cpp`): boundary zones by their record's name, each a zone of MeshInfo; zones of interior faces, whatever their type, by their id, `(interior N)`, at most 30 bytes, their record's name never copied | zones; sections of interior faces, at most 16 384 | M, T, R |

## polyMesh (`io/src/polymesh.cpp`)

| Held | Grows with | Class |
|---|---|---|
| each `FoamFile`'s text (points, faces, owner, neighbour, boundary, cellZones) | its size | S |
| header keys and values, dictionary keys and values | 4096 bytes each | T |
| `labels()` lists: sized lists checked against the text; a uniform list `N{v}` against the faces or cells it labels | faces, cells | M, R |
| points, faces, owner, neighbour and the mesh's copies | nodes, faces, face nodes | M |
| dictionary entries: only `type`, `startFace`, `nFaces` kept, one entry at a time | patches | M |
| cell zones with cells | cell zones | M |

A keyword given more than once takes its last value, as in OpenFOAM (User Guide, 4.2.14); none of the lists a superseded value holds is parsed into memory:

| List or dictionary | Repeated keyword |
|---|---|
| `points`, `faces`, `owner`, `neighbour` | not applicable: each file is its header and one list, without a keyword |
| `FoamFile` header of every file | not a gap: each value at most 4096 bytes, one held at a time, the last kept |
| `boundary` entries | not a gap: `type`, `startFace` and `nFaces` at most 4096 bytes each, the last kept; other values read past, sub-dictionaries skipped without a copy |
| `cellZones`: `cellLabels` | covered: each occurrence read past (`skip_labels`), the last parsed once its dictionary closes |
| `cellZones`: other entries | not a gap: as `boundary` entries |
| `faceZones`, `pointZones`, `sets` | not applicable: the reader does not open them (mesh.md, Readers) |

## Runtime state files (`runtime/src/run.cpp`)

| Held | Grows with | Class |
|---|---|---|
| the file's text, kept for the run's archive | its size | S |
| the scan before the parse | none | F |
| the parser's stacks and token buffer | nesting ≤ 3 or 5, a string ≤ 6 × 4096 bytes, a number ≤ 64 characters | R, T |
| `RowMap`, one entry per `centroids` or `values` member holding an array, and its numbers | at most 2 × 16 384 such members, counted by the scan; 8 bytes per number of the rows | R, M |
| the DOM of the members that are not rows | the text | S |
| the paired face states and initial fields | faces of the zone, cells | M |
| the manufactured solution's text and DOM | 1 MiB | T |
