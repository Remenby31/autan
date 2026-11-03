# Contract: mesh

The in-memory mesh that readers produce and that kernels, the partitioner and the kernel generator consume. Headers: `core/include/autan/core/{mesh,geometry,cellwise,error}.hpp`, library `autan::core`. The JSON summary sent to the SDK is a separate contract: [mesh-info.md](mesh-info.md).

## Model

A mesh is **face-based**: cells are arbitrary polyhedra known only through their faces; faces are arbitrary polygons (three or more nodes, non-planar allowed, collinear hanging nodes allowed). There is no cell→node connectivity and no cell type. A mesh is 3D and describes one partition: owned cells `[0, n_owned_cells)` followed by ghost cells `[n_owned_cells, n_cells)`, which are halo copies of cells owned by another partition. An unpartitioned mesh has no ghost cells.

| Type | Meaning |
|---|---|
| `index_t` = `int32_t` | Index of a point, face, cell or zone within one mesh. |
| `offset_t` = `int64_t` | Offset into a CSR array (face→node lists can exceed 2³¹ entries). |
| `no_cell` = `-1` | Absent cell. |

## Arrays

All arrays are structure-of-arrays, indexed by the entity they describe, and exposed both as `std::span` accessors on `autan::Mesh` and as raw pointers in `autan::MeshView` (the same struct describes host or device copies).

| Array | Size | Content |
|---|---|---|
| `x`, `y`, `z` | n_points | FP64 coordinates. |
| `face_node_offsets` | n_faces + 1 | CSR offsets, `[0] = 0`. |
| `face_nodes` | `face_node_offsets[n_faces]` | Nodes of each face, ordered so the right-hand-rule normal points from owner to neighbour (outward on the boundary). |
| `owner` | n_faces | Cell on the side the normal points away from. |
| `neighbour` | n_interior_faces | Cell on the side the normal points into. |
| `face_zone` | n_faces | Index into `face_zones`. |
| `cell_zone` | n_cells | Index into `cell_zones`. |
| `cell_global_id` | n_cells | `int64` identity of the cell: its index in the mesh as first assembled (the identity then), kept through partitioning and renumbering. |
| `boundary_zone_offsets` | n_face_zones + 1 | Owned boundary faces of face zone `z` are `[off[z], off[z+1])`; `off[0] = n_interior_faces`, `off[n_face_zones] = n_owned_faces`. Interior zones have an empty range. |
| `halo_boundary_zone_offsets` | n_face_zones + 1 | The same for the boundary faces of ghost cells, from `n_owned_faces` to `n_faces`. |
| `ghost_translation` | n_cells − n_owned_cells | Translation of each ghost cell: zero, or the shift of a periodic image ([Periodic pairs](#periodic-pairs)). |
| `cell_face_offsets`, `cell_faces` | n_cells + 1, 2·n_interior + n_boundary | CSR of the faces of each cell, in increasing face index. A face is outward for cell `c` iff `owner[f] == c`. |

## Invariants

`autan::assemble(MeshInput)` is the only way to build a `Mesh`; it establishes and checks these, and throws `autan::Error` on the first violation:

1. Faces `[0, n_interior_faces)` are interior, faces `[n_interior_faces, n_faces)` are boundary.
2. Interior faces have `owner < neighbour` and are sorted by `(owner, neighbour)`. Two cells may share several faces.
3. Boundary faces of owned cells come first, then those of ghost cells; each half is grouped by ascending zone index and sorted by owner within a zone.
4. Each face has ≥ 3 distinct nodes; each owned cell has ≥ 4 faces; each ghost cell has ≥ 1 face.
5. Every owned cell surface is closed and consistently oriented: walking each face with its normal pointing out of the cell, every edge is traversed exactly twice, once in each direction. A ghost cell carries any subset of its faces.
6. Faces that touch an owned cell come first: interior faces `[0, n_owned_interior_faces)` and boundary faces `[n_interior_faces, n_owned_faces)`; their owner is owned (orientation from the lower index puts the owned cell first). The others, halo faces, join ghost cells only, so that stencils of owned cells see the faces of their ghost neighbours. A face loop that writes residuals runs over the owned faces only: its contributions to ghost cells (`neighbour[f] ≥ n_owned_cells`) are partial sums that a reverse halo exchange adds to the owning partition, as an adjoint (transposed) kernel needs. An unpartitioned mesh has no halo faces.
7. A face zone's `boundary` flag says whether it holds boundary or interior faces, and every face matches the role of its zone (a declared zone may be empty). Face-zone names are unique; cell-zone names are unique. Global ids are ≥ 0, and no two cells share both their global id and their translation (`ghost_translation`, zero for owned cells): a periodic image repeats the id of the cell it copies.
8. Every point is referenced by a face (unreferenced points are removed).
9. Cell order is the order of the input; `assemble` never renumbers cells. Sorting is stable, so the same input always yields the same arrays bit for bit. `assemble(input, &map)` returns, in a `autan::AssemblyMap`, the input face of each face, whether it was reversed, and the input point of each point.

## Zones

Every face belongs to exactly one face zone and every cell to exactly one cell zone.

| Field | Face zone | Cell zone |
|---|---|---|
| `name` | Name in the source (Fluent zone name, Gmsh physical name, OpenFOAM patch). | Same. |
| `bc_type` / `type` | Type exactly as written in the source; empty if the format has none. | Same. |
| `kind` | `BcKind`: `unspecified`, `wall`, `thermal_wall`, `slip`, `symmetry`, `far_field`, `inlet`, `outlet`, `periodic`, normalised with the table of [mesh-info.md](mesh-info.md#normalized-boundary-types). | `CellKind`: `unspecified`, `fluid`, `solid`. |
| `boundary` | Role: boundary (true) or interior (false) faces. | — |
| `provenance.source_id` | Identifier in the source file, or −1 for a zone the reader synthesised. | Same. |
| `provenance.detail` | Human-readable origin. | Same. |

Synthesised zones: `interior` for interior faces with no zone in the source, `unassigned` for boundary faces with no zone in the source (so `autan plan` reports the missing condition), `fluid` (kind `unspecified`) for cells with no zone in the source. A source zone with one of these names collides and is rejected with E0110.

`Mesh::source()` records the format (`gmsh`, `fluent`, `polymesh`, `cgns`, or `memory` for meshes built in code) and the path as given.

## Periodic pairs

`Mesh::periodic_pairs()` lists the periodic pairs: two boundary zones of kind `periodic` whose faces coincide under a translation, `x_b = x_a + translation` for every node of zone a's faces (`MeshInput::periodic`, `CellwiseMesh::periodic`). Each pair stays two boundary zones in the mesh, so every invariant above holds unchanged; `Mesh::periodic_partners(p)` gives, for each face of zone a in order, its partner face in zone b. `assemble` checks the declaration (E0118: zones out of range, empty, not boundary zones of kind `periodic`, a zone paired with itself or in two pairs, a boundary zone of kind periodic in no pair, a translation that is zero or not finite) and matches the faces (E0119):

- each node of zone a, translated, coincides with exactly one node of zone b: closer than 1e-4 of the shortest edge of zone a's faces at that node, far below any spacing and far above single-precision coordinates;
- each face of zone a, so mapped, has the node set of exactly one face of zone b, the two zones have as many faces, and partners run through their nodes in opposite directions (both point out of the domain).

Matched nodes are then moved onto translates. The nodes that matched pairs join form classes; they are placed by forward steps `x_b = x_a + translation`, one addition each, so an exact input does not move. Every node that is zone b of no pair is a root and stays; walks from the roots, in increasing order, place the nodes not yet placed, and nodes no root reaches (classes that are cycles, as sides split into pairs of opposite translations) take the smallest of them as a further root. Every pair is then checked to 4 ulps of the larger of `|x_a|` and the translation, which is what makes two roots of one class agree: a pair the placement contradicts is E0119. A node joined by k pairs can move by up to k pair mismatches. The faces and owned cells that touch a moved node are checked against `compute_geometry`'s criteria before and after: one already degenerate is E0120 or E0121, as `compute_geometry` would report it, and one the move makes degenerate is E0119, with its index and its area or volume before and after in `data`. Moves larger than round-off (4 ulps of the larger of the coordinate and the largest translation) are reported by `Mesh::periodic_snap()` (nodes so moved, largest move of any node, largest pair mismatch relative to its tolerance, largest relative change of a cell's volume), and by the run with `W0150`; `scaled` and `renumber_cells` keep that record. The global geometry and every partition's are computed on the moved coordinates.

`renumber_cells` keeps the pairs. A partition carries none: there each periodic face is an interior face to an image ghost cell, the cell across it translated (`ghost_translation`), and the pair's zones hold interior faces ([partitioning.md](../design/partitioning.md#periodic-images)). Both cells of such a face see the zone-a face, and its geometry agrees with the part's nodes to round-off. `write_solution` records each pair as two reciprocal SIDS `GridConnectivity_t` (`Abutting1to1`, `FaceCenter`, `PointList` / `PointListDonor`, a `Periodic_t` translation in R4 as CGNS requires, and the descriptor `PeriodicTranslation`, three numbers in JSON, that keeps it to the bit), named after each side's zone, and `read_cgns` reads them back as pairs. Rotations are reserved.

## Geometry

`autan::compute_geometry(mesh)` returns a `autan::Geometry` (SoA, FP64; raw pointers via `autan::view`):

| Array | Size | Definition |
|---|---|---|
| `face_area_{x,y,z}` | n_faces | Area vector S_f = Σᵢ ½ (pᵢ − m) × (pᵢ₊₁ − m), m the mean of the face nodes. It equals ½ Σᵢ pᵢ × pᵢ₊₁, the vector area of any surface spanning the face loop. Owner → neighbour, outward on the boundary. |
| `face_area` | n_faces | ‖S_f‖. |
| `face_centroid_{x,y,z}` | n_faces | Centroids of the fan triangles (m, pᵢ, pᵢ₊₁), weighted by their area projected on S_f / ‖S_f‖. Exact for planar polygons, convex or not. |
| `cell_volume` | n_cells | Volume of the polyhedron bounded by the fan triangles, summed over tetrahedra (apex, m, pᵢ, pᵢ₊₁) with signed volumes; the apex is the mean of the face centroids. Volumes of all cells sum to the volume enclosed by the boundary triangles. |
| `cell_centroid_{x,y,z}` | n_cells | Volume-weighted centroid of the same tetrahedra. |

Ghost cells get NaN volume and centroid: the forward halo exchange fills them from the owning partition. `compute_geometry` on a partition is valid but not bitwise identical to the global computation; the bitwise path is the geometry distributed by the partitioner ([partitioning.md](../design/partitioning.md)).

On cells with planar faces, Σ_f c_f ⊗ S_f = V I (divergence theorem), so Green–Gauss with face-centroid values is exact for linear fields. On warped faces a single (c_f, S_f) pair cannot integrate a linear field over the fan exactly; the error is of the order of the warp, as in other cell-centred codes. Kernels that need exactness there must use least-squares gradients or per-triangle quadrature.

`autan::cell_closure_error(mesh, geometry)` returns ‖Σ_f ±S_f‖ / Σ_f ‖S_f‖ for each owned cell; it is below 1e-12 on every valid cell. Geometry is FP64 by design; kernels that compute in FP32 convert their own copies.

## Wall distance

`autan::WallSurface(mesh, wall_zones)` (`core/include/autan/core/wall_distance.hpp`) holds the fan triangles of the zones' boundary faces, as `compute_geometry` splits them (a strongly non-convex face is represented by its fan); `distance(p)` is the exact distance from `p` to the nearest of them and of their periodic copies, as far as the mesh spans, FP64, through a bounding-volume tree. Errors: a wall node or node mean that is not finite is E0120; periodic translations that do not reduce to three independent directions, or that would need more than 10⁶ periodic copies to cover the mesh, are E0118. `autan::wall_distance(mesh, geometry, wall_zones)` gives it for every cell centroid. On a partition, build the surface from the global mesh: a part's nearest wall may lie in another. A zone out of range or holding interior faces is E0102.

## Quality and ordering

`autan::compute_quality(mesh, geometry)` (`core/include/autan/core/quality.hpp`) returns the checkMesh metrics of OpenFOAM: per-face non-orthogonality (degrees) and skewness, per-interior-face volume ratio, per-owned-cell aspect ratio; the header gives each definition.

`autan::rcm_order(mesh)` returns a reverse Cuthill–McKee order of the owned cells (ghosts stay last); `autan::renumber_cells(mesh, new_to_old)` applies any such permutation and re-sorts the faces, so other orderings (Hilbert) plug in through the same call. Every cell keeps its global id.

## Building a mesh

Readers of face-based formats (Fluent, OpenFOAM) and the partitioner fill a `autan::MeshInput`: faces in any order, each with `face_cell0` (normal points away from it) and `face_cell1` (`no_cell` on the boundary); ghost cells last with `n_ghost_cells`, and `cell_global_id` when partitioned. Readers of element-based formats (Gmsh, CGNS) fill a `autan::CellwiseMesh` with `add_cell` (tetra, pyramid, prism, hexa in Gmsh node order), `add_polyhedron` (outward faces) and `add_labelled_face` (boundary or internal surface elements carrying a zone), then call `autan::to_face_based`, which merges coincident faces by node set. `autan::assemble` keeps only the nodes some face uses; `Mesh::source().nodes` records how many the input held. A zone that labels faces takes their role (interior or boundary); a zone that labels none keeps its declared role. `autan::scaled(mesh, scale)` multiplies a mesh's coordinates and its periodic and ghost translations by `scale`, its topology unchanged: the runtime brings a mesh to metres with the solver input's `mesh.scale`.

## Readers

`autan::io::read_mesh(path)` (library `autan::io`, header `io/include/autan/io/read_mesh.hpp`) detects the format from the content and returns an assembled mesh; `read_gmsh`, `read_fluent`, `read_cgns` and `read_polymesh` skip the detection. CGNS support needs the CGNS library (`AUTAN_ENABLE_CGNS`, on by default); a build without it reports CGNS files as E0133.

| Format | Supported | Zones | Rejected (E0133 unless noted) |
|---|---|---|---|
| Gmsh MSH 4.1, ASCII and binary | Linear tetra, pyramid, prism, hexa; triangles and quadrilaterals label faces; sparse node tags; parametric nodes; unknown sections are skipped as the format prescribes. `$Periodic` surface links with a translation become periodic pairs (master group → zone a, slave group → zone b). | Physical volumes → cell zones, physical surfaces → face zones (also when they have no element), named from `$PhysicalNames` wherever it appears, else the tag; type empty, kind `unspecified` (`periodic` for the groups of a periodic link). | MSH 2 and other versions, higher-order elements, `$PartitionedEntities`, rotational or transform-less `$Periodic` surface links, `$GhostElements`, big-endian files; non-positive physical tags (E0132); an entity in two physical groups (E0135); a periodic link of another dimension than 0 to 2, a periodic surface in no physical group, node pairs that do not follow the link's translation, or two translations for one pair of groups (E0132). |
| Fluent mesh, ASCII and binary (2010/3010, 2012, 2013) | Sections 0, 1, 2, 10, 12, 13, 39, 45; 4, 37, 38, 40 are skipped. Mixed and polygonal faces, polyhedral cells (type 7). The face normal points toward c0. | Zone sections 39/45 give names and types (they take precedence over face-section codes); a zone without one is named `zone-<id>` and typed from its code; a zone whose faces are interior, whatever its type, which nothing exposes, is named `(interior <id>)`, which no Fluent name can be, its record's name not kept; a declared zone without faces is kept. A zone's role is that of its faces. Declared element types must match the cell's face count (E0132). | 2D meshes, periodic shadow faces (18), hanging-node trees and interface parents (58, 59, 61–63), any other section. |
| CGNS/HDF5 (SIDS 4.x, header `io/include/autan/io/cgns.hpp`) | One base, one unstructured zone; cells as `NFACE_n` over `NGON_n` faces (positive face id: normal out of the cell), or linear `TETRA_4`, `PYRA_5`, `PENTA_6`, `HEXA_8` and `MIXED` with `TRI_3`/`QUAD_4` faces; point and edge elements (`NODE`, `BAR_2`), in their own sections or in `MIXED` ones, are skipped, their nodes unread; of the descriptors, the reader reads `ZoneNames` and `PeriodicTranslation`, each sized from its node first (E0122); the others are accepted whatever their size, but the CGNS library loads every descriptor when it opens the file, so `source.descriptor_bytes` counts them all (a file declaring many families only raises the `ZoneNames` cap: the memory estimate counts that descriptor's size, `source.zone_names_bytes`, not its cap); `PointList`, `PointRange`, `ElementList`, `ElementRange`. | `BC_t` → boundary zones; a BC naming a family belongs to that family's group (the root of its `FamilyName_t` chain), and BCs of one group form one zone. `BCType`, or the family's `FamilyBC` when `FamilySpecified`, is the type. A BC or `ZoneSubRegion_t` over cells gives cell zones (type empty). SIDS defines no `FamilyName_t` under `Elements_t` (CGNS refuses to read one), so a section names no zone. Whether a BC labels faces or cells follows the dimension of its elements, whatever its `GridLocation`. A `FamilyName` may be a path; its last component names the family. Files with the `ZoneNames` descriptor of results.md get their zone names back, empty boundary zones included. A face-to-face `GridConnectivity_t` of the zone with itself carrying a `Periodic_t` translation, with its reciprocal, gives a periodic pair (the first of the two is zone a), its zones of kind `periodic`. | Several bases or zones, structured zones, higher-order elements, BCs given by vertices, `NFACE_n` mixed with standard cells; data in another file, by a CGNS link, an HDF5 external link, external storage or a virtual dataset, found from the file's structure before any data is read (E0134); other connectivities and rotational periodicity; a periodic connection without its reciprocal (E0132); inconsistent offsets, cell counts or element references, unknown families (E0132); a face or cell in two zones, or BCs of one zone with different types (E0135). CGNS must be built with 64-bit `cgsize_t`. |
| OpenFOAM polyMesh, ASCII and binary, 32/64-bit labels and scalars | The `polyMesh` directory or a case directory holding `constant/polyMesh`; comments anywhere; sized, unsized and uniform label lists; ASCII faces in nested form (`faceList` or `faceCompactList`), binary faces as `faceCompactList`. | Patches → boundary zones (type as written), `cellZones` → cell zones (type empty). `faceZones`, `pointZones` and `sets` are selections, not zones, and are not read. | Compressed files, binary `faceList`, processor patches; a cell in two cell zones (E0135). |

`autan::io::write_solution(file, mesh, solution)` writes `solution.cgns` with the layout of [results.md](results.md#solutioncgns) (node names by its rule, `autan::io::zone_node_names`); files it writes read back through `read_cgns` to the same mesh.

`autan::io::mesh_hash(path)` computes the mesh hash and `autan::io::mesh_info_json(mesh, hash, bytes, logical, run)` the MeshInfo document (`logical`: `autan::io::mesh_logical_bytes(path)`), both as [mesh-info.md](mesh-info.md) defines them; `autan::io::run_counts` (library `autan_mesh_counts`, which links the kernels, the partitioner and the solvers so that `autan_io` does not) gives its counts of what a run computes on it at a `mesh.scale` (on the scaled mesh, as the run does, since round-off decides couplings at the lines' threshold): the smoother's lines and the multigrid's levels (`solvers::agglomerate`, nodes and edges per level) in a cell order, the face2 stencil and periodic ghosts of a single-rank run, and each rank's share of the run's partition. The `autan-mesh-info [--json [--scale S] [--cell-order rcm|input] [--ranks N [--groups lines]] | --surface OUT] MESH` tool prints a summary (counts, zones, quality maxima, closure) or that document, its counts at S metres per mesh unit (default 1) and its lines in the cell order given (default `rcm`, the run's) and the partition of a run on one rank, or with `--ranks N` on N ranks, the smoother's lines kept whole with `--groups lines` (N ≥ 2); or it writes the [boundary surface](#boundary-surface). It exits 1 with the E01xx code on a read or write error, or the partitioner's (`E0140`–`E0143`), and 2 on a usage error. A run does not compute MeshInfo: it writes the one it was planned with ([solver-input.md](solver-input.md), `mesh_info`).

`autan::io::write_polymesh(dir, mesh)` writes a mesh as a binary polyMesh that `read_polymesh` reads back to the same mesh; patches are typed `wall`, `symmetry` or `patch`, and each zone name is written as an OpenFOAM word of letters, digits and `_ - . : +` (each run of other characters becomes `_`, a leading digit takes a `_` before it, `interior`, the reader's zone of the internal faces, becomes `interior_`); the renamed zones are returned. The `autan-mesh-prisms [--planarity DEG] [--pillar DEG] [--aspect A] MESH OUT` tool merges the boundary-layer tetrahedra of MESH into prisms (`autan::merge_layer_prisms`, #337) and writes the result as the polyMesh OUT: advancing in layers from the wall zones, three tetrahedra become one cell when they are the only set filling a prism on the front triangle whose side quadrilaterals are planar within `--planarity` degrees (default 5), pillars within `--pillar` degrees of the wall-side cap's normal (default 30) and height at most `--aspect` times its shortest cap edge (default 1), with positive volume at every corner. The merged cell keeps each side as its two triangles, so it is exactly their union (an 8-face polyhedron); every other face and cell is kept. Before writing, the tool checks that the nodes and the boundary faces and zones are unchanged and that each merged cell's volume is its tetrahedra's within 1e-12 relative; it prints a JSON report of the cells, the faces, the renamed zones, the largest non-orthogonality and least volume ratio before and after, and, by cause, where the layers stopped.

Every container a reader holds and the bound that covers it are listed in [reader-memory.md](../design/reader-memory.md). What a reader does not use it reads past without keeping a copy, so that it costs the file's text at most: Gmsh physical names no zone takes (read again from the file once the zones are known) and element blocks of points and curves or without elements, and entities no other block names (`$Entities` read once per 65 536 distinct blocks); Fluent fluid, solid and interior zones no cell or face is in (their records read again by offset, at most 16 384 of them, E0133: the format sets no limit ([Ansys Fluent, legacy case and data file formats](https://ansyshelp.ansys.com/public/views/secured/corp/v251/en/flu_ug/flu_ug_app_form_case_data.html)), and no real Fluent mesh is at hand, the repository's and V&V's holding one fluid zone; so the limit is the CGNS node limit, a mesh's zones bounded alike whatever its format, with the same margin over the real grids, which carry at most 669 boundary zones and one fluid zone); CGNS boundary faces, in their own sections or in `MIXED` ones, that no BC, subregion or connectivity point set names (the ranges and listed ids that name them kept, not a mark per element), and every standard face beside `NGON_n` faces, which are then the mesh's faces; polyMesh dictionary entries other than `type`, `startFace` and `nFaces`, and cell zones without cells (which MeshInfo omits); and a size it is told is checked before it is allocated: a polyMesh uniform list `N{v}` holds at most the faces or cells it labels, a Fluent section header at most 8 fields (its node, cell and face headers hold five), E0132 otherwise; CGNS BC and subregion point sets of points or edges, a range judged id by id and a list read 2^17 ids at a time, never expanded.

The test meshes and how they are generated: [tests/data/mesh/README.md](../../tests/data/mesh/README.md).

### Memory of a CGNS read

Opening a CGNS file costs, beside the arrays the reader keeps, at most `source.descriptor_bytes` + `source.filtered_bytes` + 12 MiB 64 KiB of caches + 1 MiB + 16 KiB × `source.cgns_nodes` for the tree + 1 MiB of other data read at open ([mesh-info.md](mesh-info.md#document)), from the sources of CGNS 4.5.2 and HDF5 2.1.1:

- `cg_open` reads every `Descriptor_t` of the file (`cgi_read_DDD_from_list`, `cgi_read_string`, `cgns_internals.c:5483`) into a buffer of its size plus one (`cgi_read_node`, `cgns_internals.c:6758`), filled by one `H5Dread` (`adfh/ADFH.c:3665`) and held in `cgns_descr.text` (`cgns_header.h:166`) until `cg_close`: `descriptor_bytes`.
- A filter applies to whole chunks: a filtered dataset, descriptor or array, is read one chunk at a time, each of L logical bytes by the chunk's extent, which may exceed the dataset's, and C stored bytes at most the dataset's, into a buffer of C (`H5Dchunk.c:4832`). Deflate inflates it into a buffer that starts at C (`H5Zdeflate.c:85`) and doubles by `realloc` (`H5Zdeflate.c:133`), which may copy the at most L bytes inflated so far, then frees its input (`H5Zdeflate.c:152`); shuffle, szip and scale-offset allocate their output and free their input likewise (`H5Zshuffle.c:141`, `:288`). The chunk is then copied to its destination, which the estimate counts on its own. One chunk of one dataset is read at a time, so the transient is at most C + 2L of the largest: `filtered_bytes`. Only its touched pages count, as the resident memory does.
- HDF5's caches, which the CGNS library leaves at their defaults (`ADFH_CONFIG_DEFAULT`, `adfh/ADFH.c:64`): metadata at most 4 MiB (`H5C__DEFAULT_MAX_CACHE_SIZE`, `H5Cprivate.h:85`), raw-data chunks 8 MiB (`H5Pfapl.c:94`), sieve buffer 64 KiB (`H5Pfapl.c:121`).
- Other data read at open: `cg_open` reads every `DataArray_t` whole (`cgi_read_array`, `cgns_internals.c:5217`) except under `GridCoordinates_t`, `FlowSolution_t`, `Elements_t`, `ZoneSubRegion_t`, `DiscreteData_t`, `ParticleCoordinates_t`, `ParticleSolution_t` and `UserDefinedData_t` (`:5184`), never a point set's (`cgi_read_ptset`, `:3358`), and the small headers of other nodes (ranges, ordinals, rinds, conversions, iteration counts: `cgi_read_section`, `cgi_read_boco`, `cgi_read_biter`, `cgi_read_ziter`…). Beyond descriptors, at most 1 MiB, 24 times the 42 KB of the CRM-HL grid (DPW4: 2.4 and 3.2 KB), counted by the walk that counts the nodes; more is refused before `cg_open` (E0133).
- The tree: `cg_open` allocates for every node it parses (families, sections, BCs, user data, solutions, descriptors…), before the reader can drop any, and HDF5 keeps the path of every open node, so a node costs more the deeper it is. Reading a file costs at most 1 MiB + 16 KiB per node of its tree (`source.cgns_nodes`), `read_mesh` included, its own copies of families and sections among them, in a tree of at most 16 levels. Measured as the resident peak of `autan-mesh-info --json` (Release) on a one-tetrahedron file plus N nodes of one kind, against N = 0: unused `Family_t`, one-edge `BAR_2` sections (3 nodes each), `BC_t` over an edge (3), `UserDefinedData_t`, one-byte `Descriptor_t` and `FlowSolution_t` (2), at N = 16 to 16 384 kinds (49 152 nodes), the largest 15.5 KB a node from 1024 nodes up, the 1 MiB covering the page and allocator granularity below (at most 0.9 MB at 48 nodes); and 16 000 `UserDefinedData_t` or `Descriptor_t` of 32-character names under a chain of user data, 15.1 KB a node at level 18 (19.2 KB at 34: about 256 B a level). A file of more than 16 384 nodes, four times the 4032 of the CRM-HL grid (the most of the real grids counted: DPW4 332 and 442), or deeper than 16 levels, three times every real grid's 5, is refused before `cg_open` (E0133), every walk of the tree stopping at those limits.

Reading the elements adds a constant: the reader reads each section through cgio in chunks of at most 4096 offsets and 2^17 connectivity entries (1 MiB 32 KiB of buffers; a larger element is read straight into the storage it keeps), the library loading none of it (`cgi_read_array`, `cgns_internals.c:5184`), so it holds only the elements it keeps; the elements it skips are kept as runs of consecutive ids, at most one per kept element plus one, 24 bytes each. Before the sections, it marks the faces a BC, subregion or connectivity names: each range as an interval, and the ids the lists give in face and `MIXED` sections, in storage sized once at 8 bytes per list entry: 8 × `source.point_list_ids` + a 1 MiB chunk, released before the BCs and subregions are read. Their lists are then read in the same chunks and made distinct as they grow, so a list naming its faces or cells many times holds about twice the distinct ids, not its entries. A `MIXED`, `NGON_n` or `NFACE_n` section must have its `ElementStartOffset` (CGNS 4.0); one without is refused (E0133); offsets that do not start at 0, rise and end at the connectivity size, a connectivity not one-dimensional or not of its element type's size, are E0132, found before anything is sized from them.

`tests/io/test_cgns.cpp` measures the resident peak against this bound for an 8 MiB and a 128 MiB descriptor, compressible, incompressible and half of each, and for a deflated coordinate array.

## Boundary surface

`autan::io::surface_bytes(mesh, hash)`, which `autan-mesh-info --surface OUT MESH` writes to `OUT`, gives the mesh's boundary faces as an `autan.surface/1` file, for a viewer to draw them and pick zones. It checks the geometry first (`E0120`, `E0121`), since an inverted cell would turn normals inward. A boundary whose node count, total of face nodes (`face_offsets[F]`) or padded header length exceeds 2³² − 1 is `E0113`.

**Layout.** Little-endian: a `u32` byte length `n`, then `n` bytes of UTF-8 JSON (the header), padded with spaces so that `4 + n` is a multiple of 8, then the buffers. Buffer offsets count from byte `4 + n`, the end of the padded header; each is a multiple of 8.

**Header.**

| Key | Value |
|---|---|
| `schema` | `"autan.surface/1"` |
| `source.sha256` | The mesh's `source.sha256`, as in its MeshInfo ([mesh-info.md](mesh-info.md)). |
| `nodes` | Number of nodes N. |
| `faces` | Number of boundary faces F, the sum of the zones' `faces`. |
| `zones` | The MeshInfo `boundary_zones`, in their order, with their `name`, `type` and `native_type` ([mesh-info.md](mesh-info.md)), `first_face` and `faces`: zone k's faces are faces `first_face` to `first_face + faces − 1`, `faces` as in MeshInfo. |
| `buffers` | `name`, `dtype` (`f64`, `u32`), `offset` (bytes), `length` (elements), for each buffer below. |

**Buffers.**

| Name | dtype | Length | Content |
|---|---|---|---|
| `positions` | `f64` | 3N | x, y, z of each node, nodes in the order the faces first use them. |
| `face_offsets` | `u32` | F + 1 | `face_offsets[0] = 0`; face f's nodes are `face_nodes[face_offsets[f] .. face_offsets[f + 1])`. |
| `face_nodes` | `u32` | `face_offsets[F]` | Node indices into `positions`. |

Faces are the mesh's boundary faces, zone by zone, each zone's in mesh order. Each face is the polygon the mesh stores, not triangulated, its nodes in the mesh's order: by the right-hand rule its normal points out of the domain. Periodic zones are included; interior zones are not. The surface is closed and consistently oriented: each edge is traversed as often in one direction as in the other (once each way where the boundary is a manifold, more where it touches itself along an edge), and the area vectors sum to zero up to round-off.

## Errors

Every failure throws `autan::Error`; callers branch on `code()`, never on the message. Codes E0101–E0199 belong to the mesh core and readers (E0101–E0139) and to the partitioner and halo exchange (E0140–E0149); the SDK registry carries their fix and doc page.

| Code | Name | Raised when |
|---|---|---|
| E0101 | `mesh_inconsistent_sizes` | Input arrays disagree on a size, or CSR offsets are not monotone. |
| E0102 | `mesh_index_out_of_range` | A node, cell or zone index is out of range. |
| E0103 | `mesh_face_too_few_nodes` | A face has fewer than 3 nodes. |
| E0104 | `mesh_face_repeated_node` | A face lists a node twice. |
| E0105 | `mesh_face_same_cell_both_sides` | A face has the same cell on both sides. |
| E0106 | `mesh_cell_too_few_faces` | An owned cell has fewer than 4 faces, or a ghost cell none. |
| E0107 | `mesh_cell_not_closed` | Invariant 5 fails for a cell. |
| E0108 | `mesh_face_shared_by_more_than_two_cells` | More than two cells list the same face. |
| E0109 | `mesh_face_zone_role` | A face does not match its zone's role, or labels put one zone on both interior and boundary faces. |
| E0110 | `mesh_duplicate_zone_name` | Two face zones or two cell zones share a name. |
| E0111 | `mesh_labelled_face_not_found` | A labelled face matches no cell face. |
| E0112 | `mesh_empty` | No cell, face or point. |
| E0113 | `mesh_too_large` | A count exceeds the `index_t` range. |
| E0114 | `mesh_shared_face_orientation` | Two cells list their common face with the same orientation (one is inverted). |
| E0115 | `mesh_face_labelled_twice` | A face carries two labels. |
| E0117 | `mesh_duplicate_global_id` | Two cells share a global id, or one is negative. |
| E0118 | `mesh_periodic_invalid` | A periodic pair is declared wrongly: zones out of range, empty or not boundary zones of kind periodic, a zone paired with itself or in two pairs, a boundary zone of kind periodic in no pair, or a translation that is zero or not finite. |
| E0119 | `mesh_periodic_unmatched` | The faces of a periodic pair do not match one to one under its translation, the pairs place a node in two ways, or moving the matched nodes onto translates makes a face or cell degenerate. |
| E0120 | `geometry_degenerate_face` | A face has zero or non-finite area. |
| E0121 | `geometry_non_positive_volume` | An owned cell has zero, negative or non-finite volume. |
| E0122 | `mesh_text_too_long` | A name or text the reader keeps is longer than its cap: 4096 bytes for a zone, family, BC or patch name, a dictionary entry, a Gmsh section name or `PeriodicTranslation`; (families + 1) x 25 600 bytes for `ZoneNames` (a CGNS name of at most 32 bytes and a zone name of at most 4096, each escaped up to 6 bytes per byte, and their framing, per family). It is refused before the text is allocated; `data` gives what, its bytes and the limit. |
| E0130 | `io_cannot_open` | The file or directory cannot be opened or read. |
| E0131 | `io_unknown_format` | The content matches no supported format. |
| E0132 | `io_malformed` | The file violates its format specification (syntax, truncation, inconsistent counts). |
| E0133 | `io_unsupported` | The file is valid but uses a feature the reader does not support yet; the message names it. |
| E0134 | `io_external_reference` | A CGNS file the code reads (a mesh, a `start` solution, a run's solution) has data in another file: a CGNS link, an HDF5 external link, a dataset with external storage or a virtual dataset (`data.node` names the first). Files must be self-contained. |
| E0135 | `io_ambiguous_zone` | An entity belongs to several zones in the source (Gmsh entity in two physical groups, face in two OpenFOAM face zones). |
| E0140 | `partition_invalid_request` | A part count, rank, part index or cell weight is out of range, or an array has the wrong size. |
| E0141 | `partition_input_has_ghosts` | The mesh to partition already has ghost cells. |
| E0142 | `partition_backend_failed` | The partitioner (METIS, Scotch) is not built or reports an error. |
| E0143 | `partition_empty_part` | A part received no cell. |
| E0144 | `halo_inconsistent` | The communicator size differs from the partition count, or two ranks disagree on what they exchange. |
| E0145 | `halo_device_unsupported` | A halo exchange on device memory without CUDA support in the build or in MPI. |
| E0146 | `mpi_unavailable` | A launcher started several ranks of a build without MPI. |
