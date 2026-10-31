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
| `cell_global_id` | n_cells | `int64` index of the cell in the unpartitioned mesh; the identity when the mesh was not partitioned. |
| `boundary_zone_offsets` | n_face_zones + 1 | Boundary faces of face zone `z` are `[off[z], off[z+1])`; `off[0] = n_interior_faces`, `off[n_face_zones] = n_faces`. Interior zones have an empty range. |
| `cell_face_offsets`, `cell_faces` | n_cells + 1, 2·n_interior + n_boundary | CSR of the faces of each cell, in increasing face index. A face is outward for cell `c` iff `owner[f] == c`. |

## Invariants

`autan::assemble(MeshInput)` is the only way to build a `Mesh`; it establishes and checks these, and throws `autan::Error` on the first violation:

1. Faces `[0, n_interior_faces)` are interior, faces `[n_interior_faces, n_faces)` are boundary.
2. Interior faces have `owner < neighbour` and are sorted by `(owner, neighbour)`. Two cells may share several faces.
3. Boundary faces are grouped by ascending zone index and sorted by owner within a zone.
4. Each face has ≥ 3 distinct nodes; each owned cell has ≥ 4 faces; each ghost cell has ≥ 1 face.
5. Every owned cell surface is closed and consistently oriented: walking each face with its normal pointing out of the cell, every edge is traversed exactly twice, once in each direction. Ghost cells carry only the faces they share with owned cells.
6. Every face touches an owned cell, so `owner[f] < n_owned_cells` for every face: a face loop over all faces visits each owned cell's faces, and its contributions to ghost cells (`neighbour[f] ≥ n_owned_cells`) are partial sums that a reverse halo exchange adds to the owning partition, as an adjoint (transposed) kernel needs.
7. A face zone's `boundary` flag says whether it holds boundary or interior faces, and every face matches the role of its zone (a declared zone may be empty). Face-zone names are unique; cell-zone names are unique. Global ids are distinct and ≥ 0.
8. Every point is referenced by a face (unreferenced points are removed).
9. Cell order is the order of the input; `assemble` never renumbers cells. Sorting is stable, so the same input always yields the same arrays bit for bit. `assemble(input, &map)` returns, in a `autan::AssemblyMap`, the input face of each face, whether it was reversed, and the input point of each point.

## Zones

Every face belongs to exactly one face zone and every cell to exactly one cell zone.

| Field | Face zone | Cell zone |
|---|---|---|
| `name` | Name in the source (Fluent zone name, Gmsh physical name, OpenFOAM patch). | Same. |
| `bc_type` / `type` | Type exactly as written in the source; empty if the format has none. | Same. |
| `kind` | `BcKind`: `unspecified`, `wall`, `thermal_wall`, `slip`, `symmetry`, `far_field`, `inlet`, `outlet`, normalised with the table of [mesh-info.md](mesh-info.md#normalized-boundary-types). | `CellKind`: `unspecified`, `fluid`, `solid`. |
| `boundary` | Role: boundary (true) or interior (false) faces. | — |
| `provenance.source_id` | Identifier in the source file, or −1 for a zone the reader synthesised. | Same. |
| `provenance.detail` | Human-readable origin. | Same. |

Synthesised zones: `interior` for interior faces with no zone in the source, `unassigned` for boundary faces with no zone in the source (so `autan plan` reports the missing condition), `fluid` (kind `unspecified`) for cells with no zone in the source. A source zone with one of these names collides and is rejected with E0110.

`Mesh::source()` records the format (`gmsh`, `fluent`, `polymesh`, `cgns`, or `memory` for meshes built in code) and the path as given.

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

## Building a mesh

Readers of face-based formats (Fluent, OpenFOAM) and the partitioner fill a `autan::MeshInput`: faces in any order, each with `face_cell0` (normal points away from it) and `face_cell1` (`no_cell` on the boundary); ghost cells last with `n_ghost_cells`, and `cell_global_id` when partitioned. Readers of element-based formats (Gmsh, CGNS) fill a `autan::CellwiseMesh` with `add_cell` (tetra, pyramid, prism, hexa in Gmsh node order), `add_polyhedron` (outward faces) and `add_labelled_face` (boundary or internal surface elements carrying a zone), then call `autan::to_face_based`, which merges coincident faces by node set. A zone that labels faces takes their role (interior or boundary); a zone that labels none keeps its declared role.

## Readers

`autan::io::read_mesh(path)` (library `autan::io`, header `io/include/autan/io/read_mesh.hpp`) detects the format from the content and returns an assembled mesh; `read_gmsh`, `read_fluent` and `read_polymesh` skip the detection.

| Format | Supported | Zones | Rejected (E0133 unless noted) |
|---|---|---|---|
| Gmsh MSH 4.1, ASCII and binary | Linear tetra, pyramid, prism, hexa; triangles and quadrilaterals label faces; sparse node tags; parametric nodes; unknown sections are skipped as the format prescribes. | Physical volumes → cell zones, physical surfaces → face zones (also when they have no element), named from `$PhysicalNames` wherever it appears, else the tag; type empty, kind `unspecified`. | MSH 2 and other versions, higher-order elements, `$PartitionedEntities`, `$Periodic`, `$GhostElements`, big-endian files; non-positive physical tags (E0132); an entity in two physical groups (E0135). |
| Fluent mesh, ASCII and binary (2010/3010, 2012, 2013) | Sections 0, 1, 2, 10, 12, 13, 39, 45; 4, 37, 38, 40 are skipped. Mixed and polygonal faces, polyhedral cells (type 7). The face normal points toward c0. | Zone sections 39/45 give names and types (they take precedence over face-section codes); a zone without one is named `zone-<id>` and typed from its code; a declared zone without faces is kept. A zone's role is that of its faces. Declared element types must match the cell's face count (E0132). | 2D meshes, periodic shadow faces (18), hanging-node trees and interface parents (58, 59, 61–63), any other section. |
| OpenFOAM polyMesh, ASCII and binary, 32/64-bit labels and scalars | The `polyMesh` directory or a case directory holding `constant/polyMesh`; comments anywhere; sized, unsized and uniform label lists; ASCII faces in nested form (`faceList` or `faceCompactList`), binary faces as `faceCompactList`. | Patches → boundary zones (type as written), `cellZones` → cell zones (type empty). `faceZones`, `pointZones` and `sets` are selections, not zones, and are not read. | Compressed files, binary `faceList`, processor patches; a cell in two cell zones (E0135). |

The test meshes and how they are generated: [tests/data/mesh/README.md](../../tests/data/mesh/README.md).

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
| E0116 | `mesh_face_without_owned_cell` | A face touches only ghost cells. |
| E0117 | `mesh_duplicate_global_id` | Two cells share a global id, or one is negative. |
| E0120 | `geometry_degenerate_face` | A face has zero or non-finite area. |
| E0121 | `geometry_non_positive_volume` | An owned cell has zero, negative or non-finite volume. |
| E0130 | `io_cannot_open` | The file or directory cannot be opened or read. |
| E0131 | `io_unknown_format` | The content matches no supported format. |
| E0132 | `io_malformed` | The file violates its format specification (syntax, truncation, inconsistent counts). |
| E0133 | `io_unsupported` | The file is valid but uses a feature the reader does not support yet; the message names it. |
| E0134 | `io_external_reference` | The mesh refers to another file (CGNS link); meshes must be self-contained. |
| E0135 | `io_ambiguous_zone` | An entity belongs to several zones in the source (Gmsh entity in two physical groups, face in two OpenFOAM face zones). |
| E0140 | `partition_invalid_request` | A part count, rank, part index or cell weight is out of range, or an array has the wrong size. |
| E0141 | `partition_input_has_ghosts` | The mesh to partition already has ghost cells. |
| E0142 | `partition_backend_failed` | The partitioner (METIS, Scotch) is not built or reports an error. |
| E0143 | `partition_empty_part` | A part received no cell. |
| E0144 | `halo_inconsistent` | The communicator size differs from the partition count, or two ranks disagree on what they exchange. |
| E0145 | `halo_device_unsupported` | A halo exchange on device memory without CUDA support in the build or in MPI. |
