# Contract: MeshInfo

`MeshInfo` is the JSON summary of a mesh that the C++ core emits after reading it. The SDK consumes it to resolve boundary and cell zones, to check coverage and to estimate memory, without loading the mesh itself. Schema identifier: `autan.mesh-info/1`. The JSON Schema is generated from the SDK (`autan schema mesh-info`).

Producer: the mesh readers in `io/` (owner: CFD - Noyau maillage). Consumer: `py/autan` (owner: CFD - SDK & contrats). Until the core emits it, the SDK reads MeshInfo files passed explicitly (`autan plan --mesh-info FILE`).

## Document

```json
{
  "schema": "autan.mesh-info/1",
  "source": {"path": "crm-hl.cgns", "format": "cgns", "sha256": "9f2c…", "bytes": 28400000000, "logical_bytes": 28400000000,
             "descriptor_bytes": 0, "zone_names_bytes": 0,
             "filtered_bytes": 0, "cgns_nodes": 4032, "point_list_ids": 0},
  "counts": {
    "cells": 48200000, "faces": 151000000, "nodes": 52000000, "source_nodes": 52000000,
    "face_nodes": 512000000,
    "cells_by_type": {"tetra": 964000, "pyramid": 0, "prism": 13014000, "hexa": 0, "polyhedron": 34222000},
    "stencil_entries": 3856000000, "periodic_ghosts": 0
  },
  "bounds": {"min": [-1.2e3, 0.0, -2.1e2], "max": [2.9e3, 1.16e3, 5.6e2]},
  "boundary_zones": [
    {"name": "wing.main", "type": "wall", "native_type": "BCWall", "faces": 812000},
    {"name": "empennage", "type": null, "native_type": null, "faces": 95000}
  ],
  "cell_zones": [{"name": "fluid", "type": null, "native_type": null, "cells": 48200000}],
  "interfaces": [],
  "periodic": [],
  "quality": {
    "max_non_orthogonality_deg": 62.0, "max_faces_per_cell": 214,
    "lines": {"order": "rcm", "scale": 0.0254, "cells": 13014000, "long_cells": 13014000},
    "multigrid": {"cells_by_level": [48200000, 6025000, 753125, 94140, 11767, 1470, 183],
                  "edges_by_level": [150093000, 39162500, 4895312, 611910, 76485, 9555, 1189]}
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | `"autan.mesh-info/1"` | Contract version. Any other value is rejected. |
| `source.path` | string | Mesh file path as given to the reader. |
| `source.format` | `"cgns"` · `"gmsh"` · `"fluent"` · `"polymesh"` | Reader used. |
| `source.sha256` | 64 lowercase hex chars | The **mesh hash** recorded by every run: SHA-256 of the mesh file bytes. For a polyMesh directory, SHA-256 of the concatenation, for each file in the order `points`, `faces`, `owner`, `neighbour`, `boundary`, `cellZones` (when present), of `<name>\n<byte length in decimal>\n<bytes>`. |
| `source.bytes` | integer ≥ 1 | The size of what the mesh hash covers: the file's bytes, or the sum of the polyMesh files'. |
| `source.logical_bytes` | integer ≥ 0 | What the reader reads, at its stored size uncompressed: for CGNS, each HDF5 dataset it reads (the grid coordinates; element sections but their points and edges, of which it reads the header and range; BCs, connectivities, subregions, families) and every descriptor, each its element count times its element size from the metadata, whatever its compression. For every other format, `bytes`. |
| `source.descriptor_bytes` | integer ≥ 0 | The logical bytes of every CGNS `Descriptor_t` at any level, which the CGNS library loads when it opens the file and holds until it closes it (part of `logical_bytes`); 0 for other formats. Opening the file costs at most `descriptor_bytes` + `filtered_bytes` + 12 MiB 64 KiB of HDF5 caches + 1 MiB + 16 KiB × `cgns_nodes` for the tree + 1 MiB of other data read at open ([mesh.md](mesh.md#memory-of-a-cgns-read)). |
| `source.zone_names_bytes` | integer ≥ 0 | The logical bytes of the `ZoneNames` descriptor, which the reader parses (part of `descriptor_bytes`); 0 when absent and for other formats. |
| `source.filtered_bytes` | integer ≥ 0 | Of the filtered (compressed) HDF5 datasets of the file, the largest twice the logical size of one chunk (its extent, which may exceed the dataset's) plus the stored size: what HDF5 holds beside the destination while it reads one ([mesh.md](mesh.md#memory-of-a-cgns-read)); 0 for an unfiltered file and for other formats. |
| `source.cgns_nodes` | integer ≥ 0 | Every node of a CGNS file's tree, for each of which the CGNS library allocates when it opens the file: reading costs at most 1 MiB + 16 KiB per node ([mesh.md](mesh.md#memory-of-a-cgns-read)); at most 16 384 in at most 16 levels, beyond which the reader refuses the file (E0133). 0 for other formats. |
| `source.point_list_ids` | integer ≥ 0 | The entries of every `IndexArray_t` under a CGNS zone's `ZoneBC_t`, `ZoneSubRegion_t` and `ZoneGridConnectivity_t`, whatever their stored width: the reader holds 8 bytes per entry, in storage sized once, beside a 1 MiB chunk, to know which faces they name ([mesh.md](mesh.md#memory-of-a-cgns-read)); 0 for other formats. |
| `counts.cells`, `counts.faces`, `counts.nodes` | integer ≥ 1 | Global counts after reading, before partitioning. `faces` counts each face once, interior and boundary. |
| `counts.source_nodes` | integer ≥ `nodes` | The nodes the mesh file holds, those no face uses included: `nodes` counts only the nodes faces use, which assembly keeps ([mesh.md](mesh.md#building-a-mesh)). |
| `counts.face_nodes` | integer ≥ 3 × `faces` | The nodes of every face, summed over the faces: the length of the face-to-node connectivity, which sizes the mesh in memory ([mesh.md](mesh.md)). |
| `counts.cells_by_type` | object, keys `tetra` `pyramid` `prism` `hexa` `polyhedron`, integers ≥ 0 | Cells by shape. A cell is `polyhedron` unless it is exactly one of the four standard shapes. The values sum to `counts.cells`. |
| `bounds.min`, `bounds.max` | three numbers each | The least and greatest x, y and z over the mesh's nodes, in the file's units: the mesh's extent before the case's `mesh_unit` scales it. |
| `boundary_zones[]` | list, names unique | Every boundary face belongs to exactly one zone. Boundary faces the file leaves outside any group form a zone named `unassigned` with `type: null`; a file that also names a zone `unassigned` is rejected by the reader. |
| `boundary_zones[].name` | non-empty string | Zone name from the file (Fluent zone name, Gmsh physical name, CGNS `BC_t` or family name). |
| `boundary_zones[].type` | normalized type or `null` | See the mapping below. `null` means the file carries no usable type. |
| `boundary_zones[].native_type` | string or `null` | The type exactly as written in the file, for display. |
| `boundary_zones[].faces` | integer ≥ 0 | Face count. A zone the file declares without faces is kept so its name stays visible; it needs no condition. |
| `cell_zones[]` | list, names unique, at least one | Every cell belongs to exactly one zone. A file without cell zones yields one zone named `fluid` with `type: null`; cells outside every zone (Gmsh cells outside any 3D physical group) go to that zone. |
| `cell_zones[].type` | `"fluid"` · `"solid"` · `null` | Medium declared by the file. |
| `cell_zones[].native_type` | string or `null` | As written in the file. |
| `cell_zones[].cells` | integer ≥ 1 | Cell count. The values sum to `counts.cells`. |
| `interfaces[]` | list | One entry per pair of cell zones that share at least one face. |
| `interfaces[].zones` | two distinct cell-zone names, sorted | The pair. Each pair appears once. |
| `interfaces[].faces` | integer ≥ 1 | Shared faces. |
| `periodic[]` | list | One entry per periodic pair of boundary zones; each zone of type `periodic` is in exactly one pair, and no other zone is. |
| `periodic[].zones` | two distinct boundary-zone names | The pair `[a, b]`: the transform maps the faces of `a` onto those of `b`, face for face. |
| `periodic[].transform` | `{"kind": "translation", "vector": [x, y, z]}` | `x_b = x_a + vector`, in mesh units. Other kinds (rotation) are reserved. |
| `periodic[].faces` | integer ≥ 1 | Faces of each zone of the pair. |
| `quality.max_non_orthogonality_deg` | number in [0, 180] | Max over interior faces of the angle between the face normal and the vector joining the two cell centroids; over boundary faces, between the normal and the vector from the cell centroid to the face centroid. |
| `quality.max_faces_per_cell` | integer ≥ 4 | Largest face count of any cell. |
| `counts.stencil_entries` | integer ≥ 0 | Entries of the face2 gradient stencil (`kernels::cell_stencil`) on the local mesh of a single-rank run, periodic images included. |
| `counts.periodic_ghosts` | integer ≥ 0 | Ghost cells a single-rank run creates for its periodic pairs; 0 without them. |
| `quality.lines` | `{order, scale, cells, long_cells}` | The line smoother's lines (`smoother_lines`) in cell order `order` (`"rcm"`, the runtime's default, or `"input"`), on the mesh scaled by `scale` metres per mesh unit (> 0): `cells` the cells in lines of at least 2 cells, `long_cells` in lines of at least 4; `long_cells` ≤ `cells` ≤ `counts.cells`. |
| `quality.multigrid` | `{cells_by_level, edges_by_level}` | The agglomeration multigrid's hierarchy (`solvers::agglomerate`, the run's options) on one rank, in the cell order and at the scale of `quality.lines`: per level, finest first, its nodes (≥ 1, the first `counts.cells`) and edges (≥ 0, the first the interior faces); two lists of the same length ≥ 1. |
| `partition` | object, present only when `autan-mesh-info --ranks N` asks for it | The run's partition on `ranks` ≥ 2 ranks, the same as the runtime forms: `groups` (`"none"`, or `"lines"`: the line smoother's lines kept whole, `--groups lines`), and per rank `cells_by_rank` (owned, summing to `counts.cells`), `ghosts_by_rank`, `faces_by_rank`, `interior_faces_by_rank`, `stencil_entries_by_rank`, `nodes_by_rank` (the nodes of its local mesh, owned and ghost cells'), `face_nodes_by_rank` (the face-node entries of its local mesh) and `sent_cells_by_rank` (the cells it sends each exchange, one per neighbour that holds the cell as a ghost), each a list of `ranks` integers. |

Unknown fields are rejected. Readers fill every field; `partition` is the one field present only on request. The derived counts (`counts.stencil_entries`, `counts.periodic_ghosts`, `quality.lines`, `quality.multigrid`, `partition`) are for one run: `autan plan` refuses a MeshInfo whose lines were computed at another scale than the case's `mesh_unit`, or a run on several ranks without a partition for that rank count and the groups its linear solver keeps whole (`E0210`); the runtime writes the MeshInfo it was given as the run's `mesh-info.json` ([solver-input.md](solver-input.md)). A mesh must be self-contained: the mesh hash covers one file (or one polyMesh directory), so readers reject CGNS links to other files. Reader errors (malformed files, external links, invalid topology or geometry) use the codes `E0101`–`E0199` defined in [mesh.md](mesh.md).

## Normalized boundary types

| Normalized | Fluent `.msh` | CGNS `BCType_t` | polyMesh | Gmsh |
|---|---|---|---|---|
| `wall` | `wall` | `BCWall`, `BCWallViscous` | `wall` | — |
| `thermal-wall` | — | `BCWallViscousHeatFlux`, `BCWallViscousIsothermal` | — | — |
| `slip` | — | `BCWallInviscid` | — | — |
| `symmetry` | `symmetry` | `BCSymmetryPlane` | `symmetry`, `symmetryPlane` | — |
| `farfield` | `pressure-far-field` | `BCFarfield` | — | — |
| `inlet` | `velocity-inlet`, `pressure-inlet`, `mass-flow-inlet` | `BCInflow`, `BCInflowSubsonic`, `BCInflowSupersonic`, `BCTunnelInflow` | — | — |
| `outlet` | `pressure-outlet`, `outflow` | `BCOutflow`, `BCOutflowSubsonic`, `BCOutflowSupersonic`, `BCTunnelOutflow` | — | — |
| `periodic` | — | a zone of a periodic `GridConnectivity_t` pair | — | the groups of a `$Periodic` surface link |
| `null` | every other type | every other type, `FamilySpecified` resolved to the family's type first | every other type (`patch`, `empty`, `cyclic`, …) | every other physical group |

`periodic` marks the two zones of a periodic pair (`periodic[]`), whose faces the core matches ([mesh.md](mesh.md#periodic-pairs)); a periodic zone takes no condition.

`wall` carries no thermal condition: the case resolves it to an adiabatic wall. `thermal-wall` declares one whose value the case must give, so it is never resolved silently.

Fluent `interior` zones are interior faces, not boundary zones. A Fluent cell zone of type `fluid` or `solid` maps to that type; CGNS, polyMesh and Gmsh cell zones map to `null`.

How the SDK turns these types into boundary conditions is part of the case contract: [case-schema.md](case-schema.md#boundary-resolution).
