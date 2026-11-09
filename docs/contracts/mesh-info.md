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
    "stencil_entries": 3856000000, "vertex_stencil_entries": 10411200000,
    "band_entries": 37788800000, "boundary_fit_entries": 2282752000, "periodic_ghosts": 0
  },
  "bounds": {"min": [-1.2e3, 0.0, -2.1e2], "max": [2.9e3, 1.16e3, 5.6e2]},
  "boundary_zones": [
    {"name": "wing.main", "type": "wall", "native_type": "BCWall", "faces": 812000,
     "first_cell_height": {"min": 1.1e-6, "median": 2.4e-6, "max": 8.9e-6}},
    {"name": "empennage", "type": null, "native_type": null, "faces": 95000,
     "first_cell_height": {"min": 2.0e-6, "median": 3.1e-6, "max": 1.2e-5}}
  ],
  "cell_zones": [{"name": "fluid", "type": null, "native_type": null, "cells": 48200000}],
  "interfaces": [],
  "periodic": [],
  "groups": [{"name": "wing.main", "zones": ["wing.main"]}],
  "groups_source": {"path": "crm-hl.cgns.zones.json", "sha256": "4b1a…", "provenance": {"url": "https://…"}},
  "quality": {
    "max_non_orthogonality_deg": 62.0, "max_faces_per_cell": 214,
    "lines": {"order": "rcm", "scale": 0.0254, "cells": 13014000, "long_cells": 13014000},
    "multigrid": {"order": "rcm",
                  "cells_by_level": [48200000, 6025000, 753125, 94140, 11767, 1470, 183],
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
| `source.logical_bytes` | integer ≥ 0 | What the reader reads, at its stored size uncompressed: for CGNS, each HDF5 dataset it reads (the grid coordinates; element sections but their points and edges, of which it reads the header and range; the whole subtree of each BC, connectivity, subregion and family, data arrays it skips included, an upper bound on what it reads of them) and every descriptor, each its element count times its element size from the metadata, whatever its compression. For every other format, `bytes`. |
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
| `boundary_zones[].first_cell_height` | `{min, median, max}` or `null` | Over the zone's faces, the distance along each face's normal from the centroid of the cell that owns it to the face's centroid, \|(x_cell − x_face) · n̂\|, in the file's units (as `bounds`): the wall distance of a first cell's unknown. Numbers with 0 ≤ min ≤ median ≤ max (0 when a concave cell's centroid lies in the face's plane); the median of an even count is the mean of the two middle values, so one face gives three equal numbers and two faces a median halfway between them. `null` exactly for a zone without faces. |
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
| `groups[]` | list, sorted by `name` | The zone groups of the mesh's sidecar ([Zone groups](#zone-groups)); `[]` without one. |
| `groups[].name` | string matching `[A-Za-z0-9_.][A-Za-z0-9_.-]*` | Group name, unique. |
| `groups[].zones` | non-empty list of `boundary_zones` names | The group's zones, in `boundary_zones` order; groups may share zones. |
| `groups_source` | `null` or `{path, sha256, provenance}` | The sidecar read: its path, next to the mesh file with symlinks followed, the SHA-256 of its bytes, its `provenance` object. |
| `quality.max_non_orthogonality_deg` | number in [0, 180] | Max over interior faces of the angle between the face normal and the vector joining the two cell centroids; over boundary faces, between the normal and the vector from the cell centroid to the face centroid. |
| `quality.max_faces_per_cell` | integer ≥ 4 | Largest face count of any cell. |
| `counts.stencil_entries` | integer ≥ 0 | Entries of the face2 gradient stencil (`kernels::cell_stencil`) on the local mesh of a single-rank run, periodic images included. |
| `counts.vertex_stencil_entries` | integer ≥ 0 | Entries of the vertex cell stencil (`kernels::cell_stencil`, `StencilKind::vertex`) on the same mesh as `stencil_entries`: per cell, the other cells that share a node with it and the boundary faces that have one of its nodes. |
| `counts.band_entries` | integer ≥ 0 | On the same mesh, summed over its cells: the cells within two vertex rings of the cell (it excluded), and the boundary faces in the vertex stencils of the cell and of the cells within one ring. That is the stencil the quadratic gradient of a viscous run gathers when every boundary face gives a value; the run's own holds at most this many ([euler.md](../design/euler.md#quadratic-gradient)). |
| `counts.boundary_fit_entries` | integer ≥ 0 | The same with three rings (the boundary faces from the cells within two), summed over the cells that own a boundary face and their face neighbours. That is what the viscous boundary fit gathers when every boundary face is a wall; the run's own holds at most this many, whichever zones the case makes walls. |
| `counts.periodic_ghosts` | integer ≥ 0 | Ghost cells a single-rank run creates for its periodic pairs; 0 without them. |
| `quality.lines` | `{order, scale, cells, long_cells}` | The line smoother's lines (`smoother_lines`) in cell order `order` (`"rcm"`, the runtime's default, or `"input"`), on the mesh scaled by `scale` metres per mesh unit (> 0): `cells` the cells in lines of at least 2 cells, `long_cells` in lines of at least 4; `long_cells` ≤ `cells` ≤ `counts.cells`. |
| `quality.multigrid` | `{order, cells_by_level, edges_by_level}` | The agglomeration multigrid's hierarchy (`solvers::agglomerate`, the run's options) at the scale of `quality.lines`, built in cell order `order` (`"rcm"` or `"input"`; `autan-mesh-info --multigrid-order`, default its `--cell-order`). A run on one rank builds it in its cell order; an inviscid run on a partition (several ranks, or reproducibility `"ranks"`) on the mesh in RCM order, `"rcm"`, whatever order it computes in, so that it builds the one-rank RCM run's hierarchy. Per level, finest first, its nodes (≥ 1, the first `counts.cells`) and edges (≥ 0, the first the interior faces); two lists of the same length ≥ 1. |
| `partition` | object | The run's partition on `ranks` ≥ 1 ranks, the same as the runtime forms: on one rank (`autan-mesh-info` without `--ranks`, or `--ranks 1`), the one part a run extracts with periodic pairs or reproducibility `"ranks"` (the mesh itself without periodic pairs: no ghost, no cell sent; with them, the image ghosts its own halo sends), `groups` `"none"`; with `--ranks N`, N parts: `groups` (`"none"`; `"lines"`, the line smoother's lines kept whole, `--groups lines`, as a steady CPU run with the `lines` solver forms them; or `"multigrid"`, the lines of the multigrid's finest level kept whole (level 0 of `solvers::agglomerate` with the run's options on the mesh in RCM order, its lines of at least two cells, in the cells as read), `--groups multigrid`, as a steady CPU run with the multigrid on a partition forms them; a partition with other groups than the run's is refused, `E0212` by `autan plan`, `E0705` by `autan-run`); on any rank count, `ghost_stencil`, the ghosts each part holds (`"face2"`, or `"vertex"`, `"vertex2"`, `"vertex3"`, `"vertex4"` with `--ghosts`: `GhostStencil`, [partitioning.md](../design/partitioning.md#a-partition)); and per rank `cells_by_rank` (owned, summing to `counts.cells`), `ghosts_by_rank`, `faces_by_rank`, `interior_faces_by_rank`, `stencil_entries_by_rank`, `vertex_stencil_entries_by_rank`, `band_entries_by_rank`, `boundary_fit_entries_by_rank` (the `counts` of the same names on its local mesh), `nodes_by_rank` (the nodes of its local mesh, owned and ghost cells'), `face_nodes_by_rank` (the face-node entries of its local mesh) and `sent_cells_by_rank` (the cells it sends each exchange, one per neighbour that holds the cell as a ghost), each a list of `ranks` integers. |

Unknown fields are rejected. A MeshInfo written before `partition.ghost_stencil` existed, with a `partition` but no `ghost_stencil`, is refused (`E0202` from the SDK, `E0705` from `autan-run` for a solver input that embeds it): regenerate it with `autan-mesh-info`. Readers fill every field; `autan-mesh-info` always fills `partition`; a hand-written document may omit it. The derived counts (`counts.stencil_entries`, the three ring counts, `counts.periodic_ghosts`, `quality.lines`, `quality.multigrid`, `partition`) are for one run: `autan plan` refuses a MeshInfo whose lines were computed at another scale than the case's `mesh_unit`, a run on several ranks without a partition for that rank count, the groups its linear solver keeps whole and the ghost stencil its physics reads (`"face2"` for the Euler physics, `"vertex4"` for a compressible Navier–Stokes case, laminar or Spalart–Allmaras), or a run on one rank with periodic pairs or reproducibility `"ranks"` without such a one-part partition, or a steady CPU multigrid run that builds its hierarchy in another cell order than `quality.multigrid.order` (`E0212`); the runtime writes the MeshInfo it was given as the run's `mesh-info.json` ([solver-input.md](solver-input.md)). A mesh must be self-contained: the mesh hash covers one file (or one polyMesh directory), so readers reject CGNS links to other files. Reader errors (malformed files, external links, invalid topology or geometry) use the codes `E0101`–`E0199` defined in [mesh.md](mesh.md).

## Zone groups

A mesh whose file does not name its parts (the HLPW-5 CRM-HL grid has 668 zones `TRI_bdy N`) can carry a sidecar next to it, `<mesh file>.zones.json`, or `<directory>.zones.json` for a polyMesh directory (symlinks followed: next to the file itself), which `autan-mesh-info` reads into `groups` (a CGNS family is already one zone, [mesh.md](mesh.md#readers)). It is never inferred from the mesh and never modifies it:

```json
{"schema": "autan.zone-groups/1", "mesh_sha256": "9f2c…", "provenance": {"url": "https://…", "note": "from the mapbc"}, "groups": {"wing.main": ["TRI_bdy 12", "TRI_bdy 13"]}}
```

The four keys are required and no other is allowed; `provenance` maps keys to strings, shown as given. `E0136`: the file is larger than 1 MiB (checked before it is read: a hundred times a sidecar listing the 668 zones of the CRM-HL grid once), cannot be read, is not JSON, or breaks this schema (a key missing, unknown or given twice, a wrong type). `E0137`: `mesh_sha256` is not the mesh hash. `E0138`: a zone that is not a boundary zone name (exact, spaces included), a zone twice in one group, an empty group, or a name outside `[A-Za-z0-9_.][A-Za-z0-9_.-]*`. `E0139`: a group name given twice. The run records the sidecar's hash (solver-input `mesh.groups_sha256`) and archives the groups with its MeshInfo.

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
