# Contract: initial state

An initial-state file gives the flow state of every fluid cell of one mesh at the start of a run, for a field no uniform state can express (for example the isentropic vortex). A case starts from it with `initial_state` ([case-schema.md](case-schema.md#initial-state-file)). Schema identifier: `autan.initial-state/1`.

Producer: `autan.initial_state.write` in the SDK, called by the `autan-vv` exporters (owner: CFD - Euler). Consumers: `autan plan` (checks) and the solver runtime (pairing and values).

## Document

```json
{
  "schema": "autan.initial-state/1",
  "oracle": "isentropic-vortex",
  "mesh_sha256": "9f2c…",
  "variables": ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"],
  "cells": 2,
  "centroids": [[0.05, 0.05, 0.05], [0.15, 0.05, 0.05]],
  "values": [[1.21, 30.0, 4.0, 0.0, 98000.0], [1.20, 31.0, 4.2, 0.0, 97800.0]]
}
```

| Key | Content |
|---|---|
| `schema` | `"autan.initial-state/1"` |
| `oracle` | Name of the field, for display. |
| `mesh_sha256` | The mesh hash ([mesh-info.md](mesh-info.md), `source.sha256`) of the mesh the state was written for. |
| `variables` | Exactly `["density", "velocity-x", "velocity-y", "velocity-z", "pressure"]`, in SI (kg/m³, m/s, Pa), velocity in mesh axes. |
| `cells` | The number of rows: every fluid cell of the mesh, an integer ≥ 1. |
| `centroids` | One `[x, y, z]` per row (m): the centroid of the row's cell. |
| `values` | One row per cell in the order of `variables`, density and pressure > 0. Rows of `centroids` and `values` correspond; their order is free. |

A cell is identified by its centroid, not by its rank, so the file does not depend on the solver's cell order or partitioning. The centroid of a cell is `cell_centroid` of [mesh.md](mesh.md#geometry). The values are the cell's state as given, with no projection: a value at the centroid stands for the cell average, which a second-order scheme sees as an O(h²) difference. Solid cells have no row and keep their zone's state.

The solver pairs, on the whole mesh and whatever the partitioning, every fluid cell with exactly one row and every row with exactly one fluid cell, within 1e-9 times the largest extent of the mesh's bounding box. Two cells within that tolerance of each other make the pairing ambiguous (`E0511` otherwise, `data` naming the cells).

`autan plan` reads the file named by the case once, and checks those bytes: a regular file with the case's SHA-256, then a valid document (`E0508`): this schema, these variables, `cells` an integer, one row of 3 finite coordinates and one of 5 finite values per cell, density and pressure > 0. It also checks the file was written for the case's mesh (`E0509`) and has one row per fluid cell (`E0510`). The runtime makes the same checks before any directory exists, then pairs the cells. Each run archives the file as `states/<sha256>.json` in its directory ([results.md](results.md#run-directory)).

The document is JSON, about 200 bytes per cell, read whole: right for verification meshes, too large for meshes of millions of cells, which will need a binary variant.

## Writing a file

`autan.initial_state.write(path, *, oracle, mesh_sha256, centroids, values)` validates and writes a document, and returns its SHA-256 for the case:

```python
digest = autan.initial_state.write(
    Path("vortex.json"), oracle="isentropic-vortex", mesh_sha256=mesh.source.sha256,
    centroids=centroids, values=values,
)
case.initial_state = autan.StateFile(path="vortex.json", sha256=digest)
```
