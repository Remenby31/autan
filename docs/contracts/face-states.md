# Contract: face states

A face-state file gives the flow state on every face of some boundary zones of one mesh, from an oracle the solver cannot express in closed form (for example Ringleb's flow). A case imposes it with the `oracle-state` boundary condition ([case-schema.md](case-schema.md#boundaries)). Schema identifier: `autan.face-states/1`.

Producer: `autan-vv` (owner: CFD - Euler for the exporters). Consumers: `autan plan` (checks) and the solver runtime (pairing and values).

## Document

```json
{
  "schema": "autan.face-states/1",
  "oracle": "ringleb",
  "mesh_sha256": "9f2c…",
  "variables": ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"],
  "zones": {
    "inflow": {
      "faces": 2,
      "centroids": [[0.1, 0.9, 0.0], [0.2, 0.9, 0.0]],
      "values": [[1.21, 30.0, 4.0, 0.0, 98000.0], [1.20, 31.0, 4.2, 0.0, 97800.0]]
    }
  }
}
```

| Key | Content |
|---|---|
| `schema` | `"autan.face-states/1"` |
| `oracle` | Name of the oracle, for display. |
| `mesh_sha256` | The mesh hash ([mesh-info.md](mesh-info.md), `source.sha256`) of the mesh the states were exported for. |
| `variables` | Exactly `["density", "velocity-x", "velocity-y", "velocity-z", "pressure"]`, in SI (kg/m³, m/s, Pa), velocity in mesh axes. |
| `zones` | One entry per boundary zone given: `faces`, the zone's face count; `centroids`, one `[x, y, z]` per face (m); `values`, one row per face in the order of `variables`. Rows of `centroids` and `values` correspond. |

A face is identified by its centroid, not by its rank, so the file does not depend on the solver's internal face order. The centroid of a face is its area centroid: the area-weighted mean of the centroids of the triangles of a fan from its first node, the definition the core uses. The solver pairs every boundary face of the zone with exactly one row, within 1e-9 times the largest extent of the zone's bounding box (`E0421` otherwise).

`autan plan` checks the file named by the case: present, with the case's SHA-256, of this schema and these variables (`E0420`), exported for the case's mesh (`E0422`), giving every zone the condition applies to with the mesh's face count (`E0423`). Each run archives the file as `states/<sha256>.json` in its directory ([results.md](results.md#run-directory)).
