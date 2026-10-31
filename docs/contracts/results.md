# Contract: results

A run writes one directory. Its layout, the run metadata and the CGNS solution layout are the contract between the solver (and the fake backend), the results API (`cfd.Results`), the UI and V&V. Schema identifier: `cfd.results/1`.

## Run directory

```
<run_id>/
  run.json         run metadata, written at start and rewritten at every status change
  case.toml        the case exactly as run (case-schema.md)
  mesh-info.json   the MeshInfo of the mesh (mesh-info.md)
  history.jsonl    one line per recorded iteration, appended during the run
  solution.cgns    final solution (CGNS/HDF5), written when the run ends
```

All five files are required once `status` is not `running`. While running, `solution.cgns` may be absent. Readers reject a directory whose `run.json` has another `schema` value.

## run.json

```json
{
  "schema": "cfd.results/1",
  "run_id": "run-7f3a9c21e0b4",
  "status": "converged",
  "case": {"name": "crm-hl", "sha256": "…"},
  "mesh": {"path": "crm-hl.cgns", "sha256": "…"},
  "solver": {"version": "0.1.0", "commit": "d769576…", "backend": "fake"},
  "precision": "fp32+acc64",
  "deterministic": true,
  "devices": {"kind": "cpu", "count": 1},
  "started": "2026-10-31T21:40:12Z",
  "finished": "2026-10-31T21:54:40Z",
  "iterations": 3000,
  "diagnostics": []
}
```

| Field | Meaning |
|---|---|
| `run_id` | `run-` followed by 12 lowercase hex characters: the first 12 of SHA-256 over `case.sha256`, `mesh.sha256`, `solver.commit`, `precision` and `started`, joined by `\n`. |
| `status` | `running` · `converged` (stopping criterion met) · `completed` (iteration budget reached) · `stopped` (stopped by user code or signal) · `diverged` · `failed`. |
| `case.sha256` | Case hash as defined in [case-schema.md](case-schema.md#case-hash). |
| `mesh.sha256` | `source.sha256` of the MeshInfo: hash of the mesh file bytes. |
| `solver.version`, `solver.commit` | Version and full 40-character git commit of the solver build. |
| `solver.backend` | `cuda` · `hip` · `cpu` · `fake`. Results of the `fake` backend are synthetic and must never be used as reference data. |
| `precision` | Precision as run, from the case. |
| `deterministic` | Whether the run used the bitwise-deterministic mode. |
| `devices` | `kind` (`cuda` · `hip` · `cpu`) and `count`. |
| `started`, `finished` | UTC, ISO 8601, second precision, `Z` suffix. `finished` is `null` while running. |
| `iterations` | Last completed iteration (pseudo-time iterations for steady runs, time steps for unsteady ones). |
| `diagnostics` | Diagnostics raised during the run, in the format of [docs/sdk/errors.md](../sdk/errors.md); a `diverged` run carries its `D` code here. |

Mesh hash, case hash, solver commit and precision identify a run; two runs with equal identities and `deterministic: true` produce bitwise-identical `history.jsonl` and `solution.cgns` data arrays.

## History

`history.jsonl` holds one JSON object per line, in increasing `iteration`:

```json
{"iteration": 10, "time": null, "residuals": {"density": 1.2e-3, "momentum-x": 4.1e-4, "momentum-y": 2.0e-4, "momentum-z": 3.3e-4, "energy": 2.9e-3, "nu-tilde": 7.7e-5}, "monitors": {"forces": {"fx": 1.2e4, "fy": 0.0, "fz": 9.8e4, "mx": 0.0, "my": -3.1e4, "mz": 0.0, "cl": 2.04, "cd": 0.31, "cs": 0.0, "cmx": 0.0, "cmy": -0.4, "cmz": 0.0}}}
```

- `time`: physical time in seconds for unsteady runs, `null` for steady runs.
- `residuals`: for each solved equation, the root mean square over cells of the cell residual (volume-integrated, not normalized), in FP64. Names: `density`, `momentum-x`, `momentum-y`, `momentum-z`, `energy` (compressible) or `pressure`, `momentum-x`, `momentum-y`, `momentum-z` (incompressible), then `nu-tilde` (SA, DDES on SA) or `k`, `omega` (SST, DDES on SST). Every line carries every equation.
- `monitors`: one entry per monitor recorded at this iteration (iteration divisible by the monitor's `every`), keyed by monitor name. A line may carry no monitor.
- Forces monitor: `fx` `fy` `fz` in N and `mx` `my` `mz` in N·m about `moment_center`, mesh axes, pressure plus viscous contributions. With `ref_area` set: `cd`, `cs`, `cl` = force projected on the drag direction e_D = (cos α cos β, −sin β, sin α cos β), the side direction e_S = e_L × e_D and the lift direction e_L = (−sin α, 0, cos α), divided by q∞ `ref_area` with q∞ = ½ ρ∞ V∞²; and, with `freestream.ref_length` also set, `cmx` `cmy` `cmz` = moments divided by q∞ `ref_area` `ref_length`.

## solution.cgns

CGNS/HDF5, written against the CGNS SIDS 4.x and readable by the CGNS library (`cgnscheck` reports no error). All values in SI.

```
/CGNSLibraryVersion                      4.4
/Base                    CGNSBase_t      CellDimension 3, PhysicalDimension 3
  DataClass                              Dimensional
  DimensionalUnits                       Kilogram Meter Second Kelvin Radian
  RunMetadata            Descriptor_t    run.json identity fields as JSON (run_id, case, mesh, solver, precision)
  <family>               Family_t        one per boundary zone and per cell zone, named after it
    FamilyBC             FamilyBC_t      boundary zones only: BCType of the resolved condition
  Zone                   Zone_t          Unstructured; [nodes, cells, 0]
    GridCoordinates                      CoordinateX, CoordinateY, CoordinateZ (R8)
    NGonElements         Elements_t      NGON_n, ElementRange [1, F], ElementStartOffset, ElementConnectivity (I8)
    NFaceElements        Elements_t      NFACE_n, ElementRange [F+1, F+C], ElementStartOffset, ElementConnectivity (I8, signed face ids)
    ZoneBC               ZoneBC_t
      <boundary zone>    BC_t            BCType, GridLocation FaceCenter, PointList (NGON ids), FamilyName
    <cell zone>          ZoneSubRegion_t GridLocation CellCenter, PointList (NFACE ids), FamilyName
    FlowSolution         FlowSolution_t  GridLocation CellCenter, one R8 DataArray per field
```

There is one `Zone_t` holding the whole mesh, every cell as a polyhedron (`NFACE_n`), every face as a polygon (`NGON_n`). In `NFaceElements`, a positive face id means the face normal points out of the cell. Cell zones are `ZoneSubRegion_t`; boundary zones are `BC_t`. Interfaces between cell zones are interior faces and need no node.

BCType of each resolved condition: `wall` → `BCWall` (adiabatic) or `BCWallViscousIsothermal`; `slip` → `BCWallInviscid`; `symmetry` → `BCSymmetryPlane`; `farfield` → `BCFarfield`; `inlet` and `velocity-inlet` → `BCInflowSubsonic`; `outlet` → `BCOutflowSubsonic`.

`FlowSolution` fields (SIDS names), values in solid zones are those of the solid:

| Field | Unit | Written for |
|---|---|---|
| `Density` | kg/m³ | fluid cells; solid cells carry the solid density |
| `VelocityX`, `VelocityY`, `VelocityZ` | m/s | 0 in solid cells |
| `Pressure` | Pa | 0 in solid cells |
| `Temperature` | K | all cells |
| `TurbulentSANuTilde` | m²/s | SA and DDES on SA |
| `TurbulentEnergyKinetic`, `TurbulentDissipationRate` | m²/s², 1/s | SST and DDES on SST |

## Results API

`cfd.Results.open(path)` reads a run directory and checks it against this contract. `results.run` returns the run metadata, `results.case` the case, `results.mesh_info` the MeshInfo, `results.history` the residual and monitor time series, `results.forces` the forces monitor when there is exactly one, `results.field(name)` a cell-centred field as a NumPy array.
