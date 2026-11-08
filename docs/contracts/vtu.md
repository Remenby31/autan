# Contract: VTU export

`autan-vtu SOLUTION OUT` converts a run's [solution.cgns](results.md#solutioncgns) into one VTK XML UnstructuredGrid file for ParaView, VisIt or VTK. Producer: `autan::io::write_vtu` (owner: CFD - Noyau maillage). The run itself writes only `solution.cgns`, which stays the authority; the VTU is derived from it and carries nothing it does not.

## Usage

```
autan-vtu runs/<run_id>/solution.cgns solution.vtu
```

Exit status 0 on success, 1 on a coded error (printed as `Exxxx: message`), 2 on a usage error. The output is written under a temporary name and renamed, so it is whole or absent.

## File

VTK XML, `<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">`, one `Piece`. Every `DataArray` has `format="appended"`, and they all live in one `<AppendedData encoding="raw">` block after its `_`. Each array there is a UInt64 byte count followed by that many bytes, at the array's `offset` from the `_`. VTK 9 and ParaView 5.11 or later read it natively; the legacy `.vtk` format is not written (it has no 64-bit polyhedron layout).

| Element | Arrays |
|---|---|
| `Points` | `Points`, Float64, 3 components: `GridCoordinates` of the solution (m), in node order |
| `Cells` | `connectivity`, `offsets`, `faces`, `faceoffsets` (Int64), `types` (UInt8) |
| `CellData` | each `FlowSolution` field under its SIDS name; each `FlowSolutionMean` field as `<name>Mean`; `CellZone` |

**Cells.** Cell i of the solution (0-based) is VTK cell i, always a `VTK_POLYHEDRON` (type 42), whatever its shape. `connectivity` lists each cell's distinct nodes; `offsets` holds the end of each cell's list. `faces` holds, for each cell, its face count, then for each face its node count and its nodes, ordered so that the face normal (right-hand rule) points out of the cell; `faceoffsets` holds the end of each cell's entry in `faces`. Tetrahedra, hexahedra, prisms and pyramids are written as polyhedra too: that is the representation `solution.cgns` has, and it needs no node reordering, but ParaView and VTK handle polyhedra more slowly and with more memory than native cells (native types: #568).

**Cell data.** Float64, one value per cell, the same values as `solution.cgns`. `CellZone` (Int32) is the 0-based index of the cell's zone in the run's `mesh-info.json` `cell_zones` list. There is no point data: the unknowns are cell values, and interpolating them to nodes would add values the run did not compute. `RestartState` and `WallSolution` are not written.

**Ranks.** One file per run: `solution.cgns` holds the whole mesh in one zone whatever the number of ranks, so there is no `.pvtu`.

## Errors

A missing or unreadable input or an unwritable output is `E0130`. A CGNS file without a `FlowSolution` (a mesh, not a run's solution) is `E0147`. A file outside results.md's layout gets the readers' codes (`E0131` to `E0134`, [mesh.md](mesh.md#readers)).
