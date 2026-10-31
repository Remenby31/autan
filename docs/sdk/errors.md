# Diagnostic codes

Generated from `py/autan/errors.py` by `autan explain --all`; do not edit by hand.

`E` codes are errors and block the run, `W` codes are warnings, `D` codes report a divergence detected during a run. Codes `E0101` to `E0199` belong to the mesh readers ([mesh.md](../contracts/mesh.md)).

## E0001

**The case file cannot be read** (error)

The file does not exist, is not valid Python or TOML, or raised an exception while running. Python case files run as scripts; the traceback points at the failing line.

Fix: Fix the file at the reported location.

## E0002

**The case file does not define exactly one case** (error)

A Python case file must create exactly one `autan.Case`, at module level.

Fix: Create one case: `case = autan.Case("name", mesh="mesh.cgns")`.

## E0003

**The case document does not match the schema** (error)

A field is missing, has the wrong type, or is not part of `autan.case/1`. The location gives the path of the field in the document. See docs/contracts/case-schema.md.

Fix: Correct the field named in the message.

## E0004

**Unsupported case schema version** (error)

The document's `schema` key is not `autan.case/1`.

Fix: Set `schema = "autan.case/1"`.

## E0101

**Inconsistent mesh array sizes** (error)

Input arrays disagree on a size, or CSR offsets are not monotone.

Fix: Regenerate the mesh with the meshing tool; the file or partition is corrupt.

## E0102

**Mesh index out of range** (error)

A node, cell or zone index is out of range.

Fix: Regenerate the mesh; an index points past the end of its array.

## E0103

**Face with fewer than 3 nodes** (error)

A face has fewer than 3 nodes.

Fix: Remove or repair the degenerate face in the meshing tool.

## E0104

**Face with a repeated node** (error)

A face lists a node twice.

Fix: Repair the face in the meshing tool so each node appears once.

## E0105

**Face with the same cell on both sides** (error)

A face has the same cell on both sides.

Fix: Repair the mesh connectivity in the meshing tool.

## E0106

**Cell with too few faces** (error)

An owned cell has fewer than 4 faces, or a ghost cell none.

Fix: Repair the cell in the meshing tool, or regenerate the partition.

## E0107

**Cell not closed** (error)

Invariant 5 fails for a cell.

Fix: Repair the cell in the meshing tool: its faces must enclose a volume.

## D0107

**Divergence** (divergence)

The solution diverged: a residual or a state became non-finite or non-physical. `data` gives the iteration, the cell (0-based) and its cell zone.

Fix: Lower the CFL number or use a more dissipative limiter; check the mesh quality near the cell.

## E0108

**Face shared by more than two cells** (error)

More than two cells list the same face.

Fix: Repair non-manifold faces in the meshing tool.

## E0109

**Face does not match its zone's role** (error)

A face does not match its zone's role, or labels put one zone on both interior and boundary faces.

Fix: Put interior and boundary faces in separate zones in the meshing tool.

## E0110

**Duplicate zone name** (error)

Two face zones or two cell zones share a name.

Fix: Rename one of the zones in the meshing tool; `interior`, `unassigned` and `fluid` are reserved.

## E0111

**Labelled face not found** (error)

A labelled face matches no cell face.

Fix: Regenerate the mesh; a zone references a face that does not exist.

## E0112

**Empty mesh** (error)

No cell, face or point.

Fix: Check that the file is the intended mesh and was exported completely.

## E0113

**Mesh too large** (error)

A count exceeds the `index_t` range.

Fix: Partition the mesh, or use a build with 64-bit indices.

## E0114

**Inverted cell** (error)

Two cells list their common face with the same orientation (one is inverted).

Fix: Repair the inverted cell in the meshing tool.

## E0115

**Face labelled twice** (error)

A face carries two labels.

Fix: Assign each face to one zone in the meshing tool.

## E0116

**Face without an owned cell** (error)

A face touches only ghost cells.

Fix: Regenerate the partition; every face needs an owned cell.

## E0117

**Duplicate or negative global cell id** (error)

Two cells share a global id, or one is negative.

Fix: Regenerate the partition; global cell ids must be unique and non-negative.

## E0120

**Degenerate face** (error)

A face has zero or non-finite area.

Fix: Repair the face in the meshing tool; it has no area.

## E0121

**Non-positive cell volume** (error)

An owned cell has zero, negative or non-finite volume.

Fix: Repair the cell in the meshing tool; it is inverted or flat.

## E0130

**Mesh file cannot be opened** (error)

The file or directory cannot be opened or read.

Fix: Check the path and the file permissions.

## E0131

**Unknown mesh format** (error)

The content matches no supported format.

Fix: Export the mesh as CGNS, Gmsh, Fluent .msh or an OpenFOAM polyMesh.

## E0132

**Malformed mesh file** (error)

The file violates its format specification (syntax, truncation, inconsistent counts).

Fix: Re-export the mesh; the file does not follow its format specification.

## E0133

**Unsupported mesh feature** (error)

The file is valid but uses a feature the reader does not support yet; the message names it.

Fix: Export the mesh without the feature named in the message.

## E0134

**Mesh refers to another file** (error)

The mesh refers to another file (CGNS link); meshes must be self-contained.

Fix: Export a self-contained mesh (no CGNS links to other files).

## E0135

**Entity in several zones** (error)

An entity belongs to several zones in the source (Gmsh entity in two physical groups, face in two OpenFOAM face zones).

Fix: Assign each entity to one zone in the meshing tool.

## E0140

**Invalid partition request** (error)

A part count, rank, part index or cell weight is out of range, or an array has the wrong size.

Fix: Ask for 1 to N parts (N = number of cells), one rank per part, and a weight of at least 1 per cell.

## E0141

**Mesh already partitioned** (error)

The mesh to partition already has ghost cells.

Fix: Partition the global mesh, not one of its partitions.

## E0142

**Partitioner unavailable or failed** (error)

The partitioner (METIS, Scotch) is not built or reports an error.

Fix: Use a build with the partitioner enabled, or choose the other partitioning method.

## E0143

**Empty partition** (error)

A part received no cell.

Fix: Run on fewer ranks, or relax the cell weights.

## E0144

**Inconsistent halo** (error)

The communicator size differs from the partition count, or two ranks disagree on what they exchange.

Fix: Launch one MPI rank per part, each with the partition extracted for its own rank.

## E0145

**Device halo exchange unsupported** (error)

A halo exchange on device memory without CUDA support in the build or in MPI.

Fix: Use a build with CUDA enabled and a CUDA-aware MPI, such as the dev image.

## E0201

**No mesh information** (error)

Planning a case needs the MeshInfo of its mesh (docs/contracts/mesh-info.md). The C++ mesh readers are not part of this build, so it must be passed explicitly.

Fix: Pass the MeshInfo JSON: `autan plan case.py --mesh-info mesh.meshinfo.json`.

## E0202

**The MeshInfo document is invalid** (error)

The JSON does not match `autan.mesh-info/1`, or its counts are inconsistent (zone counts that do not sum to the cell count, duplicate names, unknown interface zones).

Fix: Regenerate the MeshInfo with the mesh reader of the same solver version.

## E0203

**The MeshInfo describes another mesh file** (error)

The MeshInfo `source.sha256` differs from the SHA-256 of the case's mesh file, so it describes another mesh or another version of it.

Fix: Pass the MeshInfo of the case's mesh, or change `mesh` in the case.

## W0204

**Highly non-orthogonal cells** (warning)

The maximum face non-orthogonality exceeds 70°, the threshold above which OpenFOAM's checkMesh reports severely non-orthogonal faces. Gradients and viscous fluxes lose accuracy and implicit solves may stall.

Fix: Improve the mesh near the worst cells, or lower the CFL number.

## W0205

**Mesh file not found** (warning)

The case's mesh file does not exist where the case says, so `plan` cannot check that the MeshInfo describes it (by comparing the mesh hash).

Fix: Put the mesh at the path the case gives, relative to the case file.

## E0301

**Quantity of the wrong dimension** (error)

The value has a dimension other than the one the field expects, for example a speed given for the Mach number.

Fix: Pass a value of the expected dimension, with a unit from `autan.units`.

## E0302

**Missing unit** (error)

A dimensional field received a bare number. Angles also need a unit, because degrees and radians cannot be told apart.

Fix: Multiply by a unit: `19.57 * deg`, `7.0 * m`.

## E0303

**Unknown unit** (error)

The text could not be parsed as `<number> <unit>` with Pint's default registry.

Fix: Write the quantity as `"<number> <unit>"`, for example `"101325 Pa"`.

## E0304

**Value out of range** (error)

The value must be finite, and strictly positive where the schema says `> 0` (temperatures are compared in kelvin).

Fix: Use a physically valid value.

## E0305

**Turbulence model needs an unsteady run** (error)

DDES and WMLES resolve turbulent structures in time and cannot run steady.

Fix: Set `case.time = autan.Unsteady(steps=..., dt=...)`.

## E0306

**Freestream does not match the physics** (error)

Compressible flow needs `mach` and exactly one of `pressure` or `re` (with `ref_length`); incompressible flow needs `velocity` and takes no `mach`, `re`, `pressure` or `temperature`.

Fix: Set the freestream parameters required by the physics.

## E0410

**Boundary rule matches no zone** (error)

The pattern of `case.boundary(...)` matches no boundary zone of the mesh.

Fix: Use one of the zone names listed by `autan plan`, or a glob that matches some.

## E0411

**Boundary zone matched by several rules** (error)

Two `case.boundary(...)` patterns match the same zone; neither takes precedence.

Fix: Narrow the patterns so each zone matches one rule.

## E0412

**Boundary zone without a condition** (error)

The zone is not typed in the mesh file and no rule of the case matches it.

Fix: Give it a condition: `case.boundary("name").wall()` (or `slip()`, `symmetry()`, `farfield()`, `inlet(...)`, `outlet(...)`).

## E0413

**Mesh boundary type needs values** (error)

The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values only the case can provide.

Fix: Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.

## E0414

**Boundary condition not available for this physics** (error)

`inlet` (total conditions) is compressible only, `velocity-inlet` incompressible only, and a no-slip `wall` needs viscous flow.

Fix: Use the condition matching the physics: `slip()` for inviscid walls, `inlet(...)` or `velocity_inlet(...)`.

## W0415

**Case overrides the mesh boundary type** (warning)

The case applies a condition of another kind than the type stored in the mesh file. This is allowed; the warning makes the override visible.

Fix: Check that the override is intended.

## E0501

**Cell zone rule matches no zone** (error)

The pattern of `case.cell_zone(...)` matches no cell zone of the mesh.

Fix: Use one of the cell zone names listed by `autan plan`.

## E0502

**Cell zone matched by several rules** (error)

Two `case.cell_zone(...)` patterns match the same zone.

Fix: Narrow the patterns so each zone matches one rule.

## E0503

**Solid zone without a material** (error)

The mesh declares the zone solid and the case gives it no material.

Fix: Set it: `case.cell_zone("name").solid(autan.materials.COPPER)`.

## E0504

**Material does not match the physics** (error)

Compressible fluid zones need an ideal gas; incompressible fluid zones need a constant-density fluid.

Fix: Use a material of the right kind for the physics.

## E0505

**Fluids of different materials in contact** (error)

Two fluid zones with different materials share an interface. Mixing them needs species transport, which `autan.case/1` does not provide.

Fix: Use the same material on both sides, or separate them by a solid zone.

## E0506

**Solid zone with incompressible physics** (error)

Conjugate heat transfer needs an energy equation in the fluid, which incompressible flow does not solve in `autan.case/1`.

Fix: Use `autan.Compressible(...)`, or remove the solid zone from the mesh.

## E0601

**Monitor surfaces match no boundary zone** (error)

The `surfaces` pattern of a monitor matches no boundary zone of the mesh.

Fix: Use a pattern that matches the zones to integrate over.

## E0602

**Duplicate monitor name** (error)

Two monitors share a name; results are keyed by monitor name.

Fix: Name them: `autan.forces("wing.*", name="wing")`.

## E0701

**The run directory does not conform to autan.results/1** (error)

A file is missing or unreadable, `run.json` has another schema, the stored case or MeshInfo does not match the hashes of `run.json`, or the history is out of order. See docs/contracts/results.md.

Fix: Open a directory written by a backend of this solver version.

## E0702

**No such monitor** (error)

The history has no monitor of that name, or `results.forces` was used on a case with zero or several forces monitors.

Fix: Use `results.history.monitor(name)` with a name from `results.history.monitors`.

## E0703

**No such solution data** (error)

The run has no `solution.cgns` (it failed or diverged before writing one), or the requested field is not in it.

Fix: Check `results.run.status`, and use a field name listed in the message.

## E0704

**The run failed** (error)

The backend stopped on an unexpected error; the message names it. The run directory keeps the history written so far.

Fix: Report the message with the run directory.

## W0801

**No NVIDIA GPU found** (warning)

`nvidia-smi` is missing or lists no GPU. CUDA runs are not possible on this machine.

Fix: Install the NVIDIA driver, or run on a GPU machine.

## W0802

**No MPI launcher found** (warning)

Neither `mpirun` nor `mpiexec` is on the PATH. Multi-GPU runs need MPI.

Fix: Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support).

## W0803

**Solver core not built** (warning)

This installation contains the SDK only: cases can be planned, and run only with the `fake` backend, which writes synthetic results.

Fix: Build the C++ core (see README) to run real computations.

## E0804

**No solver backend** (error)

`case.run()` needs a backend. The C++ core is not part of this build, so only the `fake` backend exists; it writes synthetic results.

Fix: Pass one explicitly: `case.run(mesh_info, backend=autan.FakeBackend())`.

## E0805

**Web UI not installed** (error)

`autan ui` needs the optional `autan-ui` package, which this installation lacks.

Fix: Install the extra: `uv add 'autan[ui]'` (or `uv sync --extra ui` in a checkout).
