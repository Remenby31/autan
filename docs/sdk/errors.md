# Diagnostic codes

Generated from `py/cfd/errors.py` by `cfd explain --all`; do not edit by hand.

`E` codes are errors and block the run, `W` codes are warnings, `D` codes report a divergence detected during a run. Codes `E0101` to `E0199` belong to the mesh readers ([mesh.md](../contracts/mesh.md)).

## E0001

**The case file cannot be read** (error)

The file does not exist, is not valid Python or TOML, or raised an exception while running. Python case files run as scripts; the traceback points at the failing line.

Fix: Fix the file at the reported location.

## E0002

**The case file does not define exactly one case** (error)

A Python case file must create exactly one `cfd.Case`, at module level.

Fix: Create one case: `case = cfd.Case("name", mesh="mesh.cgns")`.

## E0003

**The case document does not match the schema** (error)

A field is missing, has the wrong type, or is not part of `cfd.case/1`. The location gives the path of the field in the document. See docs/contracts/case-schema.md.

Fix: Correct the field named in the message.

## E0004

**Unsupported case schema version** (error)

The document's `schema` key is not `cfd.case/1`.

Fix: Set `schema = "cfd.case/1"`.

## E0201

**No mesh information** (error)

Planning a case needs the MeshInfo of its mesh (docs/contracts/mesh-info.md). The C++ mesh readers are not part of this build, so it must be passed explicitly.

Fix: Pass the MeshInfo JSON: `cfd plan case.py --mesh-info mesh.meshinfo.json`.

## E0202

**The MeshInfo document is invalid** (error)

The JSON does not match `cfd.mesh-info/1`, or its counts are inconsistent (zone counts that do not sum to the cell count, duplicate names, unknown interface zones).

Fix: Regenerate the MeshInfo with the mesh reader of the same solver version.

## E0203

**The MeshInfo describes another mesh file** (error)

The file name in the MeshInfo `source.path` differs from the case's `mesh`.

Fix: Pass the MeshInfo of the case's mesh, or change `mesh` in the case.

## W0204

**Highly non-orthogonal cells** (warning)

The maximum face non-orthogonality exceeds 70°, the threshold above which OpenFOAM's checkMesh reports severely non-orthogonal faces. Gradients and viscous fluxes lose accuracy and implicit solves may stall.

Fix: Improve the mesh near the worst cells, or lower the CFL number.

## E0301

**Quantity of the wrong dimension** (error)

The value has a dimension other than the one the field expects, for example a speed given for the Mach number.

Fix: Pass a value of the expected dimension, with a unit from `cfd.units`.

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

Fix: Set `case.time = cfd.Unsteady(steps=..., dt=...)`.

## E0306

**Freestream does not match the physics** (error)

Compressible flow needs `mach` and exactly one of `pressure` or `re` (with `ref_length`); incompressible flow needs `velocity` and takes no `mach`, `re`, `pressure` or `temperature`.

Fix: Set the freestream parameters required by the physics.

## E0410

**Boundary rule matches no zone** (error)

The pattern of `case.boundary(...)` matches no boundary zone of the mesh.

Fix: Use one of the zone names listed by `cfd plan`, or a glob that matches some.

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

Fix: Use one of the cell zone names listed by `cfd plan`.

## E0502

**Cell zone matched by several rules** (error)

Two `case.cell_zone(...)` patterns match the same zone.

Fix: Narrow the patterns so each zone matches one rule.

## E0503

**Solid zone without a material** (error)

The mesh declares the zone solid and the case gives it no material.

Fix: Set it: `case.cell_zone("name").solid(cfd.materials.COPPER)`.

## E0504

**Material does not match the physics** (error)

Compressible fluid zones need an ideal gas; incompressible fluid zones need a constant-density fluid.

Fix: Use a material of the right kind for the physics.

## E0505

**Fluids of different materials in contact** (error)

Two fluid zones with different materials share an interface. Mixing them needs species transport, which `cfd.case/1` does not provide.

Fix: Use the same material on both sides, or separate them by a solid zone.

## E0506

**Solid zone with incompressible physics** (error)

Conjugate heat transfer needs an energy equation in the fluid, which incompressible flow does not solve in `cfd.case/1`.

Fix: Use `cfd.Compressible(...)`, or remove the solid zone from the mesh.

## E0601

**Monitor surfaces match no boundary zone** (error)

The `surfaces` pattern of a monitor matches no boundary zone of the mesh.

Fix: Use a pattern that matches the zones to integrate over.

## E0602

**Duplicate monitor name** (error)

Two monitors share a name; results are keyed by monitor name.

Fix: Name them: `cfd.forces("wing.*", name="wing")`.

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

This installation contains the SDK only. Cases can be planned and run with the `fake` backend, which writes synthetic results.

Fix: Build the C++ core (see README) to run real computations.
