# Check a case before you run it

`autan plan` checks a case against its mesh before you reserve a GPU. It shows the condition of every zone and where it comes from, reports what is wrong with a code and a fix, and estimates device memory. This guide plans the heat-exchanger example, then reads and fixes a diagnostic.

## Prerequisites

- The environment of [Getting started with the Autan SDK](../getting-started/sdk.md).

## Plan the case

The heat-exchanger case has two gases and a copper plate. Plan it with the MeshInfo fixture that stands in for its mesh:

```sh
uv run autan plan py/examples/heat_exchanger.py --mesh-info py/tests/fixtures/heat-exchanger.meshinfo.json
```

```text
boundaries  5/5 zones covered
  air-inlet       4 000 faces  inlet          code
  …
  casing         96 000 faces  wall           mesh
cell zones
  air        0.5 M cells  fluid(air)               mesh
  co2        0.5 M cells  fluid(co2)               code
  plate      0.2 M cells  solid(copper)            code
memory      10.8 GB per rank (7.6 host, 3.3 cuda) x 1 · 7 equations (autan.memory/2)
warning[W0205]: mesh '…/heat-exchanger.msh' not found; the MeshInfo is taken as given
plan OK
```

Read it zone by zone:

- `casing` has no rule in the case: it is a wall because the mesh file says so (`mesh`).
- `co2` and `plate` take the materials the case gives them (`code`).
- `W0205` is expected: the mesh file is not in the repository, so `plan` takes the MeshInfo as given.
- The memory line is what one rank needs at its peak, on the host and on its device; [autan plan](../sdk/plan.md#memory-model-autanmemory2) lists what it counts. `autan-run` refuses a run that does not fit (`E0711`).

## Read a diagnostic

Every problem has a code. A code starting with `E` stops the plan with exit status 1; `W` is a warning. Print what a code means and how to fix it:

```sh
uv run autan explain E0413
```

```text
## E0413

**Mesh boundary type needs values** (error)

The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values only the case can provide.

Fix: Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.
```

The heat-exchanger mesh types its four ports as inlets and outlets, and the case gives their pressures and temperatures. Delete the `*-outlet` line from a copy of the case and `plan` reports `E0413` for both outlets.

## Use the plan in a script

`--json` prints the same plan as one JSON document, with every diagnostic in the [diagnostic shape](../contracts/case-schema.md#diagnostics):

```sh
uv run autan plan py/examples/heat_exchanger.py --mesh-info py/tests/fixtures/heat-exchanger.meshinfo.json --json
```

The exit status is 0 when the case can run and 1 when it cannot.

## Next steps

- [Boundary conditions](../concepts/boundary-conditions.md): the resolution rules behind the plan.
- [Diagnostic codes](../sdk/errors.md): every code, with its fix.
