# Getting started with the Autan SDK

Check a case against its mesh and write a run directory with the Python SDK, from a clone of the repository. This page uses the SDK's `fake` backend, which writes synthetic results in the real formats and needs no build. The solver, which `autan run --backend cpu` calls, runs inviscid compressible cases on the CPU and is built from source ([autan run](../sdk/run.md#solver-backend)); the example below uses a turbulence model, which the solver does not run yet.

> Results of the `fake` backend are synthetic. Never use them as reference data.

## Prerequisites

- A clone of [Remenby31/autan](https://github.com/Remenby31/autan).
- [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

No GPU is needed.

## Set up the environment

From the root of the clone, install the SDK and its locked dependencies:

```sh
uv sync --locked
```

## Check a case

The repository ships a case for the NASA CRM-HL high-lift configuration, `py/examples/crm_hl.py`, and a MeshInfo fixture that stands in for its mesh: the counts are realistic, the mesh file does not exist. Plan it:

```sh
uv run autan plan py/examples/crm_hl.py --mesh-info py/tests/fixtures/crm-hl.meshinfo.json
```

```text
boundaries  7/7 zones covered
  wing.slat      0.4 M faces  wall           code
  …
  fuselage       0.7 M faces  wall           mesh
  …
memory      46.6 GB total · 46.6 GB/GPU on 1 GPU · 6 equations (autan.memory/1)
warning[W0205]: mesh '…/crm-hl.cgns' not found; the MeshInfo is taken as given
plan OK
```

Every boundary zone has a condition, and the last column says where it comes from: your code or the mesh file. The warning is expected: the mesh file itself is not in the repository. For your own mesh, the C++ tool `autan-mesh-info --json MESH` writes this document ([mesh contract: readers](../contracts/mesh.md#readers)). [Boundary conditions](../concepts/boundary-conditions.md) explains the resolution.

## Run it on the fake backend

```sh
uv run autan run py/examples/crm_hl.py --mesh-info py/tests/fixtures/crm-hl.meshinfo.json --runs "${TMPDIR:-/tmp}/autan-runs"
```

The command prints the run id, its final state and its directory. The directory holds the case as run, the metadata, the residual history and a CGNS solution, as [Runs](../concepts/runs.md) describes.

## Read the results

Open the run with `autan.Results`:

```python
import os
from pathlib import Path

import autan

run_dir = max(Path(os.environ.get("TMPDIR", "/tmp"), "autan-runs").glob("run-*"))
results = autan.Results.open(run_dir)
print(results.run.status, results.run.solver.backend)
print(results.forces.cl[-1])
```

`results.run.solver.backend` is `fake`: the lift coefficient is synthetic.

## Next steps

- [Check a case before you run it](../guides/check-a-case.md): read the plan and fix what it reports.
- [Meshes](../concepts/meshes.md): how Autan stores cells, faces and zones.
- [Case schema](../contracts/case-schema.md): every field of a case.
