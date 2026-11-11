# Results in Python

`autan.Results.open(path)` reads a run directory ([results.md](../contracts/results.md)); `case.run(...)` returns the same object. Its arrays are NumPy arrays on the host, C-contiguous and writable, in the dtype below.

| Array | dtype | Shape |
|---|---|---|
| `results.field(name)`, `results.field(name, mean=True)` | float64 | one value per cell, in mesh order |
| `results.wall(name)`: the faces, then the values | int64, float64 | one per wall and slip face |
| `results.cell_zone(zone)` | int64 | the zone's cells |
| `results.checkpoint_field(name)` | float64 | one value per cell, in mesh order |
| `results.history.iterations` | int64 | one per history line |
| `results.history.residual(equation)` | float64 | one per history line |
| `results.history.provisional` | bool | one per history line, `None` for an unsteady run |
| each array of `results.history.monitor(name)`, its `iteration` included | float64, `iteration` int64 | one per line that carries the monitor |
| each quantity of `results.forces` | float64, `iteration` int64 | as `history.monitor` |

**DLPack.** Each of these arrays exports through DLPack without a copy: `x.__dlpack_device__()` is `(1, 0)` (`kDLCPU`), and the array a consumer takes, `np.from_dlpack(x)`, `torch.from_dlpack(x)` or any other, shares `x`'s memory, so a value written through one is read through the other. A field is read from `solution.cgns` once and kept, so writing into the array `results.field(name)` returned changes what later calls return. Exporting without leaving the GPU is out of the alpha: a beta item, which needs in-process bindings to the runtime ([acceptance.md](../validation/acceptance.md), M4).
