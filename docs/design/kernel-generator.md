# Kernel generator

The kernel generator turns pointwise kernels written in a restricted Python subset (`kernels/*.py`) into C++20 for CPU, CUDA and HIP, with their tangents and adjoints. Design choices and their sources: [`research/kernel-generator.md`](../research/kernel-generator.md). Code: `codegen/` (Python package `autan-codegen`, runtime headers in `codegen/include/autan/kernels/`).

## Pipeline

1. `autan_codegen.model.load` imports a kernel module and collects every loop bound to a module-level name.
2. `autan_codegen.frontend` parses each body with `ast` into a scalar SSA IR (`autan_codegen.ir`): vectors and matrices are scalarised, `for i in range(N)` is unrolled, `if` becomes `select`, calls to other bodies are inlined. Value numbering and dead-code elimination follow.
3. `autan_codegen.derive` adds, for every loop, `<loop>_tangent` and `<loop>_adjoint`, built by `autan_codegen.ad` on the IR.
4. Emitters write, per module `m`: `m.hpp` (the `Data<P>` struct and every body), `m_cpu.hpp` (serial loops), `m_gpu.cuh` (CUDA/HIP kernels and launchers, one text for both) and `m_gpu.cu` (explicit instantiations for the build's policies).

The build runs the generator as a CMake custom command (`codegen/CMakeLists.txt`); it needs Python ≥ 3.12 and nothing outside the standard library. By hand: `PYTHONPATH=codegen/src python3 -m autan_codegen kernels/roe.py --out DIR --backends cpu,gpu [--policies fp64,fp32+acc64,fp32] [--no-line]`.

| CMake option | Effect |
|---|---|
| default | CPU loops, `autan_kernels`, spike library `autan_spike` |
| `AUTAN_ENABLE_CUDA` (root) | GPU loops, `autan_kernels_gpu` (every loop × policy), `autan_spike_gpu`, `autan_loop_bench`, `autan_option_b` |
| `AUTAN_ENABLE_HIP` | the same generated GPU loops compiled with HIP into `autan_kernels_gpu`; exclusive with CUDA |

## DSL

```python
q = cells.field("q", State)                  # data: cells.field / faces.field, a role and a shape
res = cells.field("res", Residual)
velocity = param("velocity", Param.Vec3)

@pointwise
def upwind(qc: State, qn: State, n: Geom.Vec3, a: Param.Vec3) -> Residual:
    un = dot(a, n)
    return un * qc if un > 0.0 else un * qn

residual = face_loop(upwind, qc=q.left, qn=q.right, n=geometry.face_normal, a=velocity,
                     out=Inc(res, left=+1, right=-1))
```

**Types.** A role (`State`, `Grad`, `Residual`, `Geom`, `Param`) and a shape: `State`, `State.Vec3`, `State.Vec5`, `Geom.Mat3`. The precision policy maps roles to storage types ([Precision](#precision)).

**Bodies** (`@pointwise`) use: arguments, locals, numeric module constants; `+ - * /`, `**` with an integer constant 0–8, `@` (matrix times vector or matrix); `< <= > >=`, `and`, `or`, `not`; `x if c else y` and `if`/`else` (a name assigned in one branch must be assigned in the other; `return` in both or neither); `for i in range(N)` with constant bounds; indexing with constants (`v[0]`, `m[i, j]`, `v.x`); `sqrt exp log sin cos abs min max dot cross norm transpose Vec3 Vec5`; `rel(a, b)`, the difference of two `Geom` arguments in their storage type (FP64 geometry stays FP64 until the difference); calls to other `@pointwise` bodies. Anything else is rejected with a code and the `.py` line. `abs(x)` is `x if x >= 0 else -x`; `min(a, b)` is `a if a <= b else b`, `max(a, b)` is `a if a >= b else b`.

**Data.** `cells.field`, `faces.field`, `param`; built-in FP64 geometry `geometry.face_area` (area vector), `face_normal`, `face_area_magnitude`, `face_centroid`, `face_moment` (Σ over the face's fan triangles of (c_t − x_f) ⊗ S_t, zero on planar faces), `face_r_left`, `face_r_right`, `cell_centroid`, `cell_volume`, `lsq_weight` (per incidence). Geometry is computed by `autan::kernels::pack_geometry` from the core's `Geometry`.

**Loops.**

| Declaration | Iterates | Bindings | Outputs |
|---|---|---|---|
| `face_loop` | interior faces | `d.left`, `d.right` (cells), `d` (faces), params | `Inc(d, left=s_L, right=s_R)` |
| `boundary_loop` | boundary faces | `d.left`, `d` (faces), params | `Inc(d, left=s_L)` |
| `cell_loop` | owned cells | `d` (cells), params | `Write(d)` |
| `cell_reduce` | owned cells, then their faces in CSR order | `d.self_`, `d.other(boundary=f)`, `geometry.lsq_weight`, params | `Sum(d)`, `Min(d)`, `Max(d)`, optional `finalise` body |

`Inc` targets have role `Residual` (accumulation type). `Min`/`Max` ties keep the first incidence.

## Generated C++

Per module `m`, in namespace `autan::kernels::m`: `template <class P> struct Data` (a `Topology` and one pointer per dataset, structure of arrays: component `c` of entity `i` at `c * n + i`; parameters by value) and one function per body. Loops: `cpu::name(d)`; `gpu_loops::name(d, schedule)` for cell loops and reductions; `gpu_loops::name(d, strategy, schedule)` for face and boundary loops, with `gpu::Strategy` one of:

| Strategy | Mechanism | Deterministic |
|---|---|---|
| `atomic` | one thread per face, atomic increments | no |
| `warp_aggregated` | owner-sorted faces: segmented shuffle reduction per warp, one atomic per segment; neighbours atomic | no |
| `coloured` | one launch per face colour (greedy in face order), plain increments | yes |
| `store_gather` | face values to scratch, then a cell gather in CSR order | yes |
| `gather_recompute` | each cell evaluates its own faces | yes |

`gpu::Schedule` carries the block size, the colourings (`autan::kernels::colour_faces`), the store-gather scratch and the stream.

**Derivatives.** Active data are cell datasets; geometry, parameters and face datasets (boundary values) are passive. `<loop>_tangent` reads `<d>_tan` of each active input and increments or writes `<d>_tan` of the output. `<loop>_adjoint` transposes access modes (research §4.1): an `Inc` output's adjoint reads `<out>_adj` through the same map with the declared signs; each active gather increments `<d>_adj`; a `Write` output's adjoint is read, then zeroed; a reduction's adjoint recomputes the accumulators and the incidence each `Min`/`Max` selected, then sends adjoints to the incidences. Adjoint datasets have role `Residual`. The adjoint of a residual is the reverse sequence of loop adjoints (example: `advdiff_derivative` in `codegen/spike/capi.cpp`). Branches are linearised on the selected side.

## Precision

| Policy | state, grad | residual, accum | geom, param | compute |
|---|---|---|---|---|
| `fp64` | double | double | double | double |
| `fp32_acc64` (`fp32+acc64`) | float | double | double | float |
| `fp32` | float | float | float | float |

`complex_step` (host only) computes in `std::complex<double>` for complex-step tests.

## Errors

| Code | Name | Raised when |
|---|---|---|
| E1001 | `unsupported_construct` | A body uses a statement, expression or call outside the subset. |
| E1002 | `not_a_dsl_type` | An argument has no annotation, or one that is not a role type. |
| E1003 | `invalid_binding` | Loop bindings do not match the body's arguments, their types, or the loop kind. |
| E1004 | `invalid_loop_output` | A loop's outputs do not fit its kind (`Inc` signs, `Write`, accumulators). |
| E1005 | `invalid_control_flow` | A name is assigned in one branch only, `return` is in one branch only, or code follows `return`. |
| E1006 | `type_mismatch` | Shapes or kinds of operands disagree. |
| E1007 | `rel_of_computed_value` | `rel()` receives something other than `Geom` arguments as loaded. |
| E1008 | `inc_target_not_accumulation` | An `Inc` target does not have role `Residual`. |
| E1009 | `invalid_kernel_module` | A kernel module cannot be imported or declares no loop. |
| E1010 | `unknown_backend` | `--backends` names an unknown backend. |
| E1011 | `unknown_policy` | `--policies` names an unknown policy. |
| E1012 | `ad_unsupported` | A loop with several output datasets is differentiated. |
| E1013 | `policy_unsupported_by_backend` | A precision policy is requested on a backend that cannot run it (`fp64` on Metal). |
| E1100 | `runtime_failure` | A generated loop or the spike driver fails without a more specific code. |
| E1101 | `gpu_runtime_error` | A CUDA or HIP call or kernel launch fails. |
| E1102 | `unknown_precision_policy` | A driver receives an unknown policy name. |
| E1103 | `unknown_race_strategy` | A driver receives an unknown strategy name. |
| E1104 | `face_scratch_too_narrow` | Store-gather scratch has fewer components per face than the loop's outputs. |
| E1107 | `metal_runtime_error` | A Metal call, library compilation or command buffer fails. |
| E1108 | `strategy_unavailable` | A face-loop strategy needs atomics the backend lacks for the policy's accumulation type. |

## Spike (research §8)

Results, verdict and open items: [PR #44](https://github.com/Remenby31/autan/pull/44) and [kernel-generator-spike.md](kernel-generator-spike.md).
