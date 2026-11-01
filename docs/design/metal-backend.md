# Metal backend

The kernel generator's fourth backend: the loops of `kernels/*.py` on Apple GPUs, emitted from the same IR as the CPU, CUDA and HIP loops ([kernel-generator.md](kernel-generator.md)), with no per-backend edit of a kernel source. Milestone M1b, issue [#21](https://github.com/Remenby31/autan/issues/21).

## Design

`autan_codegen --backends metal` writes, per module `m`, `m_metal.hpp`: the module's Metal Shading Language (MSL) source as a string, and one host launcher per loop in `autan::kernels::m::metal_loops`. The source is the prelude (`codegen/src/autan_codegen/metal_prelude.metal`), the `Data<P>` struct, the bodies, the kernels, and one explicit instantiation per kernel and policy. Bodies, loads, reductions and adjoints come from the same emitter functions as the CUDA text (`body_lines`, `call_args`, `reduce_body`, `reduce_adjoint_body`, `twin_gather_body`); only kernel heads and launchers are Metal's.

Kernels read `Data<P>` from a constant buffer whose pointers are GPU addresses (`MTLBuffer.gpuAddress`), so loop bodies are the CUDA text (`d.q[c * d.topo.n_cells + i]`). The host fills the generated `Data<P>` with the Metal policies of `autan/kernels/metal.hpp`; the prelude's `Topology` and argument blocks mirror the host layouts.

The runtime (`autan/kernels/metal.hpp`, `codegen/cpp/metal.mm`, Objective-C++) compiles each module's library on first use with `newLibraryWithSource` in `MTLMathModeSafe` with precise functions (no reassociation: the double-float sums depend on it), caches pipelines, allocates shared-storage buffers (unified memory, no copies), records dispatches into one compute encoder with a buffer barrier between them, and makes every live buffer resident with `useResources`. Nothing needs the offline Metal toolchain, on a Mac or in CI. Errors throw E1107. First use of the spike's three modules compiled their pipelines in 4.0 s on an M4 Pro; later processes take 0.01 s from the system's shader cache.

| CMake | Effect |
|---|---|
| `-DAUTAN_ENABLE_METAL=ON` (macOS) | Metal loops, runtime `autan_metal`, driver `autan_spike_metal`, benchmarks `autan_metal_stream` and `autan_metal_loop_bench` |

## Precision

Metal has no FP64. The `fp32+acc64` policy keeps its FP64 roles (residual, accumulation, geometry, parameters) as double-float: an unevaluated sum `hi + lo` of two floats with `|lo| ≤ ulp(hi)/2`, about 48 significant bits. `fp64` is refused: by the generator (`--policies fp64` with `--backends metal`) and by the driver, with E1013.

| Policy | state, grad | residual, accum | geom, param | geom read only as a value | jac | compute |
|---|---|---|---|---|---|---|
| `fp32` | float | float | float | float | float | float |
| `fp32+acc64` | float | double-float | double-float | float | float | float |

| Operation | Algorithm | Error |
|---|---|---|
| sum of two double-floats | AccurateDWPlusDW, Joldes, Muller and Popescu (2017), algorithm 6 (TwoSum: Knuth; Fast2Sum: Dekker) | relative ≤ 3u² |
| double-float plus float (every accumulation of a body output) | DWPlusFP, same paper, algorithm 4 | relative ≤ 2u² |
| product | Dekker (1971), exact `hi·hi` error by FMA, `lo·lo` dropped | — |
| conversion to compute | `hi + lo` rounded to float | as FP64 → FP32 |

u = 2⁻²⁴. Results equal the CPU `fp32+acc64` loops to FP32 rounding (the device suite allows 200 ε).

## Strategies

Metal has device-scope atomic adds on `float`, none on 64-bit floating-point types. Strategies that need atomics on the accumulation type therefore exist in `fp32` only; the others run in both policies. A refused combination throws E1108.

| Strategy | CUDA, HIP | Metal `fp32` | Metal `fp32+acc64` |
|---|---|---|---|
| `atomic` | yes | yes (`atomic_float`) | no (E1108) |
| `warp_aggregated` | yes | yes (SIMD-group shuffles, 32 lanes) | no (E1108) |
| `coloured` | yes | yes | yes |
| `store_gather` | yes | yes | yes |
| `gather_recompute` | yes | yes | yes |
| reduction adjoint, atomic scatter | yes | yes | no (E1108) |
| reduction adjoint, two passes (deterministic strategies) | yes | yes | yes |

Residual norms on the device are fixed-order sums of squares in double-float, scaled by a power of two read from the bits so that subnormal entries (which Apple GPUs flush as operands) and squares beyond FP32's range stay exact (`autan::kernels::metal::sum_of_squares`). An FP64 sum of FP64 data is refused (E1013).

## Measurements

A dev-machine measurement, not a CI reference: Apple M4 Pro (20 GPU cores, 48 GB unified memory), macOS 26.7.1, solver commit `ce679f3` (clean tree), through `autan-bench run --max-load 1` (host load 3–9 on 14 CPUs; GPU idle before each run). On unified memory the CPU competes for the same DRAM and the GPU idle check cannot see it, hence the load limit: an earlier session under a load average near 190 ran every peak and loop about 20 % slower. Meshes: V&V families renumbered by `autan::rcm_order`; tet n=96 (3.93 M cells, the mesh of the GB10 spike: same SHA-256) and hex n=128 (2.10 M cells). Medians of 30 samples of 5 launches, GPU time of each command buffer. Results: `codegen/spike/results/metal/bench/ce679f3/`; table: `python codegen/bench/metal_summarize.py stream-pooled.json RESULTS...`. Earlier sessions, before geometry was stored in the compute type: `…/bench/` and `…/bench/39528f2/`.

**Peak.** `autan_metal_stream`, FP32 arrays of 1 GiB, float4 accesses; five runs spread over the session (three before the loops, two after), pooled (500 samples): copy 199.6 GB/s (IQR 196.5–202.7, CV 2.2 %), read 240.3 GB/s (IQR 236.6–243.4, CV 2.4 %), triad 207.0. Run medians: copy 195.5–204.5, read 238.0–245.5. Apple's nominal figure is 273 GB/s.

**Reference of the gate.** Apple GPUs read faster than they copy, and the loops mostly read: atomic advection reaches 119 % of the copy peak. Each loop is therefore measured against its own mix: a read byte costs 1/B_r, a written byte 1/B_w = 2/B_c − 1/B_r (copy moves one of each), and a loop reading R and writing W bytes has the reference (R + W) / (R/B_r + W/B_w), set mostly by the read peak. Bytes are the analytic minimum of [kernel-generator-spike.md](kernel-generator-spike.md#bandwidth), with each dataset at its stored size; W is the residual (advection) or the gradient and bounds (LSQ).

| Loop | Mesh | Policy | Best strategy | GB/s | % reference | % copy | ns/cell |
|---|---|---|---|---|---|---|---|
| advection | tet | fp32 | warp-aggregated | 238.2 | 100 % | 119 % | 0.50 |
| advection | tet | fp32+acc64 | store-gather | 162.7 | 69 % | 81 % | 0.78 |
| advection | hex | fp32 | warp-aggregated | 236.5 | 99 % | 119 % | 0.71 |
| advection | hex | fp32+acc64 | gather-recompute | 165.9 | 70 % | 83 % | 1.05 |
| LSQ + bounds | tet | fp32 | — | 209.2 | 93 % | 105 % | 0.54 |
| LSQ + bounds | tet | fp32+acc64 | — | 218.1 | 97 % | 109 % | 0.51 |
| LSQ + bounds | hex | fp32 | — | 182.2 | 80 % | 91 % | 0.83 |
| LSQ + bounds | hex | fp32+acc64 | — | 212.4 | 93 % | 106 % | 0.72 |

Geometry that bodies only convert to FP32 is stored in FP32 (kernel-generator.md, Precision). That made LSQ on hex in `fp32+acc64` 2.5 times faster (1.78 → 0.72 ns/cell): its 8-byte weight per incidence, read six times per cell, had kept it latency-bound at 55–67 % of the reference. Advection in `fp32+acc64` became 30 % faster too (tet 1.11 → 0.78, hex 1.57 → 1.05 ns/cell), but its percentage fell from 79 % to 69–70 %: its analytic bytes fell more than its time. Without 64-bit atomics on Metal, its fastest strategies are the deterministic ones, which move bytes the minimum does not count: store-gather writes and reads a scratch value per face, and both gathers read `owner[f]` per incidence to sign the flux. In `fp32` the same deterministic strategies are at 68–71 %; global colouring is at 17–19 %, as on GB10.

**MMS.** `spike/mms_study.py` with the residual evaluated by the Metal loops (gather-recompute), V&V families at levels 8, 12, 16, 24, unperturbed: fitted L2 order / finest pair, identical in `fp32` and `fp32+acc64` and to the CPU orders of the spike. Per-level errors with mesh hash, case hash, commit (`df59dd4`) and precision: `codegen/spike/results/metal/mms/`.

| Family | L2 | L∞ |
|---|---|---|
| tet | 2.14 / 2.05 | 2.00 / 2.02 |
| hex | 2.18 / 2.18 | 2.21 / 2.17 |
| hybrid | 2.14 / 2.14 | 2.02 / 2.03 |
| poly | 2.47 / 2.35 | 2.50 / 2.34 |

## Verification

`pytest tests/test_gpu.py --backend metal` with `AUTAN_SPIKE_LIB=…/libautan_spike_metal.dylib` runs the CUDA device suite on Metal: every loop, strategy and policy against the host loops (residuals, gradients, Roe flux with its tangent and adjoint, adjoints of reductions with Min/Max ties, ghost-cell partial sums), bitwise reproducibility of the deterministic strategies over ten runs and two launch shapes, norms at amplitudes 10⁻²⁰ and 10²⁰, and refusals with their codes. `tests/physics/test_euler_metal.mm` evaluates the Euler residual of `kernels/euler.py` with the Metal loops and compares it with the CPU physics (`autan::physics::Euler`): Roe, HLLC and AUSM+-up, first and second order, all five boundary kinds with the physics' gradient weights, both policies, every strategy each policy has. `tests/test_metal.py` checks the generator's refusal and compiles every DSL shape with the device's Metal compiler.

CI: `.github/workflows/metal.yml` on GitHub's `macos-26` arm64 runner builds the runtime, driver and benchmarks, then runs the device tests if the VM exposes a Metal device; if not, it says so in a warning and the step summary.

## Open items

- **Advection in `fp32+acc64` at 69–70 % of the reference** (81–83 % of copy), recorded as measured: at the threshold within the measurement's spread, with no 64-bit atomics on Metal; the M1b gate was judged met (#21). The gathers read `owner[f]` per incidence to sign each face's value, 8 bytes per face the minimum does not count; encoding the side otherwise would change the topology contract (see the follow-up issue).
- **autan-bench on macOS.** Runs record the host load and `--max-load` refuses a busy host (#121); `autan-bench baseline` refuses Metal runs for lack of a container image digest, so the pooled peak above is computed with autan-bench's statistics directly.
- **Shader compilation.** Libraries compile from source at first use (4 s for the spike modules, then the system cache); a precompiled `.metallib` (Metal Toolchain) is an option for packaged wheels.
- **Hierarchical colouring and SIMD-group transposition** are not emitted, as on CUDA.
- **Euler physics on Metal.** Only the test runs the Euler loops on Metal. A Metal physics path must take its gradient weights from `Euler<P>::gradient_weights()` and refresh them after every `set_boundary()`: they depend on the boundary kinds.
