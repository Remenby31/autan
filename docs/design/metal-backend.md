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

| Policy | state, grad | residual, accum | geom, param | compute |
|---|---|---|---|---|
| `fp32` | float | float | float | float |
| `fp32+acc64` | float | double-float | double-float | float |

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

Apple M4 Pro (20 GPU cores, 48 GB unified memory), macOS 26.7.1, through `autan-bench run` (GPU idle before the run, lock held), clean trees. Meshes: V&V families renumbered by `autan::rcm_order`; tet n=96 (3.93 M cells, the mesh of the GB10 spike: same SHA-256) and hex n=128 (2.10 M cells); loop footprints 0.32–0.81 GB. Medians of 30 samples of 5 launches (STREAM: 100), GPU time of each command buffer. Two sessions: A, solver commit `39528f2`, a quiet machine; B, commit `df59dd4` (same kernels; mesh and case hashes recorded), with other jobs keeping the host's load average near 190. On unified memory the CPU competes for the same DRAM, and the GPU idle check cannot see it: every peak and loop of session B is about 20 % slower, and percentages are taken within a session. Results: `codegen/spike/results/metal/bench/` (B) and `…/bench/39528f2/` (A); table: `python codegen/bench/metal_summarize.py stream.json RESULTS...`.

**Peak.** `autan_metal_stream`, FP32 arrays of 1 GiB, float4 accesses, before the loops. Session A: copy 202.9 GB/s (IQR 200.5–205.1, CV 1.7 %), read 238.5 GB/s (IQR 234.9–243.0, CV 2.3 %), triad 207.9. Session B: copy 169.5 (CV 1.2 %), read 186.8 (CV 3.5 %), triad 171.2. A second STREAM after the loops: copy 190.3 / read 239.9 (A), 173.4 / 191.2 (B). Apple's nominal figure is 273 GB/s.

**Reference of the gate.** Apple GPUs read faster than they copy, and the loops mostly read: atomic advection reaches 115 % of the copy peak. Each loop is therefore measured against its own mix: a read byte costs 1/B_r, a written byte 1/B_w = 2/B_c − 1/B_r (copy moves one of each), and a loop reading R and writing W bytes has the reference (R + W) / (R/B_r + W/B_w), set mostly by the read peak. Bytes are the analytic minimum of [kernel-generator-spike.md](kernel-generator-spike.md#bandwidth); W is the residual (advection) or the gradient and bounds (LSQ).

| Loop | Mesh | Policy | Best strategy | % reference A | % reference B | ns/cell A | ns/cell B |
|---|---|---|---|---|---|---|---|
| advection | tet | fp32 | atomic | 99 % | 103 % | 0.51 | 0.62 |
| advection | tet | fp32+acc64 | store-gather | 79 % | 79 % | 1.11 | 1.41 |
| advection | hex | fp32 | atomic / warp-aggregated | 99 % | 107 % | 0.71 | 0.84 |
| advection | hex | fp32+acc64 | store-gather | 79 % | 85 % | 1.57 | 1.87 |
| LSQ + bounds | tet | fp32 | — | 95 % | 99 % | 0.52 | 0.63 |
| LSQ + bounds | tet | fp32+acc64 | — | 87 % | 91 % | 0.81 | 0.97 |
| LSQ + bounds | hex | fp32 | — | 79 % | 89 % | 0.85 | 0.94 |
| LSQ + bounds | hex | fp32+acc64 | — | **55 %** | **67 %** | 1.78 | 1.83 |

Deterministic strategies in `fp32`: store-gather 70–75 %, gather-recompute 62–80 %; global colouring 18–20 %, as on GB10. In `fp32+acc64`, store-gather is the fastest strategy: its scratch holds the faces' FP32 values (exact: they are body outputs) and the gather sums them in double-float; gather-recompute (52–71 %) pays the double-float sums of every face twice.

LSQ on hex in `fp32+acc64` misses the gate in both sessions, and its time barely moves when the peak drops by 20 %: it is bound by latency, not bandwidth. Two manual experiments on the generated kernel (edits of the generated header, not kept in the repository) point at the 8-byte read of `lsq_weight` per incidence (six per cell), not at the double-float arithmetic: with its sums replaced by float sums, the time did not change; with the weights read as float, it fell from 1.80 to 0.76 ns/cell. The same loop is at 72 % on GB10. The fix belongs to the generator: geometry that bodies only convert to the compute type is stored in it, with values identical to the CPU `fp32+acc64` loops ([Open items](#open-items)).

**MMS.** `spike/mms_study.py` with the residual evaluated by the Metal loops (gather-recompute), V&V families at levels 8, 12, 16, 24, unperturbed: fitted L2 order / finest pair, identical in `fp32` and `fp32+acc64` and to the CPU orders of the spike. Per-level errors with mesh hash, case hash, commit (`df59dd4`) and precision: `codegen/spike/results/metal/mms/`.

| Family | L2 | L∞ |
|---|---|---|
| tet | 2.14 / 2.05 | 2.00 / 2.02 |
| hex | 2.18 / 2.18 | 2.21 / 2.17 |
| hybrid | 2.14 / 2.14 | 2.02 / 2.03 |
| poly | 2.47 / 2.35 | 2.50 / 2.34 |

## Verification

`pytest tests/test_gpu.py --backend metal` with `AUTAN_SPIKE_LIB=…/libautan_spike_metal.dylib` runs the CUDA device suite on Metal: every loop, strategy and policy against the host loops (residuals, gradients, Roe flux with its tangent and adjoint, adjoints of reductions with Min/Max ties, ghost-cell partial sums), bitwise reproducibility of the deterministic strategies over ten runs and two launch shapes, norms at amplitudes 10⁻²⁰ and 10²⁰, and refusals with their codes. `tests/physics/test_euler_metal.mm` evaluates the Euler residual of `kernels/euler.py` with the Metal loops and compares it with the CPU physics (`autan::physics::Euler`): Roe, HLLC and AUSM+-up, first and second order, both policies, every strategy each policy has. `tests/test_metal.py` checks the generator's refusal and compiles every DSL shape with the device's Metal compiler.

CI: `.github/workflows/metal.yml` on GitHub's `macos-26` arm64 runner builds the runtime, driver and benchmarks, then runs the device tests if the VM exposes a Metal device; if not, it says so in a warning and the step summary.

## Open items

- **LSQ on hex in `fp32+acc64` at 55–67 % of the reference.** Geometry read only through conversion to the compute type will be stored in it (CFD - Générateur kernels); re-measure then.
- **autan-bench on macOS.** The results above ran with the busy-sample fix of PR #107 (Apple's IO-registry utilisation stays at 3–18 % under a saturating load). Host load is not recorded yet, though it shares the DRAM (session B); proposed to CFD - HPC Spark.
- **Shader compilation.** Libraries compile from source at first use (4 s for the spike modules, then the system cache); a precompiled `.metallib` (Metal Toolchain) is an option for packaged wheels.
- **Hierarchical colouring and SIMD-group transposition** are not emitted, as on CUDA.
