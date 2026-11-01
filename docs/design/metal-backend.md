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

Residual norms on the device are fixed-order sums of squares in double-float, scaled by a power of two so the squares stay in FP32's exponent range (`autan::kernels::metal::sum_of_squares`). An FP64 sum of FP64 data is refused (E1013).

## Measurements

Apple M4 Pro (20 GPU cores, 48 GB unified memory), macOS 26.7.1, solver commit `93083380`, clean tree. Meshes: V&V families renumbered by `autan::rcm_order`; tet n=96 (3.93 M cells) and hex n=128 (2.10 M cells), loop footprints 0.32–0.81 GB. Medians of 30 samples of 5 launches (STREAM: 100), GPU time of each command buffer. Raw JSON with provenance: `codegen/spike/results/metal/bench/`; table: `python codegen/bench/metal_summarize.py STREAM RESULTS...`.

**Peak.** `autan_metal_stream`, FP32 arrays of 1 GiB, float4 accesses: copy 203.0 GB/s (IQR 200.7–205.0, CV 1.9 %), read 231.9 GB/s (IQR 226.3–236.7, CV 3.3 %); scale 202.5, add 210.0, triad 209.0. An earlier session on the same Mac gave copy 182.6 GB/s (CV 2 %): sessions differ by about 10 %, samples within one session by 2–3 %. Apple's nominal figure is 273 GB/s.

**Reference of the gate.** Apple GPUs read faster than they copy, and the loops mostly read: atomic advection reaches 117 % of the copy peak. Each loop is therefore measured against its own mix: a read byte costs 1/B_r, a written byte 1/B_w = 2/B_c − 1/B_r (copy moves one of each), and a loop reading R and writing W bytes has the reference (R + W) / (R/B_r + W/B_w), 220–230 GB/s for the loops below. Bytes are the analytic minimum of [kernel-generator-spike.md](kernel-generator-spike.md#bandwidth); W is the residual (advection) or the gradient and bounds (LSQ).

| Loop | Mesh | Policy | Best strategy | GB/s | % reference | % copy | ns/cell |
|---|---|---|---|---|---|---|---|
| advection | tet | fp32 | warp-aggregated | 237.7 | 104 % | 117 % | 0.50 |
| advection | tet | fp32+acc64 | store-gather | 174.6 | 76 % | 86 % | 1.18 |
| advection | hex | fp32 | warp-aggregated | 237.2 | 103 % | 117 % | 0.70 |
| advection | hex | fp32+acc64 | store-gather | 172.1 | 75 % | 85 % | 1.71 |
| LSQ + bounds | tet | fp32 | — | 213.0 | 97 % | 105 % | 0.53 |
| LSQ + bounds | tet | fp32+acc64 | — | 204.4 | 91 % | 101 % | 0.78 |
| LSQ + bounds | hex | fp32 | — | 178.6 | 80 % | 88 % | 0.85 |
| LSQ + bounds | hex | fp32+acc64 | — | 125.3 | **55 %** | 62 % | 1.79 |

Deterministic strategies in `fp32`: store-gather 73 % (tet) and 70 % (hex), gather-recompute 65 % and 71 %; global colouring 18–22 %, as on GB10. In `fp32+acc64`, store-gather is the fastest strategy (75–76 %): the double-float sums of gather-recompute (55–61 %) cost more than the scratch traffic.

LSQ on hex in `fp32+acc64` misses the gate. Not the double-float arithmetic: with its sums replaced by float sums, the time does not change. It is the 8-byte read of `lsq_weight` per incidence (six per cell): read as float, the same kernel takes 0.76 ns/cell instead of 1.80. The same loop is at 72 % on GB10. The fix belongs to the generator: geometry that bodies only convert to the compute type is stored in it, with values identical to the CPU `fp32+acc64` loops ([Open items](#open-items)).

**MMS.** `spike/mms_study.py` with the residual evaluated by the Metal loops (gather-recompute), V&V families at levels 8, 12, 16, 24, unperturbed: fitted L2 order / finest pair, identical in `fp32` and `fp32+acc64` and to the CPU orders of the spike. Per-level errors with mesh hash, commit and precision: `codegen/spike/results/metal/mms/`.

| Family | L2 | L∞ |
|---|---|---|
| tet | 2.14 / 2.05 | 2.00 / 2.02 |
| hex | 2.18 / 2.18 | 2.21 / 2.17 |
| hybrid | 2.14 / 2.14 | 2.02 / 2.03 |
| poly | 2.47 / 2.35 | 2.50 / 2.34 |

## Verification

`pytest tests/test_gpu.py --backend metal` with `AUTAN_SPIKE_LIB=…/libautan_spike_metal.dylib` runs the CUDA device suite on Metal: every loop, strategy and policy against the host loops (residuals, gradients, Roe flux with its tangent and adjoint, adjoints of reductions with Min/Max ties, ghost-cell partial sums), bitwise reproducibility of the deterministic strategies over ten runs and two launch shapes, norms at amplitudes 10⁻²⁰ and 10²⁰, and refusals with their codes. `tests/test_metal.py` checks the generator's refusal and compiles every DSL shape with the device's Metal compiler.

CI: `.github/workflows/metal.yml` on GitHub's `macos-26` arm64 runner builds the runtime, driver and benchmarks, then runs the device tests if the VM exposes a Metal device; if not, it says so in a warning and the step summary.

## Open items

- **LSQ on hex in `fp32+acc64` at 55 % of the reference.** Geometry read only through conversion to the compute type will be stored in it (generator, after #98); re-measure then.
- **autan-bench on macOS.** Its busy-sample threshold (50 % utilisation) is never reached on Apple GPUs, whose IO-registry utilisation stays at 8–18 % during a saturating STREAM; the results above are the binaries' raw JSON with autan-bench's statistics, to be re-run through `autan-bench run` once fixed (CFD - HPC Spark).
- **Shader compilation.** Libraries compile from source at first use (4 s for the spike modules, then the system cache); a precompiled `.metallib` (Metal Toolchain) is an option for packaged wheels.
- **Hierarchical colouring and SIMD-group transposition** are not emitted, as on CUDA.
