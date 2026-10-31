# Kernel generator spike: results and decision

The ten-day spike of [research §8](../research/kernel-generator.md#8-two-week-spike), run on 2026-10-31 and 2026-11-01 on PR #44 (issue #43). Generator: [kernel-generator.md](kernel-generator.md).

## Decision

**Go for option A**: Python DSL, our own SSA IR, our own tangent and adjoint by source transformation, ahead-of-time C++20 for CPU, CUDA and HIP. No fallback is triggered: the in-house adjoint passes every correctness criterion (fallback (a), Tapenade, is not needed), and Enzyme agrees with it to 2×10⁻¹⁵. Option B (hand-written templated bodies on dual numbers) is 3–7 % faster on the Roe Jacobian but needs 2.4× the source lines, 255 registers instead of 183, and gives no adjoint.

The spike does not close M1. Open for the M1 gate: execution on AMD hardware, a deterministic adjoint of cell reductions and device-side norms, the hierarchical (OP2) colouring and shared-memory transposition strategies, and the ARM NEON CPU backend ([Open items](#open-items)).

## Criteria

Hardware: DGX Spark `spark-5359` (GB10, `sm_121`, CUDA 13.0.3) in `autan-dev:local`, timings by `autan-bench run` (#8) under `/tmp/cfd-gpu.lock` with the GPU checked idle. Peak bandwidth: 239.79 GB/s, best STREAM median of `bench/baselines/gb10.json` (70 % = 167.9 GB/s). Meshes: V&V families (#9) read by `autan::io::read_mesh` and renumbered by `autan::rcm_order` (#34). Host: macOS arm64, Apple clang 21.

| Area | Criterion | Measured | Verdict |
|---|---|---|---|
| Bandwidth, advection face loop | ≥ 70 % of peak in `fp32` and `fp32+acc64`, working set ≥ 10× L2 | Best strategy (warp-aggregated): 89–98 %; atomic 88–97 %; store-gather 66–76 %; gather-recompute 57–86 %; global colouring 16–22 % ([table](#bandwidth)) | **Pass** (non-deterministic strategies); deterministic strategies 57–86 % |
| Bandwidth, LSQ gradient + bounds | same | 72–96 % | **Pass** (hex `fp32+acc64` at 72 %) |
| MMS order | L2 ≥ 1.9 on tet, hex, poly, hybrid in `fp64`, `fp32`, `fp32+acc64` | Fitted L2 order 2.08–2.49 on all ten V&V families (clean and perturbed), identical in the three policies ([table](#mms)) | **Pass** |
| Analytic | LSQ and corrected Green-Gauss exact on linear fields | LSQ ≤ 400 ε, all kinds, perturbed, three policies; corrected Green-Gauss has the exact gradient as its fixed point (the iteration itself diverges on tetrahedra) | **Pass** |
| Conservation | residual sum and adjoint of ȳ = 1 | Changing cells more than two layers from the boundary leaves Σ R unchanged to 10⁻¹³; d(Σ R)/dq vanishes there to 10⁻¹³; Roe face loop: Σ R = 0, adjoint of ȳ = 1 exactly 0. No periodic meshes: the mesh contract has none | **Pass** (adapted) |
| Adjoint correctness | dot-product gap ≤ 10⁻¹³ (`fp64`), ≤ 10⁻⁵ (`fp32`, `fp32+acc64`); tangent vs complex step ≤ 10⁻¹² | Composed residual: 2.9×10⁻¹⁷ / 2.9×10⁻⁹ / 8.3×10⁻⁹; Roe with Harten's fix: 1.4×10⁻¹⁷ / 5.2×10⁻⁹ / 6.9×10⁻⁹; min/max reductions with ties: 1.9×10⁻¹⁷ / 3.1×10⁻⁹ / 4.6×10⁻⁹; complex step 3.2×10⁻¹⁶ (residual), 4.7×10⁻¹⁶ (Roe). GPU tangents and adjoints equal the host's in all five strategies | **Pass** |
| Adjoint cost | adjoint / primal ≤ 3× | Roe face loop on GB10: 1.1–1.7× in `fp32` and `fp32+acc64`, 2.6–2.9× in `fp64` | **Pass** |
| Determinism | strategies 4–6 bitwise over 10 runs and two launch shapes: residual, gradient, norms, adjoint | Residual, LSQ and Green-Gauss gradients, face-loop adjoints (Roe): bitwise in coloured, store-gather, gather-recompute. Cell-reduction adjoints scatter with atomics; norms are not computed on the device yet | **Partial** |
| Portability | CUDA and HIP from one DSL source, no per-backend edits | One emitted text for both; CUDA runs on GB10; HIP compiles for `gfx942` in CI (`gpu-hip`, ROCm 7.2.4, HIP Clang 22, no error or warning); no AMD device to run | **Pass** for compilation; HIP execution not done |
| Build | generator ≤ 2 s; rebuild after editing one body ≤ 30 s; errors at the `.py` line | Generator 0.07 s (4 modules, CPU + GPU); rebuild 2.2 s (`autan_kernels_gpu`, all policies), 6.6 s (benchmark, all modules); DSL errors carry `file.py:line`; C++ errors land on the `.py` line through `#line` (`kernels/roe.py:53`); SASS line tables reference `kernels/*.py` | **Pass** |
| Enzyme | CPU cross-check; one pinned GPU attempt | Enzyme 0.0.303 (LLVM 23.1.4) reverse mode of the emitted host Roe body equals our adjoint to 1.95×10⁻¹⁵ over 10 000 states, half near stagnation. GPU: see [Enzyme on the Spark](#enzyme-on-the-spark) | **Pass** (CPU) |
| Option B | lines, time, registers | Roe flux + both 5×5 Jacobians: A (generated tangent, 10 directions) 1.39 ns/face, 183 registers, from 56 DSL lines that also give the adjoint; B (dual numbers by hand) 1.30–1.35 ns/face, 255 registers, 136 lines, no adjoint; results agree to 1.6×10⁻⁷ (`fp32`) | A chosen |

## Bandwidth

Useful bytes are the loop's analytic minimum (each array touched once; `res` read and written), divided by the time of the whole strategy (both passes of store-gather, all colours). Medians of 30 samples of 5 launches; CV ≤ 5 %.

| Mesh (cells) | Policy | Kernel | atomic | warp-agg. | coloured | store-gather | gather-recompute |
|---|---|---|---|---|---|---|---|
| tet n=64 (1.17 M) | fp32 | advection | 90 % | 89 % | 20 % | 66 % | 57 % |
| tet n=64 | fp32+acc64 | advection | 90 % | 90 % | 22 % | 71 % | 69 % |
| tet n=64 | fp64 | advection | 88 % | 91 % | 22 % | 72 % | 69 % |
| hex n=128 (2.10 M) | fp32 | advection | 93 % | 95 % | 18 % | 70 % | 78 % |
| hex n=128 | fp32+acc64 | advection | 97 % | 98 % | 19 % | 75 % | 83 % |
| hex n=128 | fp64 | advection | 96 % | 97 % | 19 % | 76 % | 84 % |

| Mesh | fp32 | fp32+acc64 | fp64 |
|---|---|---|---|
| LSQ gradient + bounds, tet n=64 | 95 % (0.49 ns/cell) | 94 % (0.71 ns/cell) | 94 % (0.82 ns/cell) |
| LSQ gradient + bounds, hex n=128 | 96 % (0.66 ns/cell) | 72 % (1.30 ns/cell) | 73 % (1.42 ns/cell) |

Without renumbering (cells in Gmsh order), the tetrahedral advection loop falls to 57 % and gather-recompute to 7 %: RCM is a prerequisite. One-off costs on the host: RCM 0.7 s (tet) and 1.1 s (hex), against 2.4–3.3 s to read the file. Working sets: tet ≈ 240 MB (10× L2), hex ≈ 550 MB (22× L2).

## MMS

Steady advection-diffusion with second-order reconstruction (`kernels/advection_diffusion.py`), levels n = 8, 12, 16, 24, solved by GMRES with the generated residual. Fitted L2 order / finest pair; identical to two decimals in `fp64`, `fp32+acc64` and `fp32`.

<!-- mms-table -->

## Enzyme on the Spark

<!-- enzyme-gpu -->

## Open items

- **AMD execution.** HIP compiles for `gfx942`; nothing has run on AMD hardware (decision pending with the lead).
- **Deterministic mode, complete.** Cell-reduction adjoints need per-incidence storage and a transposed gather (§6); residual norms need a fixed-order device reduction.
- **Strategies.** Hierarchical colouring (OP2) and shared-memory transposition are not emitted; the global colouring measured here is 4–5× slower than atomics, so the deterministic mode currently relies on store-gather and gather-recompute (57–86 % of peak).
- **Measurements not taken.** Measured DRAM traffic (Nsight Compute) beside the useful bytes; AoS layout; FP64 atomic throughput on GB10 in isolation.
- **CPU backend.** Serial only; the ARM NEON vectorised backend of M1 and the Metal backend of M1b (#21) build on the same IR.
- **Green-Gauss.** The iterative skewness correction diverges on tetrahedral meshes (spectral radius above 1); LSQ is the default gradient for M2.
