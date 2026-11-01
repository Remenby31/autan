# Kernel generator spike: results and decision

The ten-day spike of [research §8](../research/kernel-generator.md#8-two-week-spike), run on 2026-10-31 and 2026-11-01 on PR #44 (issue #43). Generator: [kernel-generator.md](kernel-generator.md).

## Decision

**Go for option A**: Python DSL, our own SSA IR, our own tangent and adjoint by source transformation, ahead-of-time C++20 for CPU, CUDA and HIP. No fallback is triggered: the in-house adjoint passes every correctness criterion (fallback (a), Tapenade, is not needed), and Enzyme agrees with it to 2×10⁻¹⁵. Option B (hand-written templated bodies on dual numbers) is 3–7 % faster on the Roe Jacobian but needs 2.4× the source lines, 255 registers instead of 183, and gives no adjoint.

The spike does not close M1. Open for the M1 gate: execution on AMD hardware, a deterministic adjoint of cell reductions and device-side norms, the hierarchical (OP2) colouring and shared-memory transposition strategies, and the ARM NEON CPU backend ([Open items](#open-items)).

## Criteria

Hardware: DGX Spark `spark-5359` (GB10, `sm_121`, CUDA 13.0.3) in `autan-dev:local`, timings by `autan-bench run` (#8) under `/tmp/cfd-gpu.lock` with the GPU checked idle. Peak bandwidth: 239.79 GB/s, best STREAM median of `bench/baselines/gb10.json` (70 % = 167.9 GB/s). Meshes: V&V families (#9) read by `autan::io::read_mesh` and renumbered by `autan::rcm_order` (#34). Host: macOS arm64, Apple clang 21.

| Area | Criterion | Measured | Verdict |
|---|---|---|---|
| Bandwidth, advection face loop | ≥ 70 % of peak in `fp32` and `fp32+acc64`, working set ≥ 10× L2 | Atomic 90–98 %, warp-aggregated 88–98 %; deterministic: store-gather 64–76 %, gather-recompute 60–83 %, global colouring 18–22 % ([table](#bandwidth)) | **Pass** (atomic, warp-aggregated); deterministic strategies below 70 % on tetrahedra |
| Bandwidth, LSQ gradient + bounds | same | 72–97 % | **Pass** (hex `fp32+acc64` at 72 %) |
| MMS order | L2 ≥ 1.9 on tet, hex, poly, hybrid in `fp64`, `fp32`, `fp32+acc64` | Fitted L2 order 2.08–2.49 (finest pair 1.98–2.63) on all ten V&V families (clean and perturbed), identical in the three policies ([table](#mms)) | **Pass** |
| Analytic | LSQ and corrected Green-Gauss exact on linear fields | LSQ ≤ 400 ε, all kinds, perturbed, three policies; corrected Green-Gauss has the exact gradient as its fixed point (the iteration itself diverges on tetrahedra) | **Pass** |
| Conservation | residual sum and adjoint of ȳ = 1 | Changing cells more than two layers from the boundary leaves Σ R unchanged to 10⁻¹³; d(Σ R)/dq vanishes there to 10⁻¹³; Roe face loop: Σ R = 0, adjoint of ȳ = 1 exactly 0. No periodic meshes: the mesh contract has none | **Pass** (adapted) |
| Adjoint correctness | dot-product gap ≤ 10⁻¹³ (`fp64`), ≤ 10⁻⁵ (`fp32`, `fp32+acc64`); tangent vs complex step ≤ 10⁻¹² | Composed residual: 2.9×10⁻¹⁷ / 2.9×10⁻⁹ / 8.3×10⁻⁹; Roe with Harten's fix: 1.4×10⁻¹⁷ / 5.2×10⁻⁹ / 6.9×10⁻⁹; min/max reductions with ties: 1.9×10⁻¹⁷ / 3.1×10⁻⁹ / 4.6×10⁻⁹; complex step 3.2×10⁻¹⁶ (residual), 4.7×10⁻¹⁶ (Roe). GPU tangents and adjoints equal the host's in all five strategies | **Pass** |
| Adjoint cost | adjoint / primal ≤ 3× | Roe face loop on GB10: 1.13–1.63× in `fp32` and `fp32+acc64` (2.1–2.5× with gather-recompute, which evaluates each face twice); 2.8–3.2× in `fp64`, where GB10's FP64 units bind; tangent 1.04–1.27× (`fp32`), 1.9–2.2× (`fp64`) | **Pass** in the target policies; `fp64` at the 3× mark, to investigate before M6 |
| Determinism | strategies 4–6 bitwise over 10 runs and two launch shapes: residual, gradient, norms, adjoint | Residual, LSQ and Green-Gauss gradients, face-loop adjoints (Roe): bitwise in coloured, store-gather, gather-recompute. Cell-reduction adjoints scatter with atomics; norms are not computed on the device yet | **Partial** |
| Portability | CUDA and HIP from one DSL source, no per-backend edits | One emitted text for both; CUDA runs on GB10; HIP compiles for `gfx942` in CI (`gpu-hip`, ROCm 7.2.4, HIP Clang 22, no error or warning); no AMD device to run | **Pass** for compilation; HIP execution not done |
| Build | generator ≤ 2 s; rebuild after editing one body ≤ 30 s; errors at the `.py` line | Generator 0.07 s (4 modules, CPU + GPU); rebuild 2.2 s (`autan_kernels_gpu`, all policies), 6.6 s (benchmark, all modules); DSL errors carry `file.py:line`; C++ errors land on the `.py` line through `#line` (`kernels/roe.py:53`); SASS line tables reference `kernels/*.py` | **Pass** |
| Enzyme | CPU cross-check; one pinned GPU attempt | Enzyme 0.0.303 (LLVM 23.1.4) reverse mode of the emitted host Roe body equals our adjoint to 1.95×10⁻¹⁵ over 10 000 states, half near stagnation. GPU: builds and runs for `sm_121` with clang 22 ([below](#enzyme-on-the-spark)) | **Pass** |
| Option B | lines, time, registers | Roe flux + both 5×5 Jacobians: A (generated tangent, 10 directions) 1.39 ns/face, 183 registers, from 56 DSL lines that also give the adjoint; B (dual numbers by hand) 1.30–1.35 ns/face, 255 registers, 136 lines, no adjoint; results agree to 1.6×10⁻⁷ (`fp32`) | A chosen |

## Bandwidth

Useful bytes are the loop's analytic minimum (each array touched once; `res` read and written), divided by the time of the whole strategy (both passes of store-gather, all colours). Medians of 30 samples of 5 launches; CV ≤ 5 %.

| Mesh (cells) | Policy | Footprint / L2 | atomic | warp-agg. | coloured | store-gather | gather-recompute |
|---|---|---|---|---|---|---|---|
| tet n=96 (3.93 M) | fp32 | 18.6 | 90 % | 88 % | 20 % | 64 % | 60 % |
| tet n=96 | fp32+acc64 | 32.2 | 90 % | 90 % | 22 % | 69 % | 63 % |
| tet n=96 | fp64 | 34.7 | 88 % | 89 % | 21 % | 70 % | 64 % |
| hex n=128 (2.10 M) | fp32 | 13.9 | 94 % | 95 % | 18 % | 69 % | 79 % |
| hex n=128 | fp32+acc64 | 24.5 | 98 % | 98 % | 19 % | 76 % | 83 % |
| hex n=128 | fp64 | 25.8 | 96 % | 97 % | 19 % | 76 % | 82 % |

| LSQ gradient + bounds | fp32 | fp32+acc64 | fp64 |
|---|---|---|---|
| tet n=96 | 93 % (0.50 ns/cell) | 96 % (0.70 ns/cell) | 93 % (0.82 ns/cell) |
| hex n=128 | 97 % (0.66 ns/cell) | 72 % (1.30 ns/cell) | 73 % (1.41 ns/cell) |

Advection at 90 % costs 0.56 ns/cell (`fp32`) and 0.95 ns/cell (`fp32+acc64`) on tetrahedra. Without renumbering (cells in Gmsh order), tetrahedral advection falls to 24 %, LSQ to 37 % and gather-recompute to 6 %: renumbering is a prerequisite. One-off host cost of `autan::rcm_order` + `renumber_cells`: 3.6–3.9 s for 3.93 M tetrahedra, 1.2–1.3 s for 2.10 M hexahedra; colouring: below 1 s. The footprint is the loop's analytic byte count over GB10's 24 MiB L2. A first run on tet n=64 (1.17 M cells, 5.2–5.5× L2 in `fp32`) gave the same percentages within 2 points; it does not count for the criterion. Results (autan-bench JSON with provenance): `codegen/spike/results/bench/`; summary: `python codegen/bench/summarize.py bench/baselines/gb10.json RESULTS...`.

## MMS

Steady advection-diffusion with second-order reconstruction (`kernels/advection_diffusion.py`), levels n = 8, 12, 16, 24, solved by GMRES with the generated residual. Fitted L2 order / finest pair; identical to two decimals in `fp64`, `fp32+acc64` and `fp32`.

| Family | Clean: L2 | Clean: L∞ | Perturbed 0.3: L2 | Perturbed 0.3: L∞ |
|---|---|---|---|---|
| tet (Gmsh Delaunay) | 2.14 / 2.05 | 2.00 / 2.02 | 2.08 / 2.02 | 1.96 / 1.83 |
| hex | 2.18 / 2.18 | 2.21 / 2.17 | 2.19 / 1.98 | 1.91 / 2.04 |
| hybrid (hex, pyramid, tet, prism) | 2.14 / 2.14 | 2.02 / 2.03 | 2.16 / 2.03 | 2.00 / 1.88 |
| poly (median dual) | 2.47 / 2.35 | 2.50 / 2.34 | 2.49 / 2.28 | 2.43 / 2.00 |
| agglomerated (100+ faces, non-convex) | 2.41 / 2.63 | 1.83 / 2.42 | 2.40 / 2.23 | 1.89 / 1.89 |

FP32 errors differ from FP64 by about 10⁻⁶ relative and stay above 100× the rounding floor on every level (finest-level L2 error ≥ 2×10⁻⁴). L∞ is reported, not gated: the families are not nested (V&V, [order.md](../validation/order.md)). Command: `uv run python spike/mms_study.py FAMILIES OUT` in `codegen/`, families from `autan-vv meshes KIND --levels 8,12,16,24 [--perturbation 0.3 --seed 2026]`; per-level errors with mesh hash, commit and precision in `codegen/spike/results/mms/`.

## Enzyme on the Spark

One pinned attempt, `codegen/spike/enzyme_gpu/attempt.sh`, in a throwaway `autan-dev` container on `spark-5359`: clang 22.1.8 from apt.llvm.org, Enzyme v0.0.303 built against it (needs `libzstd-dev` and `llvm-22-tools`), `clang++-22 -x cuda --cuda-gpu-arch=sm_121` with the image's CUDA 13.0.3. Clang warns that CUDA 13.0 is newer than its supported 12.9 and compiles; the reverse-mode kernel of `square.cu` (d(x²)/dx on 1024 threads) runs and is exact. So Enzyme works on our hardware as an oracle, at the price of clang for device code and an LLVM pinned to Enzyme: it stays out of the build, as research §4.3 recommends.

## Open items

- **AMD execution.** HIP compiles for `gfx942`; nothing has run on AMD hardware (decision pending with the lead).
- **Deterministic mode, complete.** Cell-reduction adjoints need per-incidence storage and a transposed gather (§6); residual norms need a fixed-order device reduction.
- **Strategies.** Hierarchical colouring (OP2) and shared-memory transposition are not emitted; the global colouring measured here is 4–5× slower than atomics, so the deterministic mode currently relies on store-gather and gather-recompute (57–86 % of peak).
- **Measurements not taken.** Measured DRAM traffic (Nsight Compute) beside the useful bytes; AoS layout; FP64 atomic throughput on GB10 in isolation.
- **CPU backend.** Serial only; the ARM NEON vectorised backend of M1 and the Metal backend of M1b (#21) build on the same IR.
- **Green-Gauss.** The iterative skewness correction diverges on tetrahedral meshes (spectral radius above 1); LSQ is the default gradient for M2.
