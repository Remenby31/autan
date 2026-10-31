# Kernel generator — design research for M1

Status: proposal for the M1 gate, 2026-10-31. Scope: the `kernels/` and `codegen/` trees of [`docs/PLAN.md`](../PLAN.md). Background facts already in [`state-of-the-art-2026-10.md`](state-of-the-art-2026-10.md) are linked, not repeated. Numbers come from the sources in [§10](#10-sources); each is tagged `[Sn]`.

## 1. Decisions in one table

| Question | Recommendation | Fallback | Evidence |
|---|---|---|---|
| DSL shape | Python-embedded DSL: pointwise bodies in a restricted Python subset parsed into our own SSA IR; loops declared with OP2-style access modes; the generator emits header-light C++20 compiled ahead of time by CMake for CUDA, HIP and CPU | Same IR, but bodies hand-written as templated `__host__ __device__` C++ functions (FLUDA style) | §3 |
| Kernel AD | Differentiate **pointwise functions, not kernels**. Tangent and adjoint by source transformation on our IR; loop-level adjoint by transposing access modes | (a) Tapenade on emitted C pointwise functions; (b) forward-mode vector dual numbers (FUN3D) → local Jacobian, adjoint = its transpose | §4 |
| Enzyme | Not on the critical path: independent oracle and benchmark only | — | §4.2 |
| Face loop | Generator chooses per loop: warp-aggregated atomics (default), hierarchical colouring (deterministic mode), store-then-gather (deterministic, no colouring) | — | §6 |
| Precision | Kernel arguments carry a precision **role**; a policy maps roles to storage / compute / accumulation types; policies are template parameters of the emitted code | Per-loop FP64 override, logged | §7 |
| Fusion | Fuse within a loop (reconstruction → flux → scatter; gradient → limiter; residual + Jacobian); no cross-loop fusion in M1; CUDA Graphs for launch overhead | — | §5 |

## 2. Constraints from the plan and the rules

- Pointwise face and cell kernels written once, emitted as native CUDA and HIP plus CPU, with tangent and adjoint ([PLAN](../PLAN.md), "Décisions figées").
- FP32 compute, FP16/BF16 storage of Jacobians and preconditioners, FP64 for accumulation, norms and geometry.
- M1 gate: MMS order 2 on tet, hex, poly and hybrid meshes in FP32 and FP64; ≥ 70 % of peak bandwidth on gradient and flux; adjoint dot-product test.
- Dev target is GB10 (DGX Spark): 273 GB/s LPDDR5X shared by CPU and GPU [S21]; CUDA 13.0. Production targets add B200/GB200 and MI300X/MI355X (wave64).
- AGENTS.md: bitwise-deterministic mode must keep working; no silent fallback; licences permissive or weak copyleft, GPL needs the lead.

## 3. DSL shape

### 3.1 Options

**A. Python-embedded DSL with its own IR.** Kernel bodies are Python functions in a restricted subset, parsed with the standard `ast` module into a typed SSA IR. Loops are declared separately with access descriptors (read, write, increment; through which map). The generator emits C++ source. Precedents: OP2 (Python translator that parses only the API calls [S10, S11]), PyFR (Mako templates of C-like bodies, compiled at run time [S9]), SoNICS (high-level generic code → per-target code, plus AD of an operator graph [S14]).

**B. Templated C++ functors compiled per backend.** Kernel bodies are `template<class Real> __host__ __device__` functions; a C++ library provides `face_loop<Strategy>(functor)`. Precedent: FUN3D's FLUDA, about 500 lines of preprocessor macros mapping to CUDA, HIP, SYCL or serial CPU, kernels templated on the physics set [S7]. Kokkos is the library-heavy version of the same idea.

### 3.2 Comparison

| Criterion | A. Python DSL + IR | B. Templated C++ functors |
|---|---|---|
| Readability for a kernel author | Physics only: the author writes `flux(qL, qR, n)`; parallel loop, races, layout and precision are absent from the body | Physics plus template plumbing; FLUDA needs a hand-written hierarchical variant to be fast on GPU (7.6× over the naïve kernel on V100) [S7] |
| Race handling and adjoint loops | Automatic: access modes are data the generator reads (§4.3, §6) | Manual per loop, or a C++ library that cannot see inside the functor |
| Reverse-mode AD | Own transformation on a restricted IR (§4) | Needs Enzyme (CUDA only through clang, §4.2) or a tape (none on GPU); forward-mode duals work |
| Fusion | IR-level: inline pointwise calls, drop intermediate arrays | Manual: someone writes the fused functor |
| Compile time | Emits only the (kernel × policy × backend) combinations a build asks for; output can be header-light | Template instantiations multiply; every TU pulls the abstraction headers |
| Debugging | Through emitted code: needs `#line` mapping and a serial CPU backend | Native: IDE, `cuda-gdb`, compiler errors point at the source |
| Cost to build | A small compiler (parser, type check, IR, emitters, AD) | Near zero for the forward path |

Compile time, measured elsewhere: NVRTC 13.3 compiles a trivial cuDF transform kernel in ~1.0 s when it includes the CCCL headers and in 68 ms with a minimal header set (2026-10, two runs × ten samples) [S20]. Headers, not kernel bodies, dominate. Option A controls exactly what each emitted file includes; option B inherits the cost of whatever abstraction headers it uses.

Debugging is the real cost of A. Giles, after building OP2, warned against the opposite failure: C++ metaprogramming whose errors are "strange cryptic compiler error messages", needing "guru-standard" developers [S10]. OP2 also abandoned its ROSE-based whole-program parser because it "could not cope with the complexity of code in HYDRA"; the Python generator that parses only API calls survived [S10].

### 3.3 Recommendation

Option A, with these limits so the compiler stays small:

- **Restricted subset.** Scalars and fixed-size `Vec`/`Mat` types (`Vec3`, `Vec5`, `Mat5`); arithmetic; a whitelist of math functions; `if`/`else` and conditional expressions on values; `for i in range(N)` with compile-time `N` (unrolled); calls to other `@pointwise` functions (inlined). No pointers, no aliasing, no heap, no `while`, no recursion. Anything else is a generator error with a code and the DSL line.
- **Loops are declarations, not code.** Example for scalar advection:

  ```python
  @pointwise
  def upwind(qL: State, qR: State, n: Geom.Vec3, area: Geom, a: Param.Vec3) -> State:
      un = dot(a, n)
      return area * (un * qL if un > 0.0 else un * qR)

  residual = face_loop(
      upwind,
      qL=cells.q.left, qR=cells.q.right,
      n=faces.normal, area=faces.area, a=params.velocity,
      out=Inc(cells.res, left=+1, right=-1),
  )
  ```

  The author never sees threads, colours, atomics, layout or precision.
- **Ahead-of-time.** The generator runs at build time (CMake custom command) and writes C++20 into the build tree; `nvcc`, `hipcc` and the host compiler build it like any other source. No run-time JIT in M1: it keeps builds reproducible (AGENTS non-negotiable 3) and avoids shipping NVRTC/hipRTC. PyFR shows run-time compilation works [S9]; we can add it later for case-specific specialisation (MFC reports ~10× from compile-time case optimisation [S16]).
- **Emitted code is plain and readable.** One function per pointwise body, templated on the precision policy; one kernel per loop and strategy; `#line` directives pointing at the `.py` source so compiler errors, `compute-sanitizer` and `cuda-gdb` report DSL lines (to verify in the spike); a serial CPU backend for stepping through in a host debugger.
- **The generator owns the data layout.** OP2's handles are opaque, so it picks SoA or AoS per platform [S10]; PyFR moved from SoA to AoSoA [S9]. We do the same: layout is a generator parameter, measured in the spike.

## 4. Automatic differentiation of kernels

### 4.1 The key choice: differentiate pointwise functions

A face kernel is a gather (cell states), a pure function (flux), and a scatter (residual increments). Only the pure function needs AD. The loop's adjoint follows mechanically from the access modes:

| Primal argument | Adjoint loop treats it as |
|---|---|
| `Read` through a map (gather) | `Inc` of its adjoint through the same map (scatter) |
| `Inc` through a map (scatter) | `Read` of its adjoint through the same map (gather) |
| `Write` direct | `Read` of its adjoint, then the adjoint is zeroed |
| `Read` direct (face data) | `Inc` direct of its adjoint (no race) |

So the adjoint of a face loop is again a face loop that reads cell data and increments cell data, and it reuses the race strategies of §6 unchanged. This is how the OP-DSL family does it: OPS builds a tape at the level of parallel loops, derives the race pattern of each adjoint loop from the reversed access pattern, and generates its CUDA and OpenMP versions; derivatives cost 3.7–9.7× the primal application [S13]. Rolls-Royce's Hydra has used Tapenade on its pointwise kernels since at least 2005 [S12]. Recent GPU work agrees: per-element differentiation kept in registers and shared memory beats PyTorch, JAX, Warp and Enzyme on mesh problems [S22].

Pointwise functions are the easy case for AD: no aliasing, no heap, bounded control flow. The reverse sweep recomputes the forward sweep in registers and stores nothing to global memory.

### 4.2 Tools, state in October 2026

| Tool | GPU status | Published cost (adjoint / primal) | Licence | Fit |
|---|---|---|---|---|
| **Enzyme** (LLVM plugin) | Reverse mode on CUDA and ROCm kernels shown at SC21 [S1]. CUDA guide still says "highly experimental", unchanged since 2021; `__enzyme_autodiff` on `__device__` code only [S2]. Releases still 0.0.x (v0.0.303, 2026-10-11) [S3]. CUDA requires compiling with clang, whose docs support CUDA 7.0–12.1 and treat newer SDKs as 12.1 [S4]; DGX Spark is CUDA 13.0, `sm_121` | 2.0× (LULESH), 3–4× (XSBench, RSBench), 6.3× (LBM), 5.4× on AMD and 18× on NVIDIA for the same DG kernel (register spills) [S1]; 3.6× with `__restrict__` vs 243× without on a Biot–Savart kernel [S5] | Apache-2.0 with LLVM exceptions | Works best on exactly our emitted functions, but couples the CUDA build to clang, the Enzyme version to the LLVM version, and HIP to ROCm's LLVM |
| **Tapenade** (source transformation, C/Fortran) | Partial CUDA support [S5]. OPS generated adjoints of all 87 CloverLeaf kernels with it; some needed local edits to remove Tapenade's branch stack before compiling for CUDA [S13] | — | MIT [S18] | Mature (Hydra, OPS); Java at generation time only; output needs post-processing for device code |
| **In-house transformation on our IR** | Whatever we emit | CuAD, a Tapenade-style tool for CUDA with the same conventions (recompute, thread-local arrays, atomics only where threads share memory): tangent 2.25–2.42×, adjoint 2.07–6.26× of primal, dot-product tests at 2–4×10⁻¹⁶ (RTX 5080, FP64) [S5] | ours | IR is already restricted; output is plain C++ for `nvcc` and `hipcc` |
| **CoDiPack** (operator overloading) | Reverse-mode tapes are CPU only; a forward type for CUDA kernels since v2.3.0 (2024-12-27) [S17] | — | GPL-3.0 (needs the lead) | No GPU reverse mode |
| **Forward dual numbers** (FUN3D) | FUN3D computes flux Jacobians on GPU by operator overloading with multivariate dual numbers; dimension 5 beat all other GPU AD variants tested; 75 % of peak FLOP/s and 61 % of peak bandwidth on V100 [S6] | Jacobian, not adjoint | — | Proven, any compiler; adjoint only as transposed local Jacobian |
| **Clad** (clang plugin) | CUDA support added recently | 60–4400×, fails beyond 8192 vortices [S5] | LGPL-3.0 | Rejected |

FUN3D: we found no publication of a GPU discrete adjoint; its published GPU AD is the forward-mode Jacobian above [S6, S7]. CuAD is two weeks old, single-author, CUDA-only and GPL-3.0: we cite its measurements, we do not depend on it.

### 4.3 Recommendation

1. **Primary: in-house tangent and adjoint transformation on the IR.** Tangent: forward sweep with dual values, vector mode with `k` directions (k = 5 gives the 5×5 BSR blocks of M2, as FUN3D found [S6]; k = 1 gives exact Jacobian-vector products for JFNK instead of finite differences). Adjoint: forward recomputation then reverse sweep, all in registers; conditions re-evaluated in the reverse sweep, which the subset guarantees is legal because a branch cannot modify its own condition (CuAD's rule [S5]). Loop-level adjoints by the table in §4.1. One mechanism serves implicit Jacobians (M2), JFNK (M2) and the adjoint (M6).
2. **Fallback (a): Tapenade** on the emitted C version of each pointwise function, post-processed by the generator into `__device__` code, as OPS did [S13]. Trigger: the in-house adjoint fails the spike criteria (§8) on one of the three spike kernels.
3. **Fallback (b): forward vector duals** to get the local Jacobian `∂F/∂(qL*, qR*)` (10 inputs for a 5-equation flux), adjoint = its transpose times the incoming adjoint. Costs more arithmetic than reverse mode but needs nothing beyond `nvcc`/`hipcc` and is what FUN3D runs in production [S6].
4. **Enzyme as oracle and benchmark, not dependency.** In CI on CPU, Enzyme's adjoint of the emitted host function is an independent check of ours. In the spike, one attempt to build an Enzyme adjoint with clang for `sm_121` and CUDA 13.0 records whether it is viable on our hardware.

This changes the fallback order in PLAN's risk table ("repli Enzyme ou adjoint écrit à la main"): Enzyme moves from fallback to oracle, Tapenade and dual numbers become the fallbacks. The lead decides in the review of this document.

**Oracles independent of the code under test** (AGENTS "Tests"): complex-step derivatives of the emitted CPU code with `std::complex<double>` (independent of our AD), central finite differences, and the dot-product identity ⟨ȳ, J ẋ⟩ = ⟨Jᵀ ȳ, ẋ⟩. The dot-product test alone checks the adjoint against our own tangent; the complex step checks the tangent.

**Known traps.** Non-smooth operations (`abs` in entropy fixes, `min`/`max` in limiters) give one-sided derivatives: the DSL provides smooth variants and a "frozen limiter" adjoint option, chosen explicitly. Register pressure in reverse sweeps caused Enzyme's 18× on CUDA [S1]; the generator reports registers and spills per emitted kernel. FP32 adjoints can be badly wrong on stiff problems (45–75 % gradient error on a stiff Burgers case in CuAD, versus < 5×10⁻⁵ with generated Jacobians) [S5]: adjoint accumulation defaults to FP64 until M6 measures otherwise.

## 5. What we copy and what we avoid

| Code | How it is organised | Copy | Avoid |
|---|---|---|---|
| **OP2 / Hydra** [S10, S11, S12] | C/C++ or Fortran API (sets, maps, datasets, `op_par_loop` with access modes); Python translator parses only API calls and emits CUDA, HIP, OpenMP, SYCL; BSD-3 | Access modes as the single source of parallel semantics; opaque data handles so the generator owns layout and halos; parse only what we define | Whole-program parsing (ROSE dropped); one thread per edge for register-heavy kernels: Hydra GPU performance was "not so good" because it "uses too many registers" [S10] |
| **OPS adjoint** [S13] | Loop-level tape; Tapenade on kernel bodies; adjoint loops generated with race handling from reversed access patterns | Loop-level tape and access-mode transposition | Hand-fixing Tapenade output per kernel |
| **PyFR** [S9] | Python outer layer; pointwise kernels in a Mako-based DSL, compiled at run time for CUDA, HIP, OpenCL, Metal, OpenMP; `reduce(op)` argument annotation turned into atomics; automatic shared-memory staging of reused read-only arguments; task graphs map to CUDA Graphs (256 s → 122 s on V100 for a 2D cylinder case); HIP graphs slower than plain launches, so emulated | Argument annotations that drive race handling; CUDA Graphs; emulate graphs on HIP until measured otherwise | Text templates without an IR: no AD, no fusion |
| **FUN3D / FLUDA** [S6, S7, S8] | ~500 lines of macros map one C++ source to CUDA, HIP, SYCL and serial CPU; kernels templated on physics; 2D thread blocks (Y = work items, X = threads within an edge) with autotuning; top 3 kernels at 78 %, 54 %, 62 % of A100 peak bandwidth (linear solver, viscous and inviscid Jacobians); warp aggregation of atomics 2.5–6× on MI100 | Hierarchical parallelism for Jacobian kernels (team of threads per face, block in shared memory, coalesced store); autotuned launch shape; dual-number Jacobians | Hand-written GPU variants per kernel; FP64 adjoint left on CPU |
| **SoNICS** [S14] | Operator graph built from user configuration; code generator from generic high-level code; linearised and adjoint code by differentiating the operator graph; CUDA Graphs; ×100 GPU vs CPU | Graph of operators as the unit of scheduling and of AD at the top level | Little is public (one NATO STO paper, undated): no detail to copy below the architecture |
| **MFC** [S15, S16] | Fortran + Fypp metaprogramming, OpenACC/OpenMP offload; one fused kernel for reconstruction, Riemann solve and flux with thread-local temporaries; FP16 storage / FP32 compute; ~10× from compile-time case specialisation | Full fusion inside a loop; mixed-precision storage as a build parameter | Directive compilers as the portability layer: native FP16/FP32 on AMD needed AMD Flang specifically, and half-precision atomics in NVHPC had "just been added" [S15] |

## 6. Face loops without the kernel author

The author declares `out=Inc(cells.res, left=+1, right=-1)`. The generator emits one or more strategies per loop and the runtime picks one per case and backend, logging the choice:

1. **Warp-aggregated atomics (default).** Faces are renumbered so that faces sharing an owner cell are contiguous (owner-sorted). Lanes of a warp with the same owner reduce their increments with a segmented shuffle reduction (head flags from comparing neighbours' keys), and the segment head issues one atomic per variable. The neighbour side is unsorted: lanes group by key with `__match_any_sync`, or issue plain atomics; the spike measures both. FUN3D's register-shuffle and shared-memory aggregation gave 2.5–6× over plain atomics on MI100 [S8]. HIP provides `__match_any_sync` and `__reduce_add_sync` with 64-bit masks; the `_sync` variants are on by default from ROCm 7.0 [S19]. Warp width (32 on NVIDIA and RDNA, 64 on CDNA) is a generator constant per target, never a literal in kernels.
2. **Hierarchical colouring (deterministic mode).** OP2's scheme: partition faces into blocks, colour blocks so no two in a launch share a cell, colour faces inside a block, stage increments in shared memory [S10, S11]. Bitwise reproducible, so it backs the deterministic mode required by AGENTS.md.
3. **Store-then-gather (deterministic, simplest).** The face loop writes fluxes to a face array; a cell loop over a CSR face list sums them in a fixed order. Costs one extra write and read of the flux per face, needs no colouring. Reference for correctness tests.

Cell loops that only gather (LSQ gradient over neighbours, limiter min/max) have no races and need no strategy. The adjoint of each loop (§4.1) is again one of these patterns, so the same three strategies apply to M6 unchanged.

Rules the generator enforces with coded errors: `Inc` targets must be in the accumulation type (no FP16/BF16 atomics; the HIP atomic table lists only integer, `float` and `double` [S19]); a loop with two `Inc` maps must declare both sides.

## 7. Precision as a parameter

Kernel arguments declare a **role**, not a C type: `State`, `Geom`, `Param`, `Jac`, `Accum`. A precision **policy** maps each role to a storage type, a compute type and, for `Inc` targets and reductions, an accumulation type:

| Role | `fp64` (reference) | `mixed` (default target) | `fp32` |
|---|---|---|---|
| `State` (solution, gradients) | f64 / f64 | f32 stored, f32 compute | f32 / f32 |
| `Geom` (centroids, normals, volumes) | f64 / f64 | f64 stored; differences like `x_f − x_c` formed in f64, then f32 | f32 / f32 |
| `Jac` (BSR blocks, preconditioners) | f64 / f64 | f16 or bf16 stored, f32 compute | f32 / f32 |
| `Accum` (residual increments, norms, dot products) | f64 | f64 | f32 |

The emitted pointwise function is `template<class P>`; loads convert from storage to compute type at the boundary, `Inc` and reductions accumulate in `P::accum`. Only policies the build lists are instantiated. A loop can override a role's type ("FP64 per kernel" from the PLAN risk table); the override is printed by `plan` and logged at run time (no silent fallback). FP16 vs BF16 for `Jac` is a policy choice to measure: FUN3D stores its Jacobian in FP16 ([state of the art §1.2](state-of-the-art-2026-10.md)); MFC found FP16 state storage visibly changed results where FP32 did not [S15], which is why `State` is never FP16 here.

## 8. Two-week spike

Goal: decide the M1 implementation path with measurements. Kernels: scalar upwind advection flux (face loop), central diffusion flux with non-orthogonal correction (face loop), Green-Gauss gradient (face loop, `Inc` of a `Vec3`), LSQ gradient (cell gather loop). Meshes: tet, hex, polyhedral, hybrid, four refinement levels each, through the M0 readers.

| Days | Work |
|---|---|
| 1–2 | Parser, IR, type check, serial C++ emitter; the four kernels; MMS on CPU in FP64 |
| 3 | CUDA emitter; the three face-loop strategies; first runs on `spark` |
| 4 | HIP emitter; build for `gfx942`; run where AMD access exists |
| 5 | Precision policies `fp64`, `fp32`, `mixed`; MMS in FP32 |
| 6–7 | Tangent and adjoint transformation; loop-level adjoint; complex-step oracle on CPU |
| 8 | Enzyme cross-check on CPU; one attempt at Enzyme + clang for `sm_121` / CUDA 13.0, result recorded |
| 9 | Benchmarks on `spark` under `flock`; fusion test (LSQ gradient + limiter in one cell loop); compile-time and register reports |
| 10 | Report against the criteria below; go / fallback decision |

Exit criteria, all measured and recorded with mesh hash, commit and precision:

| Area | Criterion |
|---|---|
| Bandwidth | Face flux and both gradients reach ≥ 70 % of the GB10 peak measured by a copy kernel on the same machine (spec 273 GB/s [S21]), bandwidth computed from an analytic minimum-bytes model of each loop, on meshes whose working set is ≥ 10× the L2 cache. ns/cell/iteration reported for each strategy (plain atomics, aggregated atomics, colouring, store-then-gather) and layout (SoA vs AoS) |
| MMS | Observed order ≥ 1.9 in L2 for the gradient and the advection-diffusion residual, on all four cell types, in FP64 and FP32 (FP32 measured on levels where the error is ≥ 100× the rounding floor) |
| Adjoint | Dot-product test relative gap ≤ 1×10⁻¹² in FP64 and ≤ 1×10⁻⁵ in FP32 for every kernel and the composed residual; tangent vs complex step ≤ 1×10⁻¹² in FP64; adjoint kernel time ≤ 3× primal (target), ≤ 6× (limit, the top of the published range [S1, S5]) |
| Determinism | Colouring and store-then-gather give bitwise-identical residuals over 10 runs and two launch configurations |
| Portability | CUDA and HIP built from the same DSL source with no per-backend edits; HIP results equal CPU results to rounding where AMD hardware is available |
| Build | Generator run ≤ 2 s for the spike kernels; rebuilding after editing one kernel ≤ 30 s; compile errors and `cuda-gdb` stops reported at the `.py` line |

Decision rules: if the in-house adjoint fails the adjoint row on any kernel by day 8, switch to fallback (a) and re-run; if no strategy reaches 50 % of peak bandwidth, the layout and renumbering are revisited before M1 continues.

## 9. Open questions

- AMD hardware for the spike and the CI runner (PLAN: "AMD via cloud", "runner AMD en CI dès M1"): without it, day 4 stops at compilation.
- Whether `cuda-gdb` honours `#line` into Python files well enough; if not, the debugging story rests on the serial CPU backend.
- FP64 atomic throughput on GB10 and on consumer Blackwell is unpublished; the spike measures it, since `Accum` is FP64 in the `mixed` policy.
- Atomics vs colouring on Hopper/Blackwell for unstructured CFD remains unstudied in public ([state of the art §4](state-of-the-art-2026-10.md)).

## 10. Sources

Accessed 2026-10-31 unless noted.

- [S1] W. S. Moses et al., "Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme", SC21, Nov 2021. https://dl.acm.org/doi/10.1145/3458817.3476165 (PDF: https://c.wsmoses.com/papers/EnzymeGPU.pdf)
- [S2] Enzyme, "CUDA Guide" (text dated 2021-01-20). https://enzyme.mit.edu/getting_started/CUDAGuide/
- [S3] EnzymeAD/Enzyme releases, v0.0.303, 2026-10-11; LICENSE. https://github.com/EnzymeAD/Enzyme/releases
- [S4] "Compiling CUDA with clang", LLVM 22.1.0 documentation. https://releases.llvm.org/22.1.0/docs/CompileCudaWithLLVM.html ; `sm_103`/`sm_121` target support with CUDA 12.9: https://github.com/llvm/llvm-project/commit/507b879b6ef09672322d6653411377fa65527160
- [S5] S. Joo, "CuAD: Source-to-Source Automatic Differentiation of CUDA Kernels for Discrete Adjoints in Computational Fluid Dynamics", arXiv:2610.26694, 2026-10-19. https://arxiv.org/abs/2610.26694 ; code (GPL-3.0): https://github.com/SEUNGHYUN-JOO/CuAD
- [S6] M. Zubair, D. Ranjan, A. Walden, G. Nastac, E. Nielsen et al., "Efficient GPU Implementation of Automatic Differentiation for Computational Fluid Dynamics", HiPC, Dec 2023. https://www.osti.gov/pages/biblio/1993463
- [S7] G. Nastac, A. Walden, L. Wang, E. Nielsen et al., "A Multi-Architecture Approach for Implicit Computational Fluid Dynamics on Unstructured Grids", AIAA SciTech 2023. https://ntrs.nasa.gov/api/citations/20220016937/downloads/2023_AIAA_SciTech_FLUDA_Paper.pdf
- [S8] "Accelerating unstructured-grid CFD algorithms on NVIDIA and AMD GPUs", IA3 workshop at SC21, 2021. https://fun3d.larc.nasa.gov/papers/SC21_AMD.pdf
- [S9] F. D. Witherden, P. E. Vincent et al., "PyFR v2.0.3: Towards Industrial Adoption of Scale-Resolving Simulations", Computer Physics Communications, 2025. https://www.sciencedirect.com/science/article/pii/S0010465525000700
- [S10] M. Giles, "Some reflections on automated code generation", NAIS workshop, 2014-07-01. https://www.cs.ox.ac.uk/files/13489/codegen.pdf
- [S11] OP-DSL/OP2-Common (BSD-3-Clause), Python translators in `translator/`. https://github.com/OP-DSL/OP2-Common
- [S12] M. Giles, airfoil test case for automatic differentiation (Hydra uses Tapenade for linear and adjoint kernels), Dec 2005. https://people.maths.ox.ac.uk/~gilesm/codes/airfoil/index.html
- [S13] G. D. Balogh, J. Lotz, J. du Toit, U. Naumann, I. Reguly, "Performance Portable Adjoints for Structured Mesh Applications with OPS", ACM TOMS 52(3). https://dl.acm.org/doi/10.1145/3815776 (draft: https://real.mtak.hu/246209/1/OPS_AD_Draft.pdf)
- [S14] ONERA, "SoNICS: a New Generation CFD Software for Satisfying Industrial Users Needs", NATO STO-MP-AVT-366, paper 10 (undated). https://publications.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-AVT-366/MP-AVT-366-10.pdf
- [S15] B. Wilfong et al., "Simulating many-engine spacecraft: Exceeding 1 quadrillion degrees of freedom via information geometric regularization", SC25 Gordon Bell finalist, arXiv:2505.07392, 2025. https://arxiv.org/abs/2505.07392
- [S16] MFC 5.0, arXiv:2503.07953, 2025; repository (MIT). https://arxiv.org/abs/2503.07953 , https://github.com/MFlowCode/MFC
- [S17] SciCompKL/CoDiPack changelog (v2.3.0, 2024-12-27; v3.1.3, 2026-10-26), `include/codi/tools/cuda/`, LICENSE (GPL-3.0). https://github.com/SciCompKL/CoDiPack
- [S18] Tapenade (MIT licence). https://gitlab.inria.fr/tapenade/tapenade
- [S19] "HIP C++ language extensions", HIP 7.16 documentation. https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- [S20] rapidsai/cudf issue #24661, "transform_kernel JIT compilation time and PCH footprint", 2026-10-17 (NVRTC 13.3.33, CCCL 3.6.0). https://github.com/rapidsai/cudf/issues/24661
- [S21] NVIDIA, "Hardware Overview — DGX Spark User Guide". https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- [S22] A. H. Mahmoud, R. Goel, J. Ragan-Kelley, J. Solomon, "Locality-Aware Automatic Differentiation on the GPU for Mesh-Based Computations", ACM TOG, July 2026 (arXiv:2509.00406). https://arxiv.org/abs/2509.00406
