# Kernel generator — design research for M1

Status: proposal for the M1 gate, 2026-10-31. Scope: the `kernels/` and `codegen/` trees of [`docs/PLAN.md`](../PLAN.md). Background facts already in [`state-of-the-art-2026-10.md`](state-of-the-art-2026-10.md) are linked, not repeated. Numbers come from the sources in [§10](#10-sources); each is tagged `[Sn]`.

## 1. Decisions in one table

| Question | Recommendation | Fallback | Section |
|---|---|---|---|
| DSL shape | Python-embedded DSL: pointwise bodies and neighbour reductions in a restricted Python subset, parsed into our own SSA IR; loops declared with OP2-style access modes; header-light C++20 emitted ahead of time for CUDA, HIP and CPU | Same loop descriptors, bodies hand-written as templated `__host__ __device__` C++ (FLUDA style) with dual-number tangents | §3 |
| Kernel AD | Differentiate loop bodies, not kernels. Tangent and adjoint by source transformation on our IR; the residual's adjoint is the reverse sequence of loop adjoints, recorded at loop level | (a) Tapenade on emitted C bodies; (b) forward dual numbers for every body, adjoint by transposed local Jacobians | §4 |
| Enzyme | Independent oracle and benchmark, not a dependency | — | §4.2 |
| Face loop | Generator emits several race strategies per loop; the default per backend is chosen by measurement, not assumed | — | §6 |
| Precision | Arguments carry a precision role; a policy maps roles to storage, compute and accumulation types, including adjoint types; policies are template parameters of the emitted code | Per-loop FP64 override, logged | §7 |
| Fusion | Within a loop only in M1 (reconstruction → flux → scatter; gradient → bounds → limiter in one cell kernel; residual + Jacobian); CUDA Graphs for launch overhead | — | §3.3, §5 |

## 2. Constraints from the plan and the rules

- Pointwise face and cell kernels written once, emitted as native CUDA and HIP plus CPU, with tangent and adjoint ([PLAN](../PLAN.md), "Décisions figées").
- FP32 compute, FP16/BF16 storage of Jacobians and preconditioners, FP64 for accumulation, norms and geometry.
- Polyhedral cells with 100+ faces (M0 gate), so cell loops have variable valence.
- M1 gate: MMS order 2 on tet, hex, poly and hybrid meshes in FP32 and FP64; ≥ 70 % of peak bandwidth on gradient and flux; adjoint dot-product test; AMD runner in CI.
- Dev target is GB10 (DGX Spark): 273 GB/s LPDDR5X shared by CPU and GPU [S21]; CUDA 13.0. Production targets add B200/GB200 and MI300X/MI355X (wave64).
- AGENTS.md: bitwise-deterministic mode; provenance (mesh hash, case hash, commit, precision) on every run; no silent fallback; licences permissive or weak copyleft, GPL needs the lead.

## 3. DSL shape

### 3.1 Options

**A. Python-embedded DSL with its own IR.** Bodies are Python functions in a restricted subset, parsed with the standard `ast` module into a typed SSA IR. Loops are declared separately with access descriptors. The generator emits C++ source. Precedents: OP2 (Python translator that parses only the API calls [S10, S11]), PyFR (Mako templates of C-like bodies, compiled at run time [S9]), SoNICS (generic high-level code → per-target code, plus AD of an operator graph [S14]).

**B. Templated C++ bodies with the same loop descriptors.** Bodies are `template<class Real> __host__ __device__` functions; loops are declared with the same access descriptors (in Python or in C++), so race handling, layout and loop-level adjoints are generated exactly as in A. Precedent for the bodies: FUN3D's FLUDA, about 500 lines of preprocessor macros mapping one source to CUDA, HIP, SYCL or serial CPU, kernels templated on the physics set [S7]; FUN3D gets flux Jacobians by instantiating them on dual-number types [S6].

Option B without external descriptors (a C++ library such as Kokkos that only sees an opaque functor) is not considered further: the generator could not derive races or adjoint loops from it.

### 3.2 Comparison

With the descriptors shared, the two options differ only in how bodies are written and differentiated.

| Criterion | A. Python bodies, own IR | B. C++ bodies |
|---|---|---|
| Reverse-mode AD of bodies | Own transformation on a restricted IR (§4) | Enzyme (CUDA only through clang, §4.2), Tapenade on C, or forward duals (adjoint by transposed local Jacobian, more arithmetic) |
| Fusion inside a loop | IR-level: inline calls, drop intermediates | Manual: the author writes the fused body |
| Precision roles (§7) | Casts inserted by the generator at each operation that needs them | Author writes casts, or a type wrapper enforces them |
| Readability | Physics only; no templates | Physics plus template syntax; to be fast on GPU, FLUDA needed a hand-written hierarchical variant (7.6× over the naïve kernel on V100) [S7] |
| Compile time | Emits only the (body × policy × backend) instances a build asks for | Same if instantiation is explicit and headers are kept light |
| Debugging | Through emitted code: needs `#line` mapping and a serial CPU backend | Native: IDE, `cuda-gdb`, compiler errors at the source line |
| Cost to build | Parser, type check, IR, emitters, AD | Near zero for the forward path |

Compile time depends mostly on included headers, not on the language bodies are written in: NVRTC 13.3 compiles a trivial cuDF transform kernel in ~1.0 s with the CCCL headers and in 68 ms with a minimal header set [S20]. Both options must therefore emit or include header-light code; this measurement does not separate A from B.

Two historical lessons bear on A. Giles, after building OP2, warned against C++ metaprogramming whose errors are "strange cryptic compiler error messages" and which needs "guru-standard" developers; OP2 abandoned its ROSE whole-program parser because it "could not cope with the complexity of code in HYDRA", while the Python generator that parses only API calls survived [S10]. A must stay a small language, not a Python compiler.

### 3.3 Recommendation

Option A, because reverse-mode AD and in-loop fusion are what the plan needs the generator for, and both are cheap on an IR we own and costly on C++ bodies. Option B is the fallback for bodies; the spike measures the effort and performance gap on one flux (§8).

- **Restricted subset.** Scalars and fixed-size `Vec`/`Mat` (`Vec3`, `Vec5`, `Mat5`); arithmetic; a whitelist of math functions; `if`/`else` and conditional expressions; `for i in range(N)` with compile-time `N` (unrolled); calls to other `@pointwise` functions (inlined). No pointers, no aliasing, no heap, no `while`, no recursion. Variables are single-assignment after parsing (SSA), so a branch cannot modify the value its predicate was computed from. Anything else is a generator error with a code and the DSL line.
- **Variable-valence reductions.** Cell loops iterate over a CSR incidence list (faces or neighbours of a cell) with declared accumulators and a finaliser:

  ```python
  @cell_reduce(over=cells.neighbours, acc={"g": Sum(Vec3), "qmin": Min(State), "qmax": Max(State)})
  def lsq_gradient_and_bounds(c, nb, acc, w: Geom.Vec3 = incidence.lsq_weight):
      dq = q[nb] - q[c]
      acc.g += w * dq
      acc.qmin = min(acc.qmin, q[nb]); acc.qmax = max(acc.qmax, q[nb])

  @finalise(lsq_gradient_and_bounds)
  def limiter(c, acc) -> (Vec3, State):
      return acc.g, venkatakrishnan(q[c], acc.g, acc.qmin, acc.qmax, geometry[c])
  ```

  The iteration count is a run-time value; the body is still restricted. Accumulators start at their identity (0 for `Sum`, +∞ for `Min`, −∞ for `Max`). LSQ weights are precomputed per incidence (geometry, passive for AD). A second reduction over the same incidence list may read the finalised values of the first: Venkatakrishnan's φ = min over faces of φ_f(∇q_c · r_f, bounds) needs the cell's own finalised gradient and bounds, so gradient, bounds and limiter run as two passes in one cell kernel. This form also covers Green-Gauss gradients and cell-based viscous terms.
- **AD rules for reductions.** `Sum`: the accumulator's adjoint is broadcast to every incidence. `Min`/`Max`: the adjoint goes to the selected incidence only; ties select the first incidence in list order, which the reverse sweep recomputes. Finaliser and pass bodies are differentiated like any body; a two-pass kernel is reversed pass by pass.
- **Face loops are declarations.** Example, second-order upwind advection:

  ```python
  @pointwise
  def upwind(qc: State, gc: State.Vec3, rc: Geom.Vec3,
             qn: State, gn: State.Vec3, rn: Geom.Vec3,
             n: Geom.Vec3, area: Geom, a: Param.Vec3) -> State:
      qL = qc + dot(gc, rc)          # reconstruction to the face
      qR = qn + dot(gn, rn)
      un = dot(a, n)
      return area * (un * qL if un > 0.0 else un * qR)

  residual = face_loop(
      upwind,
      qc=cells.q.left, gc=cells.grad.left, rc=faces.r_left,
      qn=cells.q.right, gn=cells.grad.right, rn=faces.r_right,
      n=faces.normal, area=faces.area, a=params.velocity,
      out=Inc(cells.res, left=+1, right=-1),
  )
  ```

  The author never sees threads, colours, atomics, layout or precision.
- **Ahead of time.** The generator runs at build time (CMake custom command) and writes C++20 into the build tree; `nvcc`, `hipcc` and the host compiler build it. No run-time JIT in M1: builds stay reproducible (AGENTS non-negotiable 3) and we ship no NVRTC/hipRTC. PyFR shows run-time compilation works [S9]; case-specific specialisation can come later (MFC reports ~10× from compile-time case optimisation [S16]).
- **Readable output.** One function per body, templated on the precision policy; one kernel per loop and strategy; `#line` directives to the `.py` source (whether `cuda-gdb` and `compute-sanitizer` honour them is a spike item); a serial CPU backend for host debuggers.
- **The generator owns data layout.** OP2's handles are opaque so it chooses SoA or AoS per platform [S10]; PyFR moved from SoA to AoSoA [S9]. Layout is a generator parameter, measured in the spike.

## 4. Automatic differentiation

### 4.1 Differentiate bodies; derive loop adjoints

A face loop is a gather, a body, and a scatter. The residual is a sequence of loops (gradient + limiter cell loop → face flux loop → boundary loops). Its exact Jacobian-vector product is the sequence of loop tangents; its exact vector-Jacobian product (the adjoint) is the reverse sequence of loop adjoints, each of which needs the body's adjoint plus these rules:

| Primal argument | Adjoint loop |
|---|---|
| Active `Read` through a map (gather) | `Inc` of its adjoint through the same map (scatter) |
| `Inc(target, left=+1, right=−1)` | Body's output adjoint is `s_L·t̄[L] + s_R·t̄[R]` with the declared signs, here `t̄[L] − t̄[R]`; `t̄` itself is passed on unchanged (an increment's adjoint is the identity on the accumulator) |
| `Write` direct | Read the adjoint, then zero it (the overwritten value had no influence) |
| Overwritten input needed by the reverse sweep | Saved before the primal loop or recomputed; recorded on the loop-level tape |
| Active `Read` of a global parameter | Global reduction of its adjoint |
| Passive arguments (geometry, LSQ weights, unless shape derivatives are requested) | No adjoint |

Active and passive arguments are declared on the loop, not inferred. OPS works the same way: a tape at the level of parallel loops, saved states for overwritten data, active global scalars, adjoint loops generated with their own race handling; derivatives cost 3.7–9.7× the primal application on CPUs and GPUs [S13]. Rolls-Royce's Hydra has used Tapenade on its pointwise kernels since at least 2005 [S12]. Recent GPU work keeps per-element derivatives in registers and shared memory with forward-mode operator overloading, and beats PyTorch, JAX, Warp and Enzyme on mesh problems [S22].

Reversing a loop changes its write set: the adjoint of a cell gather scatters into neighbours, so two cells that do not share a face can write the same adjoint entry. The generator recomputes the conflict structure of every adjoint loop from its reverse write set (§6), rather than reusing the primal's.

### 4.2 Tools, state in October 2026

| Tool | GPU status | Published cost | Licence | Fit |
|---|---|---|---|---|
| **Enzyme** (LLVM plugin) | Reverse mode of CUDA and ROCm kernels shown at SC21 [S1]. CUDA guide still says "highly experimental", unchanged since 2021; `__enzyme_autodiff` on `__device__` code only [S2]. Releases still 0.0.x (v0.0.303, 2026-10-11) [S3]. CUDA requires compiling device code with clang: in LLVM 22.1.0, CUDA 12.8 is fully supported, 12.9 partially, 13.0 is accepted with a "too new" warning [S4]; DGX Spark runs CUDA 13.0 on `sm_121` | Forward + reverse vs forward: 2.0× (LULESH), 3–4× (XSBench, RSBench), 6.3× (LBM), 5.4× on AMD and 18× on NVIDIA for the same DG kernel (register spills) [S1]; 3.6× with `__restrict__` vs 243× without on a Biot–Savart kernel [S5] | Apache-2.0 with LLVM exceptions | Best case is exactly our emitted functions, but it ties the CUDA build to clang, Enzyme to one LLVM version, HIP to ROCm's LLVM |
| **Tapenade** (source transformation, C/Fortran) | Partial CUDA support [S5]. OPS generated adjoints of all 87 CloverLeaf kernels with it; some needed local edits to remove Tapenade's branch stack before compiling for CUDA [S13] | — | MIT [S18] | Mature (Hydra, OPS); Java at generation time only; device output needs post-processing |
| **In-house transformation on our IR** | Whatever we emit | CuAD, a Tapenade-style CUDA tool with similar conventions (recompute, thread-local arrays, atomics only where threads share memory): tangent 2.25–2.42×, adjoint 2.07–6.26× of primal, dot-product tests at 2–4×10⁻¹⁶ (RTX 5080, FP64) [S5] | ours | IR already restricted; output is plain C++ for `nvcc` and `hipcc` |
| **CoDiPack** (operator overloading) | Reverse-mode tapes are CPU only; a forward type for CUDA kernels since v2.3.0 (2024-12-27) [S17] | — | GPL-3.0 (needs the lead) | No GPU reverse mode |
| **Forward dual numbers** (FUN3D) | Flux Jacobians on GPU by operator overloading with multivariate dual numbers; dimension 5 beat the other GPU AD variants tested; 75 % of peak FLOP/s and 61 % of peak bandwidth on V100 [S6] | Jacobian, not adjoint | — | Proven with any compiler |
| **Clad** (clang plugin) | CUDA support added recently | 60–4400×; fails beyond 8192 vortices [S5] | LGPL-3.0 | Rejected |

FUN3D: we found no publication of a GPU discrete adjoint; its published GPU AD is the forward-mode Jacobian [S6, S7]. CuAD is two weeks old, single-author, CUDA-only and GPL-3.0: we cite its measurements, we do not depend on it.

### 4.3 Recommendation

1. **Primary: in-house tangent and adjoint on the IR.** Tangent: dual values, vector mode with `k` directions. Adjoint: forward recomputation, then reverse sweep; predicates are SSA values reused by the reverse sweep; values from neighbour reductions are recomputed or saved per the loop-level tape. Register residency is not guaranteed: the generator reports registers and spills for every emitted kernel, and saves to local memory when recomputation would spill more.
2. **Uses of the same mechanism.**
   - *Exact operator* (adjoint in M6, Jacobian-vector products for JFNK in M2): full tangent or adjoint of the residual chain, including reconstruction, gradients, limiters and non-orthogonal corrections.
   - *Preconditioner Jacobian* (BSR 5×5 in M2): usually an approximation (first-order stencil, frozen gradients). For a 5-equation face flux, the two 5×5 blocks `∂F/∂q_L` and `∂F/∂q_R` need 10 tangent directions (one pass of 10 or two of 5; FUN3D found 5 best on V100 [S6]). Assembly into owner and neighbour block rows follows the same sign rules as §4.1.
3. **Fallback (a): Tapenade** on the emitted C version of each body, post-processed into device code, as OPS did [S13]. Trigger: a correctness failure of the in-house adjoint (§8) that is not fixed by day 8.
4. **Fallback (b): forward duals everywhere.** Each body's local Jacobian with respect to all its active inputs (for `upwind` above: `qc, gc, qn, gn`, 8 inputs for a scalar, 40 for 5 equations), adjoint = transposed local Jacobian times the incoming adjoint; the loop rules of §4.1 are unchanged. More arithmetic than reverse mode, but nothing beyond `nvcc`/`hipcc`, and FUN3D runs the forward half in production [S6].
5. **Enzyme as oracle and benchmark.** In CI on CPU, Enzyme's adjoint of the emitted host function cross-checks ours. In the spike, one attempt with clang for `sm_121` and CUDA 13.0, with pinned LLVM, Enzyme and toolkit versions, records whether it builds and runs on our hardware.

This reorders the fallbacks in PLAN's risk table ("repli Enzyme ou adjoint écrit à la main"): Enzyme becomes an oracle, Tapenade and dual numbers become the fallbacks. The lead decides in the review of this document.

**Non-smooth operations.** `abs` in entropy fixes, `min`/`max` in limiters and upwind switches have no classical derivative at kinks. Convention: the linearisation of the branch the primal selected, with ties resolved as the source is written (`un > 0.0` sends `un = 0` to the right state; the DSL's `abs(x)` is `x if x >= 0 else -x`, slope +1 at 0; `min`/`max` ties as in reductions above). The tangent and adjoint are exact for that selected-branch linearisation, which is what the dot-product test checks; away from kinks it is the classical derivative. Smooth variants (Venkatakrishnan limiter, smoothed `abs`) are separate DSL functions the author chooses. A "frozen limiter" adjoint treats the limiter as passive: it is the exact adjoint of a different, frozen primal, and the generator labels it approximate in logs and results.

**Oracles independent of the code under test** (AGENTS "Tests"):
- Analytic: LSQ and corrected Green-Gauss gradients are exact for linear fields; hand-derived derivatives of the scalar upwind and central diffusion fluxes.
- Conservation: on a periodic domain Σᵢ Rᵢ(q) = 0 for every q, hence 1ᵀJ = 0: the primal residual sums to zero and the adjoint with seed ȳ = 1 returns zero, both to rounding. With boundaries, the residual sum equals the boundary flux sum.
- Complex step on the emitted CPU code with `std::complex<double>`, using the standard extensions (`abs(z) = z·sign(Re z)`, comparisons on the real part) [S24]; valid only along smooth paths, so tests avoid ties.
- Dot-product identity ⟨ȳ, J ẋ⟩ = ⟨Jᵀ ȳ, ẋ⟩ (adjoint against our tangent), and central finite differences as a coarse check.
- Complex step, finite differences and the dot-product test all run on generated code and cannot catch an error the generator makes in the primal; the analytic and conservation tests can.

**Precision of derivatives.** CuAD reports adjoint gradient errors of 45–75 % on a stiff Burgers problem from FP32 **finite-difference** Jacobians, even with FP64 linear solves, against < 5×10⁻⁵ from FP32 Jacobians assembled from generated tangents [S5]. So FP32 differentiated bodies are viable. Finite differences in FP32 lost most of their accuracy in that stiff case; whether they suffice for JFNK depends on conditioning and step scaling, and tangent-mode products remove the step choice altogether, so M2 starts with tangents. Adjoint accumulation precision is a policy entry (§7), measured in M6.

## 5. What we copy and what we avoid

| Code | How it is organised | Copy | Avoid |
|---|---|---|---|
| **OP2 / Hydra** [S10, S11, S12] | C/C++ or Fortran API (sets, maps, datasets, `op_par_loop` with access modes); Python translator parses only API calls and emits CUDA, HIP, OpenMP, SYCL; BSD-3 | Access modes as the single source of parallel semantics; opaque handles so the generator owns layout and halos; parse only what we define | Whole-program parsing (ROSE dropped); one thread per edge for register-heavy kernels: Hydra on GPU was "not so good" because it "uses too many registers" [S10] |
| **OPS adjoint** [S13] | Loop-level tape; Tapenade on bodies; adjoint loops with race handling from reversed access patterns; saved states | Loop-level tape, access-mode transposition, explicit active globals | Hand-fixing Tapenade output per kernel |
| **PyFR** [S9] | Python outer layer; pointwise kernels in a Mako-based DSL compiled at run time for CUDA, HIP, OpenCL, Metal, OpenMP; `reduce(op)` argument annotation turned into atomics; automatic shared-memory staging of reused read-only arguments; task graphs map to CUDA Graphs (256 s → 122 s on V100 for a 2D cylinder); HIP graphs slower than plain launches, so emulated | Argument annotations that drive race handling; CUDA Graphs; emulate graphs on HIP until measured otherwise | Text templates without an IR: no AD, no fusion |
| **FUN3D / FLUDA** [S6, S7, S8] | ~500 lines of macros map one C++ source to CUDA, HIP, SYCL and serial CPU; kernels templated on physics; 2D thread blocks (Y = work items, X = threads within an edge) with autotuning; top 3 kernels at 78 %, 54 %, 62 % of A100 peak bandwidth (linear solver, viscous and inviscid Jacobians); several atomic-update strategies benchmarked per GPU (§6) | Hierarchical parallelism for Jacobian kernels (a team of threads per face, block in shared memory, coalesced store); autotuned launch shape; dual-number Jacobians; strategy chosen per GPU by measurement | Hand-written GPU variants per kernel |
| **SoNICS** [S14] | Operator graph built from user configuration; code generator from generic high-level code; linearised and adjoint code by differentiating the operator graph; ×100 GPU vs CPU | Operator graph as the unit of scheduling and of top-level AD | Little is public (one NATO STO paper, undated): nothing to copy below the architecture |
| **MFC** [S15, S16] | Fortran + Fypp metaprogramming, OpenACC/OpenMP offload; reconstruction, Riemann solve and flux in one kernel with thread-local temporaries; FP16 storage / FP32 compute; ~10× from compile-time case specialisation | Full fusion inside a loop; storage precision as a build parameter | Directive compilers as the portability layer: native FP16/FP32 on AMD needed AMD Flang specifically, and half-precision atomics in NVHPC had "just been added" [S15] |

## 6. Race strategies without the kernel author

The author declares `out=Inc(...)`. The generator emits several strategies per loop; the runtime picks one per backend and case from a table filled by benchmarks, and logs the choice.

FUN3D's measurements show why the default cannot be fixed in advance [S8, Tables III–IV, CUDA 11.2 / ROCm 4.2, 1.1 M-node mesh, FP64]:

| Kernel, method (speed-up vs plain global atomics) | V100 | A100 | MI100 |
|---|---|---|---|
| Edge Roe flux, shared-memory transposition (coalesced atomics) | 1.64× | 1.73× | 1.81× |
| Edge Roe flux, warp aggregation (left and right) | n/a | 1.72× | 2.40× |
| LSQ gradient, shared-memory transposition | 2.20× | 2.12× | 2.78× |
| LSQ gradient, warp aggregation | 1.27× | 1.98× | 4.40× |
| LSQ gradient, hybrid aggregation + transposition | 1.26× | 2.94× | 6.01× |

Warp aggregation pays most where FP64 atomics are slow (MI100 had no FP64 atomic-add hardware) and helps little on V100; in a cell-based viscous kernel it slowed V100 to 0.49× [S8, Table V]. Strategies the generator emits:

1. **Plain atomics.** Baseline, always available.
2. **Shared-memory transposition.** A team of threads per face stages the increments in shared memory, then contiguous threads issue the atomics for the owner and neighbour cells (FUN3D's original roe-flux method [S8]).
3. **Warp aggregation.** Faces are renumbered owner-sorted, so equal owners are contiguous in a warp; a segmented shuffle reduction (head flags from comparing neighbour keys) leaves one atomic per segment and variable. Neighbour keys are unsorted: lanes group with `__match_any_sync` or fall back to transposition. Reductions are written with shuffles in FP32/FP64: CUDA's `__reduce_add_sync` is integer-only, and on AMD floating-point types need `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES` [S19]. Masks are 64-bit on HIP [S19]; warp width (32 on NVIDIA and RDNA, 64 on CDNA) is a per-target constant. Every lane of the warp takes part in each sync primitive with the full mask; lanes past the end of the face range carry the identity value and a sentinel key. Tests cover partial warps, segments crossing warp boundaries, and neighbour keys that match in non-contiguous lanes.
4. **Hierarchical colouring.** OP2's scheme: blocks of faces coloured so no two blocks in a launch share a target, faces coloured within a block, increments staged in shared memory [S10, S11]. Race-free without atomics.
5. **Store-then-gather.** The face loop writes fluxes to a face array; a cell loop over the CSR face list sums them in a fixed order. One extra write and two reads of the flux per interior face, no colouring.
6. **Gather-recompute.** A cell loop computes every face flux of the cell itself: two flux evaluations per interior face, no atomics, no face array. Candidate for cheap fluxes (scalar advection, Green-Gauss).

Adjoint loops get their own conflict analysis (§4.1): the adjoint of a cell gather is a scatter to neighbours, so colouring needs distance-2 colours on the cell graph, and strategy 5 needs per-incidence storage with a transposed incidence list.

**Determinism.** Strategies 4–6 sum in an order fixed by the mesh, the partition and the face/cell numbering, independent of the launch configuration. The deterministic mode guarantees bitwise-identical results for a given mesh, partition, build and strategy, across runs and launch shapes; it does not promise equality across strategies or partitions. Norms and dot products use a fixed-order tree reduction in the same mode.

Rules the generator enforces with coded errors: `Inc` targets are in their accumulation type, never FP16/BF16 (the HIP atomic table lists integer, `float` and `double` only [S19]); a two-sided `Inc` declares both signs.

## 7. Precision as a parameter

Arguments declare a role, not a C type. A policy maps each role to a storage type, a compute type and an accumulation type; adjoint variables have their own entry:

| Role | `fp64` (reference) | `mixed` (default target) | `fp32` |
|---|---|---|---|
| `State` (solution) | f64 | store f32, compute f32 | f32 |
| `Grad` (gradients, limiter values) | f64 | store f32; Green-Gauss `Inc` accumulates in f64 scratch, then converts | f32 |
| `Residual` (`Inc` targets of flux loops) | f64 | accumulate f64 | f32 |
| `Geom` (centroids, normals, volumes, LSQ weights) | f64 | store f64; `rel(x_f, x_c)` subtracts in f64, then converts to f32 | f32 |
| `Param` (free-stream values, constants) | f64 | store f64, compute f32 | f32 |
| `Jac` (BSR blocks, preconditioners) | f64 | store bf16 or fp16 with an f32 scale per block, compute f32 | f32 |
| Adjoint of each role | f64 | accumulate f64, store f64 until M6 measures f32 | f32 |
| Reductions (norms, dot products) | f64 | f64 | f64 |

The emitted body is `template<class P>`; loads convert storage → compute, `Inc` and reductions use `P::accum`, and the operations that need more than the compute type (`rel` on geometry, reductions) are DSL functions with defined promotion, not implicit casts. Only policies the build lists are instantiated. A loop can override a role's type ("FP64 per kernel" in the PLAN risk table); `plan` prints the override and the run logs it.

`Jac` storage serves only the preconditioner. Each block is scaled by its largest magnitude before conversion; values that overflow or underflow the 16-bit format are counted and logged per assembly. Convergence and adjoint checks always use the exact operator (tangent or adjoint of the residual chain) and FP64 residual norms, never the stored `Jac`. FP16 vs BF16 is measured: FUN3D stores its Jacobian in FP16 ([state of the art §1.2](state-of-the-art-2026-10.md)); MFC found FP16 storage of the state visibly changed results where FP32 did not [S15], which is why `State` is never 16-bit here.

## 8. Two-week spike

The spike chooses the implementation path. It does not close M1: the M1 gate keeps PLAN's criteria, including ≥ 70 % of peak bandwidth (50–70 % does not pass) and an AMD runner in CI.

Kernels: second-order upwind advection face loop (§3.3), central diffusion with non-orthogonal correction (face loop), LSQ gradient + limiter bounds (cell reduction), Green-Gauss gradient (face loop, `Inc` of a `Vec3`), and, for AD only, a 3D Euler Roe flux body with entropy fix (nonlinear, branches, `Vec5`, register pressure). Meshes: tet, hex, polyhedral (including cells with 100+ faces), hybrid; four refinement levels each, through the M0 readers.

| Days | Work |
|---|---|
| 1–2 | Parser, IR, type check, serial C++ emitter; face loops and cell reductions; MMS and analytic gradient tests on CPU in FP64 |
| 3 | CUDA emitter; strategies 1–6 of §6; first runs on `spark` |
| 4 | HIP emitter; build for `gfx942`; run where AMD access exists |
| 5 | Policies `fp64`, `fp32`, `mixed`; MMS in FP32 and `mixed` |
| 6–7 | Tangent and adjoint of bodies and loops, Roe flux included; analytic, conservation, complex-step and dot-product tests |
| 8 | Enzyme cross-check on CPU; one pinned attempt with clang for `sm_121` / CUDA 13.0 |
| 9 | Benchmarks; option B comparison (Roe flux and its 10-direction dual-number Jacobian hand-written in C++ against the generated ones: lines written, time, registers); compile time; `#line` debugging check |
| 10 | Report against the criteria; go or fallback decision |

Every run records mesh hash, case hash, commit and precision policy. Timings come from the `bench/` harness on `spark` under `flock /tmp/cfd-gpu.lock`, with the GPU checked idle, per AGENTS.md. The spike's results become the first tracked benchmarks, so later changes fall under the 5 % regression rule.

| Area | Criterion |
|---|---|
| Bandwidth | For the advection face loop and the LSQ cell loop in the `fp32` and `mixed` policies: useful bytes (analytic minimum per loop) divided by the complete time of the strategy (all its kernels, e.g. both passes of store-then-gather) ≥ 70 % of the GB10 peak measured by a copy kernel on the same machine (spec 273 GB/s [S21]). Measured DRAM traffic is reported beside it for diagnosis; extra traffic cannot pass the gate because the numerator is fixed. `fp64` is reported only (GB10 FP64 throughput is a separate ceiling). Working set ≥ 10× L2. Reported for every strategy, layout (SoA, AoS) and policy, with ns/cell/iteration and the one-off cost of renumbering and colouring |
| MMS | Steady advection-diffusion with second-order reconstruction: observed order of the discrete solution error in L2 ≥ 1.9 on all four cell types, in FP64, FP32 and `mixed` (FP32 measured on levels where the error is ≥ 100× the rounding floor). Gradient orders reported separately against the literature: on meshes where the line between cell centroids misses the face centroid, LSQ is first order and the common Green-Gauss variant (linear interpolation to the face) is zeroth order [S23]; "corrected" Green-Gauss below means the iterative skewness correction of [S23] |
| Analytic and conservation | LSQ and corrected Green-Gauss exact to rounding on linear fields; periodic residual sum and adjoint of ȳ = 1 zero to rounding. At least the composed residual and its adjoint are exercised through a C++ solver-API function, not only through generated-kernel test drivers |
| Adjoint correctness | Dot-product gap `|⟨ȳ,Jẋ⟩ − ⟨Jᵀȳ,ẋ⟩| / (‖ȳ‖‖Jẋ‖ + ‖Jᵀȳ‖‖ẋ‖)` ≤ 1×10⁻¹³ in FP64, ≤ 1×10⁻⁵ in FP32 and `mixed`, random seeds, every loop and the composed residual; dedicated cases for branch merges (Roe entropy fix), loop-carried values (unrolled loops, reductions with ties) and names reassigned before a predicate (SSA renaming); tangent vs complex step ≤ 1×10⁻¹² relative in FP64 on smooth paths. A failure here triggers fallback (a) if unfixed on day 8 |
| Adjoint cost | Adjoint loop time alone / primal loop time alone: target ≤ 3×; above 6× is investigated before M6 (published range: 2.07–6.26× for CuAD [S5]; forward + reverse 2.0–18× for Enzyme [S1]). Not a fallback trigger on its own |
| Determinism | Strategies 4–6: bitwise-identical residual, gradient, norms and adjoint over 10 runs and two launch configurations |
| Portability | CUDA and HIP built from the same DSL source with no per-backend edits; HIP equals CPU to rounding where AMD hardware is available |
| Build | Generator ≤ 2 s for the spike kernels; rebuild after editing one body ≤ 30 s; compile errors reported at the `.py` line |

If no strategy reaches 50 % of peak on day 9, layout and renumbering are revisited before M1 continues.

## 9. Open questions

- AMD hardware for the spike and the M1 CI runner (PLAN: "AMD via cloud"): without it, day 4 stops at compilation and the M1 gate cannot close.
- Whether `cuda-gdb` and `compute-sanitizer` honour `#line` into Python files; if not, debugging rests on the serial CPU backend.
- FP64 atomic throughput on GB10 is unpublished; it decides between FP64 atomics and FP32 in-warp partial sums with FP64 atomics per segment.
- Atomics vs colouring on Hopper/Blackwell for unstructured CFD remains unstudied in public ([state of the art §4](state-of-the-art-2026-10.md)); the FUN3D data above stop at A100 and MI100.

## 10. Sources

Accessed 2026-10-31 unless noted.

- [S1] W. S. Moses et al., "Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme", SC21, Nov 2021. https://dl.acm.org/doi/10.1145/3458817.3476165 (PDF: https://c.wsmoses.com/papers/EnzymeGPU.pdf)
- [S2] Enzyme, "CUDA Guide" (text dated 2021-01-20). https://enzyme.mit.edu/getting_started/CUDAGuide/
- [S3] EnzymeAD/Enzyme releases, v0.0.303, 2026-10-11; LICENSE. https://github.com/EnzymeAD/Enzyme/releases
- [S4] LLVM 22.1.0, `clang/include/clang/Basic/Cuda.h` (`FULLY_SUPPORTED = CUDA_128`, `PARTIALLY_SUPPORTED = CUDA_129`, `CUDA_130` listed). https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/clang/include/clang/Basic/Cuda.h ; the prose in "Compiling CUDA with clang" (still "CUDA 7.0 through 12.1") is stale: https://releases.llvm.org/22.1.0/docs/CompileCudaWithLLVM.html
- [S5] S. Joo, "CuAD: Source-to-Source Automatic Differentiation of CUDA Kernels for Discrete Adjoints in Computational Fluid Dynamics", arXiv:2610.26694, 2026-10-19. https://arxiv.org/abs/2610.26694 ; code (GPL-3.0): https://github.com/SEUNGHYUN-JOO/CuAD
- [S6] M. Zubair, D. Ranjan, A. Walden, G. Nastac, E. Nielsen et al., "Efficient GPU Implementation of Automatic Differentiation for Computational Fluid Dynamics", HiPC, Dec 2023. https://www.osti.gov/pages/biblio/1993463
- [S7] G. Nastac, A. Walden, L. Wang, E. Nielsen et al., "A Multi-Architecture Approach for Implicit Computational Fluid Dynamics on Unstructured Grids", AIAA SciTech 2023. https://ntrs.nasa.gov/api/citations/20220016937/downloads/2023_AIAA_SciTech_FLUDA_Paper.pdf
- [S8] C. P. Stone, A. Walden, M. Zubair, E. J. Nielsen, "Accelerating unstructured-grid CFD algorithms on NVIDIA and AMD GPUs", IA3 workshop at SC21, 2021. https://fun3d.larc.nasa.gov/papers/SC21_AMD.pdf
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
- [S19] "HIP C++ language extensions", HIP 7.16 documentation (warp sync, warp reduction and atomic functions). https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
- [S20] rapidsai/cudf issue #24661, "transform_kernel JIT compilation time and PCH footprint", 2026-10-17 (NVRTC 13.3.33, CCCL 3.6.0). https://github.com/rapidsai/cudf/issues/24661
- [S21] NVIDIA, "Hardware Overview — DGX Spark User Guide". https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- [S22] A. H. Mahmoud, R. Goel, J. Ragan-Kelley, J. Solomon, "Locality-Aware Automatic Differentiation on the GPU for Mesh-Based Computations", ACM TOG, July 2026 (arXiv:2509.00406). https://arxiv.org/abs/2509.00406
- [S23] A. Syrakos, S. Varchanis, Y. Dimakopoulos, A. Goulas, J. Tsamopoulos, "A critical analysis of some popular methods for the discretisation of the gradient operator in finite volume methods", Physics of Fluids 29, 127103, 2017 (arXiv:1606.05556). https://arxiv.org/abs/1606.05556
- [S24] J. R. R. A. Martins, P. Sturdza, J. J. Alonso, "The complex-step derivative approximation", ACM TOMS 29(3), 245–262, 2003. https://dl.acm.org/doi/10.1145/838250.838251
