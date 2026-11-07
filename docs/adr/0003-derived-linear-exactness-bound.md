# 0003: Linear exactness is judged against certified per-cell radii, not a fixed 64 ε

**Status:** proposed, 2026-11-07, revision 4, amended 2026-11-07 for √ and `hypot` (after the third review on #577; accepted once these changes are in, the orchestrator decided, with Codex re-checking them in the freeze-commit review). Not accepted. The orchestrator has it reviewed again before any verdict changes, and no PASS is recorded before that review and the held-out validation below. **Decided by:** to be decided by the orchestrator, for the lead's delegate. **Issues:** #43, #566 (scope decision 3).

## Context

`acceptance.md` line 18 requires LSQ and corrected Green-Gauss gradients "exact to 64 ε" on linear fields. Read as written, that is an **absolute** bound per component: |ĝ_u − G_u| ≤ 64 ε of the policy. The M1 spike did not meet it: LSQ ≤ 400 ε, and the tests allow 9600 ε. Scope decision 3 of #566 forbids relaxing the threshold without first finding the cause.

The cause was analysed on paper before any value (#43, step A, revised after review), then explored (#43, step B), on main b3683284, host only, in exact rational arithmetic. **This ADR comes after that exploration.**

Units: step B's tables report |ĝ_u − G_u| / (ε ‖G‖∞) with ‖G‖∞ = 1.2. They are *relative to ‖G‖∞*, and absolute values are 1.2 times larger. They are kept distinct from the original absolute 64 ε below.

What step B observed, limited to the discrepancies it tested and to the arithmetic model below:
- **D1, as step B defined it** (the kernel against its formula evaluated exactly on its *stored* inputs): no observed deviation exceeded its bound, at worst 0.44 (LSQ) and 0.25 (GG) of it. That check reads the stored inputs, so it cannot detect corrupted stored geometry or weights. It is replaced below by an oracle built from the nodes.
- **LSQ:** the gap to G is dominated by the rounding of the stored field values. It roughly doubles with each doubling of the cells per direction and is proportional to the field's offset a.
- **GG, the exact same-cell decomposition** (https://github.com/Remenby31/autan/issues/43#issuecomment-6710945756):
  - In the cell of largest input error, the field data dominates on clean meshes and the closure term on perturbed ones.
  - Coordinate rounding is ≤ 17.4 ε, fp32 only; the injected gradient's rounding is ≤ 0.32 ε.
  - The kernel's own arithmetic is of the same order as the input terms. Its maximum over all cells is 267.2 ε with a = 1, and 174 307.3 ε with a = 10³ (hybrid, perturbed, fp64, N = 16), in units of ε‖G‖∞.
  - The norm columns are maxima of absolute values, not an additive identity, and the selected cell maximises input error, not total error.

A fixed absolute 64 ε cannot hold for these operators on these inputs as the mesh is refined or the field offset grows. The proposal replaces it with checks against radii that are **proven** from a declared operation sequence, not fitted.

## Arithmetic model and definitions

- **Format and rounding.** IEEE 754 binary64 and binary32, round to nearest even, no FMA contraction on the host (`-ffp-contract=off`, CMakeLists.txt). Unit roundoff u_t of format t: 2⁻⁵³ or 2⁻²⁴. Machine epsilon ε_t = 2u_t.
- **Range and limits.** IEEE 754 round to nearest even **with gradual underflow**: flush-to-zero and denormals-are-zero are excluded, and so are evaluation changes such as x87 extended precision or an FMA the compiler contracts unless the declared sequence states it as one operation.
  - Overflow, a divisor its radius may reach, or a widening cast from an undeclared format makes the instance **invalid**: reported, never a pass.
  - Zero results and zero inputs are allowed. Where an interval reaches the subnormal band, the rounding term adds Higham's underflow constant η = 2^(emin − p), an exact rational per format: 2⁻¹⁰⁷⁵ in binary64, 2⁻¹⁵⁰ in binary32. It is kept for + and − too, conservatively (approved in the review of #579).
  - The certificate bounds a distance: the sign of a zero result is outside it.
- **Roles and casts per policy** (`runtime.hpp`, `emit_cpp.py`):

| policy | state | compute | accumulation | gradient | geometry read through `rel` | other geometry in bodies |
|---|---|---|---|---|---|---|
| fp64 | 64 | 64 | 64 | 64 | 64 | 64 |
| fp32+acc64 | 32 | 32 | 64 | 32 | 64 | 32 (`geom_value`) |
| fp32 | 32 | 32 | 32 | 32 | 32 | 32 |

  The geometry is built in binary64 (`core/src/geometry.cpp`, `codegen/cpp/geometry.cpp`), then cast to the role's type.
- **Comparisons.** Per cell c and per component u, on absolute values. No ratio is formed. A radius of 0 requires exact equality.
- **Exact evaluation.** Every value, reference and radius is evaluated in exact rational arithmetic. Floating-point sums are never a reference.

### Certified error propagation

Every quantity X carries its **exact value** x and a **radius** r ≥ 0. The radius is proven to bound |X̂ − x| for the value X̂ that any evaluation of the declared operation sequence produces under the model above. For an operation whose exact result on the computed operands is rounded to format t (unit u):

| operation | radius of the result |
|---|---|
| input given exactly | 0 |
| X ± Y | (1 + u)(r_x + r_y) + u·\|x ± y\| (+ η near underflow, for every rule) |
| X · Y | (1 + u)(\|x\| r_y + \|y\| r_x + r_x r_y) + u·\|x y\| |
| X / Y, valid only if r_y < \|y\| | (1 + u)·(r_x + \|x/y\| r_y)/(\|y\| − r_y) + u·\|x/y\| |
| cast to a narrower format (unit u_t) | (1 + u_t) r_x + u_t \|x\| |
| widening cast | r_x |

**Exactness refinement.** When r_x = r_y = 0 and the exact result is representable in t, the radius is 0.

**Proof sketch.** Write X̂ = x + δ_x with |δ_x| ≤ r_x. The computed result is fl(X̂ op Ŷ) = (X̂ op Ŷ)(1 + η), |η| ≤ u. Then |fl(X̂ op Ŷ) − (x op y)| ≤ |X̂ op Ŷ − x op y| + u|X̂ op Ŷ| ≤ (1 + u)|X̂ op Ŷ − x op y| + u|x op y|. The first factor is bounded exactly:
- by r_x + r_y for ±;
- by |x|r_y + |y|r_x + r_x r_y for ·;
- by (r_x + |x/y| r_y)/(|y| − r_y) for /, using |Ŷ| ≥ |y| − r_y > 0.

The refinement holds because rounding a representable value under round-to-nearest is exact. No first-order truncation and no safety factor is used: the earlier "first order × 2 under a 1/8 condition" is dropped.

**Two operations beyond the four rules** (amendment, 2026-11-07, decided on #43). The geometry construction (`core/src/geometry.cpp`) uses `std::sqrt` and `std::hypot`. The rational-only semantics are extended to exactly these two operations.

- **√, correctly rounded (IEEE 754).** Let x be the executed finite input, X the exact value, |X − x| ≤ r and X − r > 0. Take rationals 0 ≤ L ≤ U with L² ≤ X − r and U² ≥ X + r, and ℓ, m > 0 with ℓ² ≤ X − r and m² ≤ X. Then the executed y = fl(√x) satisfies

  |y − √X| ≤ r/(ℓ + m) + u·U + η.

  √X is irrational in general, so the certified value is a rational centre μ with radius ρ ≥ |μ − √X|, validated by exact squaring: (μ − ρ)² ≤ X ≤ (μ + ρ)², with μ − ρ ≥ 0. Floating-point roots only *propose* the witnesses μ, ρ, ℓ, m, L, U; exact squaring validates them, or the instance is invalid. The result may be re-centred on the executed y with radius ρ + |y − μ|.
- **`hypot`, a declared sequence replayed bit for bit** (decided on #43, replacing a per-instance (1 ± k·u) check: stricter, not a relaxation).
  - In the build's toolchain the three-argument `std::hypot` is not a library call. It is libstdc++'s inline `__hypot3` (GCC 13, `/usr/include/c++/13/cmath`:3550–3563): a = max(|x|, |y|, |z|) by the source's comparisons, then x/a, the squares, their sum in the source order, √, and × a. It is declared in `sequences.py`, with the libstdc++ version and lines pinned in FREEZE.md.
  - Per instance, the sequence is replayed by exact emulation of binary64 round-to-nearest-even from the stored area components. The stored magnitude, as the kernels read it, must equal the replay **bit for bit**. A mismatch means the build does not run the declared sequence (another library, an FMA, another branch), and the instance is **invalid**, fail-closed.
  - The radius comes from the certified rules on that sequence (÷, ×, +, √, with η).
  - a = 0 (a zero face) is an exact zero only when every exact area component is certainly zero; otherwise invalid.
- **Correspondence through these operations.** The intended values are rational. The face normal's magnitude cancels in the centroid's weighted ratio, and the length scale cancels in volume and centroid. That identity is proven once, tied to the frozen sequences and their branches, and written in `sequences.py` and FREEZE.md. Per instance, its hypotheses are checked: each branch is the one the identity assumes (e.g. magnitude > 0 certified by a positive lower bound), and every denominator is separated from zero by its certified radius. The rational parts are checked exactly.
- **The radius covers the whole computed result.** Cancellation in exact arithmetic does not remove the rounding that the executed sequence actually performs, so every rounding along the executed path (√, `hypot`, the divisions by magnitude and length) is propagated into the radius.

**tidy** (decided on #43, before any held-out run, PR #585). After every rule, a certificate is re-centred on the nearest dyadic of 128 significant bits (ties to even), and its radius plus the shift is rounded up, exactly, to 64 significant bits. This keeps values and radii of bounded size. It is sound by the triangle inequality and never shrinks an interval. **tidy enlarges R by ≤ 2⁻⁶³·R + 2⁻¹²⁷·|v| per operation, declared before held-out.** Measured on the step-B geometry: at most 1.08 × 10⁻¹⁹ relative over 2.53 million calls.

**Declared operation sequences.** The radius of a quantity is propagated through a declared sequence, frozen with the code below:
- the field: the fixture's evaluation of a + x·G in binary64, then storage;
- the geometry construction, operation by operation as in `core/src/geometry.cpp` (face area, centroid and volume by fans and the apex decomposition) and `codegen/cpp/geometry.cpp` (moments, LSQ weights with `inverse3`);
- the kernel bodies, driven by the generator's own IR instruction list for each body, with the `C()` conversions and storage types of `emit_cpp.py`, and the loop's accumulation order (incidence order for `cell_reduce` and the gathers, face order for the scatter).

The **values**, unlike the radii, never come from the code under test.

**Correspondence obligation.** A radius bounds the distance from the computed value to the *exact evaluation of the declared sequence*. The checks centre their intervals on the *intended operator's* exact value. The two must be the same number.

So, on every instance and for every quantity that enters a check (each geometric array entry, each weight, each cell's gradient component), the oracle evaluates the declared sequence **with every rounding removed**, in exact rationals on the exact data, and verifies that it equals the intended operator's exact value. Any mismatch makes the instance **invalid**, reported as a correspondence failure (the code computes another function), never a pass.

This obligation is checked, not assumed. A proof for a whole operator does not replace it.

## Decision (proposed)

The gate "linear fields exact to 64 ε" becomes four mandatory checks per instance. Each fails on its own; a failure of one never enlarges another's tolerance.

1. **Topology and orientation, derived independently.** The oracle derives the topology from the **cells' own node lists**: the outward polygons each builder of `codegen/spike/meshes.py` returns per cell, before `to_faces` merges them. It does not use the face-based arrays.
   - Each cell's polygons must form a closed surface, every edge traversed exactly twice, once in each direction (`docs/contracts/mesh.md`, rule 5).
   - The polygons must form **exactly one connected component** (connected through shared edges), and that component's exact signed volume, by the divergence theorem, must be positive. Cancelling edges plus a positive *total* volume would not suffice: two disjoint shells of +8 and −1 pass that test.
   - **Boundary validity:** every boundary face belongs to exactly one cell. The domain's boundary surface is closed and consistently oriented, every edge traversed as often in one direction as in the other (`docs/contracts/mesh.md`, surface output). Its exact signed volume equals the sum of the cells'.
   - Each face of the kernels' mesh must match polygons of exactly two cells, with opposite node orders (interior), or of one cell (boundary).
   - Owner and neighbour follow the contract's convention: the face's right-hand normal points from owner to neighbour, interior faces have owner < neighbour, and a face is outward for c iff owner[f] = c.
   - After agglomeration, the merged cell's polygons are those of its parts minus the shared ones.

   The kernels' owner, neighbour, cell_faces, incidence sides and face node orders must equal this derivation. Any difference fails.
2. **Geometry against the nodes.** The *intended* geometry is evaluated exactly from the node coordinates by the construction's definitions: face vector areas and centroids, face moments, cell volumes and centroids, LSQ weights. Every stored array the kernels read must lie within its certified radius of these exact values.
   This is an independent check: a common scaling of s, M and V, which keeps the closure zero and the gradient unchanged, fails here. The closure terms z and Z are no longer used as allowances.
3. **D1, the kernel against the intended operator from the nodes.** The oracle evaluates the intended operator exactly, with the exact node-defined geometry and weights, on the kernel's stored field values q̂.
   - For LSQ: Σₖ w_k (q̂_k − q̂_c), with w = M⁻¹d/|d|² exact.
   - For GG: the face reconstruction, face sum and division, with the exact areas, moments, centroids and volumes.

   The check is |ĝ_u − g_int,u| ≤ R_D1,c,u, where R_D1 propagates the geometry construction, the weights, the casts and the kernel arithmetic, but not the field's rounding.
4. **The linear-exactness bound.** First, the oracle verifies exactly that the intended operator reproduces G on the exact field with the exact geometry. This holds by construction for LSQ, Σw dᵀ = I. For GG it is the design premise of the face moment, and is checked per cell; a cell where it fails is reported as a property of the operator, never covered by a tolerance.
   Then |ĝ_u − G_u| ≤ R_c,u, where R propagates everything from the exact data (nodes, a, G): the field's evaluation and storage, the geometry, the weights, the casts and the kernel arithmetic.
   R replaces "T_arith + T_data + T_geom". The coordinate and injected-gradient terms of GG are inside R by construction, not added by hand. For reporting, R is also propagated with one source class at a time (field, geometry, gradient injection, arithmetic). Each such radius bounds that class alone; the gate uses the joint R.

**Fail-closed.** PASS requires all of the following:
- every instance in the manifest is valid and passes checks 1–4;
- every positive fixture passes;
- every corruption control is demonstrated and fails as stated.

Any invalid instance, any control "not demonstrated", or any failed or missing item **blocks PASS, and M1 stays FAIL**. Nothing is skipped, substituted or weakened to obtain a PASS.

No check claims to detect every defect. A PASS establishes consistency with the intended operator within the certified envelope, for the declared sequences, on the instances run.

### Positive fixtures (must PASS)

These are separate from the corruption controls below. The nonzero-difference rule of the controls does not apply to them.

- **Exact fixtures:** uniform hexahedral meshes with dyadic node coordinates, dyadic G and a, and constant fields (G = 0).
  - A fixture is *certified exact* when the propagation, with its exactness refinement, gives R = 0 for every cell and component. The check is then ĝ = G exactly. Dyadic inputs alone certify nothing.
  - The manifest lists an approved set of exact fixtures. That set **must be non-empty**, and every fixture in it must certify R = 0. A listed fixture that does not certify, or that fails ĝ = G, blocks acceptance.
- **Boundary cases:** cells with only boundary faces, a single boundary face, a zero gradient component, the constant field. Each must pass checks 1–4.

### Defect controls

Each control must FAIL the stated check, and its failure is **proven at the output** before it is run. A primitive change larger than its own allowance does not by itself move the output: a corrupted weight times a zero field difference is invisible.

So, for every control, the oracle evaluates exactly the declared sequence on the *corrupted* inputs (exact value y_c) and certifies its radius r_c. It proves that the whole interval [y_c − r_c, y_c + r_c] lies outside the stated check's acceptance interval [centre − R, centre + R]: |y_c − centre| > R + r_c. Only then is the control run, and the run must fail.

The fields of the corruption controls have every neighbour difference q_k − q_c nonzero, checked exactly per cell, so no corrupted weight can be multiplied by zero. A control whose proof cannot be established on a declared case is reported as **not demonstrated**, never as passed.

| # | corruption | must fail |
|---|---|---|
| 1 | one stored LSQ weight component doubled | 2, and 3 where the output proof holds |
| 2 | one weight's sign flipped | 2, and 3 where the output proof holds |
| 3 | one incidence dropped from a cell | 1 and 3 |
| 4 | one face area component moved by K units in the last place of its stored type. K is the smallest integer ≥ 1 with \|s′ − s\| > 2 r_s, computed exactly after rounding (K = 1 when r_s = 0). The target must be finite and valid in the model: the corrupted component is chosen with \|s\| ≥ the smallest normal, and the step must not cross zero or reach a subnormal | 2 |
| 5 | s, M and V of every face and cell scaled by 2 together (closure stays zero) | 2 |
| 6 | one cell volume changed by more than 2 r_V | 2 |
| 7 | one face moment changed by more than 2 r_M | 2 |
| 8 | one cell centroid moved by more than 2 r_x | 2 |
| 9 | one interior face's owner and neighbour swapped, its area not negated | 1 |
| 10 | the kernel's output offset to the representable value beyond G ± R (the next float past the interval's far side, so it is outside whatever its sign or R = 0) | 4 |
| 11 | wrong precision: the gradient evaluated with its accumulation narrowed to binary32 under fp32+acc64. Inexact binary32 intermediates do not imply a wrong output ((1, 2⁻²⁵, −2⁻²⁵) sums to 1 in both precisions). So a fixture qualifies only if **exact emulation** of the narrowed sequence, step by step in binary32 under round-to-nearest, gives a final output outside both the D1 interval and the bound interval of fp32+acc64. Its witness is that **exactly emulated, rounded output itself**: a singleton, radius 0, not the generic unrounded centre with its radius. The generic rule cannot show the failure of a mutation that changes only precision. Fixtures are searched in a declared, frozen family; if none qualifies, the control is not demonstrated | 3 and 4 |

### Prospective protocol

Before the held-out run, one commit freezes the items below. That **freeze commit is reviewed (Codex) and approved by the orchestrator before any held-out case is executed**. No choice of a declared sequence (operation order, cast placement, accumulation order) may enlarge a radius without that review. The commit freezes:
- this ADR's equations and tables, as reviewed;
- the oracle and radius code, with its path and SHA-256, including the declared operation sequences and the correspondence checks;
- the build: the commit of the kernels, compiler and version, flags, and the FP assumptions above;
- the complete **case manifest**: every family, seed, level, policy, a, G, fixture and control.

The held-out cases are disjoint from step B:
- seeds 11 and 13 (B: 3, 9);
- N = 6, 12, 24 (B: 4, 8, 16);
- the agglomerated family (not in B);
- G = (−0.9, 0.4, 1.3) and a ∈ {−7.5, 250} (B: (0.3, −1.2, 0.7), {1, 10³}).

None of these cases has been evaluated by the kernels or the oracle in this analysis. The commit states so.

Every run is kept, failed ones included, with its raw output. Any amendment after the freeze, to an equation, the code, the build or the manifest, needs a new prior approval and a new disjoint case set. The validation passes only under the fail-closed rule above, on the frozen manifest.

## Consequences

- `acceptance.md` line 18 states the four checks and links here. The PR that changes it reports the held-out validation and the controls.
- `codegen/tests/test_gradients.py` replaces its fixed 9600 ε by these checks, through a test helper that holds the oracle and the propagation, never the kernels.
- **Scope:** host only. The device contracts FMAs (#571), so its sequences differ; it is decided there.
- Conservation keeps its 64 ε Σ|F| gate, unaffected by this ADR. Its PASS on #43 is limited to the tested scope (host, interior-only, source-free advection face loop). Boundary and source terms, the device, MPI, time integration, converged GG, and N = 32 or agglomerated meshes are not covered.
