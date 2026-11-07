# 0003: Linear exactness is judged against a derived per-cell bound, not 64 ε

**Status:** proposed, 2026-11-07. Not accepted: the orchestrator has it reviewed (Codex) before any verdict changes, and no PASS is recorded before that review and the held-out validation below. **Decided by:** to be decided by the orchestrator, for the lead's delegate. **Issues:** #43, #566 (scope decision 3).

## Context

`acceptance.md` requires LSQ and corrected Green-Gauss gradients "exact to 64 ε" on linear fields (line 18). The M1 spike did not meet it: LSQ ≤ 400 ε, and the tests allow 9600 ε. Scope decision 3 of #566 forbids relaxing the threshold without first finding the cause.

The cause was analysed on paper before any value was computed (#43, step A, revised after review), then measured as an exploration (#43, step B). Both were on main b3683284, on the host, in exact rational arithmetic on the kernels' stored inputs.

- **The kernels are not at fault.** In every instance the LSQ and GG kernels lie within the arithmetic bound of their generated operation graph, at worst 0.44 (LSQ) and 0.25 (GG) of it (D1). The stored geometry closes at the roundoff of its storage type (G2).
- **Where the 64 ε gap comes from.** For LSQ it is the rounding of the stored field values, e_data = Σₖ ŵₖ(ηₖ − η_c). It roughly doubles with each doubling of the cells per direction and is proportional to the field's offset a, while the stencil conditioning h·Σ|ŵ| stays between 1.1 and 5.5.
- **For GG it is the inputs and the kernel's own rounding together.** The exact same-cell decomposition (https://github.com/Remenby31/autan/issues/43#issuecomment-6710945756) attributes the worst cell's input error to the field data on clean meshes and to the closure term a z + Z G on perturbed ones. Coordinate rounding (fp32 only, ≤ 17.4 ε) and the injected gradient's rounding (≤ 0.32 ε) are smaller. The kernel's arithmetic, within its bound, is of the same order as the input terms (up to 252.8 ε in one cell).

For LSQ, a fixed 64 ε relative to |G| is therefore not a property any kernel reading these inputs can guarantee: the inputs already deviate from the linear field by more than that. For GG the same holds of the inputs, and the kernel's rounding, which grows with a, adds to it. Hence a per-cell bound with an arithmetic term, rather than a fixed multiple of ε.

**This change comes after exploration.** It is proposed after step B's values were seen. To keep it from being fitted to them:
- every constant below is derived from operation counts, not from B's values;
- every constant is frozen in this document before the held-out validation;
- the held-out meshes and fields are disjoint from B's.

## Definitions

- **Roundoff.** u is the unit roundoff of a format: 2⁻⁵³ (binary64), 2⁻²⁴ (binary32). ε = 2u is its machine epsilon. γₙ(u) = n u / (1 − n u), defined only for n u < 1/2 (checked).
- **Arithmetic model.** IEEE 754 round to nearest even, no FMA contraction on the host (`-ffp-contract=off`, CMakeLists.txt). No underflow or overflow: every input and intermediate magnitude is checked to lie in the normal range of its type, or the instance is invalid, which is reported and not counted as a pass.
- **Roles per policy** (`codegen/include/autan/kernels/runtime.hpp`; emission in `codegen/src/autan_codegen/emit_cpp.py`):

| policy | state u_s | compute u_c | accumulation u_a | gradient u_g | geometry read through `rel` u_x | other geometry reaching bodies u_v |
|---|---|---|---|---|---|---|
| fp64 | 2⁻⁵³ | 2⁻⁵³ | 2⁻⁵³ | 2⁻⁵³ | 2⁻⁵³ | 2⁻⁵³ |
| fp32+acc64 | 2⁻²⁴ | 2⁻²⁴ | 2⁻⁵³ | 2⁻²⁴ | 2⁻⁵³ (double storage) | 2⁻²⁴ (`geom_value`) |
| fp32 | 2⁻²⁴ | 2⁻²⁴ | 2⁻²⁴ | 2⁻²⁴ | 2⁻²⁴ | 2⁻²⁴ |

  A value's "stored" form is its double rounded to the role's type (round to nearest). Casts from a narrower to a wider type are exact.
- **Norms.** Every comparison is **per cell c and per Cartesian component u**, with absolute values. A gate passes when every (c, u) passes. No ratio is formed, so a zero bound is not a division.
- **Zero cases.** If T_{c,u} = 0, the gate requires ĝ_u = G_u exactly. If the measured error is 0, the cell passes whatever its bound.
- **Exact evaluation.** Every reference, bound and decomposition is evaluated in exact rational arithmetic (`fractions.Fraction` of each double). Floating-point sums (fsum, ExactSum) are never a reference.
- **Field.** q(x) = a + G·x. The fixture evaluates it in binary64 at the binary64 centroids x (or face centroids) and stores q̂ = fl_s(fl₆₄(a + x·G)).
  b_q,i bounds |q̂ᵢ − q(xᵢ)|: b_q,i := u_s |q̂ᵢ| + (1 + u_s) γ₄(2⁻⁵³)(|a| + Σⱼ |Gⱼ xᵢⱼ|).

## Decision (proposed)

The gate "linear fields exact to 64 ε" becomes four mandatory checks per instance. Each fails on its own; none enlarges another's tolerance.

### 1. D1, the kernel against its exact reference

g_ref is the kernel's formula evaluated exactly on its stored inputs. The check is |ĝ_u − g_ref,u| ≤ T_arith,c,u, where T_arith is the arithmetic bound of the generated operation graph (#43, revised step A, D1):

- **LSQ:** T_arith = u_g |g_ref,u| + (1 + u_g) θ Σₖ |ŵ_uk| |q̂ₖ − q̂_c|, with θ = (1 + u_c)³(1 + γ_{m−1}(u_a)) − 1.
- **GG:** per face, the `rel` subtraction in u_x then the conversion in u_c, two 3-term dot products and the face sum in γ₇(u_c), q̃ŝ and M̂ᵀĝ in γ₄(u_c). Per cell, the accumulation γ_{m_c}(u_a), then conversion, division and storage, (1 + u_c)²(1 + u_g) − 1.

A doubled weight, a wrong sign or a missing incidence fails here, because g_ref reads the same stored inputs.

### 2. G1, LSQ weights, componentwise

The exact weights w are computed in rationals from the stored binary64 centroids, by the operations of `store_lsq_weights` and `inverse3` (`codegen/cpp/geometry.cpp`). For every weight component, |ŵ_uk − w_uk| ≤ τ_w,uk. τ_w is the first-order forward bound of that operation graph:

- propagate γ through each operation from the absolute values of its exact operands (differences d, squared norms, the 9 accumulated products of M, the 2×2 cofactors, the determinant, the reciprocal, the products with d);
- multiply by 2 for second-order terms;
- add u_v |w_uk| for the cast to the body's type.

τ_w is absolute, built from operand magnitudes, so a near-zero weight gets the absolute tolerance its operands imply, not a relative one. The factor 2 is valid while every stage's relative bound is at most 1/8 (checked; otherwise the instance is invalid).

### 3. G2, geometry closure

Exactly, z_c = Σ_f σ_f ŝ_f and Z_c = Σ_f σ_f (ŝ_f x̂_fᵀ + M̂_fᵀ) − V̂_c I. These are zero for exact geometry:
- the fan vector area of `core/src/geometry.cpp` equals the polygon's vector area for any fan centre;
- the vector areas of a closed surface sum to zero;
- the face moment is defined so that s x_fᵀ + M is the face's first moment.

So z and Z are pure rounding. Their margins τ_z,c and τ_Z,c are the first-order forward bounds of the construction's operation graph, propagated from the stored node coordinates (exact rationals), times 2:
- face area: n-node fan, node mean, differences, cross products, sum over n triangles, `core/src/geometry.cpp` lines 46–70;
- face centroid and the cell's apex decomposition for volume and centroid: lines 79–110;
- face moments: fan of `codegen/cpp/geometry.cpp`, lines 208–264;
- plus the casts to the body types (u_v, u_x).

|z_c| > τ_z,c or |Z_c| > τ_Z,c is a **geometry defect**: it fails, and never enlarges a tolerance.

### 4. The linear-exactness bound

For every cell and component, |ĝ_u − G_u| ≤ T_c,u := T_arith,c,u + T_data,c,u + T_geom,c,u:

- **LSQ:**
  - T_data = Σₖ |ŵ_uk| (b_q,k + b_q,c).
  - T_geom = |(E_c G)_u|, with E_c = Σₖ ŵₖ (xₖ − x_c)ᵀ − I exact on the stored weights and centroids. It is admitted only if G1 passes for the cell.
- **GG:**
  - T_data = Σ_f |ŝ_fu| b̄_f / |V̂_c|, with b̄_f = ½(b_q,l + b_q,r) on interior faces and b_q,b on boundary faces.
  - T_coord = Σ_f |ŝ_fu| ½ Σⱼ |Gⱼ| (u_x |x_lj| + u_x |x_rj|) / |V̂_c| (the field evaluated at x, the reconstruction reading the stored X).
  - T_grad = Σ_f [ |ŝ_fu| ½ Σⱼ u_g |Gⱼ| (|r_lj| + |r_rj|) + Σⱼ |M̂_ju| u_g |Gⱼ| ] / |V̂_c|.
  - T_geom = |a z_c + Z_c G|_u / |V̂_c|, admitted only if G2 passes for the cell.

T_data, T_coord and T_grad bound the propagated input errors. T_geom is the exact closure term, admitted only for validated geometry.

### Fixtures that stay tight, and controls that must fail

- **Exact fixtures.** On uniform hexahedral meshes with dyadic coordinates, a dyadic G and a, every stored input is exact, so T_data = T_coord = T_grad = 0. The bound reduces to T_arith + T_geom, both computed. A constant field (G = 0) must give ĝ = 0 exactly for LSQ (every difference is exactly 0).
- **Defect controls**, each required to FAIL the stated check:
  1. stored LSQ weights doubled in one cell: D1 passes (same stored inputs), G1 fails;
  2. one weight's sign flipped: G1 fails;
  3. one incidence dropped from the gradient: D1 fails;
  4. one face area scaled by 1 + 10⁻⁶: G2 fails;
  5. a kernel output offset by 2 T_c in one cell: the bound fails.

### Validation before acceptance

The constants above are frozen as written here, before the validation runs. These are γ indices, the factor 2, the 1/8 validity limit, b_q, and the definitions of τ_w, τ_z and τ_Z.

The held-out set is disjoint from step B:
- mesh seeds 11 and 13 (B used 3 and 9);
- levels N = 6, 12, 24 (B: 4, 8, 16);
- the agglomerated family (not in B);
- G = (−0.9, 0.4, 1.3) and a ∈ {−7.5, 250} (B: G = (0.3, −1.2, 0.7), a ∈ {1, 10³});
- the three policies.

The ADR is accepted only if every held-out instance is valid and passes checks 1–4, every exact fixture meets its threshold, and every defect control fails as stated. A failure is reported as is. It is not answered by changing a constant: any change would need a new ADR and a new held-out set.

## Consequences

- `acceptance.md` line 18 states the four checks and links here. The PR that changes it reports the held-out validation and the controls.
- `codegen/tests/test_gradients.py` replaces its fixed 9600 ε by these checks. They need exact evaluation, so the per-cell computation runs in a test helper, not in the kernels.
- **Scope:** host only. The device kernels contract FMAs (#571); their bounds and determinism are decided there.
- Conservation keeps its 64 ε Σ|F| gate. That gate is unaffected by this ADR, and its PASS on #43 is limited to the tested scope (host, interior-only, source-free advection face loop).
