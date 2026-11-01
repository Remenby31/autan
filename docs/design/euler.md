# Compressible Euler discretisation

The M2 physics of [PLAN.md](../PLAN.md): cell-centred finite volumes on arbitrary polyhedra, written once in the kernel DSL (`kernels/euler.py`, [kernel-generator.md](kernel-generator.md)) and driven by `autan::physics::Euler<P>` (`physics/`). Gates: [acceptance.md §M2](../validation/acceptance.md#m2--compressible-euler).

## Residual

R_c = Σ_f F(W_L, W_R)·S_f − s_c V_c over the faces of cell c, in conserved variables U = (ρ, ρu, ρv, ρw, ρE) of an ideal gas.

| Step | Choice | Source |
|---|---|---|
| Variables reconstructed | primitive W = (ρ, u, v, w, p) | |
| Gradients | weighted least squares (1/d²) over the **face2 stencil**: face neighbours, their face neighbours, and the boundary faces of both | [Stencil](#gradient-stencil) |
| Limiter | Venkatakrishnan, bounds over the face neighbours, ε_k² = (K/L)³ V q_k², K = 5, L the reference length, q_k the freestream scale of W_k | Venkatakrishnan (1995), eq. 5 |
| Face states | W_f = W_c + φ ∘ (∇W · (x_f − x_c)) | |
| Warped faces | flux of the linear reconstruction through the fan triangles: Σ_j dF_j[W; ∇W · M_{:,j}], M = Σ_t (c_t − x_f) ⊗ S_t | [Warped faces](#warped-faces) |
| Fluxes | Roe with Harten's fix on the acoustic waves only; HLLC with Einfeldt speeds; AUSM+-up with M_co the freestream Mach number | Roe (1981); Toro, Spruce, Speares (1994); Batten et al. (1997); Liou (2006) |

Harten's fix stays off the contact and shear eigenvalue so that every flux keeps a stationary contact exact (`tests/physics/test_euler.cpp`).

## Boundary conditions

Each boundary face carries a code and five values (`BoundaryKind`). The boundary state is computed from the reconstructed interior state at the face, then the numerical flux takes (W_f, W_b); slip faces take the pressure flux (0, p_f n, 0).

| Kind | Values | Boundary state |
|---|---|---|
| `state` | W | given (manufactured solutions) |
| `farfield` | freestream W | Riemann invariants normal to the face; entropy and tangential velocity from the upwind side; supersonic faces take the upwind state |
| `slip` (slip wall, symmetry) | — | mirror state for the gradients; pressure flux |
| `inlet` | p₀, T₀ | flow normal to the face; R⁺ from the interior and c_p T₀ give c_b (Carlson 2011) |
| `outlet` | p | ρ and v from the outgoing characteristics; supersonic outflow extrapolates |

The gradient's boundary values are the boundary states of the first-order interior state (`boundary_states`, a frozen loop: its derivatives are dropped, so the generated adjoint of the gradient is approximate at boundary faces).

## Gradient stencil

Least squares over the face neighbours alone make the unlimited second-order scheme **linearly unstable** on the tetrahedra of `autan-vv meshes tet`: the Jacobian of the residual has eigenvalues with negative real part. The same holds for the scalar advection of the generator spike, so the stencil, not the Euler flux, is at fault. Widening the stencil removes every unstable eigenvalue, as Diskin and Thomas (2011, AIAA J. 49(4), "inviscid fluxes") recommend for cell-centred schemes.

Spectrum of the dense Jacobian (finite differences) at the manufactured solution, tet n = 5 (762 cells), Roe, unlimited:

| Gradient stencil | Eigenvalues with Re < 0 | min Re (scaled) |
|---|---|---|
| face neighbours | 42 | −3.53 |
| face2 | 0 | +0.87 |
| vertex (cells sharing a node) | 0 | +0.91 |

Face neighbours stay stable on the hexahedral, prismatic, hybrid and polyhedral families (no eigenvalue with Re < 0 at n = 5–6). Cost on GB10 (tetrahedra n = 96, 3.9 M cells, `fp32`, measured by CFD - Générateur kernels): face neighbours 0.50 ns/cell (93 % of peak bandwidth), face2 1.82 ns/cell (58 %), vertex 20 ns/cell (25 %). face2 is the default: stable, second order, 11× cheaper than vertex. Both stencils give order 2 ([MMS](#manufactured-solution)).

## Warped faces

The faces of polyhedral duals are not planar. A one-point flux at the face centroid then commits an O(h) relative error per face, and the solution converges at first order (measured: finest-pair order 1.05–1.26 on the `poly` family). Integrating the linear reconstruction over the fan triangles through the face moment M restores order 2 (2.1–2.6).

## Manufactured solution

Steady `euler` solution of autan-vv (Roy et al. 2004 form), boundary faces at the exact state, cell sources at centroids; Newton-Krylov on the generated residual (`physics/mms/euler_mms.py`). The table is filled from the study of PR #99.

