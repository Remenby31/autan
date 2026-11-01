# Compressible Euler discretisation

The M2 physics of [PLAN.md](../PLAN.md): cell-centred finite volumes on arbitrary polyhedra, written once in the kernel DSL (`kernels/euler.py`, [kernel-generator.md](kernel-generator.md)) and driven by `autan::physics::Euler<P>` (`physics/`). Gates: [acceptance.md §M2](../validation/acceptance.md#m2--compressible-euler).

## Residual

R_c = Σ_f F(W_L, W_R)·S_f − s_c V_c over the faces of cell c, in conserved variables U = (ρ, ρu, ρv, ρw, ρE) of an ideal gas.

| Step | Choice | Source |
|---|---|---|
| Variables reconstructed | primitive W = (ρ, u, v, w, p) | |
| Gradients | weighted least squares (1/d²) over the **face2 stencil**: face neighbours, their face neighbours, and the boundary faces of both | [Stencil](#gradient-stencil) |
| Limiter | Venkatakrishnan, bounds over the gradient stencil, ε_k² = (K/L)³ V q_k², K = 5, L the reference length, q_k the freestream scale of W_k | Venkatakrishnan (1995), eq. 5 |
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

The gradient's boundary values are the boundary states of the first-order interior state (`boundary_states`, a frozen loop: its derivatives are dropped, so the generated adjoint of the gradient is approximate at boundary faces). They enter the least-squares fit only on `state` faces, where the condition gives the value at the face; on the other kinds the value is the adjacent cell's, which belongs at the cell centroid, and the fit uses the interior stencil alone (a cell whose interior stencil cannot determine a gradient keeps every boundary face). Offset components within 64 ulps of the coordinates count as zero: across a one-cell-thick slab the centroids differ only by round-off, and the fit would otherwise turn it into a gradient across the layer. The fit is determined when det A > 10⁻¹⁰ A₁₁A₂₂A₃₃, which is independent of the cells' aspect ratio. Where the linear reconstruction would make ρ or p non-positive, the face takes the cell value. The warped-face correction applies to interior faces and to the pressure of slip faces.

## Gradient stencil

Least squares over the face neighbours alone make the unlimited second-order scheme **linearly unstable** on the tetrahedra of `autan-vv meshes tet`: the Jacobian of the residual has eigenvalues with negative real part. The same holds for the scalar advection of the generator spike, so the stencil, not the Euler flux, is at fault. Widening the stencil removes every unstable eigenvalue, as Diskin and Thomas (2011, AIAA J. 49(4), "inviscid fluxes") recommend for cell-centred schemes.

Spectrum of the dense Jacobian (finite differences) at the manufactured solution, tet n = 5 (762 cells), Roe, unlimited:

| Gradient stencil | Eigenvalues with Re < 0 | min Re (scaled) |
|---|---|---|
| face neighbours | 42 | −3.53 |
| face2 | 0 | +0.87 |
| vertex (cells sharing a node) | 0 | +0.91 |

Face neighbours stay stable on the hexahedral, prismatic, hybrid and polyhedral families (no eigenvalue with Re < 0 at n = 5–6). Cost on GB10 (tetrahedra n = 96, 3.9 M cells, `fp32`, measured by CFD - Générateur kernels): face neighbours 0.50 ns/cell (93 % of peak bandwidth), face2 1.82 ns/cell (58 %), vertex 20 ns/cell (25 %). face2 is the default: stable, second order, 11× cheaper than vertex. Both stencils give order 2 ([MMS](#manufactured-solution)). The limiter's bounds use the same stencil: bounds over the four face neighbours of a tetrahedron are narrower than the values the face2 gradient extrapolates, so the limiter clips smooth solutions and the limited scheme falls to first order on tetrahedra (measured: finest-pair order 0.58–1.20).

## Warped faces

The faces of polyhedral duals are not planar. A one-point flux at the face centroid then commits an O(h) relative error per face, and the solution converges at first order (measured: finest-pair order 1.05–1.26 on the `poly` family). Integrating the linear reconstruction over the fan triangles through the face moment M restores order 2 (2.1–2.6).

## Precision

Cells store U − U_ref, the reference being the freestream, and the primitive pressure is the gauge pressure p − p_ref, computed from the differences only: p − p_ref = (γ − 1)(ΔE − (|m|²/ρ − |m_ref|²/ρ_ref)/2), the kinetic difference written with Δm and Δρ. Momentum fluxes take the gauge pressure (p_ref sums to zero over a closed cell), and the slip mirror keeps it, so that FP32 does not round a small gauge pressure away through p_ref; every other use of p adds p_ref back. FP32 storage resolves U to ε₃₂|U_ref| in absolute terms: a cell whose density falls below about 10⁻⁷ ρ_ref cannot be stored, its residual is not finite, and the run ends `diverged`. In `fp32+acc64` the residual evaluated in FP32 has a rounding floor: on the MMS (hex, n = 16, from the freestream) the residuals drop 5.9–6.0 orders with absolute storage and 6.3–6.4 orders with the reference state, then stay there. The rest of the floor is the FP32 rounding of the fluxes themselves, ε₃₂ times the face flux, so how many orders a case drops in `fp32+acc64` depends on how far its first residual stands above it (3.8 orders on a subsonic NACA 0012 slab from the freestream, where `fp64` drops 12 orders and the lift agrees to 3·10⁻⁶; the limiter plays no part). The physics' residual scale per equation is that flux, Φ_k = f_k √(Σ_i A_i² / N) over the history's N cells, A_i the surface of cell i and f_k the freestream's ρq, ρq² (each momentum) and ρHq, q = |v| + c. Measured plateaus sit at 0.01–0.04 ε₃₂Φ_k on the NACA 0012 (M 0.5, α 0° and 2°, limited or not), up to 0.07 on the MMS (hex, n = 16) and up to 0.12 on Ni's transonic bump (M 0.675, three meshes, rising with refinement; measured by CFD - Multigrille), so a steady `fp32+acc64` run converges at 0.5 ε₃₂Φ_k at the latest ([case-schema.md](../contracts/case-schema.md#time)). A ten-order drop is therefore measured in `fp64`; in `fp32+acc64` the gate is the integrated forces and the surface pressure against `fp64`, and the plateau is reported ([acceptance.md](../validation/acceptance.md#m2--compressible-euler)). The surface pressure must then agree with `fp64` to max |ΔCp| ≤ 10⁻³ over the wall faces, shock cells excepted (reported there): the plateau leaves state differences of order 10⁻⁶–10⁻⁵ relative, so 10⁻³ is a margin of 100 or more, and it is ten times finer than the 0.01 to which Cp distributions are compared with experiment.

## Manufactured solution

Steady `euler` solution of autan-vv (Roy et al. 2004 form), boundary faces at the exact state, cell sources at centroids; Newton-Krylov on the generated residual (`physics/mms/euler_mms.py`). The table is filled from the study of PR #99.
