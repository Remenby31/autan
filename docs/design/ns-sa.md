# Navier-Stokes and Spalart-Allmaras discretisation

The M3 physics of [PLAN.md](../PLAN.md): the viscous fluxes of the compressible Navier-Stokes equations and, to come, the Spalart-Allmaras model in its negative variant. They extend the [Euler discretisation](euler.md): the residual is the Euler residual plus the viscous terms, written once in the kernel DSL (`kernels/navier_stokes.py`, [kernel-generator.md](kernel-generator.md)) and driven by `autan::physics::NavierStokes<P>` (`physics/`). Gates: [acceptance.md §M3](../validation/acceptance.md#m3--navier-stokes-and-spalart-allmaras). Issue: #137.

## Residual

R_c = R_c^Euler − Σ_f F_v · S_f, with F_v = (0, τ n, u · τ n + k ∇T · n):

| Term | Choice | Source |
|---|---|---|
| Stress | τ = μ_e (∇u + ∇uᵀ − ⅔ (∇·u) I), μ_e = μ(T) + μ_t | Stokes' hypothesis, Boussinesq; TMR, "Implementing turbulence models into the compressible RANS equations" |
| Viscosity | Sutherland, μ = μ_ref (T/T_ref)^{3/2} (T_ref + S)/(T + S), or constant | case gas (`viscosity.law`) |
| Conduction | k = c_p (μ / Pr + μ_t / Pr_t), Pr_t = 0.9 | TMR |
| Face gradients | average of the two cells' least-squares gradients, its component along the line of centres replaced by the difference quotient: ∇φ_f = ∇̄φ + ((φ_R − φ_L)/\|d\| − ∇̄φ · e) e, e = d/\|d\| | Blazek (2015), eq. 5.69 |
| Face values | average of both linear reconstructions at the face centroid (unlimited) | |
| Warped faces | the stress work through the fan triangles, τ : (∇u M), M the face moment of [euler.md](euler.md#warped-faces) | |
| ∇T | from the gradients of ρ and p: ∇T = (∇p − R T ∇ρ)/(ρ R) | |

The least-squares gradients are the Euler ones (unlimited, face2 stencil); the limiter acts on the convective reconstruction only.

## Boundary conditions

A boundary face's gradient comes from a **cubic** least-squares fit of T and the velocity (19 unknowns: gradient, Hessian, third derivatives) over the two rings of cells around the boundary cell (cells sharing a node with it or with one of those) and the boundary faces among them that give T and the velocity (manufactured and wall faces), evaluated at the face, with its component along r = x_f − x_c set to the secant slope (φ_b − φ_c)/|r|, which the cubic reproduces. Only cells owning such a face carry the fit. Each unknown is scaled by the stencil's extent along its directions and the weights are 1/d² in those scaled coordinates, so that a wall-resolved layer (aspect ratio 10⁴ in the TMR grids) keeps the normal matrix well conditioned. Monomials in a direction where no stencil entry extends (the span of a one-cell-thick 2D mesh) are left out. A cell whose two rings leave the rest undetermined, such as a wall's last cell before an outlet (three positions along the wall), takes three rings; one whose three rings do not either is refused with an error.

Measured (MMS `navier_stokes`, hex and poly): the linear least-squares gradient of a boundary cell is first order (one-sided stencil), so the half-cell difference built on it makes the wall stress first order; a quadratic fit makes it second order face by face (1.98) and the cubic fit third order (2.95–2.98 on hex 32–96, 3.09–3.12 on poly 16–48). The density's convergence on these families is almost the same with the three (fitted order 1.73 on hex 32–96, 1.79 on poly 16–48; [acceptance.md](../validation/acceptance.md#m3--navier-stokes-and-spalart-allmaras)), so the boundary faces are not what keeps it below 2 there. On one ring of cells the quadratic fit made the hybrid family weakly unstable (the residual grew from 10⁻¹⁰ after convergence); the two-ring cubic fit converges.

The boundary state is that of the Euler boundary condition ([euler.md](euler.md#boundary-conditions)), with one more kind:

| Kind | Values | Boundary state | Viscous flux |
|---|---|---|---|
| `wall` | T_w, or 0 (adiabatic) | (p_i / (R T_w) or ρ_i, 0, 0, 0, p_i): no slip, zero normal pressure gradient | full stress; heat flux from T_w (isothermal) or none (adiabatic, whose wall temperature is the cell's extrapolated to the face) |
| `slip` | — | mirror state | none |
| others | as Euler | as Euler | full, at the boundary state |

The convective flux of a `wall` face is the pressure flux of a slip face (v · n = 0 on both). Wall faces are not in the least-squares stencil, like slip faces: only the velocity is known there.

## Implicit solver

The viscous face loops write their Jacobian blocks dF_v/dW_L, dF_v/dW_R in the same pass as the residual, the cell gradients held fixed: the difference quotients of the face gradients give the compact part of the viscous Jacobian. `NavierStokes<P>::assemble` adds them, times dW/dU, to the Euler blocks. The local time step adds four times the viscous spectral radius Σ_f max(4/3, γ) (μ/Pr + μ_t/Pr_t) A² / (ρ V) to the convective one (Blazek 2015, eqs. 6.20 and 6.22).

## Verification

| Test | Oracle | Where |
|---|---|---|
| Linear velocity and temperature, every cell type, `fp64` and `fp32+acc64`, with and without a no-slip wall (isothermal and adiabatic) | viscous residual = −V (0, 0, 0, 0, τ : ∇u) to round-off | `tests/physics/test_navier_stokes.cpp` |
| Cubic velocity vanishing on a wall, every cell type and a one-layer slab stretched 10⁴:1 at the wall ending at an outlet, both precisions | wall shear stress μ ∂u_t/∂y at each face to round-off | same |
| MMS `navier_stokes` through `autan-run` and `autan-vv verify`, gate `ns-mms-order` | Roy et al. (2004) form, μ = 10, Pr = 0.72 | `physics/mms/study.sh` |
| MMS `rans_sa`, gate `rans-sa-mms-order` | Eça et al. (2007), Oliver et al. (2012) principles, wall y = 0 | same |

Levels: [acceptance.md §M3](../validation/acceptance.md#m3--navier-stokes-and-spalart-allmaras). The `rans_sa` fields have a boundary layer of thickness 0.15 at y = 0, under three cells on the coarse levels of the cube families, hence its Cartesian family (n, 4n, n) (`autan_vv.meshes.families.cartesian`, written with `write_fluent`); its aspect ratio is the same on every level, so h = (V/N)^(1/3) measures the refinement.
