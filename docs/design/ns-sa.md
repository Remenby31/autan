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

A boundary face's gradient joins the cell value to the boundary state at the face centroid, with the same correction along r = x_f − x_c. The boundary state is that of the Euler boundary condition ([euler.md](euler.md#boundary-conditions)), with one more kind:

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
| MMS `navier_stokes` through `autan-run` and `autan-vv verify`, gate `ns-mms-order` | Roy et al. (2004) form, μ = 10, Pr = 0.72 | `physics/mms/study.sh` |
