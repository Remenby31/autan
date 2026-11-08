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
| Warped faces | the stress work through the fan triangles, τ : (∇u M), M the face moment of [euler.md](euler.md#warped-faces); none on a wall at rest, whose velocity is zero over the whole face | |
| ∇T | from the gradients of ρ and p: ∇T = (∇p − R T ∇ρ)/(ρ R) | |

The viscous terms take the Euler linear least-squares gradient (unlimited, face2 stencil). The convective reconstruction takes the quadratic gradient in every cell ([euler.md](euler.md#quadratic-gradient)), and the limiter acts on it only.

On a CUDA device, the stepper `make_device_stepper` of a `NavierStokes<P>` (`physics/src/navier_stokes_device.cu`) runs the same generated loops after the device Euler residual, from the host's boundary fit and eddy viscosity. Under the mesh's own numbering as global keys (identity keys, one rank), both sides gather each cell's faces in the same order. With the generated loops built with `--fmad=false` (#571), the device residual then has the host's bits wherever no `exp` or `log` is evaluated: state, slip, outlet and wall conditions (#586, `tests/physics/test_navier_stokes_device.cpp`). The Spalart-Allmaras model's device copy (`make_device_spalart_allmaras`) runs on that stepper's arrays and writes the eddy viscosity into it. Its residual has the host's bits where nt < 0, and stays within #586's tolerance where `fw` and `ft2` evaluate `exp` and `log`. `autan-run --backend cuda` runs laminar cases on the first; SA runs stay on the host for the alpha ([ADR 0005](../adr/0005-gpu-scope-alpha-laminar-beta-rans.md)). An unsteady laminar run on the device repeats the host's state bit for bit at every step under identity keys, and stays within #586's declared bound of the host's default run.

### Limits on a CUDA device

What a device run does not claim, for the alpha ([ADR 0005](../adr/0005-gpu-scope-alpha-laminar-beta-rans.md), #586):

- **Several steps against the host's default run:** agreement over several steps is not bounded. A device run repeats the host's state bit for bit at every step only when the host sums in the device's order (identity keys, one rank). Against `autan-run --backend cpu`, which sums in another order, the difference is measured and reported but no bound is claimed (#586 criterion 4b, not demonstrated; #619).
- **Farfield and inlet faces in `fp32+acc64`:** the device residual is shown tight to the host's up to stencil gain G = 4.79, the largest measured. No repo mesh exceeds it, by construction, so W0717 guards meshes beyond the measured coverage. A run beyond it warns W0717; use `fp64` (#608).
- **SA:** RANS runs stay on the host. SA on a device is `E0706`.

## Boundary conditions

A boundary face's gradient comes from a **cubic** least-squares fit of T and the velocity (19 unknowns: gradient, Hessian, third derivatives) over the two rings of cells around the boundary cell (cells sharing a node with it or with one of those) and the boundary faces among them that give T and the velocity (manufactured and wall faces), evaluated at the face and corrected along e = r/|r|, r = x_f − x_c, by the difference between the values' rise φ_b − φ_c and the cubic's over r, divided by |r|: zero on cubic fields. Cells owning such a face carry the fit, and so do their face neighbours, whose least-squares stencils reach those boundary faces on one side only; an interior face takes from each side its fit where it carries one, its least-squares gradient otherwise. The weights are 1/d² with each axis scaled by the stencil's extent along it, so that the entries across a wall-resolved layer (aspect ratio 10⁴ in the TMR grids) do not outweigh the others. The normal matrix is solved in the frame of the wall (the normal of the stencil's boundary face nearest to the cell, completed by the global axis least aligned with it), each unknown scaled by the stencil's extent along its local directions, and its derivatives are rotated back: on a sloped or curved wall the global axes mix the wall-normal spacing with the slope and the matrix scaled along them is singular to round-off (#448). The polynomial space and the weights do not depend on the frame, so neither does the fit; on walls along an axis the frame is the global one. Monomials in a direction where no stencil entry extends (the span of a one-cell-thick 2D mesh) are left out. A cell whose two rings leave the rest undetermined, such as a wall's last cell before an outlet (three positions along the wall), takes three rings; one whose three rings do not either is refused with an error.

Measured (MMS `navier_stokes`, hex and poly): the linear least-squares gradient of a boundary cell is first order (one-sided stencil), so the half-cell difference built on it makes the wall stress first order; a quadratic fit makes it second order face by face (1.98) and the cubic fit third order (2.95–2.98 on hex 32–96, 3.09–3.12 on poly 16–48). The density's convergence on these families is almost the same with the three (fitted order 1.73 on hex 32–96, 1.79 on poly 16–48; [acceptance.md](../validation/acceptance.md#m3--navier-stokes-and-spalart-allmaras)), so the boundary faces are not what keeps it below 2 there. On one ring of cells the quadratic fit made the hybrid family weakly unstable (the residual grew from 10⁻¹⁰ after convergence); the two-ring cubic fit converges.

The boundary state is that of the Euler boundary condition ([euler.md](euler.md#boundary-conditions)), with one more kind:

| Kind | Values | Boundary state | Viscous flux |
|---|---|---|---|
| `wall` | T_w, or 0 (adiabatic) | (p_i / (R T_w) or ρ_i, 0, 0, 0, p_i): no slip, zero normal pressure gradient | full stress; heat flux from T_w (isothermal) or none (adiabatic, whose wall temperature is the cell's extrapolated to the face) |
| `slip` | — | mirror state | none |
| others | as Euler | as Euler | full, at the boundary state |

The convective flux of a `wall` face is the pressure flux of a slip face (v · n = 0 on both). Wall faces are not in the least-squares stencil, like slip faces: only the velocity is known there.

Spalart-Allmaras takes ν̃ = 0 on a `wall`, the data on `state` and `farfield` faces, `nu_tilde_ratio` times the laminar ν of the boundary state on `inlet` faces (case-schema.md#turbulence), and the adjacent cell's value on `slip` and `outlet` faces, which then carry no diffusive flux of ν̃ (zero normal derivative). No ν̃ is convected through a `wall` or a `slip` face.

## Implicit solver

The viscous face loops write their Jacobian blocks dF_v/dW_L, dF_v/dW_R in the same pass as the residual, the cell gradients held fixed: the difference quotients of the face gradients give the compact part of the viscous Jacobian. The boundary loop's block also carries the boundary state's dependence on the cell (dF_v/dW_b dW_b/dW_L): it evaluates the boundary state from the cell, the same values as `prim_b`. `NavierStokes<P>::assemble` adds them, times dW/dU, to the Euler blocks, with the boundary fit's entries that fall in the block pattern. `NavierStokes<P>::gradient_tangent` gives the rest as a product, per face set: the cell gradients' and the fits' generated tangents, the fits' boundary entries at adiabatic walls (the cell's temperature extrapolated to the face), less the fit blocks already assembled; with both face sets, A v + T_interior v + T_boundary v is the residual's derivative without a limiter (`tests/physics/test_navier_stokes.cpp`). The steady operator, like Euler's, adds the boundary faces' product only ([euler.md](euler.md#time-stepping)). The local time step adds four times the viscous spectral radius Σ_f max(4/3 (μ + μ_t), γ (μ/Pr + μ_t/Pr_t)) A² / (ρ V) to the convective one (Blazek 2015, eqs. 6.20 and 6.22, with the momentum and heat diffusivities bounded apart, so that a large Pr does not hide the first). With Spalart-Allmaras, every flow residual reads the μ_t of its own state, from that state's primitives and ρν̃, which costs one more flow evaluation per step: the residual is then a function of the state alone, and a continued run repeats the uninterrupted one bit for bit (#430).

## Verification

**Status.** `ns-mms-order` passes in FP64 on every family, clean and perturbed, and in FP32 on all but the clean hexahedral one: there, ρv's error at n = 96 and 128 falls under the gate's evidence threshold of 100 times the FP32 rounding floor, as it does for any scheme this accurate (#524). `rans-sa-mms-order` failed on ρE on its former family (n, 4n, n) at n = 24–64, a failure recorded as such; the family's anisotropy, not the scheme or the model, keeps pressure pre-asymptotic there, and the gate's family for later runs is in [ADR 0002](../adr/0002-rans-sa-order-gate-family.md) (#497). Measurements and the choice of the quadratic gradient: [#137](https://github.com/Remenby31/autan/issues/137).

| Test | Oracle | Where |
|---|---|---|
| Linear velocity and temperature, every cell type, `fp64` and `fp32+acc64`, with and without a no-slip wall (isothermal and adiabatic) | viscous residual = −V (0, 0, 0, 0, τ : ∇u) to round-off | `tests/physics/test_navier_stokes.cpp` |
| Cubic velocity vanishing on a wall, every cell type and a one-layer slab stretched 10⁴:1 at the wall ending at an outlet, both precisions | wall shear stress μ ∂u_t/∂y at each face to round-off | same |
| Slab closed by adiabatic walls at rest, one with warped faces, and by slip faces, both precisions | the energy residuals sum to zero | same |
| ν̃ linear, positive or negative, with a gradient normal to a slip or outlet face, on a sheared slab, both precisions | the ν̃ residuals sum to the boundary diffusion through the other faces plus the cb2 volume term | same |
| Uniform ν̃, a flow and its reverse, every cell type closed by slip faces, both precisions | no ν̃ convected through the boundary | same |
| Uniform flow through an inlet at its own total state and an outlet, ν̃ = 3ν, both precisions | every ν̃ residual zero | same |
| Uniform negative ν̃ at rest, every cell type, both precisions | each residual is the SA-neg destruction −V ρ c_w1 (ν̃/d)²; no eddy viscosity | same |
| Fluid at rest, Pr = 100, every cell type, both precisions | viscous spectral radius Σ_f 4/3 μ A²/(ρV) | same |
| MMS `navier_stokes` through `autan-run` and `autan-vv verify`, gate `ns-mms-order` | Roy et al. (2004) form, μ = 10, Pr = 0.72 | `physics/mms/study.sh` |
| MMS `rans_sa`, gate `rans-sa-mms-order` | Eça et al. (2007), Oliver et al. (2012) principles, wall y = 0 | same |

Levels: [acceptance.md §M3](../validation/acceptance.md#m3--navier-stokes-and-spalart-allmaras). The `rans_sa` fields have a boundary layer of thickness 0.15 at y = 0, under three cells on the coarse levels of the cube families, hence its Cartesian families (`autan_vv.meshes.families.cartesian`, written with `write_fluent`), whose aspect ratio is the same on every level, so that h = (V/N)^(1/3) measures the refinement ([ADR 0002](../adr/0002-rans-sa-order-gate-family.md)).

On the poly family, ρv's fitted L2 order exceeds 2.6 for every convective gradient measured (the band, its average-fit variant, and the quadratic fit in every cell, #137), so `ns-mms-order` reports it for investigation. The excess is pre-asymptotic. ρv's error fits A h² + B h³ with the h³ term as large as the h² term at n = 64, and with the same ratio B/A ≈ 60–68 for every gradient. One level finer the pairwise order falls as that fit predicts: on the x86 poly family (`autan-vv meshes poly --seed 0`, Gmsh 4.15.2; n = 64 sha256 46da8c94…, n = 96 sha256 52e1d808…), ρv's pair 64→96 is 2.44 against a predicted 2.47, where a single power law would give 2.67 (#514). The gate still flags it at its levels.
