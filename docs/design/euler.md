# Compressible Euler discretisation

The M2 physics of [PLAN.md](../PLAN.md): cell-centred finite volumes on arbitrary polyhedra, written once in the kernel DSL (`kernels/euler.py`, [kernel-generator.md](kernel-generator.md)) and driven by `autan::physics::Euler<P>` (`physics/`). Gates: [acceptance.md §M2](../validation/acceptance.md#m2--compressible-euler).

## Residual

R_c = Σ_f F(W_L, W_R)·S_f − s_c V_c over the faces of cell c, in conserved variables U = (ρ, ρu, ρv, ρw, ρE) of an ideal gas.

| Step | Choice | Source |
|---|---|---|
| Variables reconstructed | primitive W = (ρ, u, v, w, p) | |
| Gradients | weighted least squares (1/d²) over the **face2 stencil**: face neighbours, their face neighbours, and the boundary faces of both | [Stencil](#gradient-stencil) |
| Limiter | Venkatakrishnan, bounds over the gradient stencil, ε_k² = (K/L)³ V q_k², K = 10, L the reference length, q_k the freestream scale of W_k | Venkatakrishnan (1995), eq. 5 |
| Face states | W_f = W_c + φ ∘ (∇W · (x_f − x_c)) | |
| Warped faces | flux of the linear reconstruction through the fan triangles: Σ_j dF_j[W; ∇W · M_{:,j}], M = Σ_t (c_t − x_f) ⊗ S_t | [Warped faces](#warped-faces) |
| Fluxes | Roe with Harten's fix on the acoustic waves only; HLLC with Einfeldt speeds; AUSM+-up with M_co the freestream Mach number; the case's `physics.flux` picks one, Roe by default ([case-schema.md](../contracts/case-schema.md#physics)) | Roe (1981); Toro, Spruce, Speares (1994); Batten et al. (1997); Liou (2006) |

Harten's fix stays off the contact and shear eigenvalue so that every flux keeps a stationary contact exact (`tests/physics/test_euler.cpp`).

## Boundary conditions

Each boundary face carries a code and five values (`BoundaryKind`). The boundary state is computed from the reconstructed interior state at the face, then the numerical flux takes (W_f, W_b); slip faces take the pressure flux (0, p_f n, 0).

| Kind | Values | Boundary state |
|---|---|---|
| `state` | W | given (manufactured solutions) |
| `farfield` | freestream W, or W plus a [point vortex](#point-vortex) | Riemann invariants normal to the face; entropy and tangential velocity from the upwind side; supersonic faces take the upwind state |
| `slip` (slip wall, symmetry) | — | mirror state (normal velocity reflected) in the limiter's bounds and the spectral radius, in the gradient fit only when the fit falls back; pressure flux from the reconstructed state |
| `inlet` | p₀, T₀ | flow normal to the face; R⁺ from the interior and c_p T₀ give c_b (Carlson 2011) |
| `outlet` | p | ρ and v from the outgoing characteristics; supersonic outflow extrapolates |

The gradient's boundary values are the boundary states of the first-order interior state (`boundary_states`, a frozen loop: its derivatives are dropped, so the generated adjoint of the gradient is approximate at boundary faces). They enter the least-squares fit only on `state` faces, where the condition gives the value at the face; on the other kinds the value is computed from the adjacent cell's state, which belongs at the cell centroid, and the fit uses the stencil's cells and `state` faces alone (a cell whose fit is then undetermined keeps every boundary face, as a one-cell-thick slab does: [Time stepping](#time-stepping)). Every boundary value, whatever its weight, enters the limiter's bounds (the minimum and maximum over the stencil). Offset components within 64 ulps of the coordinates count as zero: across a one-cell-thick slab the centroids differ only by round-off, and the fit would otherwise turn it into a gradient across the layer. The fit is determined when det A > 10⁻¹⁰ A₁₁A₂₂A₃₃, which is independent of the cells' aspect ratio. Where the linear reconstruction would make ρ or p non-positive, the face takes the cell value. The warped-face correction applies to interior faces and to the pressure of slip faces.

### Point vortex

A lifting 2D body leaves the circulation's vortex in the far field, so a uniform farfield state makes the lift depend on the outer radius. With `point_vortex`, each farfield face takes the freestream plus the compressible vortex of Thomas and Salas (1986, AIAA J. 24(7)) centred at `center`, in the x–z lift plane (steady cases, zero sideslip, subsonic freestream). The velocity is the Prandtl-Glauert vortex (β = √(1 − M∞²)):

u = u∞ + q sin θ, w = w∞ − q cos θ, q = Γβ / (2πr (1 − M∞² sin²(θ − α))).

Here r and θ are the face centroid's distance and polar angle from the vortex in x–z. The speed of sound follows from the freestream total enthalpy; density and pressure are isentropic (`physics/include/autan/physics/point_vortex.hpp`). The circulation is Kutta–Joukowski's Γ = L′/(ρ∞V∞), where L′ is the lift on the `surfaces` zones per unit of their extent along y. It is measured on the residual of each step and applied from the next, so a converged solution satisfies it exactly. The implicit Jacobian omits it, and restarts carry it (`VortexCirculation<i>` in `RestartState`).

## Gradient stencil

Least squares over the face neighbours alone make the unlimited second-order scheme **linearly unstable** on the tetrahedra of `autan-vv meshes tet`: the Jacobian of the residual has eigenvalues with negative real part. The same holds for the scalar advection of the generator spike, so the stencil, not the Euler flux, is at fault. Widening the stencil removes every unstable eigenvalue, as Diskin and Thomas (2011, AIAA J. 49(4), "inviscid fluxes") recommend for cell-centred schemes.

Spectrum of the dense Jacobian (finite differences) at the manufactured solution, tet n = 5 (762 cells), Roe, unlimited:

| Gradient stencil | Eigenvalues with Re < 0 | min Re (scaled) |
|---|---|---|
| face neighbours | 42 | −3.53 |
| face2 | 0 | +0.87 |
| vertex (cells sharing a node) | 0 | +0.91 |

Face neighbours stay stable on the hexahedral, prismatic, hybrid and polyhedral families (no eigenvalue with Re < 0 at n = 5–6). Cost on GB10 (tetrahedra n = 96, 3.9 M cells, `fp32`, measured by CFD - Générateur kernels): face neighbours 0.50 ns/cell (93 % of peak bandwidth), face2 1.82 ns/cell (58 %), vertex 20 ns/cell (25 %). face2 is the default: stable, second order, 11× cheaper than vertex. Both stencils give order 2 ([MMS](#manufactured-solution)). The limiter's bounds use the same stencil: bounds over the four face neighbours of a tetrahedron are narrower than the values the face2 gradient extrapolates, so the limiter clips smooth solutions and the limited scheme falls to first order on tetrahedra (measured: finest-pair order 0.58–1.20).

Next to `state` boundaries the face2 fit is one-sided: a cell whose stencil reaches a boundary face takes the exact value at the face centroid, half a cell away, and its gradient carries an O(h) error. The truncation R_h(u)/V − S at the exact MMS state, on the `hex` family 32–96 (Roe, unlimited), therefore falls in the band of cells reconstructing with those gradients. Excluding two rings of cells from the boundary, continuity converges at 1.86, 1.69 and 1.62 between levels (1.2 × 10⁻⁴ at 96³, about h^1.5 in L2 for an O(h) error one cell thick). Excluding three rings, it converges at 2.18, 1.97 and 1.91 (5.8 × 10⁻⁵), and deeper sets give the same level.

### Quadratic gradient

The Navier-Stokes physics reconstructs its convective face states with the gradient of a quadratic least-squares fit in every cell (`GradientFit::quadratic`, `band_gradients`). Its viscous terms keep the linear face2 gradient (`linear_gradient`), and the wall shear stress has its own fit ([ns-sa.md](ns-sa.md)). The Euler physics keeps the linear gradient: a quadratic fit near the boundary lowered the coarse levels' errors but cost the `euler-mms-order` gate (hex 12–32, ρu fit/finest pair 1.85/1.90 with Roe against 2.06/1.99).

Why every cell ([#137](https://github.com/Remenby31/autan/issues/137), `ns-mms-order`, μ = 10): the linear fit is first order in the cells whose stencil reaches the boundary (above). With the quadratic fit in face rings 1 and 2 only, the tetrahedral family fails the gate (ρ 1.876, ρE 1.845 on the finest pair), and so does the exact gradient in face rings 1–3: the switch in accuracy at the edge of the fitted cells limits the order. With the fit in every cell, there is no switch and the gate passes on every family ([ns-sa.md](ns-sa.md#verification)).

The fit is over two vertex rings, with weights 1/|d|² in coordinates scaled per direction by the stencil's extent. It is fitted to the values of those cells, each taken as the point value at its centroid, and to the states of their boundary faces that give one, also as point values. The fit to cell averages (each cell's second moment about its centroid added, since the average of a quadratic over a cell adds H : M/2) was measured and not kept (#490). It makes the boundary-face normal derivative second order on exact averages, but it lowers the hexahedral family's momentum orders (ρv's finest pair 1.849 on hex 32–128, against 1.934 with the point fit) and changes nothing on tetrahedra. Directions where no entry extends (a one-cell-thick slab) are left out. A cell with fewer than 1.5 entries per unknown keeps the linear fit, and the run logs how many such cells there are (none on the MMS families or the TMR grids).

The cell's own boundary faces are weighted entries, not equality constraints as Ollivier-Gooch and Van Altena (2002) recommend. As constraints, they make the scheme linearly unstable. Spectrum of the dense Jacobian at the MMS, Roe, unlimited, as above:

| Gradient | tet n = 5: Re < 0, min Re | hybrid n = 5: Re < 0, min Re |
|---|---|---|
| face2 only | 0, +0.87 | 0, +0.56 |
| quadratic, faces weighted | 0, +0.87 | 0, +0.55 |
| quadratic, own faces as constraints | 49, −1.56 | 20, −2.73 |

The constrained fit diverges on the hybrid Euler MMS (n = 16) as reconstruction; the weighted fit converges there in 41 implicit steps.

## Warped faces

The faces of polyhedral duals are not planar. A one-point flux at the face centroid then commits an O(h) relative error per face, and the solution converges at first order (measured: finest-pair order 1.05–1.26 on the `poly` family). Integrating the linear reconstruction over the fan triangles through the face moment M restores order 2 (2.1–2.6). On a mesh whose faces are all proven planar, M is an exact zero (`PackedGeometry::all_planar`: tetrahedra, axis-aligned and extruded faces), and the residual runs the face loops without that term (`*_planar`), the same values wherever the full loops' arithmetic stays finite, reading no face moments. The choice is one per run, the minimum over the ranks, and logged; the Jacobian and tangent loops keep their full form. Navier-Stokes runs always take the full loops: the planar boundary loops have no no-slip wall flux (#540).

## Precision

Cells store U − U_ref, the reference being the freestream, and the primitive pressure is the gauge pressure p − p_ref, computed from the differences only: p − p_ref = (γ − 1)(ΔE − (|m|²/ρ − |m_ref|²/ρ_ref)/2), the kinetic difference written with Δm and Δρ. Momentum fluxes take the gauge pressure (p_ref sums to zero over a closed cell), and the slip mirror keeps it, so that FP32 does not round a small gauge pressure away through p_ref; every other use of p adds p_ref back. FP32 storage resolves U to ε₃₂|U_ref| in absolute terms: a cell whose density falls below about 10⁻⁷ ρ_ref cannot be stored, its residual is not finite, and the run ends `diverged`. In `fp32+acc64` the residual evaluated in FP32 has a rounding floor: on the MMS (hex, n = 16, from the freestream) the residuals drop 5.9–6.0 orders with absolute storage and 6.3–6.4 orders with the reference state, then stay there. The rest of the floor is the FP32 rounding of the fluxes themselves, ε₃₂ times the face flux, so how many orders a case drops in `fp32+acc64` depends on how far its first residual stands above it (3.8 orders on a subsonic NACA 0012 slab from the freestream, where `fp64` drops 12 orders and the lift agrees to 3·10⁻⁶; the limiter plays no part). The physics' residual scale per equation is that flux, Φ_k = f_k √(Σ_i A_i² / N) over the history's N cells, A_i the surface of cell i and f_k the freestream's ρq, ρq² (each momentum) and ρHq, q = |v| + c. Measured plateaus sit at 0.01–0.04 ε₃₂Φ_k on the NACA 0012 (M 0.5, α 0° and 2°, limited or not), up to 0.07 on the MMS (hex, n = 16), up to 0.12 on Ni's transonic bump (M 0.675, three meshes, rising with refinement; measured by CFD - Multigrille) and 0.75 on the Navier-Stokes manufactured solution (tetrahedra, n = 16, #467). A steady `fp32+acc64` run converges on its plateau by the rule of [case-schema.md](../contracts/case-schema.md#time). A ten-order drop is therefore measured in `fp64`; in `fp32+acc64` the gate is the integrated forces and the surface pressure against `fp64`, and the plateau is reported ([acceptance.md](../validation/acceptance.md#m2--compressible-euler)). The surface pressure must then agree with `fp64` to max |ΔCp| ≤ 10⁻³ over the wall faces, shock cells excepted (reported there): the plateau leaves state differences of order 10⁻⁶–10⁻⁵ relative, so 10⁻³ is a margin of 100 or more, and it is ten times finer than the 0.01 to which Cp distributions are compared with experiment.

## Time stepping

Steady runs march in pseudo-time with an implicit step, (V/Δt + J̃) ΔU = −R(U), Δt = CFL V / Σ_f (|v·n| + c) A local to each cell:

- J̃ is the approximate Jacobian: dF/dW of every face with the reconstruction's gradients and limiter held fixed, written by the face loops in the same pass as the residual (`*_jacobian` loops of `kernels/euler.py`), times dW/dU of each cell; it is assembled in 5 × 5 blocks (`BlockMatrix`) and the step's system is solved to a relative residual of 0.1 by the [agglomeration multigrid](multigrid.md) (`linear_solver = "multigrid"`, one rank), or by GCR preconditioned with four symmetric block Gauss-Seidel sweeps over the runtime's colouring by global id (`"gauss-seidel"`, to the same 0.1), which runs the same algorithm on any number of ranks: ghost cells take ΔU after each colour, and dW/dU and the other cell data the coupling blocks read are filled into them first.
- On every path (`multigrid`, `lines`, `gauss-seidel`), inviscid or viscous, the operator also carries the boundary faces' derivative through the reconstruction's gradients, (∂F/∂∇W)(∂∇W/∂U) with the limiter held, applied as a product (`Euler::gradient_tangent`), not as blocks: the gradient stencil reaches second neighbours. The boundary values a cell's fit keeps when its interior stencil cannot determine its gradient (every cell of a one-cell-thick slab whose faces across the layer are not `state` faces) are functions of their owner cell's state; the product carries their derivative too (`boundary_states_jacobian`). Without it, the step in the large far-field cells of ONERA M6 is unstable at the CFL numbers the SER reaches, a period-2 mode three orders above the wing's residual (#244). Navier-Stokes adds its viscous part, through the cell gradients and the boundary fit (`NavierStokes::gradient_tangent`, [ns-sa.md](ns-sa.md)).
- The run starts at first order (unless the case sets `time.start_order = "second"`, [case-schema.md](../contracts/case-schema.md#time)), where J̃ is the exact Jacobian, and solves second order from the step after the one whose first-order residual has dropped three orders, or that is the 201st (`FirstOrderStart`). In between, an `fp64` run also switches once its drop has stalled (100 steps without a new reference, the reference being the first drop, then each drop below 0.9 times the reference before it; no floor involved); transonic NACA 0012 at α = 0° never drops three orders and switches there, and converging `fp64` starts go at most 42 steps without a new reference (#611). An `fp32+acc64` run, whose three orders can lie below round-off and whose stall followed summation order (#259), switches instead once every flow equation's first-order RMS residual level (the median of its last 100 values, `runtime::plateau_window`) is at most `runtime::round_off_band` (10) times its round-off floor C·ε₃₂·Φ_k (case-schema.md, Time), equations exempt at their floor left out. The level is first defined at the 100th step, so this is not an adaptive length: on every NACA 0012 case measured (M 0.5 and M 0.8 at α = 1.25°, 64² and 256²) the levels are already within the band then, and the start lasts **in practice 100 first-order steps, set by the window**, the same on any summation order; where the residual rises (NACA 0012 M 0.8 α = 0°, ONERA M6) it lasts 201, the cap, 100 first-order steps more than the stall gave, and the run takes 29–43 % more steps to converge ([#259](https://github.com/Remenby31/autan/issues/259)). These steps, at most 201, are provisional: they do not count towards a round-off plateau of the run. The log names the rule that ended the start (`first-order start ended`: `drop`, `stall`, `level` or `cap`). From a field near its solution the first-order start moves the field away before the second order brings it back. Measured on NACA 0012 256² (fp64, #636 X636b), started from its own converged solution, the forces leave their 1 % band and settle again after 37–67 iterations. With `start_order = "second"` they stay in band from iteration 1, and restarts from a neighbouring Mach and incidence settle in 5–54 iterations instead of 20–57.
- CFL = 1000 / (the largest, over the flow equations, of the residual of the order being solved relative to its value at the first step of that order, which a continued run takes from its checkpoint; an equation still exempt is left out, and a woken one is measured from the reference of its kind, [case-schema.md](../contracts/case-schema.md#time)) (switched evolution relaxation, Mulder and van Leer 1985), times a penalty, between 1 and 10⁶. The penalty halves after a step that had to be shortened or whose largest relative residual rose by more than 10 % over the step before's (so also on the first second-order step, where it restarts at 1), and otherwise recovers by half, up to 1. Above a case-dependent CFL number the defect correction with the approximate Jacobian stops converging, so the CFL number follows that success instead of a constant bound. On the TMR flat plate (137 × 97, SA, `fp64`, #185) the unbounded law climbs to about 1.5 × 10⁴ and stalls one order down; constant bounds of 1000, 2000 and 3000 reach ten orders in 2487, 2104 and 1964 steps, 5000 diverges, and this law reaches them in 1729 steps without a case-tuned bound. The 10 % is a noise margin: halving on every rise ratchets the penalty down on the small rises of a converging run, and the flat plate then stalls 6.4 orders down. The device's steady step keeps a fixed CFL number of 0.8.
- A step is shortened so that no cell loses more than half its density or pressure: each cell's admissible fraction (its density bound, then halved until its pressure bound holds) independently, the least over every rank. It is logged (`solver` warning with alpha and the CFL number); the next CFL number is then at most half the one used.

Steps to ten orders: [multigrid.md](multigrid.md#measurements). A Newton-Krylov variant (FGMRES on finite differences of the full residual, the limiter frozen, block Gauss-Seidel sweeps as preconditioner) was less robust far from the solution: on the MMS from the freestream (hex, n = 16) its steps were shortened at almost every iteration and the run stalled after 1.3 orders, where the defect correction converged ten orders in 177 steps (Gauss-Seidel, CFL = 10 / drop). It is left for M3 as an end phase, switched on once the residual has dropped.

Unsteady runs use SSP-RK3 (Shu and Osher 1988) with one global time step from `dt` or `cfl`.

Start: each cell takes its zone's `initial` state, or, with an initial-state file ([initial-state.md](../contracts/initial-state.md)), its row of the file; `autan_vv.initial_state.write_initial_state` writes one from an oracle (the isentropic vortex) at the cell centroids.

Restart: `restart_fields` holds the conserved variables bit for bit (5 arrays), after averaging the sums and their weight (7), for a steady case the solver's state (penalty, the last step's relative residual, order, first-order start, reference residuals and CFL bound: 14) as uniform fields, and one circulation per point vortex; an `fp32+acc64` steady case also writes its first-order start's window (the sample count and 100 values per flow equation) as the `SolverState` array `SolverStartWindow`, so that `continue` reproduces the uninterrupted run (`tests/physics/test_euler_run.cpp`).

## Partitions

On a partition (several ranks, or one with periodic image ghosts) the residual fills the ghost cells' primitives, gradients and limiter between its loops (`Euler<P>::set_ghosts`), and the physics updates, sums and reports its owned cells only; norms, residual scales and loads are exact sums over the ranks. The guarantee across rank counts is in [runtime.md](../contracts/runtime.md#ranks).

## Conservation check

With `AUTAN_CONSERVATION=1`, the inviscid physics checks, on the state a run starts from (before its first step) and after every host step, that the face part R^F of the residual (the residual without its source) sums to the boundary fluxes, Σ_c R^F_c = Σ_b F_b, which holds when every interior flux enters its two cells with opposite signs. R^F comes from the residual's own face loops (a step that diverges logs no event, so the last after-step event is the written solution's for a run that does not diverge), with the current scheme, flux and planar choice (`Euler::face_parts`); the loads and fields re-evaluate their own state, so the check changes no output; the face fluxes come from write-only loops around the same bodies (`*_face_fluxes`, `*_boundary_fluxes` in `kernels/euler.py`). The check costs one more residual evaluation and one face pass per step, and 5 × (cells + faces) FP64 values allocated at the first check; without the variable nothing runs. Each check logs, collectively on every rank, an `info` event, source `solver`, message `conservation`, whose data hold, per equation (`equations`, the history's names), exact sums over the owned cells of every rank, each term added on its own:

| Key | Content |
|---|---|
| `at`, `iteration` | `start` on the state the run starts from, `step` after a step; the history iteration whose state it is (the start's: the one before the first step) |
| `faces` | Σ_c R^F_c |
| `boundary` | Σ_b F_b over the boundary faces of owned cells |
| `E` | `faces` − `boundary`, one exact sum of the signed terms |
| `D` | Σ \|F\| over the incidences of owned cells: an interior face once per owned cell it bounds, a boundary face once |
| `max_faces`, `max_terms` | the largest number of faces of a cell, and of face terms the loops add into one owned cell, over the ranks |
| `flux_mismatch`, `first_mismatch` | how many partition-interface faces have fluxes that differ in any bit between their two ranks (in the orientation of the mesh as read), and the least global id among them, or null |

A non-finite sum is written null. Under reproducibility `"run"` the two ranks may evaluate an interface face in opposite orientations, so its fluxes can differ in their last bits. On a device, for Navier-Stokes, with periodic pairs, or with another value, the check does not run and a `solver` warning gives the reason (`data.reason`).

Two more variables serve D2 gate 7's controls ([#259](https://github.com/Remenby31/autan/issues/259)); every rank must see the same values, or the run fails at setup. `AUTAN_CONSERVATION_FACE=x,y,z` adds `selected_face`: the interior faces whose centroid lies within 10⁻⁶ of the point in every coordinate (`matches`) and, for exactly one, its global id, centroid, global-left cell, the rank that owns that cell, whether it lies on a partition interface, and its flux in its global orientation. `AUTAN_CONSERVATION_FACES=<file>` makes rank 0 write, at the start event, the event's `equations`, `E`, `D` and `max_faces` and the counts of interior and boundary faces as a JSON line (schema `autan.conservation-faces/1`), then every face once as CSV (global id, 1 for a boundary face, centroid, the five signed fluxes); the event names the file in `faces_file`. `tools/gate7.py` reads a run's start event, or its last after-step event, which counts only for a converged run at that event's iteration, and gives gate 7's verdict ([acceptance.md](../validation/acceptance.md)); `tools/gate7_control.py` checks a mutant's evidence against the intact build's (`tests/physics/gate7_mutants.json`); `tools/gate7_report.py` reports each interior face's flux relative to D against the controls' scale, 100 τ.

## Limiter and verification

Venkatakrishnan's threshold ε² ∝ h³ shrinks faster than the squared variations ∝ h² of a smooth solution, so the limiter clips smooth extrema more as the mesh is refined; K sets where that starts (Venkatakrishnan 1995; Michalak and Ollivier-Gooch 2008, AIAA 2008-776, discuss the same trade-off). On the MMS (tet, levels 12–32, Roe, host study), fitted / finest-pair orders of ρv with the limiter on: K = 5 gives 1.88/1.78, K = 10, 20 and 40 give 1.99/1.97, the unlimited values. K = 10 is the default: smooth flows keep the design order at these resolutions, and the limiter still acts at shocks (Sod gate, below). A manufactured solution is smooth by construction and measures the design order of the scheme; verification cases (`verification` in the solver input) run without the limiter, every other case with it.

## Manufactured solution

Steady `euler` solution of autan-vv (Roy et al. 2004 form), every boundary face at the exact state, cell sources at centroids, solved by `autan-run` and checked by `autan-vv verify --gate euler-mms-order` (Roe, unlimited, implicit steady solver; commit a6b557c, PR #134). Fitted / finest-pair L2 orders:

| Family, levels | Precision | ρ | ρu | ρv | ρw | ρE |
|---|---|---|---|---|---|---|
| tet, 12–32 | fp64, fp32+acc64 | 1.99/2.04 | 2.01/1.98 | 1.99/1.98 | 1.99/1.97 | 2.05/1.98 |
| hex, 12–32 | fp64, fp32+acc64 | 2.20/2.18 | 2.06/1.99 | 1.96/1.97 | 2.17/2.15 | 1.91/1.96 |
| poly, 12–32 | fp64, fp32+acc64 | 2.56/2.42 | 2.53/2.34 | 2.55/2.44 | 2.53/2.41 | 2.14/2.08 |
| hybrid, 16–48 | fp64, fp32+acc64 | 2.11/2.12 | 1.97/2.07 | 1.95/2.00 | 1.99/2.10 | 1.98/2.01 |

The hybrid family needs one level more: its pyramids, prisms and tetrahedra fill corner blocks that hold few cells on coarse levels, and the momentum orders between successive levels are 1.76–1.86 (12→16), 1.94–1.98 (16→24), 1.87–1.90 (24→32), 2.00–2.10 (32→48): they fluctuate by ±0.1, as on every non-nested family ([order.md](../validation/order.md)), and reach 2 on the finest pair. Polyhedral duals supraconverge (above 2.1: a warning of the gate, not a failure). The host study (`physics/mms/euler_mms.py`) gives the same orders for HLLC and AUSM+-up.
