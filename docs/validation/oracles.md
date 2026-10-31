# Oracles

Source: `vv/src/autan_vv/oracles/`. An oracle maps positions (and a physical time for unsteady flows) to the primitive fields of [results.md](../contracts/results.md) — `Density`, `VelocityX`, `VelocityY`, `VelocityZ`, `Pressure`, `Temperature` and, for SA, `TurbulentSANuTilde` — in SI. `autan-vv oracles` lists them; `autan_vv.oracles.registry.get_oracle(name)` builds one.

| Name | Physics | Dim | Kind | Definition |
|---|---|---|---|---|
| `mms-euler`, `mms-navier-stokes`, `mms-rans-sa`, `mms-rans-sa-noft2` | Euler, laminar NS, RANS-SA | 3 | exact with the manufactured source | [mms.md](mms.md) |
| `ringleb` | Euler | 2 | exact, steady | Hodograph solution (Chiocchia 1985; HiOCFD1 C1.2): (x − J/2)² + y² = 1/(4ρ²V⁴) solved for the speed V, then u = V cos θ sign(y), v = V sin θ with sin θ = V/k. Domain k ∈ [0.7, 1.5], V ≥ 0.5 |
| `isentropic-vortex` | Euler | 2 | exact, unsteady | Vortex of strength 5 convected at Mach 0.5 (Shu 1998) on the periodic [−10, 10]² domain recommended by Spiegel, Huynh and DeBonis (2015) |
| `sod`, `lax` | Euler | 1 | exact, unsteady | Exact Riemann solver (Toro 2009, ch. 4), diaphragm at x = 0.5 |
| `shu-osher` | Euler | 1 | **reference only** | 1D MUSCL–HLLC–SSP-RK3 solution on 12 800 cells at t = 1.8, cached in `~/.cache/autan-vv`; regression, never an exact gate |
| `couette` | compressible NS | 1 | exact | Plane Couette flow with viscous heating, constant μ and Pr (White 2006) |
| `poiseuille` | incompressible NS | 1 | exact (incompressible) | u = G y (H − y)/(2μ); holds to O(M²) for the compressible solver |
| `blasius` | boundary layer | 2 | asymptotic | Blasius profile by shooting to f″(0) = 0.332057336215196; approximates NS for Re_x ≫ 1 at low Mach |

The Ringleb, vortex and shock-tube oracles use units where the stagnation (Ringleb) or reference (vortex, shock tubes) density, speed of sound and temperature are 1: γ = 1.4 and R = 1/γ, so T = c². A run verified against them uses the same gas constant; the harness checks it.

## How they are verified

`vv/tests/test_oracles.py`, against what defines each one rather than against the code:

- Ringleb and the vortex: residual of the steady or unsteady Euler equations by fourth-order finite differences, below 10⁻⁹; Ringleb also recovers its hodograph speed, its stagnation temperature, and is continuous across y = 0; the vortex returns to its initial state after one period.
- Riemann solver: the star states of Toro's Table 4.3, tests 1–5, within the rounding of the printed digits (for test 5, whose inputs are themselves rounded outputs, plus the propagated input rounding); the Rankine–Hugoniot conditions across the Sod shock.
- Shu–Osher: the 1D solver that produces it converges to the exact Sod solution at L1 order 0.6–1.1, as a discontinuous solution must (Banks, Aslam and Rider 2008), which is the second source that makes it usable for regression; the reference places the shock near x = 2.4.
- Blasius: f″(0) and the displacement constant 1.72078765752246 of Ganapol (2010) to 10⁻¹³ and 10⁻¹⁰; wall shear and edge velocities.
- Couette and Poiseuille: their momentum and energy balances by finite differences, and the boundary values.

## References

- F. Chiocchia, Exact solutions to transonic and supersonic flows, AGARD-AR-211, 1985; HiOCFD1 problem C1.2, https://cfd.ku.edu/hiocfd/case_c1.2.html
- C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws, LNM 1697, 1998. doi:10.1007/BFb0096355
- S. C. Spiegel, H. T. Huynh and J. R. DeBonis, A survey of the isentropic Euler vortex problem using high-order methods, AIAA 2015-2444.
- E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009.
- G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, *J. Comput. Phys.* 27:1–31, 1978. doi:10.1016/0021-9991(78)90023-2
- P. D. Lax, Weak solutions of nonlinear hyperbolic equations and their numerical computation, *Comm. Pure Appl. Math.* 7:159–193, 1954.
- C.-W. Shu and S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, II, *J. Comput. Phys.* 83:32–78, 1989. doi:10.1016/0021-9991(89)90222-2
- J. W. Banks, T. Aslam and W. J. Rider, On sub-linear convergence for linearly degenerate waves in capturing schemes, *J. Comput. Phys.* 227:6985–7002, 2008. doi:10.1016/j.jcp.2008.04.002
- F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill, 2006.
- H. Blasius, Grenzschichten in Flüssigkeiten mit kleiner Reibung, *Z. Math. Phys.* 56:1–37, 1908; B. D. Ganapol, Highly accurate solutions of the Blasius and Falkner-Skan boundary layer equations via convergence acceleration, arXiv:1006.3888, 2010.
