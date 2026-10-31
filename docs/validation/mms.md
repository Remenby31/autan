# Manufactured solutions

Source: `vv/src/autan_vv/mms/`. Derived symbolically with sympy; exported with `autan-vv mms <name|all> --out DIR`.

## Convention

Every solution is steady and defined on the unit cube [0, 1]³. For a system ∂U/∂t + ∇·(F_c(U) − F_v(U, ∇U)) = Q(U, ∇U) + S, the manufactured source is S = ∇·(F_c − F_v) − Q evaluated on the manufactured fields (Roache 2002; Salari and Knupp 2000). Q holds the model's own sources (Spalart-Allmaras production, destruction, cb2 and density-gradient terms). The solver adds S × V_cell, evaluated with the same quadrature as its other volume terms; boundary values are Dirichlet from the exact fields unless a test says otherwise.

## Catalogue

| Name | Equations | Fields | Notes |
|---|---|---|---|
| `advection_diffusion` | ∇·(aφ) − ∇·(ν∇φ) = S | φ = 1 + ½ sin(1.1πx + 0.3) cos(0.9πy + 0.1) sin(0.7πz + 0.6) | a = (1, 0.7, 0.4), ν = 0.1 |
| `euler` | Compressible Euler, γ = 1.4, R = 287 | ρ, u, v, w, p in the trigonometric form of Roy et al. (2004), extended to 3D | Subsonic (checked), ρ and p bounded away from 0 |
| `navier_stokes` | Laminar compressible NS, constant μ = 10, Pr = 0.72, Stokes hypothesis | Same fields as `euler` | Re ≈ 10 on the unit cube so viscous and convective terms are both exercised |
| `rans_sa` | Compressible RANS + SA-neg, conservative form of Allmaras et al. (2012) as given on the [TMR SA page](https://tmbwg.github.io/turbmodels/spalart.html); Pr_t = 0.9, μ = 10⁻³ | No-slip boundary-layer-like velocity and ν̂ ∝ y near the wall y = 0; d = y | ν̂ ≥ 0, so SA-neg equals standard SA with S̃ from Note 1(c) |
| `rans_sa_noft2` | Same with f_t2 = 0 | Same | |

The RANS fields follow the principles of Eça et al. (2007) and Oliver et al. (2012): a wall with no-slip velocity and ν̂ = 0, ν̂ linear in the wall distance near it, smooth elsewhere. The wall is the mesh zone `ymin` and the wall distance is exactly d = y. Each solution lists **validity conditions** (positive ρ, p, ν̂; subsonic; S̄ ≥ −c₂Ω so the S̃ clipping branch never activates; S̃ > 0; r < 10). Generation samples the domain, including a near-wall layer down to y = 1/2048, and fails with `V0201` if one is violated, so the source is smooth and the solver must take the same branches. On these fields χ reaches 13.6 and r reaches 0.75, so f_v1, f_v2 and f_w are all exercised away from their limits.

## Exports

**C++20 header** (`<name>.hpp`, namespace `autan::mms::<name>`): `Parameters` holds the constants; for each group `fields`, `gradients` (∂/∂x, ∂/∂y, ∂/∂z of each field), `conserved`, `sources` and, for RANS, `auxiliary` (wall distance, μ_t, χ, r, S̃) there is

```cpp
template <typename T> AUTAN_MMS_HD void sources(T x, T y, T z, T* out);   // out[i] is sources_names[i]
```

`AUTAN_MMS_HD` is `__host__ __device__` under CUDA or HIP. Literals are written `T(p)/T(q)` and π as `std::numbers::pi_v<T>`, so one body serves `float` and `double` without promotion. Common subexpressions are shared.

**JSON** (`<name>.json`, schema `autan.mms/1`): `name`, `equations`, `coordinates` (`x, y, z`), `domain`, `parameters`, `validity`, `references`, and one object per group with `subexpressions` (ordered `[name, expr]` pairs, each may use earlier ones) and `outputs` (`{name: expr}`). Expressions use C99 syntax restricted to `+ - * /`, unary minus, `pow`, `sqrt`, `sin`, `cos`, `exp`, `log`, decimal literals and the constant `pi`. `autan_vv.mms.parse_json_expressions` is a reference reader: it builds expressions from that grammar only and rejects anything else with `V0202`, executing nothing.

## Verification of the generator

Tests use oracles that do not reuse the derivation (`vv/tests/test_mms.py`, `test_mms_export.py`):

- the advection-diffusion source of φ = sin(πx), derived by hand;
- exact steady solutions whose source must vanish: Shu's isentropic vortex for Euler, compressible Couette flow with viscous heating for Navier-Stokes;
- the RANS-SA source re-evaluated by fourth-order finite differences of an independent implementation of the TMR equations, with the model constants typed from the TMR page, at points down to y = 0.03 (agreement 10⁻¹⁰–10⁻⁸ relative, for SA-neg and SA-noft2);
- the header compiled with `-Wall -Wextra -Werror` and evaluated in `double` and `float` against sympy: errors stay below 32 ε of the precision relative to each output's magnitude (measured: ≤ 1×10⁻¹⁵ in FP64, ≤ 5×10⁻⁷ in FP32, sources included);
- the JSON parsed back and evaluated, within 10⁻¹².

## References

- P. J. Roache, Code verification by the method of manufactured solutions, *J. Fluids Eng.* 124(1):4–10, 2002. doi:10.1115/1.1436090
- K. Salari and P. Knupp, Code verification by the method of manufactured solutions, Sandia SAND2000-1444, 2000. doi:10.2172/759450
- C. J. Roy, C. C. Nelson, T. M. Smith and C. C. Ober, Verification of Euler/Navier-Stokes codes using the method of manufactured solutions, *Int. J. Numer. Meth. Fluids* 44(6):599–620, 2004. doi:10.1002/fld.660
- P. R. Spalart and S. R. Allmaras, A one-equation turbulence model for aerodynamic flows, *La Recherche Aérospatiale* 1:5–21, 1994.
- S. R. Allmaras, F. T. Johnson and P. R. Spalart, Modifications and clarifications for the implementation of the Spalart-Allmaras turbulence model, ICCFD7-1902, 2012.
- L. Eça, M. Hoekstra, A. Hay and D. Pelletier, On the construction of manufactured solutions for one and two-equation eddy-viscosity models, *Int. J. Numer. Meth. Fluids* 54(2):119–154, 2007. doi:10.1002/fld.1387
- T. A. Oliver, K. C. Estacio-Hiroms, N. Malaya and G. F. Carey, Manufactured solutions for the Favre-averaged Navier-Stokes equations with eddy-viscosity turbulence models, AIAA 2012-0080. doi:10.2514/6.2012-80
- C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws, in *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, LNM 1697, 1998 (isentropic vortex).
- NASA Turbulence Modeling Resource, [The Spalart-Allmaras turbulence model](https://tmbwg.github.io/turbmodels/spalart.html) (CC0).
