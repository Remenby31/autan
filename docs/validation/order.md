# Observed order of accuracy

Source: `vv/src/autan_vv/order.py`, `study.py`, `reference_fv.py`.

## Definitions

- **Mesh size** h = (V/N)^(1/d) for N cells filling volume V in d dimensions.
- **Observed order from errors** (exact solution known, e.g. MMS): least-squares fit of log e = log C + p log h over all levels, with the slope's standard error, its 95 % confidence interval (Student t, N − 2 degrees of freedom), R², and the order of each successive pair (`pairwise_orders`, coarse to fine, the last one being the finest pair).
- **Observed order from a functional** (exact value unknown, e.g. a drag coefficient): three grids, Celik et al. (2008) eqs. 3a–3c solved by fixed-point iteration, which handles non-constant refinement ratios. Richardson extrapolation φ_ext = (r²¹ᵖ φ₁ − φ₂)/(r²¹ᵖ − 1). Oscillatory (`V0107`) or divergent (`V0108`) triplets are rejected: their order is meaningless.
- **GCI** (Roache 1994, 1998): GCI_fine = F_s |φ₂ − φ₁|/(r²¹ᵖ − 1), with F_s = 1.25 on three grids and 3 on two grids with an assumed order (`gci_two_grids`). Judge the asymptotic range by comparing the observed order with the formal order: the classic ratio GCI₃₂/(rᵖ GCI₂₁) is identically 1 when p is the observed order of the same triplet.

The implementation reproduces Table 1 of Celik et al. (2008) (p = 1.53, φ_ext = 6.1685, GCI = 2.2 %) and recovers exact power laws to 10⁻¹².

## Report

`autan-vv order errors.json --out order.json --plot order.png` reads `{"label", "dim", "levels": [{"n_cells", "volume", "errors": {norm: value}, ...}]}` and writes schema `autan.order-study/1`: the levels with h added (extra keys such as `mesh_sha256` are kept) and, per norm, `order`, `order_stderr`, `order_ci95`, `constant`, `r_squared`, `pairwise_orders`. The plot shows each norm against h with its fit.

## Reference finite-volume solve

`autan_vv.reference_fv.solve` solves the `advection_diffusion` manufactured problem in FP64 on any `FaceMesh`. It proves the order tool and every mesh family end to end, and gives the solver a second-order baseline:

- gradients by weighted least squares over face neighbours and boundary face centroids, weights 1/|d|²;
- convection by the upwind cell value reconstructed with its gradient and integrated over each face's triangle fan, which is exact for linear fields even on warped faces;
- diffusion by the face gradient ∇φ_f = ḡ + α (Δφ − ḡ·d) d/|d|², α = 4/3 inside (Nishikawa 2010) and α = 2 on boundary faces, where the cell-to-face quadratic needs it for a second-order face derivative. The correction along d rather than along the face normal keeps the coefficients bounded on non-convex agglomerated cells, where d·n can be negative;
- Dirichlet data from the exact solution, the source at centroids, and GMRES preconditioned by the LU factors of the compact first-order operator.

Checks (`vv/tests/test_reference_fv.py`): a linear solution is reproduced to a few ulps on every kind, perturbed; the L2 order on the finest pair and the fitted order exceed 1.85 on every kind, clean and perturbed; `autan-vv verify` writes the report, plot and mesh manifest.

### Measured orders

`autan-vv verify <kind> --levels … [--perturbation 0.3] --seed 2026`, FP64, macOS arm64, Gmsh 4.15.2. Fitted order over four levels / order on the finest pair.

| Kind | Perturbation | Levels n | Finest cells | L2 | L∞ |
|---|---|---|---|---|---|
| tet | 0 | 8, 16, 24, 32 | 149,401 | 2.14 / 2.03 | 1.85 / 1.23 |
| tet | 0.3 | 8, 16, 24, 32 | 149,401 | 2.07 / 1.95 | 1.70 / 0.74 |
| hex | 0 | 8, 16, 32, 48 | 110,592 | 2.13 / 2.09 | 2.18 / 2.11 |
| hex | 0.3 | 8, 16, 32, 48 | 110,592 | 2.12 / 2.02 | 1.92 / 1.78 |
| prism | 0 | 8, 16, 24, 32 | 76,672 | 2.31 / 2.35 | 2.02 / 0.56 |
| prism | 0.3 | 8, 16, 24, 32 | 76,672 | 2.23 / 2.22 | 1.90 / 1.54 |
| hybrid | 0 | 6, 12, 24, 36 | 146,448 | 2.07 / 2.06 | 1.97 / 1.98 |
| hybrid | 0.3 | 6, 12, 24, 36 | 146,448 | 2.04 / 2.11 | 1.88 / 1.85 |
| poly | 0 | 8, 16, 24, 32 | 27,543 | 2.40 / 2.04 | 2.42 / 2.01 |
| poly | 0.3 | 8, 16, 24, 32 | 27,543 | 2.41 / 2.05 | 2.41 / 2.36 |
| agglomerated | 0 | 8, 16, 24, 32 | 13,985 | 2.56 / 2.33 | 1.92 / 2.13 |
| agglomerated | 0.3 | 8, 16, 24, 32 | 14,055 | 2.48 / 2.55 | 1.73 / 1.89 |

L∞ fluctuates on unstructured families: the meshes are not nested and the worst cell changes from level to level. Use L1 or L2 as the acceptance norm and report L∞.

## References

- I. B. Celik, U. Ghia, P. J. Roache, C. J. Freitas, H. Coleman and P. E. Raad, Procedure for estimation and reporting of uncertainty due to discretization in CFD applications, *J. Fluids Eng.* 130(7):078001, 2008. doi:10.1115/1.2960953
- P. J. Roache, Perspective: a method for uniform reporting of grid refinement studies, *J. Fluids Eng.* 116(3):405–413, 1994. doi:10.1115/1.2910291
- P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa, 1998.
- H. Nishikawa, Beyond interface gradient: a general principle for constructing diffusion schemes, AIAA 2010-5093. doi:10.2514/6.2010-5093
- D. J. Mavriplis, Revisiting the least-squares procedure for gradient reconstruction on unstructured meshes, AIAA 2003-3986. doi:10.2514/6.2003-3986
