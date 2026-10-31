# Validation dataset catalogue

This page lists, for every milestone M1 to M9 of [`docs/PLAN.md`](../PLAN.md), the reference data that will prove our results are right. For each case it gives the source, licence and access, format and size, the quantities available, the published uncertainty, a proposed acceptance tolerance with its reason, and a priority. The framework that downloads, runs and scores these cases is issue #2 (`vv/`). The frozen acceptance numbers belong in `docs/validation/acceptance.md`; the tolerances below are proposals for it.

Every URL was opened on 2026-10-31 and returned HTTP 200 unless stated otherwise. "(unverified)" marks a value that we could not confirm against a primary source that day, usually because the paper is paywalled. Confirm it before it becomes a gate.

## How to read an entry

**Kind**
- **V-exact**: verification against an exact or manufactured solution. The error must go to zero at the design order.
- **V-ref**: verification against converged numerical references from independent codes, such as the NASA CFL3D and FUN3D runs on the Turbulence Modeling Resource (TMR). Different discretisations (cell-centred, node-centred) follow different refinement paths on the same grid family and meet only in the continuum limit. A V-ref gate therefore compares continuum estimates: our value extrapolated from at least three levels in the asymptotic range, with its own discretisation-error band (observed order and grid-convergence index), against the reference's extrapolated value and its band. Comparisons level by level are diagnostics, not gates.
- **Val**: validation against an experiment. This measures model-form error as well as code error.

**Priority**
- **Must**: the milestone gate depends on it.
- **Useful**: it adds coverage or diagnostics, but the gate passes without it.

**Gate or report.** A *gate* fails the build. A *report* is published on the dashboard but never fails the build. A turbulence model's own error against experiment is usually larger than the experimental uncertainty, so most Val comparisons for RANS are reports. The gate on those cases is the V-ref comparison with CFL3D and FUN3D. This follows `AGENTS.md`: expected values come from the specification or independent sources, never from our own code.

**Precision.** Unless an entry says otherwise, a tolerance applies to both FP32 (FP32 state and fluxes, FP64 accumulation) and FP64.
- FP64 accumulation does not remove the rounding of FP32 states and fluxes, so FP32 has its own error floor. Measure that floor per case, as the error plateau on the finest levels, instead of assuming a value.
- An order-of-accuracy gate needs at least three levels in the asymptotic range above the floor, in each precision. If FP32 cannot provide three, shift the mesh family towards coarser levels rather than drop evidence, and record the plateau.
- Round-off criteria (exactness, conservation) are written as a multiple of the machine epsilon ε of the precision under test, as `AGENTS.md` requires. `acceptance.md` fixes the multiplier per case after measuring it.
- Residual convergence is measured on the scaled residual: each equation divided by its free-stream flux scale, L2 over cells. The PLAN's "10 orders" is demonstrated in FP64. In FP32 the gate is that the residual reaches its measured plateau, and that forces stop changing within the M5 criterion (0.1 %) before it does.

**Statistics.** For unsteady cases, discard the initial transient, compute each mean with a 95 % batch-means confidence interval, and pass only if the whole interval lies inside the tolerance band. Report the averaging duration. An averaging length on its own (for example "50 cycles") is not a statistical criterion.

## Access and licences

Policy for the case library:
- Permission to *use* data and permission to *redistribute* it are separate questions. Record the rights evidence for each asset (licence text, terms page, date read) in the case library.
- Vendor only assets with recorded CC0, public-domain, CC-BY or CC-BY-SA rights. Every other asset is stored as URL plus SHA-256 and downloaded at test time.
- Downloading at test time does not lift a use restriction. Non-commercial (NC) terms and reserved automated-analysis or AI uses still apply to how we use the data; those cases need permission or a replacement (see [decisions](#gaps-and-decisions-for-the-lead)).

| Source | Licence / access | Vendor in repo? |
|---|---|---|
| NASA TMR (grids, CFL3D/FUN3D results, experiment transcriptions), [github.com/TMBWG/turbmodels](https://github.com/TMBWG/turbmodels) | **CC0-1.0** (GitHub licence field). `.gz`/`.zip`/`.x` files are in Git LFS, which has a bandwidth quota. Plain files come from `raw.githubusercontent.com`. | Yes |
| NASA NTRS reports, NPARC/Wind-US archive, CRM, DPW and HLPW pages and grids | Per asset. Works by NASA staff are not under US copyright, but contractor reports, contributed experimental data and workshop meshes can carry third-party rights. NASA states that its permission to publish such material does not pass to recipients ([sti.nasa.gov/disclaimers](https://sti.nasa.gov/disclaimers/)). | Only with rights evidence for that asset; otherwise download by hash |
| AGARD AR-138, AR-303 (NATO STO) | Free download, no registration. No licence text. | No: store pointers and transcribed tables, with citation |
| ERCOFTAC Classic Collection, [cfd.mace.manchester.ac.uk/ercoftac](http://cfd.mace.manchester.ac.uk/ercoftac/) | **CC BY-NC-SA 4.0**. No login. **HTTP only**: HTTPS refuses the connection. Commercial use needs ERCOFTAC's permission. | No. Use is blocked pending a decision. |
| ERCOFTAC KBwiki, [kbwiki.ercoftac.org](https://kbwiki.ercoftac.org/) | [Terms and Conditions](https://kbwiki.ercoftac.org/w/index.php?title=Terms_and_Conditions), revised 2026-10-25. Human reading and reuse with attribution are allowed, commercial included. **Text and data mining, and use to train, evaluate or otherwise develop AI systems, are reserved**: a licence from admin@cado-ercoftac.org is required before any bulk download or mining. | No. Automated scoring and M11 use are blocked pending a decision. |
| DrivAerML, AhmedML, WindsorML (Hugging Face `neashton/*`) | **CC-BY-SA-4.0** | Yes (share-alike for derived data) |
| ADODG CRM grids, [Mendeley 10.17632/7jnyjdbvf9.1](https://data.mendeley.com/datasets/7jnyjdbvf9/1) | **CC BY 4.0** | Yes |
| Lee & Moser, Madrid channel DNS, HiOCFD files, Featflow, TNF archive | No licence stated. The files ask for citation. | No: download by hash |
| DrivAer CAD (TU München) | Free, **registration required**, no licence text | No |
| Loughborough DrivAer experiment | **CC-BY-NC-4.0**. Commercial use needs permission. | No. Use is blocked pending a decision. |
| ONERA OAT15A (DPW-8) | ONERA property, conditions of use. The Jacquin 2009 paper is for personal use only. | No |
| MASA (MMS library) | **LGPL-2.1** (`COPYING`) | Test-only dependency |
| LANL ExactPack (exact Riemann, Noh, Sedov) | **BSD-3-Clause** (`LICENSE.txt`) | Test-only dependency |
| SU2 tutorials (ONERA M6 design mesh) | LGPL-2.1 | Test-only data |
| preCICE tutorials | LGPL-3.0 | Test-only data |
| Journal papers (Roy, Eça, Ghia, Botella–Peyret, Schäfer–Turek, …) | Paywalled | No: transcribe only the numbers we need, with the citation |

### Hosts that have moved

Several well-known references point to dead hosts. Use the current ones:

| Old | Current |
|---|---|
| `turbmodels.larc.nasa.gov/*` (301 to a nasa.gov notice since 2026) | `https://tmbwg.github.io/turbmodels/*`, same page names |
| `aiaa-dpw.larc.nasa.gov` (connection refused) | `https://www.aiaa-dpw.org/`; grids at `https://dpw.larc.nasa.gov/` |
| `hiliftpw.larc.nasa.gov` | `https://aiaa-hlpw.org`; past workshops as zip archives (HLPW-4 1.0 GB, HLPW-5 1.75 GB) |
| `www.featflow.de` (times out) | `https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/` |
| `www.sto.nato.int` (403 to scripts) | `https://publications.sto.nato.int/publications/AGARD/…` |
| `cfdval2004.larc.nasa.gov` (down) | TMR mirror `Other_exp_Data/CFDVAL2004/case3expdata.html` |
| `cavityflow.com` (Erturk, 404) | arXiv [cs/0411047](https://arxiv.org/abs/cs/0411047) |
| TUM DrivAer `mw.tum.de` pages | [epc.ed.tum.de/en/aer/research-groups/automotive/drivaer](https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer/) |

## Summary by milestone

| M | Must | Useful |
|---|---|---|
| M1 | Scalar advection–diffusion MMS, linear-field gradient exactness, adjoint dot-product test of one kernel | — |
| M2 | Euler MMS, Ringleb, isentropic vortex, Sod/Lax, NACA 0012 inviscid (Vassberg–Jameson), ONERA M6 inviscid | Shu–Osher, HiOCFD Gaussian bump, Billig shock standoff, NPARC wedge/cone/expansion |
| M3 | NS MMS, RANS-SA and SA-neg MMS, Couette/Poiseuille/Blasius, TMR flat plate, bump-in-channel (2D and 3D), NACA 0012 numerics (V-ref) | NACA 0012 validation, Bachalo–Johnson, NASA hump, Driver–Seegmiller step, RAE 2822, ONERA M6 viscous, Schülein Mach 5 SWBLI, Settles ramps, Mach 2/5 flat plate, ACC-SBLI Mach 7 |
| M4 | The M3 TMR cases driven through the SDK | — |
| M5 | ONERA M6 viscous (reference forces), NACA 0012 numerics | DPW CRM |
| M6 | Adjoint dot-product and complex-step checks, ADODG case 2 (RAE 2822), SU2 ONERA M6 constrained design | ADODG case 4.1 (CRM wing), ADODG case 1, Giles–Pierce quasi-1D adjoint |
| M7 | SST MMS, BDF2 temporal order, DDES shielding, Taylor–Green Re 1600, channel DNS (Lee & Moser), cylinder Re 3900, Driver–Seegmiller step, Ahmed 25°/35°, HLPW-5 CRM-HL (TC1 and TC2.4) | Le–Moin–Kim DNS, periodic hill, DPW CRM, AGARD AR-303 cases |
| M8 | Incompressible MMS (pressure order, discrete divergence), lid-driven cavity (Botella–Peyret, Ghia), Schäfer–Turek 2D-1/2D-2/2D-3/3D, DrivAerML baseline (code-to-code) | Erturk high-Re cavity, Loughborough DrivAer |
| M9 | Species MMS with per-species balance, 1D diffusion, He/air shock tube, conjugate MMS and composite wall, Sandia propane jet, Cutler He/air coaxial jet, NASA C3X vane | Brown–Roshko layer, preCICE heated plate, Vynnycky plate, O'Hern He plume, Panchapakesan–Lumley jet |

---

## M1 — Generator spike: scalar advection–diffusion

The gate is "MMS order 2 on tet/hex/poly/hybrid in FP32 and FP64".

### Scalar advection–diffusion MMS — V-exact · Must
- **Source:** the method of manufactured solutions as set out by Roache (2002), *J. Fluids Eng.* 124:4–10, [doi:10.1115/1.1436090](https://doi.org/10.1115/1.1436090) (paywalled). Also Salari & Knupp, SAND2000-1444, [doi:10.2172/759450](https://doi.org/10.2172/759450). That report is free from OSTI, but `osti.gov` timed out from our network on 2026-10-31.
- **Data:** generated symbolically in `vv/` (#2). No external files.
- **Quantities:** L2 and L∞ error of the scalar and of its LSQ and Green–Gauss gradients.
- **Mesh families:** refinement and perturbation must be deterministic. The perturbed ("dirty") family moves nodes by a seeded fraction of the local spacing, so every level has the same quality statistics. Redrawing the perturbation independently at each level does not make a mesh family.
- **Acceptance (gate):**
  - Observed L2 order of the solution 2 ± 0.1 on at least three levels in the asymptotic range, for each cell type and precision, on the clean and the perturbed families.
  - The LSQ gradient reproduces a linear field exactly (to round-off) on every cell type and on the perturbed meshes. This check does not depend on the mesh.
  - LSQ gradient order ≥ 0.9 on all families; report whether it reaches 2 on smooth hexahedra.
  - Green–Gauss with simple face interpolation is inconsistent on skewed meshes (Syrakos et al., *Phys. Fluids* 29:127103, 2017, [doi:10.1063/1.4997682](https://doi.org/10.1063/1.4997682)). Gate its order (≥ 0.9) only on families whose skewness stays bounded under refinement; report it elsewhere.
  - Report the L∞ order; gate it only on hex and tet, because polyhedral L∞ is dominated by the worst cell.
- **Why:** 2 is the design order of a second-order finite-volume scheme, and ±0.1 matches common practice for clean asymptotic ranges (Roache 2002). The gradient criteria follow what each operator can actually achieve on each mesh type.

### Adjoint kernel prototype — V-exact · Must
- The PLAN's M1 gate includes a dot-product test of one adjoint kernel. Use the procedure and the ε-scaled tolerances of [Gradient correctness](#gradient-correctness--v-exact--must) (M6), applied to the generated flux or gradient kernel alone, on every cell type, in FP32 and FP64.

---

## M2 — Compressible Euler

The gate is "MMS Euler, Ringleb, NACA0012 and ONERA M6 inviscid; 10 orders of residual convergence".

### Euler MMS — V-exact · Must
- **Source:** Roy, Nelson, Smith & Ober (2004), "Verification of Euler/Navier–Stokes codes using the method of manufactured solutions", *IJNMF* 44:599–620, [doi:10.1002/fld.660](https://doi.org/10.1002/fld.660). Paywalled; the sine/cosine field coefficients in its tables are what we need. An independent implementation is [MASA](https://github.com/manufactured-solutions/MASA) (LGPL-2.1), with `euler_2d`/`euler_3d`, transient and axisymmetric variants.
- **Data:** symbolic (`vv/`). Cross-check our source terms against MASA's at random points to 1e-12 (FP64). This gives the independent oracle that `AGENTS.md` asks for.
- **Quantities:** L2 and L∞ error of ρ, ρu, ρv, ρw, ρE.
- **Acceptance (gate):** L2 order 2 ± 0.1 for every conserved variable, on all four cell types, for each flux (Roe, HLLC, AUSM+-up), FP32 and FP64. Run both supersonic and subsonic fields, so that both upwind branches of each flux are exercised.
- **Why:** the design order. Agreement with MASA proves the source terms are right.

### Ringleb flow — V-exact · Must
- **Source:** Chiocchia, AGARD-AR-211 (1985) is the classical reference; no online copy was found (unverified). The usable open definition is HiOCFD1 problem C1.2, [cfd.ku.edu/hiocfd/case_c1.2.html](https://cfd.ku.edu/hiocfd/case_c1.2.html): q0 = 0.5, k_min = 0.7 (outer wall), k_max = 1.5 (inner wall), γ = 1.4. The flow is transonic, irrotational and isentropic. The page states M = 1.5 on the inner wall at y = 0, but 1.5 is the speed q = k_max there; with a² = 1 − (γ − 1)q²/2 the Mach number is ≈ 2.02. Take the equations as the definition, not the prose.
- **Data:** analytic. The site also has five Gmsh grids (`ringleb0-4.gmsh`), with no licence stated.
- **Quantities:** L2 error in density and entropy.
- **Acceptance (gate):** L2 order 2 ± 0.1. Run twice: with the exact state imposed on all four boundaries (tests the interior scheme alone), then with slip walls (tests the wall boundary condition). Entropy error must also fall at order 2.
- **Why:** the solution is exact and smooth, and the flow is transonic without a shock.

### Isentropic vortex — V-exact · Must
- **Sources:**
  - Shu (1998), *Lecture Notes in Mathematics* 1697, [doi:10.1007/BFb0096355](https://doi.org/10.1007/BFb0096355) (paywalled).
  - Spiegel, Huynh & DeBonis (2015), AIAA 2015-2444, free on NTRS: [20150018403](https://ntrs.nasa.gov/citations/20150018403).
  - HiOCFD5 case VI1, [how5.cenaero.be/content/vi1-vortex-transport-uniform-flow](https://how5.cenaero.be/content/vi1-vortex-transport-uniform-flow): PDF definition and meshes (`C1.4_gridfiles_0.zip`, 6.2 MB).
- **Quantities:** L2 error of the conserved variables after one convection period; error over time.
- **Acceptance (gate):**
  - Spatial order 2 ± 0.1 after one period on a domain of at least [−10, 10]².
  - Temporal order equal to the Runge–Kutta order ± 0.1, on a fixed fine mesh.
- **Why:** Spiegel et al. show that Shu's original [−5, 5]² periodic box creates an artificial shear layer that floors the error. Their recommended setup gives a clean asymptotic range.

### Sod, Lax and Shu–Osher shock tubes — V-exact (Sod, Lax) · Must; regression (Shu–Osher) · Useful
- **Sources:**
  - Sod (1978), *JCP* 27:1–31, [doi:10.1016/0021-9991(78)90023-2](https://doi.org/10.1016/0021-9991(78)90023-2); free copy at [hal-01635155](https://hal.science/hal-01635155).
  - Lax (1954) and Shu–Osher (1989): take the initial states from Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. (2009). The original DOIs are unverified.
  - Exact solver: [LANL ExactPack](https://github.com/lanl/ExactPack), BSD-3-Clause (`riemann`, also `noh`, `sedov`).
- **Data:** exact for Sod and Lax. Shu–Osher has no exact solution; its reference is a very fine 1D run, which is a regression check only.
- **Quantities:** ρ, u, p profiles; shock, contact and rarefaction positions; global conservation.
- **Acceptance (gate):**
  - Global density L1 error decreases at an observed order ≥ 0.6. A contact discontinuity converges at about m/(m + 1) for a scheme of formal order m: 2/3 for second order, below the shock's order 1 (Banks, Aslam & Rider, *JCP* 227:6985–7002, 2008, [doi:10.1016/j.jcp.2008.04.002](https://doi.org/10.1016/j.jcp.2008.04.002)).
  - Measure the metrics separately: shock position within 1 cell of exact on every level; contact width (cells across the 10–90 % jump) reported; error in the smooth part of the rarefaction reported with its order.
  - Conservation as a discrete balance: the change in total mass, momentum and energy equals the time-integrated boundary fluxes, to round-off (a multiple of ε). Normalise each quantity by a physical scale (for momentum, p_L·t), because the initial momentum is zero.
  - Run on quasi-1D strips of each cell type.
- **Shu–Osher:** report only.

### NACA 0012 inviscid (Vassberg & Jameson 2010) — V-ref · Must
- **Source:** Vassberg & Jameson, "In pursuit of grid convergence for two-dimensional Euler solutions", *J. Aircraft* 47(4):1152–1166 (2010), [doi:10.2514/1.46737](https://doi.org/10.2514/1.46737). Author PDF: [aero-comlab.stanford.edu/Papers/AIAA-46737-375.pdf](http://aero-comlab.stanford.edu/Papers/AIAA-46737-375.pdf).
- **Grids:** the sharp-trailing-edge NACA 0012 is defined analytically. Its O-grids (Karman–Trefftz, farfield about 150 c) are defined by equations in the paper, from 32² to 4096² cells. We regenerate them; there is no file to download.
- **Reference values:** continuum estimates from FLO82, OVERFLOW and CFL3D (paper Tables 10–11, read from the PDF):

| Case | CL | CD | Spread between codes |
|---|---|---|---|
| M 0.5, α 1.25° | 0.17978–0.18035 | 0 (subcritical; extrapolated CD ≤ 1.2e-5) | CL 5.7e-4 |
| M 0.8, α 0° | 0 | 0.0083415–0.0083423 | CD 8e-7 |
| M 0.8, α 1.25° | 0.34823–0.35621 | 0.022453–0.022685 | CL 0.008, CD 2.3e-4; the point-vortex runs give CD 0.022675–0.022685 |

- **Farfield matters.** The CL band in the table mixes runs with and without a point-vortex farfield correction. We use the point-vortex farfield at about 150 c on the paper's own geometry and grids. The two point-vortex references then give **CL 0.35160–0.35621, CD 0.022675–0.022685** (FLO82, CFL3D with vortex).
- **Acceptance (gate, on our continuum estimate from at least three levels):**
  - M 0.8, α 0°: CD within 1 count of 0.008342.
  - M 0.8, α 1.25°: CL inside 0.3516–0.3562 widened by our own grid-convergence index; CD within 1 count of 0.02268.
  - M 0.5 (subcritical): CD decreases monotonically under refinement and the extrapolated |CD| is below 0.5 count. Report the observed order instead of gating on it: the paper shows sub-second-order force convergence for some codes, attributed to the trailing-edge singularity.
- **Why:** the published spread between codes defines what "converged" means for this case.

### ONERA M6 inviscid — Val (qualitative) · Must
- **Source:** Schmitt & Charpin, AGARD AR-138 (1979) chapter B1, [publications.sto.nato.int/…/AGARD-AR-138.pdf](https://publications.sto.nato.int/publications/AGARD/AGARD-AR-138/AGARD-AR-138.pdf) (17.6 MB, 612 pages, scanned).
- **Machine-readable data:**
  - NPARC archive, [grc.nasa.gov/www/wind/valid/m6wing/m6wing.html](https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html): Test 2308 (M 0.8395, α 3.06°, Re_MAC 11.72e6); Cp files `cp{1..7}{u,l}.ex`.
  - TMR, [onerawingnumerics_val.html](https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html): sharp-TE CAD (IGS/STP), `case_2308.dat` with Cp at η = 0.20, 0.44, 0.65, 0.80, 0.90, **0.96** (not 0.95, as often quoted), 0.99. Also a Fortran grid generator for tets, prisms or hexes that writes `.su2` directly.
- **Uncertainty:** the NPARC files carry a uniform ±0.02 Cp error bar, which looks nominal rather than measured (unverified).
- **Acceptance (report for Euler):**
  - λ-shock pattern and shock positions at η = 0.20–0.65 within ±0.03 x/c of experiment.
  - Cp RMS away from shocks ≤ 0.05.
  - The Euler gate itself is 10 orders of residual drop on this case, with a grid-converging CL; inviscid forces against experiment are not meaningful.

### Supporting inviscid checks — Useful
- **HiOCFD Gaussian bump** (HiOCFD5 VI2; HiOCFD1 C1.1 triangle grids `cfd.ku.edu/hiocfd/bump/`): the entropy error must converge at order 2 ± 0.15. A cheap check of curved walls.
- **NPARC analytic cases** ([archive](https://www.grc.nasa.gov/www/wind/valid/archive.html)): Mach 1.3 normal shock, Mach 2 and 2.5 wedges, Mach 2.35 10° cone, Mach 2.5 Prandtl–Meyer expansion. Shock angle within 0.2° and post-shock state within 0.5 % (oblique-shock and Taylor–Maccoll relations).
- **Billig shock standoff** (*J. Spacecraft Rockets* 4:822, 1967, [doi:10.2514/3.28969](https://doi.org/10.2514/3.28969), paywalled):
  - Correlation: sphere Δ/R = 0.143 exp(3.24/M²); cylinder 0.386 exp(4.67/M²). The coefficients are from secondary sources (unverified).
  - Acceptance: Δ/R within ±5 % for M 3–10.
  - This is the first robustness check for strong shocks before hypersonic work.

---

## M3 — Navier–Stokes + Spalart–Allmaras

The gate is "TMR flat plate, bump and NACA 0012 within tolerance; convergence on degraded meshes".

### Laminar NS MMS — V-exact · Must
- **Source:** Roy et al. 2004 (above); MASA `navierstokes_3d_transient_sutherland`.
- **Acceptance (gate):** L2 order 2 ± 0.1 on all four cell types and the perturbed meshes, FP32 and FP64, with the non-orthogonal correction active. The Reynolds number of the manufactured field must make viscous and convective terms comparable; check this by comparing the magnitudes of the two source-term parts.

### RANS-SA MMS — V-exact · Must
- **Sources:**
  - Eça, Hoekstra, Hay & Pelletier (2007), *IJNMF* 54:119–154, [doi:10.1002/fld.1387](https://doi.org/10.1002/fld.1387), paywalled. Wall-bounded 2D MMS for SA, k-ε and k-ω. Revision: Eça, Hoekstra & Vaz (2012), *IJCFD* 26:313–332, [doi:10.1080/10618562.2012.717617](https://doi.org/10.1080/10618562.2012.717617).
  - The TMR page for the Lisbon verification workshops, [eca1.html](https://tmbwg.github.io/turbmodels/eca1.html), hosts the workshop reports and the Fortran sources of the manufactured functions.
  - Compressible version: Oliver, Estacio-Hiroms, Malaya & Carey, "Manufactured solutions for the Favre-averaged Navier–Stokes equations with eddy-viscosity turbulence models", AIAA 2012-0080, [doi:10.2514/6.2012-80](https://doi.org/10.2514/6.2012-80). Implemented in MASA as `fans_sa` (named free-shear and wall-bounded solutions) and `rans_sa`.
  - TMR has no SA MMS page of its own.
- **SA-neg has no published oracle.** MASA implements standard SA with ν̃ > 0, not the SA-neg branch, which changes production, destruction, diffusion and the eddy viscosity when ν̃ < 0 ([TMR SA page, `#saneg`](https://tmbwg.github.io/turbmodels/spalart.html#saneg)). For that branch, manufacture a field in which ν̃ changes sign, and check the `vv/` source terms against a second derivation written independently from the TMR equations.
- **Quantities:** L2 and L∞ errors of the mean-flow variables and ν̃.
- **Acceptance (gate):**
  - Positive branch: pin the MASA solution identifier and check our source terms against it to round-off.
  - Coupled order with second-order turbulence advection: L2 order 2 ± 0.1 for the mean flow and for ν̃, positive and SA-neg branches.
  - A configuration with first-order turbulence advection is reported separately. Its error passes into the mean flow through ν_t, so every variable is then expected at order 1.

### Couette, Poiseuille and Blasius — V-exact · Must
- **Compressible Couette with viscous heating** (White, *Viscous Fluid Flow*; constant μ and k, isothermal walls): the velocity is linear, so it must be reproduced to round-off (a multiple of ε) on every cell type. Viscous heating makes the temperature quadratic, which is outside the linear reconstruction, so the gate for T and for the wall heat flux is order 2 ± 0.1.
- **Poiseuille:** order 2 ± 0.1 on non-aligned tet and polyhedral meshes, which exercises the non-orthogonal correction.
- **Blasius:**
  - Reference: our own ODE solve to 1e-12, checked against Ganapol, arXiv [1006.3888](https://arxiv.org/abs/1006.3888) (f''(0) = 0.332057336215196; digits unverified). HiOCFD1 C1.4 ([case_c1.4.html](https://cfd.ku.edu/hiocfd/case_c1.4.html)) gives a ready setup.
  - Acceptance at M ≤ 0.1 and Re_x ≥ 1e4: Cf·√Re_x = 0.664 within 1 %, u/U profile within 0.5 %.
  - The remaining error comes from the leading-edge singularity and compressibility.

### TMR SA verification cases — V-ref · Must

Common to all cases on [tmbwg.github.io/turbmodels](https://tmbwg.github.io/turbmodels/):
- CC0; PLOT3D, with 3D CGNS versions for some cases. Five nested levels.
- Reference results from CFL3D (cell-centred, structured) and FUN3D (node-centred, unstructured). Their numerical settings differ between datasets: on the flat plate both use first-order turbulence advection, but for the NACA 0012 point-vortex results FUN3D uses second order. Record the settings from each page with its reference values.
- Each page gives a grid-convergence index (Celik et al. 2008) and `.dat` result files.

| Case (page) | Conditions | Grids | Reference values | Acceptance (gate) |
|---|---|---|---|---|
| 2D zero-pressure-gradient flat plate (`flatplate.html`) | M 0.2, Re 5e6 | 35×25 to 545×385; wall spacing 5e-7; the coarsest level is 42 kB of raw text, the bundles are in LFS | Cf at x = 0.97008 and 1.90334; CD; ν_t and u⁺ profiles. Cf convergence index: CFL3D 0.017 %, FUN3D 0.028 % | Our continuum Cf within 0.5 % of the TMR extrapolated value on hexes; within 1 % on triangles or polyhedra (FUN3D's own uncertainty on the triangulated grids is 0.59 %). |
| 2D bump-in-channel (`bump.html`) | M 0.2, Re 3e6 | 89×41 to 1409×641 | Cf at x = 0.632, 0.75, 0.868; CL, CD. Cf convergence index at 0.75: CFL3D 0.086 %, FUN3D 0.120 % | Continuum Cf and CD within 0.5 % |
| 3D bump-in-channel (`bump3d.html`) | M 0.2, Re 3e6 | 5×45×21 to 65×705×321 | CFL3D, FUN3D | Same as 2D. This is the first real 3D RANS gate and the M5 scaling seed. |
| NACA 0012 numerics (`naca0012numerics_val.html`) | M 0.15, Re 6e6, α 0, 10, 15° | Three families, up to 7169×2049; structured and unstructured CGNS | α 10°, family II, extrapolated, with point-vortex farfield: **CL 1.0912–1.0913, CD 0.01221–0.01222, CDp 0.00601–0.00602, Cm 0.00677–0.00679**. DPW-6 gives CL 1.0909–1.0911. | Our extrapolated value (family II, at least three levels): ΔCL ≤ 0.002, ΔCD ≤ 1 count, ΔCm ≤ 0.0002. The two NASA codes agree to about 1e-4 in CL. |
| Airfoil near-wake verification (`airfoilwakeverif500c.html`) | DSMA661, M 0.088, Re 1.2e6, farfield at 500 c | Supplied | CFL3D, FUN3D | Useful: wake ν_t and velocity defect match within the CFL3D/FUN3D spread. |

**Degraded meshes.** The M3 gate also asks for convergence on degraded meshes. Run the flat plate and bump on the perturbed polyhedral variants that #2 generates. Require the continuum Cf within 1 % of the TMR value and the same residual drop. Perturbation changes the grid, not the converged answer.

### TMR validation cases (experiment) — Val · Useful (report)

These cases use the same TMR conventions and CC0 files. Gate code correctness against the CFL3D/FUN3D SA results published on each page, as continuum estimates: separation and reattachment points within ±0.01 x/c (or x/H), and Cp/Cf curves within the code-to-code spread. Report the difference to experiment without gating on it.

| Case (page) | Conditions | Data | Published uncertainty | Report targets |
|---|---|---|---|---|
| NACA 0012 validation (`naca0012_val.html`) | M 0.15, Re 6e6, α 0 to stall; sharp-TE rescaled 0012; C-grids 113×33 to 1793×513 | Ladson (NASA TM-4074) CL/CD, tripped; Gregory & O'Reilly Cp (R&M 3726); Abbott, McCroskey | None numerical. Tripping strongly changes CD (McCroskey: tripped CD0 at 3e6 is about 10 % above 6e6). | CL within 0.02 for α ≤ 10° (unverified as a published tolerance) |
| Bachalo–Johnson axisymmetric transonic bump (`axibump_val.html`) | M 0.875; Re_c 2.763e6 (unverified); grids 91×41 to 1441×641 | Cp; u, u'v', k profiles at x/c = −0.25 to 1.375. New Sandia re-test: Lynch et al., *AIAA J* 61(4), 2023 | Separation x/c ≈ 0.7, reattachment ≈ 1.1. A larger tunnel moved the shock 0.01–0.015 c forward. | Shock x/c within ±0.02 |
| NASA wall-mounted hump, CFDVAL2004 case 3 (`nasahump_val.html`) | M 0.1, Re_c 936,000; grids up to 1633×433 (705k cells, 8.7 MB gz); CGNS and AFLR3 versions | Cp, oil-film Cf, PIV profiles | **Reattachment x/c = 1.11 ± 0.003**; 2D PIV vs hot-wire within 2–3 % | SA over-predicts the bubble length; report Δx_reatt |
| Driver–Seegmiller backward-facing step (`backstep_val.html`; also ERCOFTAC case 030) | M 0.128, Re_H ≈ 36,000; 4-zone grids, finest 7.7 MB gz | Cp, Cf, velocity and stress profiles | **x_reatt/H = 6.26 ± 0.10**; Cf ±8 % (95 %), ±15 % in the separated region | x_reatt/H within ±0.3 |
| ACC-SBLI Mach 7 cone–cylinder flare (`axiswblim7_val.html`) | Kussoy & Horstman, M 7.11, T_w 311 K | p_w, q_w, upstream profiles | **p_w and q_w ±10 %**; profiles u ±3 %, T ±6 % (NASA TM-101075, [NTRS 19890010729](https://ntrs.nasa.gov/citations/19890010729)) | p_w, q_w within ±10 % outside separation; peak q_w within ±20 % |
| Supersonic flat plate (`ZPGflatplateSS_val.html`) | M 2 and 5, T_w/T_aw 0.2–1.0 | van Driest II theory (not an experiment) | — | Cf within 3 % of van Driest II at M 2. This is the first wall heat-flux check before CHT. |

### RAE 2822 — Val · Useful (report); also the M6 baseline
- **Source:** Cook, McDonald & Firmin, AGARD AR-138 chapter A6. The PDF is above; its accuracy table is unreadable in the scan (unverified).
- **Data:** NPARC [raetaf.html](https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf.html) has `geom.txt` and study #4 (case 6, corrected **M 0.729, α 2.31°**, 369×65 C-grid). Its stated Cp measurement error is **≈ 0.0026**. A new Airbus/NWTF pETW dataset (2025) adds wall pressures: [nwtf.ac.uk/dataset/2687](https://www.nwtf.ac.uk/dataset/2687/); licence not stated.
- **Conditions (AR-138 headers, tripped):**
  - Case 6: M 0.725, α 2.92°, Re 6.5e6.
  - Case 9: M 0.730, α 3.19°, Re 6.5e6.
  - Case 10: M 0.750, α 3.19°, Re 6.2e6.
- **Wind-tunnel corrections are not unique.** ADODG case 2 and HiOCFD case C2.2 use case 9 at M 0.734, α 2.79°. 2.54° is also widely quoted (unverified). Proposed: **case 9 at M 0.734 with fixed CL = 0.824** (the ADODG definition), which removes the α ambiguity. This awaits the lead's sign-off ([decision 2](#gaps-and-decisions-for-the-lead)).
- **Acceptance (report):** Cp within 0.03–0.05 away from the shock; shock x/c within ±0.02. Do not gate on CL or CD against experiment: the α correction itself is uncertain by about ±0.25°.

### ONERA M6 viscous — V-ref · Useful in M3, Must in M5
- **Source:** TMR [onerawingnumerics_val.html](https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html). Cases 2308 (α 3.06°) and 2565 (α 6.06°), M 0.84, Re_root 14.6e6.
- **SA reference forces, case 2308** (`combined_forces_pitchmom_maxmut.dat`):

| Code | Points | CL | CD |
|---|---|---|---|
| FUN3D | 0.96 M | 0.26585 | 0.017227 |
| FUN3D | 7.6 M | 0.26788 | 0.016923 |
| FUN3D | 60.8 M | 0.26955 | 0.016947 |
| CFL3D | 69.2 M | 0.26932 | 0.016956 |
| USM3D (tet) | 363 M | 0.27062 | 0.017050 |

- **Acceptance:** on the TMR grid family built with its own Fortran generator, our continuum CL within 0.5 % and CD within 2 counts of the FUN3D and CFL3D finest-grid values. The spread between codes on the finest grids is about 0.4 % in CL and 1 count in CD. Level-by-level differences from FUN3D are reported. Cp against experiment at the seven stations is a report.

### Supersonic and hypersonic shock–boundary-layer interaction — Val · Useful

These cases fill the high-speed gap. None is a milestone gate, but they are the first compressible-RANS cases a customer in high speed will ask for.

**Schülein Mach 5 impinging shock (DLR) — preferred SWBLI case.**
- **Sources:** Schülein, *AIAA J* 44(8):1732–1741 (2006), [doi:10.2514/1.15110](https://doi.org/10.2514/1.15110), paywalled. The tables are reproduced, with Schülein's permission, in **NASA TM-2013-216604** (Marvin, Brown & Gnoffo), [NTRS 20140010683](https://ntrs.nasa.gov/citations/20140010683) (1.3 MB, 127 pages). Cite both.
- **Conditions:** M 5, Re 37e6/m, p0 2.12 MPa, T0 410 K, T_w 300 K. Shock-generator angles 6°, 10°, 14° (pressure ratio 3.78, 7.64, 13.65). Separation at 334 mm and 314 mm for 10° and 14°; impingement at ≈ 350 mm.
- **Data:** 67 wall taps, Cf (oil-film interferometry), q_w (thin skin), pitot profiles. Transcribe from the TM.
- **Uncertainty:** **p ±2 %, Cf ±4–10 %, q_w ±5 %**, probe position ±0.05 mm.
- **Acceptance (report):** p_w within ±5 % outside the interaction and ±10 % at the peak; separation point within ±5 mm; q_w within ±15 % upstream and ±25 % at the reattachment peak, where RANS over-predicts.
- The same TM also tabulates Coleman Mach 9 corners, Holden CUBRC Mach 8–13 corners and cone-flares, HIFiRE, and the Kussoy Mach 8 impinging shock: a ready-made hypersonic set for later.

**Settles compression corners, Mach 2.85 (8°, 16°, 20°, 24°).**
- **Sources:** Settles et al., *AIAA J* 14(12), 1976, [doi:10.2514/3.61513](https://doi.org/10.2514/3.61513); Settles, Fitzpatrick & Bogdonoff, *AIAA J* 17(6), 1979, [doi:10.2514/3.61180](https://doi.org/10.2514/3.61180). Both paywalled.
- **Use NASA CR-177638 (1994), "new and corrected data"**, full text at [archive.org/details/nasa_techdoc_19940032012](https://archive.org/details/nasa_techdoc_19940032012). It replaces the 1991 Settles & Dodson database (CR-177577). The 1991 Cf column was normalised wrongly ("essentially worthless") and its wall pressure is now p_w/p∞. **Do not use the 1991 numbers.**
- **Data:** p_w, Cf, pitot and static profiles (ASCII tables in the PDF, about 270 kB of text). The same report adds Kuntz/Amatucci/Addy Mach 2.94 LDV profiles.
- **Inflow:** the report warns that no experiment fully defines the inflow. Match δ* (or δ, δ*, θ and Cf within 15 %).
- **Acceptance (report):**
  - p_w within ±5 % upstream and on the plateau.
  - Separation and reattachment within ±0.5 δ0 for 16° and 20°.
  - Cf ±15 % downstream.
  - 24° is three-dimensional in the experiment: trends only.

**Other SWBLI references.**
- *Laminar:* Degrez, Boccadoro & Wendt, *JFM* 177:247–263 (1987), [doi:10.1017/S0022112087000727](https://doi.org/10.1017/S0022112087000727); M 2.15, paywalled, figures must be digitised. Open alternative: Hakkinen et al., NASA Memo 2-18-59W, [NTRS 19980231989](https://ntrs.nasa.gov/citations/19980231989), M 2.0, with p_w and Cf. Acceptance: grid-converged p_w within ±3 %, separation length within ±10 %. A laminar interaction separates scheme error from turbulence-model error, which makes it the best first SWBLI check.
- *Holden CUBRC double cone (25°/55°) and hollow cylinder–flare,* M 9.5–11.4 in N₂. Open description in [NTRS 20010004101](https://ntrs.nasa.gov/citations/20010004101), but no open data tables. **Deferred** until the solver has vibrational nonequilibrium.
- *Dupont/Dussauge IUSTI Mach 2.3:* unsteady, data in figures only. Low priority.
- *Sajben transonic diffuser:* weak and strong shock, NPARC [transdif.html](https://www.grc.nasa.gov/www/wind/valid/transdif/transdif.html). Useful for internal flows with shock-induced separation.

---

## M4 — SDK + CLI v0

No new data. The gate has three parts:
- **TMR cases driven by the SDK.** The M3 flat plate, bump and NACA 0012 numerics cases run from a case file through `plan / run / diff`, using the TMR CGNS files (CGNS is a declared reader format), with the M3 acceptance numbers.
- **Boundary conditions are stated in the case.** TMR warns that the boundary conditions embedded in its CGNS files are guidance, not necessarily the correct setup. The case file states each boundary condition, and `plan` shows where each one comes from.
- **Missing and invalid boundary conditions are caught.** Variants built from the same files, run through the public CLI: one boundary zone with no condition (`plan` must report the coded missing-condition error and its fix) and one with an invalid type.

AFLR3 (`.ugrid` with `.mapbc`) is not a declared reader format. It stays out of scope unless an importer is scoped separately.

---

## M5 — Multi-GPU + precision

This section covers only the numerical-reference part of the M5 gate: "CL/CD identical within 0.1 % between precisions". The scaling part (≥ 80 % efficiency at 8 GPUs, measured up to 64) and the published ns/cell/iteration are measured with `bench/` and are outside this catalogue.

- **ONERA M6 viscous** (M3 entry, Must): the 7.6 M- and 60.8 M-point TMR levels are the multi-GPU scaling cases. Gate: |CL_FP32 − CL_FP64| ≤ 0.1 % of CL_FP64 and |CD_FP32 − CD_FP64| ≤ 0.1 % of CD_FP64, on the same mesh, both converged to their residual plateau. CL and CD must also be bitwise identical between 1 and N GPUs in deterministic mode.
- **NACA 0012 numerics at α 15°** (Must): near CLmax, where LAVA reported FP32 sensitivity. The same 0.1 % gate checks that our mixed precision holds where it is hardest.
- **DPW CRM** (Useful; see M7): Vassberg grids from 5.3 M to 41.6 M points (L3.M is a 998 MB `.p3d`). This is the first production-sized precision study.

---

## M6 — GPU discrete adjoint

The gate is "dot-product test + finite differences; RAE 2822 and ONERA M6 drag minimisation".

### Gradient correctness — V-exact · Must
- **Precedent:** Nielsen & Kleb, *AIAA J* 44(4), 2006, [NTRS 20050060975](https://ntrs.nasa.gov/citations/20050060975). On a turbulent ONERA M6 (16,391 nodes), dCL and dCD from complex-step, hand-coded adjoint and complex-variable adjoint agree to about 11 digits. That is one implementation's result on FUN3D's grid and parameterisation: it shows what is achievable, not a universal tolerance.
- **Analytic adjoint:** Giles & Pierce, *JFM* 426:327–345 (2001), free PDF at [people.maths.ox.ac.uk/gilesm/files/jfm01.pdf](https://people.maths.ox.ac.uk/gilesm/files/jfm01.pdf). Analytic quasi-1D adjoint, the only cross-code analytic adjoint reference. Useful.
- **Conditions:**
  - Complex-step needs complex-safe versions of abs, min/max, limiters and every branch, following Martins, Sturdza & Alonso, *ACM TOMS* 29:245–262 (2003), [doi:10.1145/838250.838251](https://doi.org/10.1145/838250.838251). The branch conventions are fixed in the kernel generator.
  - Primal and adjoint are converged far enough that their residuals do not limit the gradient error: at least two decades below the tolerance, propagated through the objective.
  - Compare with a mixed norm: |g − g_ref| ≤ tol · max(|g_ref|, 1e-3 ‖g_ref‖∞), so that near-zero sensitivities do not produce meaningless relative errors.
- **Acceptance (gate):**
  - Dot-product test ⟨Jv, w⟩ = ⟨v, Jᵀw⟩ to a multiple of ε (with random v, w), per kernel and for the whole residual, in FP32 and FP64.
  - Gradient against complex-step on 5–10 design variables: tol = 1e-10 in FP64. In FP32 the tolerance is set after measuring the plateau, starting at 1e-3.
  - Against central differences: agreement to the finite-difference truncation error, with the step chosen by a step sweep.
  - At least one end-to-end geometric sensitivity (FFD variable → mesh deformation → objective), not only flow-variable sensitivities.

### ADODG case 2: RAE 2822 drag minimisation — Must
- **Definition:** AIAA Aerodynamic Design Optimization Discussion Group (ADODG). Official PDFs at [sites.google.com/view/mcgill-computational-aerogroup/adodg](https://sites.google.com/view/mcgill-computational-aerogroup/adodg), mirrored on the [MDO Lab wiki](https://mdolab.engin.umich.edu/wiki/aerodynamic-design-optimization-workshop).
  - Conditions: RAE 2822 at M 0.734, Re 6.5e6, **CL = 0.824**.
  - Constraints: cm ≥ −0.092, area ≥ initial.
- **References:** He et al. 2019 (MDO Lab) report a drag *reduction* of **88.66 counts**. Lee & Zingg (Jetstream) report an *absolute* optimised drag of 119.22 counts on their finest mesh (their baseline is unverified). These figures differ in mesh, baseline and design space, so they do not define a scatter band for the optimum. A frozen-turbulence adjoint converges poorly here (He 2019): the SA adjoint must be complete.
- **Acceptance (gate):** gradients pass [Gradient correctness](#gradient-correctness--v-exact--must); the optimiser converges (optimality and feasibility tolerances recorded) with every constraint satisfied; drag decreases. Report the achieved reduction next to He 2019. Gating on the optimum value would require reproducing a reference's mesh and parameterisation exactly.

### SU2 ONERA M6 constrained design — Must
- **Source:** [su2code.github.io/tutorials/Inviscid_3D_Constrained_ONERAM6](https://su2code.github.io/tutorials/Inviscid_3D_Constrained_ONERAM6/), LGPL-2.1. Euler, M 0.8395, fixed CL 0.286, drag objective, thickness constraints at five stations, FFD box. Mesh `mesh_ONERAM6_inv_FFD.su2` (108,396 points, 26.6 MB).
- **Acceptance (gate):** gradients against complex-step as above. The optimisation reduces CD with all constraints met. Compare the converged CD with SU2's published history as a report: there is no community reference optimum.
- **Why:** this is the PLAN's ONERA M6 drag minimisation, on a mesh that someone else built.

### ADODG case 4.1: CRM wing — Useful
- **Definition:** Lyu, Kenway & Martins, *AIAA J* 53(4), 2015. M 0.85, Re 5e6, CL 0.5, CM ≥ −0.17, volume ≥ initial, thickness ≥ 25 % of the initial value. Grids and geometry on **Mendeley Data, CC BY 4.0** (link above).
- **Reference:** baseline **199.7 counts** on the 28.8 M-cell L0 grid (201.7 on L1 at 3.6 M cells; 211.1 on L2 at 450 k). Optimised **182.8 counts (−8.5 %)** with 720 FFD variables; multistart optima agree within 0.1 count.
- **Acceptance:** the baseline is a primal V-ref: CD within ±1 count of 201.7 on the Mendeley L1 grid. The optimum is comparable only with the same grids and the same 720 FFD variables; in that setup, report the reduction against 8.5 %.
- Useful rather than Must because it needs the full SA adjoint on a 3.6 M-cell mesh. It becomes the M6 stretch goal and the showcase.

### ADODG case 1: NACA 0012, Euler — Useful (gradient check only)
- M 0.85, α 0, minimise CD with thickness ≥ baseline. Published optima range from 25 to 297 counts and are not unique, so the optimum cannot be a pass/fail gate. Use the case to check gradients near shocks.

---

## M7 — SST, DDES, WMLES, low-dissipation flux, BDF2

The PLAN gate is "cylinder, backward-facing step, Ahmed; CRM-HL near CLmax compared with HLPW-5". Experiments cannot prove new equations or a new time integrator correct, so verification cases come first. The canonical LES cases (Taylor–Green, channel) are added as Must because they are the only clean proof that the low-dissipation flux and the wall model work.

### SST MMS — V-exact · Must
- Manufacture mean flow, k and ω in `vv/`, following the wall-bounded construction of Eça et al. (2007) (M3 entry). Check the source terms against a second, independent derivation.
- Gate: L2 order 2 ± 0.1 for the mean flow, k and ω, on all cell types and precisions, with second-order turbulence advection.

### BDF2 temporal order — V-exact · Must
- Isentropic vortex (M2 entry) on a fixed fine mesh with dual-time BDF2, refining Δt. Converge the dual-time residual so that iteration error stays at least two decades below the temporal error; spatial error must also be negligible on that mesh.
- Gate: temporal order 2 ± 0.1.

### DDES shielding — V-ref · Must
- The DDES shielding function must keep an attached boundary layer in RANS mode (Spalart et al., *Theor. Comput. Fluid Dyn.* 20:181–195, 2006, [doi:10.1007/s00162-006-0015-0](https://doi.org/10.1007/s00162-006-0015-0)).
- Run the TMR flat plate with DDES on a grid fine enough in the streamwise and spanwise directions to trigger LES mode. Gate: Cf within 1 % of our own SA result on the same grid; the shielding function stays near 1 across the boundary layer.
- Repeat for each DDES/IDDES branch we ship.

### Taylor–Green vortex, Re 1600 — V-ref · Must
- **Definition:** HiOCFD5 case WS1, [how5.cenaero.be/content/ws1-dns-taylor-green-vortex-re1600](https://how5.cenaero.be/content/ws1-dns-taylor-green-vortex-re1600). Periodic box [−πL, πL]³, M0 = 0.1, Pr = 0.71, t_final = 10 t_c (case PDF `BS1_TaylorGreenVortexRe1600_0.pdf`, which places peak dissipation "at t ≈ 8 t_c"). Outputs: E_k, dE_k/dt and enstrophy over time; vorticity on the face x = −πL and energy spectra at t = 8 t_c.
- **Reference:** 512³ spectral DNS, [cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag](https://cfd.ku.edu/hiocfd/spectral_Re1600_512.gdiag).
  - ASCII, 102 kB; columns t, E_k, −dE/dt, enstrophy; Δt = 0.01.
  - Downloaded and read: **peak −dE/dt = 0.0128575 at t = 8.97**.
  - No licence stated. Commonly attributed to van Rees et al., *JCP* 230 (2011) (attribution unverified).
- **Supporting:** DeBonis, AIAA 2013-0382, [NTRS 20130011044](https://ntrs.nasa.gov/citations/20130011044) (free). At t* = 9, 128³–512³ grids are within 2 % of the reference and 64³ is 9.8 % off.
- **Acceptance (gate, provisional):**
  - Peak dissipation within 5 % at 256³ and within 2 % at 512³-equivalent, with peak time within ±0.2 t_c. Also compare E_k(t) and the energy spectrum at t = 8 t_c, because peak dissipation alone can agree through cancellation between physical and numerical dissipation.
  - Close the compressible kinetic-energy budget before attributing anything to numerics: −dE_k/dt equals solenoidal (enstrophy-based) dissipation plus dilatational dissipation minus pressure–dilatation work (DeBonis 2013, eqs. 11–16). Only the remainder is numerical dissipation. Report it, because that is what the low-dissipation flux must reduce.
  - Run hex and polyhedral meshes; HiOCFD supplies perturbed meshes on request.
- **Why:** 2 % is the level that DeBonis reached with a high-order scheme. The thresholds for our second-order scheme are fixed in `acceptance.md` after our own 64³–512³ refinement study.

### Channel flow DNS — V-ref · Must
- **Source:** Lee & Moser, *JFM* 774:395 (2015), [turbulence.oden.utexas.edu/channel2015/content/Data_2015_5200.html](https://turbulence.oden.utexas.edu/channel2015/content/Data_2015_5200.html). Re_τ 180, 550, 1000, 2000, 5200.
  - ASCII mean, Reynolds stress and budget profiles, each with a `_stdev` file; HDF5 spectra (220 MB at 5200).
  - No licence; the file headers ask for citation.
- **Lower-Re alternatives:** Moser, Kim & Mansour (1999), `turbulence.oden.utexas.edu/data/MKM/`; Hoyas & Jiménez, [torroja.dmt.upm.es/channels/data](https://torroja.dmt.upm.es/channels/data/).
- **Uncertainty:** per-point statistical standard deviation is published. At Re_τ 5200, max σ(U⁺) = 0.036, about 0.13 % of the centreline value.
- **Forcing:** run at constant bulk flow rate (matched Re_b), so that u_τ and Cf are predictions. With a fixed pressure gradient the wall stress is set by the forcing, and only the profiles test anything.
- **Acceptance:**
  - Wall-resolved LES at Re_τ 180/550 (gate): u_τ within ±2 %; mean velocity and Reynolds-stress profiles within the DNS σ plus our own sampling interval.
  - WMLES at Re_τ 5200 (report with band): Cf within ±5 %, U⁺ within ±3 % for y/δ > 0.1.
  - RANS (SA, SST) (report): U⁺ within ±3 %.
  - WMLES and RANS against DNS include model error, so a miss there is first a model or wall-model finding, then a code bug.

### Circular cylinder — Val · Must
- **Re 3900 (DDES/LES):** Parnaudeau, Carlier, Heitz & Lamballais, *Phys. Fluids* 20:085101 (2008), [doi:10.1063/1.2957018](https://doi.org/10.1063/1.2957018) (paywalled; author PDF on irisa.fr). No public data files (unverified).

| Quantity | Value | Uncertainty |
|---|---|---|
| Strouhal number | 0.208 | ±0.002 (hot-wire) |
| Recirculation length L_r/D | 1.51 (PIV), 1.56 (their LES) | Statistical: ±30 % after 12 shedding periods, ±6 % after 120 |
| Cd | 0.99 | ±0.05 (compiled by Kravchenko & Moin 2000; unverified) |

- **Acceptance (gate):** St 0.208 ± 0.010; L_r/D 1.5 ± 0.2; Cd 0.99 ± 0.08. Each is judged by its confidence interval (see [Statistics](#how-to-read-an-entry)), and averaging continues until the 95 % half-width of L_r is below 0.05 D. Spanwise domain ≥ πD; results shown on two grid levels.
- **Why:** the published statistical uncertainty of L_r is large unless the average is long, so the gate is on a converged interval, with a band wider than the hot-wire uncertainty.
- **High Re** (Achenbach 1968, *JFM* 34:625, paywalled): report only, ±15 % on Cp and Cf (unverified values).

### Backward-facing step — Val · Must
- **RANS and DDES:** Driver–Seegmiller (M3 entry). Gate: x_reatt/H = 6.26 ± 0.3 for SST and DDES.
- **LES:** Le, Moin & Kim (1997) DNS, ERCOFTAC case 031 ([case031](http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case031), `bwst-allfiles.zip`). Re_h 5100, expansion ratio 1.20, profiles and Reynolds-stress budgets at x/h = −3 to 19.
  - Reattachment about 6.28 h (unverified).
  - The DNS averaged over only about 4.5 flow-throughs, so compare budgets qualitatively.
  - Acceptance: x_R/h = 6.28 ± 0.3; profile RMS ≤ 0.04 U₀. Useful.

### Ahmed body, 25° and 35° — Val · Must
- **Source:** Lienhart & Becker (LSTM Erlangen), ERCOFTAC case 082, [case082](http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case082). CC BY-NC-SA 4.0: use is blocked until [decision 3](#gaps-and-decisions-for-the-lead) is settled. If permission is refused, fall back on the Ahmed (1984) drag only plus code-to-code comparison with AhmedML.
  - Data: `ahby-allfiles.zip` (2.4 MB, ASCII): inlet profiles, LDA in the symmetry plane and in cross planes, hot-wire boundary layers, rear Cp.
  - Conditions: U 40 m/s, Re_h 768,000, blockage 4 %.
- **Drag:** the drag reference is Ahmed, Ramm & Faltin (1984), SAE 840300. Values are read from a figure: about 0.285 at 25° and 0.257 at 35° (unverified to three digits). Their force error is ±0.2 N (about ±0.001 Cd), but facilities disagree by more than that: later 25° measurements reach about 0.30 (unverified).
- **Open alternative:** AhmedML (Hugging Face `neashton/ahmedml`, CC-BY-SA-4.0) gives 500 hybrid RANS-LES variants. These are another code's results, usable only for code-to-code comparison.
- **Acceptance:**
  - 35°: Cd within ±5 % of 0.257; wake U/U∞ RMS error ≤ 0.10.
  - 25°: DDES/WMLES must reattach on the slant, with Cd within ±8 %. Report only for RANS.
- **Why:** 25° sits just below the critical slant angle (about 30°), where RANS is known to fail and DDES is sensitive to the RANS-to-LES transition. The bands reflect the scatter between facilities, not the balance accuracy.

### HLPW-5 CRM-HL — V-ref (TC1) and Val (TC2.4) · Must
- **Sources:** [aiaa-hlpw.org](https://aiaa-hlpw.org). HLPW-5 archive `hlpw5.s3.us-east-1.amazonaws.com/website/Workshop5.zip` (1.75 GB): geometry, grid links, experimental data, case definitions. Summary paper: [NTRS 20240014255](https://ntrs.nasa.gov/citations/20240014255) (read). Rights per asset (see the NASA row of the licence table); no licence on the S3 grids.
- **TC1, CRM-HL wing–body verification (V-ref gate):**
  - Conditions: M 0.20, Re_c 5.6e6, α 11°, T_ref 521 °R. Cuboid domain ±65,000 in. (full-scale inches, MAC 275.8 in.).
  - Reference: 12 SA-neg results from different codes converge, as N^(−1/3) → 0, to a range of **ΔCL ≈ 0.005, ΔCD ≈ 0.0005, ΔCM ≈ 0.003**. SA-neg-R(crot1)-QCR2000: 7 results within ΔCL 0.004, ΔCD 0.0006, ΔCM 0.002.
  - Gate: our SA-neg continuum estimate lies inside that range, widened by our own grid-convergence index.
- **TC2, configuration build-up (Val):**
  - Configurations 2.1 (WBHV), 2.2 (+ slat), 2.3 (+ flaps), 2.4 (+ nacelle, pylon, chine = CRM-HL-LDG). M 0.20; Re_c 5.4e6 (2.1) and 5.9e6 (2.2–2.4); α sweep up to 23.6° for 2.4; T_ref 518.67 °R.
  - Experiment: ONERA F1 data for the as-designed geometry, corrected to free air.
  - Scatter between submissions at α 23° (TC2.2): **≈ 0.15 in CL for HRLES and WMLES, ≈ 0.5 for RANS**. Many WMLES entries under-predicted CLmax; transition modelling on the slat changes the result strongly.
  - Gate (WMLES, TC2.4): CLmax within ±0.075 of ONERA F1 (half the scale-resolving scatter), α at CLmax reported. RANS near CLmax: report only, because no RANS model predicted this flow.
- **Background:** HLPW-4 (Rumsey et al., [NTRS 20220005271](https://ntrs.nasa.gov/citations/20220005271)) ran the QinetiQ 5 m semispan model at M 0.2, Re_MAC 5.49e6, with RANS scatter ≈ 0.1 in CL at α 7.05° and ≈ 0.68 at 19.57°. HLPW-6 takes place in June 2027; its test-case documents and committee grids (`.b8.ugrid`, `.cgns`, `.mapbc`, up to 18.4 GB per level) are already published.
- **Capacity check:** RANS meshes for these cases run from 100 M to 500 M+ nodes, and HLPW-5 included meshes above a billion cells. That tests the GB10's 128 GB memory.

### Periodic hill, Re_h 10,595 — V-ref (LES) · Useful
- **Source:** Breuer, Peller, Rapp & Manhart, *Comput. Fluids* 38:433–457 (2009), and the Rapp & Manhart experiments. ERCOFTAC KBwiki [UFR 3-30](https://kbwiki.ercoftac.org/w/index.php/UFR_3-30). About 145 `.dat` files on S3, no login. Columns y/h, U, V, uu, vv, uv. Re_h 5600, 10,600, 19,000, 37,000.
- **Reference:** LESOCC gives separation x/h = 0.19 and reattachment 4.69; MGLET about 0.22 and 4.72 (unverified).
- **Acceptance (LES/WMLES):** x_reatt/h = 4.69 ± 0.25, profile RMS ≤ 0.05 U_b. Report only for RANS.
- **Licence:** KBwiki terms reserve text and data mining and AI development or evaluation (see the licence table). Automated scoring by the harness needs ERCOFTAC's licence first ([decision 3](#gaps-and-decisions-for-the-lead)).

### DPW CRM — Val + workshop scatter · Useful
- **Sources:** [aiaa-dpw.org](https://www.aiaa-dpw.org/), grids at [dpw.larc.nasa.gov](https://dpw.larc.nasa.gov/) (DPW1–DPW8). Geometry and NTF/Ames/ETW data at [commonresearchmodel.larc.nasa.gov](https://commonresearchmodel.larc.nasa.gov/): IGES/STP; WB, WBNP and WBT configurations; aeroelastic shapes; PSP.
- **Cases:**
  - DPW-6: CRM WB and WBNP at M 0.85, Re 5e6, fixed CL = 0.5, with a grid-convergence study.
  - DPW-8: verification cases V1 ONERA OAT15A (buffet; see the licence note above), V2 Joukowski airfoil, V3 DPW-III wing. Its "scatter" group fixes the protocol: SA-neg, Sutherland, adiabatic wall, residuals to 1e-10. Reusing that protocol gives an exact code-to-code comparison.
- **Uncertainty:** NTF repeatability target about 0.5 count; tunnel-to-tunnel CD differences up to about 10 counts (both unverified).
- **Acceptance:** CD at CL 0.5 on the committee medium grid within the DPW core-solution band, about ±5 counts around the median (unverified; check Tinoco et al., *J. Aircraft* 2018, [doi:10.2514/1.C034409](https://doi.org/10.2514/1.C034409)). Grid-convergence order consistent with the workshop. Gate against workshop scatter rather than the tunnel.

### AGARD AR-303 — Val · Useful
- **Source:** "A selection of experimental test cases for the validation of CFD codes" (1994). Vol 1 (4.3 MB) and Vol 2 (38 MB, 602 pages) at [publications.sto.nato.int](https://publications.sto.nato.int/publications/AGARD/AGARD-AR-303-VOL-2/AGARD-AR-303-Vol-2.pdf). Also on NTRS (19950011431).
- **Data:** printed tables only; machine-readable copies exist only as third-party transcriptions (NPARC, TMR).
- **Use:** a source to transcribe additional airfoil, wing, delta-wing and SWBLI cases into the library when a customer case needs them. No case from it is a gate.

---

## M8 — Pressure-based solver

The gate is "cavity, cylinder, DrivAer compared with references". Cavity and cylinder results cannot prove the pressure–velocity coupling correct on their own, so a manufactured solution comes first.

### Incompressible MMS — V-exact · Must
- A divergence-free manufactured velocity with a smooth pressure (MASA has incompressible NS solutions to cross-check against). Fix the pressure gauge (zero mean) before comparing.
- **Gate:**
  - L2 order 2 ± 0.1 for velocity, ≥ 1.9 for pressure.
  - Discrete divergence (continuity residual) at round-off after convergence; boundary mass balance at round-off.
  - Odd–even (checkerboard) pressure mode amplitude decreases under refinement.
  - Run on all cell types and precisions, and with each linear-solver backend (AmgX, hypre).

### Lid-driven cavity — V-ref · Must
| Reference | Access | Data | Accuracy |
|---|---|---|---|
| Botella & Peyret, *Comput. Fluids* 27:421–433 (1998), [doi:10.1016/S0045-7930(98)00002-4](https://doi.org/10.1016/S0045-7930(98)00002-4) | Paywalled | Tables, Re 1000, Chebyshev N = 160 | Velocities converged to about 7 digits (from the abstract; unverified) |
| Ghia, Ghia & Shin, *JCP* 48:387–411 (1982), [doi:10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4) | Paywalled | Centreline velocity tables, about 17 points each, Re 100–10,000 | Errors up to about 4 % at Re 1000, according to Botella & Peyret |
| Erturk, Corke & Gökçöl, *IJNMF* 48:747–774 (2005), [arXiv cs/0411047](https://arxiv.org/abs/cs/0411047) | Free | 601² grid, Re up to 21,000 | Second order; steadiness above Re about 8000 is debated |

- **Acceptance (gate):**
  - Re 1000 centreline velocities within 0.1 % of the lid speed against Botella & Peyret on the finest grid, with observed order 2 ± 0.1 on the grid sequence.
  - Ghia within ±1 % (±2 % at Re ≥ 5000).
  - Erturk at high Re is qualitative only.
  - The density-based solver may run the same case at M ≤ 0.05 with low-Mach preconditioning, as a cross-check.

### Schäfer–Turek cylinder (DFG 1996) — V-ref · Must
- **Source:** Schäfer & Turek, *NNFM* 52:547–566 (1996), [doi:10.1007/978-3-322-89849-4_39](https://doi.org/10.1007/978-3-322-89849-4_39) (paywalled). Reference values on the Featflow pages [dfg_benchmark1_re20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html), `dfg_benchmark2_re100`, `dfg_benchmark3_re100` and `dfg_flow3d`. No licence stated.
- **Setup:** channel 2.2 × 0.41, cylinder of radius 0.05 at (0.2, 0.2), ν = 1e-3.

| Case | Reference (Featflow pages) | Acceptance (gate) |
|---|---|---|
| 2D-1, steady, Re 20 | CD **5.57953523384**, CL **0.010618948146**, Δp **0.11752016697** | CD within 0.1 %, Δp within 0.2 %, CL within 2 % |
| 2D-2, periodic, Re 100 | The page's original results are superseded by the higher-order force evaluation in `20141124_dfg_bench2_official.zip` (downloaded): Q2 level 6, CD max **3.22739**, CL from **−1.02129** to **0.98657** (level 5 differs by 0.02 % in CD max). Paper bounds: CD max 3.22–3.24, CL max 0.99–1.01, St 0.295–0.305 (unverified) | St within ±1 %, CD max within 0.5 % |
| 2D-3, unsteady inflow | Featflow level 6, Δt = 1/1600: CD max **2.9437637214**, CL max **0.47748781595**, Δp(t = 8) **0.11154138872**. These are finite-level results, not proven continuum values to every printed digit. | CD max within 0.5 %, CL max within 2 % |
| 3D-1Z, steady, Re 20 | CD **6.18533**, CL **0.009401** (Braack & Richter 2006) | CD within 0.5 % |

- **Why:** the Featflow values are converged far beyond our target; their level-to-level change is much smaller than our tolerance. The tolerances allow a second-order finite-volume scheme to reach them on meshes of a few hundred thousand cells. For the compressible solver at M ≤ 0.1, use the wider 1996 paper bounds.

### DrivAerML baseline — V-ref · Must; DrivAer experiments — Val · Useful
- **DrivAerML:** Ashton et al. (2024), arXiv 2408.11969, [huggingface.co/datasets/neashton/drivaerml](https://huggingface.co/datasets/neashton/drivaerml). **CC-BY-SA-4.0**, **31.2 TB** in total.
  - Content: 500 morphed notchback variants (484 public). Incompressible OpenFOAM, SA σ-DDES, about 160 M cells, U∞ 38.889 m/s, Re_L 7.19e6, A_ref 2.17 m².
  - Per run: STL, surface VTP (about 500 MB), volume VTU (about 50 GB), mean-force CSV, slices.
  - The baseline was compared with Ford/Pininfarina full-scale experiments.
- **TU München DrivAer geometry:** [epc.ed.tum.de](https://www.epc.ed.tum.de/en/aer/research-groups/automotive/drivaer/). Registration required. Heft et al., SAE 2012-01-0168; the TUM Cd and Cp data are in that paywalled paper (unverified).
- **Loughborough experiment:** [doi:10.17028/rd.lboro.12881213.v1](https://doi.org/10.17028/rd.lboro.12881213.v1). CC-BY-NC-4.0. 25 % scale, 40 m/s, Re_L 2.94e6. Cd: fastback 0.311, notchback 0.312, estate 0.334 (uncorrected). CSV forces, rear Cp, PIV wake. No uncertainty stated.
- **Acceptance:**
  - DrivAerML baseline (V-ref): same geometry, domain, boundary conditions and turbulence model class. Cd within ±3 %, surface Cp L2 difference ≤ 0.05, our values judged on their confidence intervals.
  - DrivAerML gives one mesh level only, so its own numerical uncertainty is unknown. The ±3 % band is a proposal to tighten once a second independent code result exists.
  - Against experiment (report): |ΔCd| ≤ 0.010, in the same facility and configuration. The Loughborough data are NC-licensed ([decision 3](#gaps-and-decisions-for-the-lead)).
- **Why:** `AGENTS.md` excludes golden files produced by our own code, not independent published results. The baseline is an independent numerical reference; its comparison with experiment does not prove each of its outputs numerically accurate.

---

## M9 — Multi-species mixing and conjugate heat transfer

The gate is "jet mixing, reference CHT case; .msh zones mapped automatically". These were the largest gaps in public data, so each entry says what is missing.

### Species verification — V-exact · Must
- **Species MMS:** extend the Euler/NS MMS with n species mass fractions and mixture properties, with non-trivial mixing everywhere, so that a scheme that only preserves a uniform composition cannot pass. Gate:
  - L2 order 2 ± 0.1 for each Y_k.
  - Each species' integrated mass changes exactly by its boundary fluxes and sources, to round-off; likewise mixture mass and energy. ΣY_k = 1 and positivity alone are not enough: clipping and renormalisation can satisfy both while creating or destroying individual species.
  - ΣY_k = 1 and Y_k ≥ 0 to round-off, without clipping.
- **1D Fickian diffusion:** Y = Y₀ erfc(x / 2√(Dt)). Gate: order 2 in space and the time-integrator order in time.
- **Stefan tube (steady evaporation):** N = (cD/L) ln[(1 − x_L)/(1 − x_0)] (Bird, Stewart & Lightfoot). Gate: flux within 0.5 % on the finest grid.
- **Two-gas shock tube** (He/air Sod with a jump in γ): exact Riemann solution with a contact between gases of different γ. ExactPack's `riemann` solver takes separate γ_l and γ_r. Gate:
  - Same as Sod.
  - No pressure or velocity oscillation at the contact beyond 1e-3 relative.
- **Why the two-gas shock tube:** spurious pressure oscillations at material interfaces are the classic failure of conservative multi-species schemes, and this is the cheapest test that exposes them.

### Sandia propane jet (TNF archive) — Val · Must
- **Source:** [tnfworkshop.org/data-archives/simplejet/propanejet](https://tnfworkshop.org/data-archives/simplejet/propanejet/). `SandiaPropaneJet.zip` (463 kB) plus a documentation PDF (377 kB).
- **Licence:** a US-government disclaimer and a request to contact the author (R. Schefer). No licence text: download by hash, cite, do not vendor.
- **Conditions:** D 5.26 mm, U_j 53 m/s, coflow 9.2 m/s, Re 68,000, propane into air (density ratio about 1.5).
- **Data:** mixture fraction f and its RMS, PDFs, U, V, u′, v′, u′v′.
- **Uncertainty:** **f ±2 %, f′ ±3 %, U ±1 %, V ±1.5 %, u′ ±2 %, v′ ±2.5 %, u′v′ ±5 %**.
- **Acceptance (report for RANS, gate for LES):** centreline decay of mean f and jet half-width within ±10 %. Report the turbulent Schmidt number used.
- **Why it is the reference:** it is the only open, machine-readable, non-reacting variable-density jet with stated uncertainties that we found.

### Cutler supersonic coaxial helium–air jet — Val · Must
- **Configuration:** centre jet of 95 % He + 5 % O₂ **by volume** (about 70.4 % He by mass), air coflow, both at Mach 1.8.
- **Sources:**
  - [NTRS 20040086859](https://ntrs.nasa.gov/citations/20040086859) (AIAA 99-3588) and [NTRS 20060020224](https://ntrs.nasa.gov/citations/20060020224) (NASA chapter), both free.
  - The *AIAA J* paper (doi:10.2514/1.5781) is paywalled.
- **Data:** pitot pressure, T0, sampled mole fraction, RELIEF velocity and u_rms. **No electronic data found online**: digitise from the NTRS reports, or ask the authors (NASA Langley).
- **Uncertainty:** probe pressure ±0.5 %, T ±2 K, mole fraction ±1–1.5 %, RELIEF velocity ±3 % (95 % bands).
- **Scored quantity:** the sampled composition measures the He–O₂ jet gas mixed with air, not helium alone. Score the jet-gas mole fraction as measured, converting our species fields to that basis.
- **Acceptance (report):** centreline jet-gas mole fraction within ±0.05 absolute; pitot within ±10 %; potential-core length within ±10 %. Report Sc_t and any compressibility correction.
- **Why Must:** it is the only helium/air case that also tests compressible mixing, which is our differentiator, even though it needs digitising.

### Other mixing data — Val · Useful
- **Brown & Roshko (1974):** He/N₂ mixing layer, [doi:10.1017/S002211207400190X](https://doi.org/10.1017/S002211207400190X). Spreading rate only, no tables. Spreading rate within ±15 %.
- **O'Hern et al. (2005), 1 m helium plume:** *JFM* 544, paywalled; ERCOFTAC KBwiki [UFR 1-07](https://kbwiki.ercoftac.org/w/index.php/UFR_1-07). Puffing, unsteady LES case; data files not found on the page.
- **Panchapakesan & Lumley (1993) helium jet:** *JFM* 246, [doi:10.1017/S0022112093000102](https://doi.org/10.1017/S0022112093000102), paywalled. Far-field decay constants within ±10 %.
- **Burrows–Kurkov Mach 2.44 H₂ injection:** NPARC `bk`. A reacting case, but its non-reacting setup could serve; data content unverified.

### Conjugate heat transfer verification — V-exact · Must
- **Conjugate MMS:** manufactured temperature continuous across the fluid/solid interface, with continuous heat flux k∂T/∂n and different conductivities on each side. Gate: L2 order 2 ± 0.1 in both zones; interface flux balance and global energy balance to round-off (a multiple of ε), in FP32 and FP64. This is the primary proof.
- **Composite wall (1D, steady):** the exact solution is piecewise linear. Meshes fit the interface (our zones are cell zones, so the interface is always a set of faces). Gate: round-off error on orthogonal meshes; order 2 ± 0.1 on non-orthogonal, interface-fitted polyhedral meshes. Interfaces that cut cells are out of scope.
- **Zone mapping:** a zone type in a Fluent `.msh` says "fluid" or "solid", not which gas or solid it is. Gate in two steps:
  1. The reader keeps every zone's identity and type, checked against the zone list in the file.
  2. The case assigns a material to each zone explicitly, and an active problem then runs on it: two different gases plus one solid, with heat crossing both interfaces, judged on the energy balance above.

### Conjugate heat transfer references — Useful
- **Luikov (1974):** conjugate flat plate, *IJHMT* 17:257–265, [doi:10.1016/0017-9310(74)90087-8](https://doi.org/10.1016/0017-9310(74)90087-8), paywalled. Integral-method correlation in the Brun number; trend within ±10 %.
- **Vynnycky, Kimura, Kanev & Pop (1998):** *IJHMT* 41:45–59, [doi:10.1016/S0017-9310(97)00113-0](https://doi.org/10.1016/S0017-9310(97)00113-0), paywalled; Navier–Stokes plus slab conduction. Code-to-code after digitising.
- **preCICE "flow over heated plate":** [precice.org/tutorials-flow-over-heated-plate.html](https://precice.org/tutorials-flow-over-heated-plate.html), LGPL-3.0. Inspired by Vynnycky; no reference solution. Compare with the OpenFOAM and SU2 tutorial results within ±1 %.

### NASA C3X turbine vane — Val · Must
- **Source:** Hylton et al., NASA CR-168015 (1983), [NTRS 19830020105](https://ntrs.nasa.gov/citations/19830020105). Public; 11.8 MB, 225 pages; data in Appendix A as scanned tables.
- **Hardware:** stainless-steel vane with 10 radial cooling holes and about 80 midspan thermocouples.
- **Conditions:** T_t ≈ 780–818 K, exit Mach 0.89–1.06, Re₂ 1.45–2.53e6, Tu 6.5 % or 8.3 %.
- **Benchmark runs:**
  - Run 112: exit M 0.90, Re₂ 2.01e6, Tu 8.3 %.
  - Run 158: exit M 0.91, Re₂ 1.47e6, Tu 8.3 %.
- **Data:** wall temperature, heat-transfer coefficient h (derived through an internal conduction model), surface pressure, per-hole coolant conditions.
- **Uncertainty:**
  - **Surface T ±1 °C**, gas T ±11 °C, vane conductivity ±3 %, pressure ±0.7 kPa.
  - **h ±6.2–7.3 %** over the first half of the arc, rising to **±23.5 %** at the trailing edge.
- **Transcription:** a community transcription exists in pull request #886 of the GitHub repository `thiemom/combaero` (record the commit hash when used); its accuracy and licence are unverified. Transcribe from the NASA PDF ourselves and use that transcription only as the independent cross-check.
- **Acceptance (report for RANS):**
  - Wall temperature within ±2 % (about ±15 K) over 0–60 % of the surface arc.
  - Pressure distribution within ±2 % of p_t.
  - Do not use h for acceptance: it is a derived quantity with a large and non-uniform uncertainty.
  - Suction side after transition: report only.
- **Why:** this is the only open, well-documented CHT experiment with metal temperatures and a published error budget that we found.

---

## Gaps and decisions for the lead

1. **Digitising.** These references have no machine-readable data and need transcription before #2 can parse them. The transcription itself needs an independent check: two people, or one transcription compared with a second source.
   - Settles CR-177638, Schülein (TM-2013-216604), Cutler, C3X Appendix A.
   - AR-138 RAE 2822 and ONERA M6 tables. NPARC and TMR files partly cover these.
   - Ghia, Botella & Peyret and Schäfer–Turek paper bounds.
2. **RAE 2822 corrected conditions.** Proposal: case 9 at M 0.734 with fixed CL 0.824 (the ADODG definition). This needs the lead's sign-off, since it changes the case hash.
3. **Restricted data.** These need the lead's decision before integration:
   - Non-commercial licences: ERCOFTAC Classic Collection (CC BY-NC-SA; Ahmed case 082, Le–Moin–Kim case 031) and Loughborough DrivAer (CC BY-NC). The options are to request permission or to replace them.
   - KBwiki (periodic hill, O'Hern plume): the terms reserve automated analysis and AI evaluation and require a licence from ERCOFTAC before bulk download.
   - No licence stated: TNF, Lee & Moser, Madrid, HiOCFD and Featflow files. These are downloaded by hash and never committed.
   - NASA-hosted material: rights evidence must be recorded per asset before anything is vendored.
4. **Still missing.**
   - No open hypersonic laminar SWBLI data tables (Holden double cone).
   - No open CHT case other than C3X.
   - No open variable-density helium jet with data files.
   - None of these blocks a milestone. Each is a candidate for a data request to the authors.
5. **Unverified values to confirm before freezing `acceptance.md`:**
   - Botella & Peyret digits.
   - Schäfer–Turek 1996 bounds.
   - Ahmed Cd.
   - The DPW core-solution band.
   - Cylinder Cd at Re 3900.
   - Le–Moin–Kim reattachment.
   - The DrivAerML ±3 % band, which has no published numerical uncertainty behind it.
