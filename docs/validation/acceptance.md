# Acceptance criteria

The numbers each milestone gate in [PLAN.md](../PLAN.md) must meet, where they come from, and the `autan-vv` tool that measures them. A threshold marked *ours* is a project decision; its rationale is in the row. Dataset details (files, conditions, licences, further cases) live in [validation-datasets.md](../research/validation-datasets.md); this page states the gate once.

Conventions for every gate:

- **Order** is the least-squares order of the L2 error over at least four levels and the order of the finest pair, both from `autan-vv order` ([order.md](order.md)). L∞ is reported, not gated: on non-nested unstructured families it fluctuates by ±1 (measured, [order.md](order.md#measured-orders)).
- **Code-to-code** gates compare our **continuum estimate** (Richardson from the three finest of at least four levels, with its GCI, `autan_vv.order.richardson`) with the published continuum estimate, never level by level.
- **Round-off** tolerances are multiples of the machine epsilon of the precision under test (AGENTS.md, Tests).
- **FP32** orders are taken on levels whose error is at least 100× the FP32 rounding floor ([kernel-generator.md](../research/kernel-generator.md) §8).
- Every reported run records mesh hash, case hash, commit and precision.

## M1 — generator spike

| Criterion | Threshold | Source |
|---|---|---|
| MMS `advection_diffusion`, second-order reconstruction | L2 order ≥ 1.9 (fit and finest pair) on tet, hex, poly (incl. `agglomerated`, 100+ faces) and hybrid families, clean and perturbed, in `fp64`, `fp32`, `fp32+acc64` | PLAN M1; kernel-generator.md §8. The reference solve reaches 1.95–2.55 on the same families ([order.md](order.md#measured-orders)), so the bar is attainable |
| Linear fields | LSQ and corrected Green-Gauss gradients exact to 64 ε; the reference solve reproduces a linear solution to ≤ 256 ε | Exactness of linear reconstruction; kernel-generator.md §4.3 |
| Conservation | Σ residual = Σ boundary fluxes to 64 ε × Σ|flux| | Discrete divergence theorem |
| Adjoint dot product | gap ≤ 1×10⁻¹³ (FP64), ≤ 1×10⁻⁵ (FP32, `fp32+acc64`) | kernel-generator.md §8 |
| Bandwidth | ≥ 70 % of measured GB10 peak on the gradient and flux loops | PLAN M1, AGENTS.md |

## M2 — compressible Euler

| Criterion | Threshold | Source |
|---|---|---|
| MMS `euler` | L2 order 2 ± 0.1 for each conserved variable, 4 cell families, Roe, HLLC, AUSM+-up, FP32 and FP64 | Roy et al. 2004 (design order); [datasets §M2](../research/validation-datasets.md#m2--compressible-euler) |
| Ringleb flow | L2 order 2 ± 0.1 in density and entropy | HiOCFD1 C1.2; datasets §M2 |
| Isentropic vortex, one period, domain ≥ [−10, 10]² | spatial order 2 ± 0.1; temporal order = RK order ± 0.1 | Spiegel, Huynh and DeBonis, AIAA 2015-2444 |
| NACA 0012 inviscid, M 0.8, α 0° | continuum CD within 1 count of 0.008342 | Vassberg and Jameson 2010, Tables 10–11 (via datasets §M2) |
| NACA 0012 inviscid, M 0.8, α 1.25° | CL inside 0.3516–0.3562 widened by our GCI; CD within 1 count of 0.02268 | idem, point-vortex farfield runs |
| NACA 0012 inviscid, M 0.5, α 1.25° | CD decreases monotonically; extrapolated \|CD\| < 0.5 count | idem (subcritical, CD → 0) |
| ONERA M6 inviscid (run 2308 conditions) | residual drop ≥ 10 orders; shock x/c within ±0.03 of experiment at η = 0.20–0.65 (report) | PLAN M2; Schmitt and Charpin 1979 (`onera-m6` case) |

## M3 — Navier-Stokes and Spalart-Allmaras

| Criterion | Threshold | Source |
|---|---|---|
| MMS `navier_stokes` | L2 order 2 ± 0.1, 4 families + perturbed, FP32 and FP64, non-orthogonal correction on | Roy et al. 2004 |
| MMS `rans_sa`, `rans_sa_noft2` | L2 order 2 ± 0.1 for the mean flow and ν̂ with second-order turbulence advection | Eça et al. 2007; Oliver et al. 2012; [mms.md](mms.md) |
| Compressible Couette | velocity exact to 64 ε; T and wall heat flux order 2 ± 0.1 | Exact solution (`test_mms.py` uses the same flow) |
| TMR flat plate (`tmr-2dzp-flatplate`) | continuum cf(x = 0.97) within 0.5 % of 2.7054×10⁻³ (hex) and 1 % (tet, poly); continuum CD within 0.5 % of 2.859×10⁻³ | CFL3D and FUN3D finest grid cf 2.70562 / 2.70540×10⁻³, CD 2.85985 / 2.85247×10⁻³; their Richardson estimates 2.7052–2.7060×10⁻³ and 2.8586–2.8592×10⁻³. *Ours*: 0.5 % is ten times the CFL3D–FUN3D cf spread (0.03 %) |
| TMR bump-in-channel (`tmr-2dbc-bump`) | continuum CD and cf(0.75) within 0.5 % of CFL3D/FUN3D | CD 3.5724 / 3.5611×10⁻³, cf(0.75) 6.1494 / 6.1514×10⁻³ (finest grid); *ours*: the CD spread between codes is 0.3 % |
| TMR NACA 0012, α 10°, family II (`tmr-2dn00-naca0012-numerics`) | \|ΔCL\| ≤ 0.002 from 1.0912–1.0913; \|ΔCD\| ≤ 1 count from 0.01221–0.01222 | Diskin et al. 2015 continuum range; *ours*: below the 12-code SA-neg spread (ΔCL ≈ 0.005, ΔCD ≈ 5 counts, datasets §M7) |
| NACA 0012 validation, α 0/10/15° (`tmr-2dn00-naca0012`) | report CL, CD, Cp, cf against CFL3D and Ladson, Gregory and O'Reilly; no gate on experiment | TMR 2DN00 page |
| Degraded meshes | residual drop ≥ 8 orders on the perturbed families (`--perturbation 0.3`) | PLAN M3 ("convergence on degraded meshes"); *ours*: 8 orders leaves forces converged to < 0.01 % |

## M5 — multi-GPU and precision

| Criterion | Threshold | Source |
|---|---|---|
| Precision invariance | CL and CD in `fp32+acc64` within 0.1 % of `fp64` on the M3 TMR cases and ONERA M6 | PLAN M5 |
| Scaling | parallel efficiency ≥ 80 % at 8 GPUs, measured to 64 | PLAN M5 |

## M6 — discrete adjoint

| Criterion | Threshold | Source |
|---|---|---|
| Dot product, every kernel and the composed residual | as M1 | kernel-generator.md §8 |
| Gradient vs central finite differences (FP64) | relative difference ≤ 10⁻⁶ at the step minimising truncation plus round-off | *ours*: FD truncation O(h²) with h ≈ ε^(1/3) gives ≈ 10⁻¹⁰–10⁻⁶ |
| RAE 2822 case 9 and ONERA M6 drag minimisation | report baseline against experiment: RAE 2822 case 9 CN 0.803, CD 0.0168 (M 0.73, α 2.79° corrected, Re 6.5×10⁶) | Cook, McDonald and Firmin 1979, decoded from the 1981 tape (`rae2822` case) |

## M7 — SST, DDES, WMLES

| Criterion | Threshold | Source |
|---|---|---|
| Driver–Seegmiller backward-facing step | x_reatt/H within ±0.3 of 6.26 | Driver and Seegmiller 1985 (6.26 ± 0.10), via TMR; datasets §M3 |
| CRM-HL near CLmax | inside the HLPW-5 SA-neg range ΔCL ≈ 0.005, ΔCD ≈ 5 counts, ΔCM ≈ 0.003 | HLPW-5 summary (NTRS 20240014255); datasets §M7 |

Milestones M4 and M8–M11 gate on functionality or on cases still to be added; their numbers will be set here when those cases enter the library.
