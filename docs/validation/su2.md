# Comparison with SU2

A direct comparison of Autan with [SU2](https://su2code.github.io/) on the same meshes: iterations and wall time to one convergence criterion, and accuracy against published references (issue #195). Source: `vv/src/autan_vv/su2/`, `vv/su2/`.

**Status.** Phase 1: SU2 built, protocol fixed, NACA 0012 run by SU2 alone and checked against Vassberg and Jameson (2010) ([results](#phase-1-su2-on-the-naca-0012-cases)). Phase 2: forces and wall Cp of Autan against SU2 on the same meshes ([results](#phase-2-autan-against-su2-forces-and-wall-cp)). TMR flat plate: SU2 alone, against CFL3D and FUN3D ([results](#tmr-flat-plate-su2-alone)). Times are not compared until the measured protocol of #387 runs.

## SU2, an external tool

SU2 is LGPL-2.1. It is built and run apart, never vendored in this repository and never linked with Autan; the repository holds only the image recipe, our scripts, configurations and results.

| Item | Value |
|---|---|
| Version | v8.6.0, commit `7048fd35f56d4bf52e586429d6746a36a10255f8` (checked at build) |
| Image | `vv/su2/Dockerfile`: Ubuntu 24.04 (pinned digest), GCC 13.3; `meson --buildtype=release -Dwith-mpi=disabled -Dwith-omp=true`, no AD, no Python wrapper, no CGNS or TecIO |
| No MPI | SU2's MPI build compiles and links its bundled ParMETIS, whose licence is non-commercial (AGENTS.md); SU2 runs as one process with OpenMP threads (`SU2_CFD -t N`) |
| Build | `docker build -t autan-su2:v8.6.0-omp vv/su2` on the DGX Spark (aarch64); every run records the image id |
| Run | `vv/su2/campaign.sh THREADS IMAGE_ID CASE...` inside the image (the spark-5359 campaign lane, `autan-campaign start NAME --cpus 4 --dir ROOT --image autan-su2:v8.6.0-omp`), or `vv/su2/run.sh ROOT CASE THREADS` on a host with Docker; both: `SU2_CFD -t THREADS`, `nice -n 10`, the stop rules below, write `log.txt` and `provenance.json` (`autan.su2-run/1`: image id, SU2 version, threads, SHA-256 of the configuration and mesh it ran, CPU model, wall time, exit status) |

## Fairness protocol

Fixed on 2026-11-01, before any measurement, and revised by the [fairness audit](#fairness-audit) the same day, still before any Autan measurement. Each code solves the same problem on the same mesh with its developers' own published settings. SU2 runs twice: **SU2-JST**, its reference scheme for this case (`inv_NACA0012.cfg` and the QuickStart of v8.6.0), which is the SU2 that Autan's accuracy is compared with; and **SU2-Roe**, the closest scheme to Autan's (`inv_NACA0012_Roe.cfg`, limiter threshold matched), reported alongside. The rows below describe SU2-Roe where the schemes differ.

| Item | Autan | SU2 | Difference that remains |
|---|---|---|---|
| Mesh | the `naca0012` MSH file ([meshes.md](meshes.md#naca-0012-o-meshes)) | the same file converted (below); 2D: the slab's back plane | none: `plane_sha256` proves the same grid bit for bit |
| Control volumes | cells: NC² | median-dual of the nodes: NC(NC + 1) | cell-centred against vertex-centred, inherent to SU2 |
| Gas, freestream | the case: γ, R = R_u/M, p∞, T∞, M, α | read from the same case object | none |
| Flux | Roe, Harten's fix (width 0.1 c) on the acoustic waves only | SU2-Roe: `ROE`, entropy fix \|λ\| ≥ 0.001 λ_max on every wave (`roe.cpp`). SU2-JST: central flux with JST scalar dissipation (sensor coefficients 0.5, 0.02), no reconstruction or limiter | the entropy fixes differ in form; JST is another scheme altogether |
| Reconstruction | primitive (ρ, u, v, w, p), least squares 1/d², face2 stencil | MUSCL κ = 0 on (T, v, p, ρ), `WEIGHTED_LEAST_SQUARES` over edge neighbours | variables and stencils differ |
| Limiter | Venkatakrishnan, ε_k² = (K/L)³ V q_k², K = 10 per cell | `VENKATAKRISHNAN`, one ε² = (K_SU2 · 1)³ for the whole mesh on variables scaled by the freestream (`FREESTREAM_PRESS_EQ_ONE`) | SU2's threshold cannot vary with the cell; it is matched to Autan's on the median cell within one chord of mid-chord: K_SU2 = K V_med^(1/3)/L. Velocity is scaled by c∞ in Autan and c∞/√γ in SU2 |
| Wall, far field | slip; characteristic far field 148 c away, uniform freestream at α = 0°, with the point vortex of the lift at c/4 on the lifting conditions ([cases.md](cases.md#naca-0012-inviscid)) | `MARKER_EULER`; `MARKER_FAR`, uniform freestream: SU2 v8.6.0 has no point-vortex correction | on the lifting conditions only; phase 2 also runs Autan's lifting cases without the vortex, so that the two codes solve one problem, while the gates keep Autan's vortex case |
| Forces | chord 1, area chord × span, moment about (c/4, 0) nose-up | per unit span; nose-up moment = −CMz in 2D, CMy in 3D | none |
| Convergence | every residual 10 orders below its first value ([case-schema.md](../contracts/case-schema.md#time)) | the same rule, applied by `autan_vv.su2.history` to SU2's history | none: both residuals are the RMS of the integrated flux balances over the control volumes |
| Solver controls | the defaults of the measured commit | SU2-Roe: `inv_NACA0012_Roe.cfg`: implicit Euler, CFL 30, BiCGSTAB (3 iterations, 10⁻¹) with ILU(0), 3-level W-cycle multigrid. SU2-JST: `inv_NACA0012.cfg`: implicit Euler, CFL 1000, FGMRES (5 iterations, 10⁻³) with LU-SGS, 3-level full multigrid | each code as its developers publish it; identical for every level |
| Precision | `fp64`; `fp32+acc64` | FP64; the mixed-precision build (`-Denable-mixedprec=true`: linear solver and preconditioner in FP32) | timed pairwise, `fp64` against FP64 to the criterion; `fp32+acc64` against mixed precision to one endpoint for both, the first iteration from which CL and CD stay within 0.01 % of that code's own converged `fp64` values (a ten-order drop is out of reach in FP32 storage, [case-schema.md](../contracts/case-schema.md#time)) |

SU2 measures its relative residual from the largest value so far, not the first, and its FP64 residuals stall 10.5–11 orders below their largest value at M 0.5 (64²–256²), where a rise of up to 0.7 orders in the first iterations (32²) leaves no single margin that works on every level. So `run.sh` applies the protocol's criterion to `history.csv` while SU2 runs and stops it with SIGTERM, which SU2 handles by finishing the iteration and writing its output; it also stops a run whose residuals have stalled (the drop of the worst equation has not grown by 0.1 orders in 2 000 iterations; with `STALL_SKIP_ZERO=1`, the worst of the equations that did not start identically zero, so that one woken at the round-off floor cannot stop the run), reported not converged with the drop it reached. SU2's own rule is set two orders beyond as a backstop, and the 20 000-iteration cap ends any other run. The criterion's iteration is then read from the history.

**Timing (phase 2).** On spark-5359 with the host quiet (load < 2, GPU idle), each measurement under `autan-gpu-lock -m "CFD - Comparatif SU2: <run>" -t 600` (never a raw `flock`; ≤ 10 min, 2 min between takes), so that no GPU job runs during an SU2 or Autan CPU timing, alternating Autan and SU2, three repetitions, median reported. The same cores: SU2 with n OpenMP threads against Autan's CPU backend with n threads, n = 1 and n = the count Autan's CPU backend uses, pinned with `taskset`; Autan's GPU backend against SU2 at its fastest n. Iterations are counted up to and including the one whose residual meets the criterion, so a run given that budget ends on it. Time to solution is the wall time of a run given that budget (found by an untimed run), from process start to exit, mesh reading included, writing only the final solution. Reported per level: iterations, wall time, time per iteration per control volume.

**Published GPU reference (phase 2).** SU2 publishes no GPU timing (only a 2024 GSoC prototype), so Autan's GPU backend is also measured against FUN3D on the public DPW4 CRM grid (3 675 916 vertices, 9.79 M mixed elements, SA-QCR2000): 7 min to machine zero on one A100 (Nastac et al. 2023). Autan runs the same grid, with the same mixed elements and turbulence model, to the same residual drop and the same CD as that paper's Table 3, and reports wall time, iterations and throughput normalised by bandwidth: bytes of peak bandwidth per control volume and iteration. FUN3D counts vertices and Autan cells, which on this grid differ in number and in neighbours per control volume; the comparison is by time to the same solution, the normalised throughput is reported alongside.

**Accuracy.** Against SU2-JST, SU2-Roe reported alongside: CL, CD, CM per level at the criterion, and their continuum estimates from the three finest of at least four converged levels with `autan_vv.cases.naca0012.continuum` (the paper's eqs. 20–24), against the gates of [acceptance.md](acceptance.md#m2--compressible-euler).

## Fairness audit

Done on 2026-11-01 because SU2-Roe, as first configured, left a spurious drag of 8–25 counts in the shock-free M 0.5 flow, falling at order 0.5 instead of 2: more likely a fault of our setup than of SU2, and to be settled before any comparison. Runs on the shared Spark, SU2 v8.6.0 image `sha256:5eecb8fd…`; forces only.

1. **Build.** SU2's own regression cases (`serial_regression.py`: `naca0012` and `naca0012_FMG`, iteration 20 of `inv_NACA0012_Roe.cfg` and `inv_NACA0012.cfg` on `mesh_NACA0012_inv.su2`, one thread) give its reference values to every printed digit: rms[ρ], rms[ρE], CL, CD = −4.031446, −3.524530, 0.273056, 0.009703 and −4.160287, −3.553906, 0.296154, 0.016890. The image builds SU2 as SU2 tests it.
2. **Published results.** SU2 publishes no grid-convergence study of this case that we found (tutorials and test cases, not Vassberg–Jameson); its reference configurations are the published practice, and they are what the protocol now runs.
3. **Mesh.** Native 2D SU2 files (not the slab), points bit for bit those of the generator, far field 148 c, markers `airfoil` (slip) and `farfield`. SU2 reported reorienting the 256 edges of one marker at 256²: the converter now writes every boundary edge with the domain on its left, and SU2 reorients nothing. The conversion is checked directly (an independent parser compares the converted file with the generator's slab, [Tests](#tests)), and with SU2-JST these meshes give the paper's values (next item), which excludes errors in geometry, markers, conditions or force reference that would affect every scheme.
4. **Scheme.** Our first SU2-Roe configuration differed from `inv_NACA0012_Roe.cfg` only in the limiter coefficient (and the gas constant, 287.055 against SU2's default 287.058 J/(kg K)). With SU2 v8.6.0 and its published `inv_NACA0012_Roe.cfg` (SHA-256 `2592a85c04e0039f95b9c282401281efb82b2ca8a8bd5cca8a7f86b35d50a882`), run as published on these meshes apart from the mesh, the flight condition and the outputs, the residuals stall below one order at M 0.8, α 0° (128²–512²) and the drag at M 0.5 is −14, +14 and +11 counts (128²–512²). On 256² and 512² at M 0.5, every MUSCL upwind variant tried leaves 10–25 counts: HLLC, Green–Gauss gradients, unweighted least squares, the matched threshold, the limiter off, the threshold ÷ 4. With the same SU2 and its published `inv_NACA0012.cfg` (JST, SHA-256 `ef5ae43fb9249202d09c8c07057bf9b3a3f87978dcd1190ea081d00ba90f8943`), the drag at M 0.5 decreases in magnitude at every level, from −16.7 to −0.11 counts (128²–1024²), at order 2, and the continuum CD at M 0.8, α 0° is 0.008340 against the paper's 0.008342.
5. **Consequence.** These findings concern SU2 v8.6.0, the published configurations and the variants listed above, on the sharp-trailing-edge O-meshes of Vassberg and Jameson; they say nothing of SU2's upwind schemes elsewhere, nor of variants not tried. At M 0.8 the published Roe configuration stalls, which is an iterative failure, not a measured spatial order; at M 0.5 the drag of every upwind variant falls by a factor of 1.3–1.7 from 256² to 512² (an order of 0.4–0.75 if its limit is zero), where SU2-JST's falls by a factor of 4.7. So the protocol compares Autan's accuracy with SU2-JST and reports SU2-Roe alongside as the closest scheme.

Reproduce (SU2 configurations from the v8.6.0 tag, `TestCases/euler/naca0012/`; one run at a time, 4 threads):

```sh
autan-vv su2 prepare --family family --published inv_NACA0012_Roe.cfg --out su2-published-roe
autan-vv su2 prepare --family family --published inv_NACA0012.cfg --out su2-published-jst
vv/su2/run.sh su2-published-roe m050-a125/n0256 4        # each case in turn
autan-vv su2 history su2-published-roe/m050-a125/n*      # iterations and forces at the criterion
```

`--published` keeps every option of the file except those in `autan_vv.su2.config.ADOPTED_KEYS` (mesh, Mach number, incidence, freestream, iteration cap, stop rule, outputs), refuses a file whose markers or force reference differ from the case's, and records the scheme as `published:<SHA-256 of the file>`. The variants are the published Roe file with one option changed (`CONV_NUM_METHOD_FLOW= HLLC`, `NUM_METHOD_GRAD= GREEN_GAUSS`, `NUM_METHOD_GRAD_RECON= LEAST_SQUARES`).

## Mesh conversion

`autan_vv.su2.mesh.convert` reads the MSH 4.1 file (`autan_vv.su2.msh`, the format of the Gmsh manual; anything else is `V0901`), checks that it is one layer of hexahedra between two planes y = const, each a straight extrusion whatever its node order (`V0902` otherwise), and writes SU2's native format with every coordinate in 17 significant digits:

- 2D: the back plane, (x, z) → (x, y), quadrangles counterclockwise; the lateral zones become edge markers, the planar zones (`back`, `front`, Autan's symmetry planes) are dropped.
- 3D: the slab itself, every zone a marker, symmetry on `back` and `front`. Used as a check: on 32² at M 0.8, α 1.25°, it gives the 2D run's CL and CD to ten digits.

The record `<mesh>.json` (`autan.su2-mesh/1`) holds the SHA-256 of source and output, counts per marker, and `plane_sha256`, a hash of the plane's quadrangles and marker edges as coordinates, independent of numbering; it is the same from the 2D file, the 3D file and the source.

## Running

```sh
autan-vv meshes naca0012 --levels 32,64,128,256,512,1024 --out family   # macOS or x86-64 (Gmsh)
autan-vv su2 prepare --family family --scheme jst --out su2-jst            # meshes, case.cfg, level.json
vv/su2/run.sh su2-jst m080-a125/n0256 4                                       # on the Spark, 4 threads
autan-vv su2 study su2-jst/m080-a125/n* --out m080-a125.json               # criterion, continuum, gates
```

`prepare` writes `OUT/meshes/` and one directory per condition and level with `case.cfg` and `level.json` (`autan.su2-level/1`: scheme, K_SU2, configuration hash, Autan case hash, source mesh hash, conversion record). `study` refuses a run whose configuration or mesh, as hashed when it ran and as on disk, is not the prepared one, or whose SU2 failed (`V0904`), and runs that mix conditions, schemes or SU2 images (`V0905`).

SU2-Roe's K_SU2 per level: 2.29 (32²), 1.44, 0.90, 0.57, 0.36, 0.23 (1024²). In freestream units the threshold ε² = K_SU2³ is 12 at 32², 0.18 at 256² and 0.012 at 1024²; the squared density jump across the M 0.8 shock is about 0.2, so in both codes the limiter barely acts below 256². Autan's threshold uses h³ = V, and on a one-layer slab V includes the span (1 chord here): the same grid with another span limits differently.

## Phase 1: SU2 on the NACA 0012 cases

SU2 alone, on the six levels 32²–1024² of [the O-mesh family](meshes.md#naca-0012-o-meshes), each case prepared and studied as above (4 OpenMP threads on the shared DGX Spark; wall times are not measurements and are not reported). Reports: `vv/su2/results/naca0012/` (`autan.su2-study/1` per scheme and condition). CL, CD and CM (nose-up, about c/4) at the iteration that meets the criterion; *it.* is that iteration's count. Continuum estimates come from the three finest levels.

### SU2-JST, the accuracy reference

| Level | M 0.8, α 0°: it. | CD | M 0.8, α 1.25°: it. | CL | CD | CM | M 0.5, α 1.25°: it. | CL | CD |
|---|---|---|---|---|---|---|---|---|---|
| 32² | 560 | 0.002683 | 557 | 0.296699 | 0.014153 | −0.024564 | 273 | 0.155056 | −0.013512 |
| 64² | 367 | 0.006085 | 389 | 0.337825 | 0.019849 | −0.034309 | 170 | 0.172758 | −0.005328 |
| 128² | 328 | 0.007504 | 286 | 0.349362 | 0.021546 | −0.037310 | 209 | 0.177844 | −0.001673 |
| 256² | 409 | 0.008124 | 443 | 0.351659 | 0.022237 | −0.037936 | 268 | 0.179315 | −0.000411 |
| 512² | 632 | 0.008292 | 670 | 0.352357 | 0.022431 | −0.038119 | 373 | 0.179668 | −0.000087 |
| 1024² | 1 284 | 0.008329 | 1 402 | 0.352232 | 0.022459 | −0.038070 | 487 | 0.179750 | −0.000011 |
| Continuum (order) | | 0.008340 (2.15) | | — | 0.022464 (2.77) | — | | 0.179775 (2.11) | +0.000012 (2.10) |
| Vassberg–Jameson | | 0.008342 | | 0.3516–0.3562 | 0.02268 | | | 0.180346 (FLO82) | → 0 |

- **M 0.8, α 0°:** CD converges at order 2.15 to 0.008340, 0.02 count from the paper: gate passed.
- **M 0.5, α 1.25°:** the drag of this shock-free flow decreases in magnitude at every level, at order 2.1, to +0.12 count: gate passed. CL converges at order 2.1 to 0.17977, 0.3 % below FLO82's continuum.
- **M 0.8, α 1.25°:** CD converges at order 2.8 to 0.022464, 2.2 counts below the paper: the one-count gate fails. CL lies in the paper's band from 256² but changes direction between 512² and 1024², so it has no Richardson estimate (`V0107`) and its gate cannot be evaluated. SU2's far field is the uniform freestream, where the paper's lifting values come from point-vortex far-field runs; the missing vortex lowers the circulation the boundary sees and can plausibly account for part or all of the gap, but no run here varies the far field to show it. Autan's lifting cases carry the vortex, so this gate is not a like-for-like check of SU2.

**What this says about the cases.** SU2's reference scheme passes, at second order, the two gates that the far-field circulation hardly affects (the drag at zero lift, and the vanishing drag of the subcritical flow): the meshes, the conditions, the slip wall and the force reference are consistent with the paper. The lifting gate is not tested like for like, since SU2 has no point-vortex far field.

### SU2-Roe, the closest scheme

Roe with the matched limiter threshold, the controls of `inv_NACA0012_Roe.cfg` (`su2-roe/`); the [audit](#fairness-audit) gives the reasons for the differences from SU2-JST.

| Level | M 0.8, α 0°: it. | CD | M 0.8, α 1.25°: it. | CL | CD | CM | M 0.5, α 1.25°: it. | CL | CD |
|---|---|---|---|---|---|---|---|---|---|
| 32² | 459 | −0.004477 | 455 | 0.249619 | 0.005854 | −0.014450 | 180 | 0.115582 | −0.020010 |
| 64² | 451 | 0.008512 | 474 | 0.316844 | 0.021396 | −0.029516 | 195 | 0.152902 | −0.001639 |
| 128² | 662 | 0.011083 | 683 | 0.344790 | 0.025040 | −0.036798 | 316 | 0.169238 | 0.002933 |
| 256² | 916 | 0.010440 | 1 381 | 0.352372 | 0.024654 | −0.038565 | 515 | 0.175435 | 0.002505 |
| 512² | 1 972 | 0.009574 | 2 072 | 0.352651 | 0.023760 | −0.038453 | 1 326 | 0.177912 | 0.001517 |
| 1024² | stalls at 7.4 orders | 0.009005 | 8 235 | 0.353147 | 0.023198 | −0.038467 | 3 316 | 0.178959 | 0.000831 |
| Continuum (order) | | — | | — | 0.022246 (0.67) | — | | 0.179728 (1.24) | −0.000732 (0.53) |

No gate passes: at M 0.8, α 0° the 1024² run stalls and the three finest converged levels give no estimate (`V0108`); at M 0.5 the drag falls at order 0.53 and is still 8.3 counts at 1024². With SU2's published `inv_NACA0012_Roe.cfg` run as it is (`su2-published-roe/`, scheme `published:2592a85c…`), M 0.8, α 0° stalls below one order at 128²–512² and M 0.5 gives −14.0, +13.7 and +10.5 counts; the other upwind variants of the audit are in `upwind-variants.json` and the limiter study in `limiter-sensitivity.json`. A rerun of the SU2-Roe campaign from scratch gave the same forces and iteration counts at every level.

## Phase 2: Autan against SU2, forces and wall Cp

No timing yet (issue #387). Autan at main `880dcce`, `fp64`, CPU: the verdict runs of the NACA 0012 gates (point vortex on the lifting conditions) and the same lifting cases without the vortex, which SU2 v8.6.0 does not have, so that both codes solve one problem; levels 128²–1024² of the same mesh files (identical SHA-256). The 1024² runs without the vortex were stopped once for host memory and continued (`autan run --start`), judged against their original run's first residuals. Both sides at the iteration that meets the protocol's criterion; continua from the three finest levels both codes converged, listed in each report (`continuum.levels`); each report also names the SHA-256 of the SU2 study and of every Autan `run.json` and `history.jsonl` it read. `autan-vv su2 compare`; reports: `vv/su2/results/naca0012/autan-880dcce/`.

| Condition (far field) | Code | 1024²: CL | CD | CM | Continuum CL | CD | CM |
|---|---|---|---|---|---|---|---|
| M 0.8, α 0° (uniform in both) | Autan | 0 | 0.0083503 | 0 | 0 | 0.0083434 | — |
| | SU2-JST | 0 | 0.0083293 | 0 | 0 | 0.0083402 | — |
| M 0.5, α 1.25° (uniform in both) | Autan | 0.179681 | 0.0000189 | −0.0022418 | 0.179770 | +0.13 count | −0.0022623 |
| | SU2-JST | 0.179750 | −0.0000110 | −0.0022617 | 0.179775 | +0.12 count | −0.0022618 |
| M 0.8, α 1.25° (uniform in both) | Autan | 0.354966 | 0.0226227 | −0.0388761 | — | 0.0225868 | — |
| | SU2-JST | 0.352232 | 0.0224593 | −0.0380703 | — | 0.0224642 | — |

- **Forces.** On one problem the two codes meet in the continuum: at M 0.5 to 5 × 10⁻⁶ in CL, 0.01 count in CD and 5 × 10⁻⁷ in CM; at M 0.8, α 0° to 0.03 count in CD. At M 0.8, α 1.25° their CDs are 1.2 counts apart in the continuum and their CLs 0.0027 apart at 1024²; neither CL converges monotonically on these levels (`V0107`, `V0108`), so the CL gap has no continuum value. Level by level the drag differences shrink by about four per refinement where both converge (ΔCD at M 0.5: +27.2, +6.1, +1.3, +0.3 counts from 128² to 1024²); those of CL and CM at M 0.5 shrink by two to three (orders 1.1–1.5).
- **Far field.** Without the vortex, Autan's continuum CD at M 0.8, α 1.25° falls by 1.7 counts (0.0227597 with it, the gate's value, against 0.0225868) and its CL at 1024² by 0.0028; at M 0.5 its continuum CL falls by 0.00057, to SU2-JST's. Autan's response to the vortex at M 0.8, α 1.25°, 1.7 counts, is of the size of SU2's 2.2-count gap to the paper there, which is consistent with the far-field explanation of phase 1; SU2's own response is not measured (it has no vortex), and without the vortex the two codes still differ by 1.2 counts.
- **Wall Cp.** Face by face (Autan's face value against the mean of SU2's at the face's nodes), over the body (95 % of the chord, away from the shocks), the largest difference at 1024² is 0.0039 at M 0.5, 0.025 at M 0.8, α 0° and 0.042 at M 0.8, α 1.25° (RMS 0.0006, 0.0013, 0.0025); at M 0.5 it falls by three to four per refinement (0.135, 0.048, 0.014, 0.004 from 128²). The shocks, located at the steepest rise of Cp, lie on the same face or one face apart in both codes: on the upper surface at x = 0.504 (α 0°, from 256²) and 0.646 then 0.640 (α 1.25°, 256² then 512²–1024²), on the lower surface at 0.337–0.340 (α 1.25°, 512²–1024²). Over all faces, wall Cp differs by 0.21–0.51 at every level; at M 0.5, 1024², the eight largest differences are the last faces before the sharp trailing edge, a stagnation point where SU2 has a node and Autan a face beside it.
- **Against SU2-Roe** the differences are larger and shrink at low order, as SU2-Roe's own convergence does (phase 1): ΔCD at M 0.5 is −18.9, −23.0, −14.7, −8.1 counts from 128² to 1024².

## TMR flat plate: SU2 alone

The 2D zero-pressure-gradient flat plate of the NASA Turbulence Modeling Resource (`tmr-2dzp-flatplate`, M 0.2, Re 5 × 10⁶ per unit length, SA), run by SU2 before Autan has SA (#152), as a check of the case and of SU2 on it; pre-registered on #195 before any run. The five TMR grids, 35×25 to 545×385, come from the case library with their SHA-256 and are converted to SU2's 2D format with TMR's boundaries (`autan_vv.su2.flatplate`): `inlet` (i = 1), `outlet` (i = max), `farfield` (j = max), `symmetry` (j = 1, x < 0) and `wall` (j = 1, 0 ≤ x ≤ 2). The converted 137×97 is SU2's own `mesh_flatplate_turb_137x97.su2` (TestCases `v8.6.0`): the same points to 2.2 × 10⁻¹⁶, the same marker edges, two markers written in the other direction. SU2 v8.6.0 (image checked against its regression values, including `turb_flatplate`) runs its published `TestCases/rans/flatplate/turb_SA_flatplate.cfg` (SHA-256 `8d5f6f7c…2e917`) with only the mesh, iteration cap, convergence fields and outputs changed (`flatplate.ADOPTED_KEYS`). That file is SA-noft2 in v8.6.0; the same with `SA_OPTIONS= WITHFT2` is TMR's standard SA. Every run met the protocol's criterion (all residuals, ν̃ included, 10 orders below their first value); cf at x = 0.970084071 from SU2's surface file and CD (reference area 2) from its history, both at the run's last iteration; continua from the three finest grids with h = N^(−1/2), N the number of cells as in TMR's files, and none if one of those three did not converge; reports: `vv/su2/results/flatplate/`.

| Grid | cf × 10³ | CD × 10³ | Δcf to CFL3D, FUN3D | ΔCD to CFL3D, FUN3D |
|---|---|---|---|---|
| 35×25 | 2.70057 | 2.80529 | −3.40 %, +1.24 % | −5.05 %, +1.13 % |
| 69×49 | 2.71496 | 2.83605 | −0.49 %, +0.74 % | −1.68 %, +0.47 % |
| 137×97 | 2.71909 | 2.85023 | +0.29 %, +0.63 % | −0.56 %, +0.36 % |
| 273×193 | 2.72074 | 2.85851 | +0.52 %, +0.60 % | −0.10 %, +0.37 % |
| 545×385 | 2.72155 | 2.86392 | +0.59 %, +0.60 % | +0.14 %, +0.40 % |
| Continuum | 2.72234 (p = 1.03) | 2.87409 (p = 0.62) | | |

- **Gate.** The M3 gate of [acceptance.md](acceptance.md#m3--navier-stokes-and-spalart-allmaras), applied to SU2: continuum cf within 0.5 % of 2.7056 × 10⁻³ and CD within 0.5 % of 2.8589 × 10⁻³. **Fail** with either model: cf +0.62 %, CD +0.53 %.
- **References.** Recomputed from TMR's files with `autan_vv.order.richardson`, the continua are those acceptance.md quotes: cf 2.70524 / 2.70600 × 10⁻³ and CD 2.85924 / 2.85861 × 10⁻³ (CFL3D / FUN3D). On the finest grid the two codes agree to 0.01 % in cf; SU2 is 0.6 % above both.
- **Order.** SU2's cf and CD rise monotonically with refinement, at observed orders 1.0 and 0.6, against 2.0 and 1.75 (CFL3D) and 1.3 and 0.8 (FUN3D), towards a limit of its own 0.6 % above the references: refinement does not close the gap.
- **Model.** The table is SA-noft2, SU2's published setting. Standard SA (`WITHFT2`, the log names the model) gives the same cf and CD to 10⁻⁵ at every grid and the same continua: the ft2 term is not the difference.
- **Reference pressure.** SU2 takes its freestream from the Reynolds number: p∞ = 114 455 Pa, which sets the dynamic pressure q∞ = ½ γ p∞ M² that wall shear and drag are divided by (drag also by the reference area). The published file writes its boundary conditions for 115 056 Pa (outlet and far field 115 056, inlet pt = 1.02828 × 115 056), and the wall pressure near x = 0.97 is 115 057: the flow's dynamic pressure is 0.525 % above the reference one and its Reynolds number 0.525 % above 5 × 10⁶. TMR states the boundary conditions relative to the reference pressure. Run with the boundary pressures set for SU2's p∞ (outlet and far field 114 455 Pa, inlet pt 117 691.787 Pa, nothing else changed; hypothesis, prediction and test declared on #195 before the run), cf and CD fall by 0.443–0.448 % on every grid, the −0.44 % predicted; on 545×385 cf is then +0.14 % / +0.15 % from CFL3D / FUN3D and CD −0.31 % / −0.05 %. Report: `h1-reference-pressure.json`.
- **Criterion on that run.** Its first residuals are 3.3 orders lower than the published runs' (the initial freestream now matches the boundaries) and its residuals end no higher than theirs (35×25: rms[ρ] reaches −13.9 against −13.8), so no grid reaches 10 orders below its first value: the stall rule stops them with their worst equation 8.3–9.1 orders below its first value, and under the registered rule there is no estimate. The amendment declared on #195 before any of its forces was read is an absolute endpoint, not an equivalent of the ten-order drop: each residual at or below the published run's on the same grid when that run met the criterion. Under it every grid converges, in 645 to 3 356 iterations; the continuum is cf 2.71017 × 10⁻³ (+0.17 %) and CD 2.86127 × 10⁻³ (+0.08 %), inside the gate. The observed orders do not change (1.03, 0.62).
- **Other hypotheses** (#195, each on top of the corrected pressures, judged the same way, every run converged by the amended criterion). A Riemann far field on top, TMR's boundary, instead of the pressure outlet lowers cf by 0.19 % and CD by 0.17 % on every grid: on 545×385 cf is then −0.05 % / −0.04 % from CFL3D / FUN3D, and the continuum is cf 2.70507 × 10⁻³ (−0.02 %) and CD 2.85632 × 10⁻³ (−0.09 %), as predicted before the run (`h3-far-field-rerun.json`; its first attempt, `h3-far-field.json`, was cut at 2 000 iterations by the stall rule, its ρv residual starting identically zero and then rising, hence `STALL_SKIP_ZERO`). Second-order SA convection raises the observed orders (cf 1.03 → 1.39, CD 0.62 → 0.67) and moves the finest grid by less than 0.01 % (`h5-sa-convection.json`); weighted least-squares gradients change the coarse grids but not the finest, and their CD has no power law (`V0108`; `h4-gradients.json`). So the published configuration's +0.6 % is its boundary conditions: with TMR's (pressures relative to the reference pressure, Riemann far field), SU2 matches CFL3D and FUN3D within 0.05 % on the finest grid.

```sh
autan-vv su2-flatplate prepare plate --published turb_SA_flatplate.cfg --su2-mesh mesh_flatplate_turb_137x97.su2
CPUSET=10-17 NICE=19 MEMORY=2g vv/su2/run.sh plate sa-noft2/545x385 8   # each case in turn
autan-vv su2-flatplate results plate --out report.json                    # criterion, cf, CD, continua, gate (exit 1 on a fail)
autan-vv su2-flatplate results variant --models sa-noft2 --reference plate   # a variant, both criteria
```

## Tests

`vv/tests/test_su2.py`: the converted file parsed independently and compared with the slab the generator writes (points bit for bit, counterclockwise cells, area equal to the annulus between the far-field and airfoil polygons, airfoil nodes on eq. 1 of the paper, hashes from both files); refusals `V0901`, `V0902`, `V0903`; the configuration against the case (R from CODATA and the molar mass of air); K_SU2 on a Cartesian slab against (a² s)^(1/3) K; the criterion on geometric residual sequences; SU2 v8.6.0's history columns and axes from real 2D and 3D runs (excerpts in `vv/tests/data/`); the study reproducing the paper's FLO82 extrapolation (Table 3). Flat plate: a PLOT3D grid converted with TMR's boundaries (markers, edge orientation, area, `V0902` for a grid that is not the plate), the same grid renumbered and partly reversed recognised as the same and a moved point not, the configuration changing only `ADOPTED_KEYS` (`V0903`), cf and the SA residual read from SU2 v8.6.0's own 35×25 outputs (excerpts in `vv/tests/data/`), and the report on runs carrying CFL3D's values, which gives CFL3D's continuum from TMR's files and passes, gives no estimate when the finest run has not converged, judges a variant that stops above ten orders but at the published runs' residual levels converged only by the amended criterion, and refuses a configuration changed after its run.

## References

- T. D. Economon, F. Palacios, S. R. Copeland, T. W. Lukaczyk and J. J. Alonso, SU2: an open-source suite for multiphysics simulation and design, *AIAA J.* 54(3):828–846, 2016. doi:10.2514/1.J053813
- J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, *J. Aircraft* 47(4):1152–1166, 2010. doi:10.2514/1.46737
- G. Nastac, A. Walden, L. Wang, E. Nielsen, Y. Liu, M. Opgenorth, J. Orender and M. Zubair, A multi-architecture approach for implicit CFD on unstructured grids, AIAA 2023-1226. doi:10.2514/6.2023-1226
- V. Venkatakrishnan, Convergence to steady state solutions of the Euler equations on unstructured grids with limiters, *J. Comput. Phys.* 118:120–130, 1995. doi:10.1006/jcph.1995.1084
