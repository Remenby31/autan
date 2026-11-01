# Comparison with SU2

A direct comparison of Autan with [SU2](https://su2code.github.io/) on the same meshes: iterations and wall time to one convergence criterion, and accuracy against published references (issue #195). Source: `vv/src/autan_vv/su2/`, `vv/su2/`.

**Status.** Phase 1: SU2 built, protocol fixed, NACA 0012 run by SU2 alone and checked against Vassberg and Jameson (2010). No Autan-against-SU2 number is published until Autan's implicit solver converges on these O-meshes (it stalls from 64², [cases.md](cases.md#naca-0012-inviscid)): a comparison against a solver that stalls says nothing.

## SU2, an external tool

SU2 is LGPL-2.1. It is built and run apart, never vendored in this repository and never linked with Autan; the repository holds only the image recipe, our scripts, configurations and results.

| Item | Value |
|---|---|
| Version | v8.6.0, commit `7048fd35f56d4bf52e586429d6746a36a10255f8` (checked at build) |
| Image | `vv/su2/Dockerfile`: Ubuntu 24.04 (pinned digest), GCC 13.3; `meson --buildtype=release -Dwith-mpi=disabled -Dwith-omp=true`, no AD, no Python wrapper, no CGNS or TecIO |
| No MPI | SU2's MPI build compiles and links its bundled ParMETIS, whose licence is non-commercial (AGENTS.md); SU2 runs as one process with OpenMP threads (`SU2_CFD -t N`) |
| Build | `docker build -t autan-su2:v8.6.0-omp vv/su2` on the DGX Spark (aarch64); every run records the image id |
| Run | `vv/su2/run.sh ROOT CASE THREADS`: `SU2_CFD -t THREADS` in the image, `nice -n 10`, writes `log.txt` and `provenance.json` (`autan.su2-run/1`: image id, SU2 version, threads, SHA-256 of the configuration and mesh it ran, CPU model, wall time, exit status) |

## Fairness protocol

Fixed on 2026-11-01, before any measurement. Each code solves the same problem on the same mesh, with the closest discretisation SU2 offers and each code's own published solver settings.

| Item | Autan | SU2 | Difference that remains |
|---|---|---|---|
| Mesh | the `naca0012` MSH file ([meshes.md](meshes.md#naca-0012-o-meshes)) | the same file converted (below); 2D: the slab's back plane | none: `plane_sha256` proves the same grid bit for bit |
| Control volumes | cells: NC² | median-dual of the nodes: NC(NC + 1) | cell-centred against vertex-centred, inherent to SU2 |
| Gas, freestream | the case: γ, R = R_u/M, p∞, T∞, M, α | read from the same case object | none |
| Flux | Roe, Harten's fix (width 0.1 c) on the acoustic waves only | `ROE`, entropy fix \|λ\| ≥ 0.001 λ_max on every wave (`roe.cpp`) | the entropy fixes differ in form; neither can take the other's |
| Reconstruction | primitive (ρ, u, v, w, p), least squares 1/d², face2 stencil | MUSCL κ = 0 on (T, v, p, ρ), `WEIGHTED_LEAST_SQUARES` over edge neighbours | variables and stencils differ |
| Limiter | Venkatakrishnan, ε_k² = (K/L)³ V q_k², K = 10 per cell | `VENKATAKRISHNAN`, one ε² = (K_SU2 · 1)³ for the whole mesh on variables scaled by the freestream (`FREESTREAM_PRESS_EQ_ONE`) | SU2's threshold cannot vary with the cell; it is matched to Autan's on the median cell within one chord of mid-chord: K_SU2 = K V_med^(1/3)/L. Velocity is scaled by c∞ in Autan and c∞/√γ in SU2 |
| Wall, far field | slip; characteristic far field, uniform freestream, no point vortex, 148 c away | `MARKER_EULER`; `MARKER_FAR` (same) | none |
| Forces | chord 1, area chord × span, moment about (c/4, 0) nose-up | per unit span; nose-up moment = −CMz in 2D, CMy in 3D | none |
| Convergence | every residual 10 orders below its first value ([case-schema.md](../contracts/case-schema.md#time)) | the same rule, applied by `autan_vv.su2.history` to SU2's history | none: both residuals are the RMS of the integrated flux balances over the control volumes |
| Solver controls | the defaults of the measured commit | `inv_NACA0012_Roe.cfg` of SU2 v8.6.0: implicit Euler, CFL 30, BiCGSTAB (3 iterations, 10⁻¹) with ILU(0), 3-level W-cycle multigrid | each code as its developers publish it; identical for every level |
| Precision | `fp64`; `fp32+acc64` | FP64; the mixed-precision build (`-Denable-mixedprec=true`: linear solver and preconditioner in FP32) | timed pairwise, `fp64` against FP64 and `fp32+acc64` against mixed precision, each accepted only if its CL and CD agree with its own `fp64` to 0.1 % |

SU2 measures its relative residual from the largest value so far, not the first, and its FP64 residuals stall 10.5–11 orders below their largest value at M 0.5 (64²–256²), where a rise of up to 0.7 orders in the first iterations (32²) leaves no single margin that works on every level. So `run.sh` applies the protocol's criterion to `history.csv` while SU2 runs and stops it with SIGTERM, which SU2 handles by finishing the iteration and writing its output; it also stops a run whose residuals have stalled (the drop of the worst equation has not grown by 0.1 orders in 2 000 iterations), reported not converged with the drop it reached. SU2's own rule is set two orders beyond as a backstop, and the 20 000-iteration cap ends any other run. The criterion's iteration is then read from the history.

**Timing (phase 2).** On spark-5359 with the host quiet (load < 2, GPU idle), each measurement under `autan-gpu-lock -m "CFD - Comparatif SU2: <run>" -t 600` (never a raw `flock`; ≤ 10 min, 2 min between takes), so that no GPU job runs during an SU2 or Autan CPU timing, alternating Autan and SU2, three repetitions, median reported. The same cores: SU2 with n OpenMP threads against Autan's CPU backend with n threads, n = 1 and n = the count Autan's CPU backend uses, pinned with `taskset`; Autan's GPU backend against SU2 at its fastest n. Iterations are counted up to and including the one whose residual meets the criterion, so a run given that budget ends on it. Time to solution is the wall time of a run given that budget (found by an untimed run), from process start to exit, mesh reading included, writing only the final solution. Reported per level: iterations, wall time, time per iteration per control volume.

**Published GPU reference (phase 2).** SU2 publishes no GPU timing (only a 2024 GSoC prototype), so Autan's GPU backend is also measured against FUN3D on the public DPW4 CRM grid (3 675 916 vertices, RANS-SA): 7 min to machine zero on one A100 (Nastac et al. 2023). Autan runs the same grid to the same residual drop and the same CD as that paper's Table 3, and reports wall time, iterations and throughput normalised by bandwidth: bytes of peak bandwidth per control volume and iteration. FUN3D counts vertices and Autan cells, which on this grid differ in number and in neighbours per control volume; the comparison is by time to the same solution, the normalised throughput is reported alongside.

**Accuracy.** CL, CD, CM per level at the criterion, and their continuum estimates from the three finest of at least four converged levels with `autan_vv.cases.naca0012.continuum` (the paper's eqs. 20–24), against the gates of [acceptance.md](acceptance.md#m2--compressible-euler).

## Mesh conversion

`autan_vv.su2.mesh.convert` reads the MSH 4.1 file (`autan_vv.su2.msh`, the format of the Gmsh manual; anything else is `V0901`), checks that it is one layer of hexahedra between two planes y = const, each a straight extrusion whatever its node order (`V0902` otherwise), and writes SU2's native format with every coordinate in 17 significant digits:

- 2D: the back plane, (x, z) → (x, y), quadrangles counterclockwise; the lateral zones become edge markers, the planar zones (`back`, `front`, Autan's symmetry planes) are dropped.
- 3D: the slab itself, every zone a marker, symmetry on `back` and `front`. Used as a check: on 32² at M 0.8, α 1.25°, it gives the 2D run's CL and CD to ten digits.

The record `<mesh>.json` (`autan.su2-mesh/1`) holds the SHA-256 of source and output, counts per marker, and `plane_sha256`, a hash of the plane's quadrangles and marker edges as coordinates, independent of numbering; it is the same from the 2D file, the 3D file and the source.

## Running

```sh
autan-vv meshes naca0012 --levels 32,64,128,256,512,1024 --out family   # macOS or x86-64 (Gmsh)
autan-vv su2 prepare --family family --out su2                            # meshes, case.cfg, level.json
vv/su2/run.sh su2 m080-a125/n0256 4                                       # on the Spark, 4 threads
autan-vv su2 study su2/m080-a125/n0* --out m080-a125.json                 # criterion, continuum, gates
```

`prepare` writes `su2/meshes/` and one directory per condition and level with `case.cfg` and `level.json` (`autan.su2-level/1`: K_SU2, configuration hash, Autan case hash, source mesh hash, conversion record). `study` refuses a run whose configuration or mesh, as hashed when it ran and as on disk, is not the prepared one, or whose SU2 failed (`V0904`), and runs that mix conditions or SU2 images (`V0905`).

K_SU2 per level: 2.29 (32²), 1.44, 0.90, 0.57, 0.36, 0.23 (1024²). In freestream units the threshold ε² = K_SU2³ is 12 at 32², 0.18 at 256² and 0.012 at 1024²; the squared density jump across the M 0.8 shock is about 0.2, so in both codes the limiter barely acts below 256². Autan's threshold uses h³ = V, and on a one-layer slab V includes the span (1 chord here): the same grid with another span limits differently.

## Tests

`vv/tests/test_su2.py`: the converted file parsed independently and compared with the slab the generator writes (points bit for bit, counterclockwise cells, area equal to the annulus between the far-field and airfoil polygons, airfoil nodes on eq. 1 of the paper, hashes from both files); refusals `V0901`, `V0902`, `V0903`; the configuration against the case (R from CODATA and the molar mass of air); K_SU2 on a Cartesian slab against (a² s)^(1/3) K; the criterion on geometric residual sequences; SU2 v8.6.0's history columns and axes from real 2D and 3D runs (excerpts in `vv/tests/data/`); the study reproducing the paper's FLO82 extrapolation (Table 3).

## References

- T. D. Economon, F. Palacios, S. R. Copeland, T. W. Lukaczyk and J. J. Alonso, SU2: an open-source suite for multiphysics simulation and design, *AIAA J.* 54(3):828–846, 2016. doi:10.2514/1.J053813
- J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions, *J. Aircraft* 47(4):1152–1166, 2010. doi:10.2514/1.46737
- G. Nastac, A. Walden, L. Wang, E. Nielsen, Y. Liu, M. Opgenorth, J. Orender and M. Zubair, A multi-architecture approach for implicit CFD on unstructured grids, AIAA 2023-1226. doi:10.2514/6.2023-1226
- V. Venkatakrishnan, Convergence to steady state solutions of the Euler equations on unstructured grids with limiters, *J. Comput. Phys.* 118:120–130, 1995. doi:10.1006/jcph.1995.1084
