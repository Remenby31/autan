# Plan d'implémentation — solveur CFD

Validé le 2026-10-31. Sources : `research/etat-de-l-art-2026-10.md`, `research/dx.html`, `mockup/ux.html`.

## Décisions figées

- Volumes finis centrés cellules, adressage par faces, **polyèdres arbitraires**, maillages hybrides, nœuds non conformes.
- **Compressible d'abord** (solveur densité implicite), pression/incompressible ensuite.
- C++20. Kernels ponctuels face/cellule décrits une fois → **générateur** émettant CUDA + HIP natifs (+ CPU) **et leurs adjoints**. Kokkos en repli CPU. Pas de Rust dans le cœur.
- Précision : calcul FP32, stockage FP16/BF16 des Jacobiens/préconditionneurs, FP64 pour accumulation, normes, géométrie, niveaux grossiers.
- Solveurs linéaires : BSR 5×5 + relaxation par blocs multicolore + multigrille d'agglomération + line-implicit + JFNK (maison) ; AmgX / hypre derrière une interface pour la pression.
- SDK Python (nanobind), schéma de cas typé avec unités, CLI `plan / run / logs / diff / doctor / verify`. Le cas est du code, l'UI en est une vue.
- Conditions aux limites : **types lus dans le maillage** (Fluent .msh, Gmsh, CGNS), valeurs et surcharges en Python, provenance affichée par `plan`.
- « Plusieurs gaz » : mélange multi-espèces **et** zones de cellules à fluides différents + solides (CHT), jalon M9.
- Matériel : dev RTX PRO 6000 / DGX Spark ; prod B200/GB200, MI300X/MI355X. Apple Silicon : backend CPU optimisé ARM dès M1, backend GPU Metal en M1b (pas de FP64 sur Metal : `fp32+acc64` émulé en double-float, `fp64` refusé avec un code d'erreur).

## Arborescence

```
cfd-solver/
  core/       maillage (faces, zones), partition METIS, halos, géométrie FP64
  kernels/    sources DSL des kernels ponctuels (flux, gradients, limiteurs, BC)
  codegen/    générateur → CUDA / HIP / CPU + tangent / adjoint
  solvers/    pas de temps, JFNK, BSR 5×5, agglomération MG, backends AmgX/hypre
  physics/    flux (Roe, HLLC, AUSM+-up), SA, SST, DDES, WMLES, espèces
  io/         lecteurs CGNS, Fluent .msh, Gmsh, polyMesh ; écritures CGNS/VTK
  py/         SDK, schéma, CLI
  vv/         MMS, cas TMR/HLPW/DPW, benchmarks perf, tableaux de bord
  ui/         application web (à partir de M10)
```

Build : CMake + scikit-build-core + nanobind. CI : GitHub Actions + runners GPU auto-hébergés (RTX PRO 6000, DGX Spark) ; AMD via cloud.

## Jalons

Durées indicatives pour 2–3 ingénieurs à plein temps assistés d'agents. Chaque jalon se ferme sur une **porte** : critères chiffrés + revue adversariale (Codex gpt-6) + cas ajoutés à la CI.

| # | Jalon | Contenu | Porte (critère de sortie) | Mois |
|---|---|---|---|---|
| M0 | Fondations | Repo, build, CI GPU ; structure de maillage par faces ; lecteurs Gmsh, Fluent .msh (zones typées), CGNS ; partition + halos ; rapport qualité | Lecture des 4 types de cellules + polyèdres à 100+ faces ; zones et types BC conservés ; 50 M cellules partitionnées | 0–1,5 |
| M1 | Spike générateur + perf | DSL kernels → CUDA/HIP/CPU (CPU vectorisé ARM NEON, testé sur Mac) ; gradients LSQ et Green-Gauss ; advection-diffusion scalaire ; atomiques vs coloration ; prototype adjoint d'un kernel | MMS ordre 2 sur tét/hex/poly/hybride en FP32 et FP64 ; ≥ 70 % du pic BW sur gradient et flux ; test produit scalaire de l'adjoint OK | 1–3 |
| M1b | Backend Metal (Apple Silicon) | Émetteur Metal Shading Language depuis l'IR du générateur ; accumulations FP64 émulées en double-float ; stratégie de boucle sur les faces mesurée (pas d'atomiques 64 bits) ; runner macOS Apple Silicon en CI | Mêmes portes que M1 sur Metal : MMS ordre 2 sur les 4 types de maillage en `fp32` et `fp32+acc64` ; ≥ 70 % du pic de bande passante mesuré sur la machine | 3–4 (parallèle à M2) |
| M2 | Euler compressible | Roe/HLLC/AUSM+-up, BC caractéristiques ; explicite puis implicite (Jacobien BSR 5×5 fusionné, GS bloc multicolore, JFNK) | MMS Euler, Ringleb, NACA0012 et ONERA M6 non visqueux ; convergence de 10 ordres | 2,5–5 |
| M3 | Navier-Stokes + SA | Flux visqueux avec corrections de non-orthogonalité ; SA-neg ; distance à la paroi GPU ; agglomération MG + line-implicit | TMR plaque plane, bosse, NACA0012 dans la tolérance ; convergence sur maillages dégradés | 4,5–8 |
| M4 | SDK + CLI v0 | Schéma typé + unités ; zones du maillage avec provenance ; `plan` (couverture BC, mémoire, y+) ; `run`, `logs`, `diff` ; erreurs codées ; API résultats + DLPack ; sorties CGNS/VTK | Cas TMR entièrement pilotés par le SDK ; `plan` détecte les BC manquantes ; erreurs avec code + correction | 3–7 (parallèle) |
| M5 | Multi-GPU + précision | MPI GPU-aware, recouvrement, CUDA Graphs ; étude FP32+acc64 vs FP64 | Efficacité ≥ 80 % à 8 GPU, mesurée jusqu'à 64 ; CL/CD identiques à 0,1 % entre précisions ; ns/cellule/itération publiés | 7–9 |
| M6 | Adjoint discret GPU | Adjoint des kernels générés, solve transposé, FFD, `cfd.grad` | Test produit scalaire + différences finies ; minimisation de traînée RAE2822 et ONERA M6 | 8–12 |
| M7 | SST, DDES, WMLES | SST, DDES/IDDES, loi de paroi WMLES, flux peu dissipatif, BDF2 temps dual | Cylindre, marche descendante, Ahmed ; CRM-HL près de CLmax comparé à HLPW-5 | 9–14 |
| M8 | Solveur pression | SIMPLE/PISO puis couplé ; AmgX/hypre | Cavité, cylindre, DrivAer comparés aux références | 12–16 |
| M9 | Multi-gaz + CHT | Transport d'espèces, propriétés de mélange ; zones fluides distinctes ; zones solides et interfaces couplées | Mélange de jets, cas CHT de référence ; zones du .msh mappées automatiquement | 14–17 |
| M10 | UI web v1 + MCP | Viewer 3D (depuis la maquette), suivi en direct, éditeur sur le schéma ; serveur MCP | Cas CRM-HL monté de bout en bout dans l'UI, code généré identique au SDK | 10–16 (parallèle) |
| M11 | IA native | Usine de données, warm start par surrogate, résidu comme signal de confiance | Gain de temps mesuré sur une famille de cas ; fausse confiance détectée | 15–18 |

**v1 ≈ mois 18.** Après v1 : adaptation de maillage GPU, overset, convertisseur tét → poly, mailleur.

## Chemin critique

M0 → M1 → M2 → M3 → M5 → M6 / M7. M4 et M10 avancent en parallèle dès M1.

## Risques

| Risque | Parade |
|---|---|
| Générateur + AD trop complexe | Spike M1 avec porte ; AD maison sur l'IR du générateur, replis Tapenade puis nombres duaux, Enzyme comme oracle de vérification (voir `research/kernel-generator.md`) |
| FP32 insuffisant (CLmax) | Précision en paramètre de template dès M1, étude M5, repli FP64 par kernel |
| AMG/solveurs lents | Harnais de matrices réelles dès M2 ; backends interchangeables |
| Portage AMD fragile | Runner AMD en CI dès M1 |
| Accès matériel de prod | Cloud B200/MI300X pour M5 ; dev sur RTX PRO 6000 et DGX Spark |
| Données de validation | Grilles et données TMR, DPW et HLPW publiques |

## Exécution

- Session principale = juge ; agents Opus implémentent par lots ; une PR par lot.
- Revue Codex gpt-6 à chaque porte.
- Tableau d'avancement mis à jour à chaque passage.
