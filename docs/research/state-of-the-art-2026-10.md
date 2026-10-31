---
lang: fr
---

# Autan — état de l'art et opportunités (2026-10-31)

Synthèse de 9 recherches parallèles (commercial, open source/recherche, frontière numérique, IA×CFD, marché, stacks GPU, solveurs linéaires, matériel/précision, techniques de kernels).
Les chiffres viennent des sources citées ; ceux marqués *(non vérifié)* viennent de résumés ou de mémoire.

---

## 1. État de l'art par axe

### 1.1 Codes commerciaux

| Code | GPU aujourd'hui | Absent sur GPU |
|---|---|---|
| **Ansys Fluent 2026 R1** (Synopsys) | Solveur **pression seul**, 3D ; k-ε, SST, GEKO, LES, WMLES, SBES ; VOF, DPM, FGM. 2,2 Md cellules sur 1024 MI250X : 38,5 h → 1,5 h ([Ansys 04/2025](https://www.ansys.com/news-center/press-releases/4-1-25-fluent-acceleration-with-amd-bh-ornl)). Mode hybride ~+40 % vs FP64 | **Solveur densité**, 2D, **adjoint**, overset, transition/DES incertains |
| **STAR-CCM+ 2606** (Siemens) | Ségrégé + **couplé (compressible)**, multiphasique, chimie ; 1 GPU ≈ 115–384 cœurs ([blog](https://blogs.sw.siemens.com/simcenter/simcenter-star-ccm-2606-released/)) | **Overset** (remplacé par Virtual Body), **adjoint** |
| **Cadence Charles** | LES explicite compressible 100 % GPU, Voronoi (Stitch) ; HiLiftAeroML : 1800 WMLES de 300–500 M cellules ([arXiv 2605.19565](https://arxiv.org/html/2605.19565v1)) | RANS stationnaire, adjoint *(non vérifié)* |
| **PowerFLOW / XFlow / ultraFluidX** | LBM GPU, cartésien, surtout bas Mach isotherme | Maillage conforme, compressible élevé |
| **CONVERGE 6** | RANS/LES + chimie sur GPU (07/2026) | Son maillage autonome pas encore porté |

Licences opaques (SM comptés, tokens, Millennium M2000 ≈ 2 M$ pour 32 GPU). Aucun benchmark GPU public standardisé.

### 1.2 Open source / codes gouvernementaux

- **NASA FUN3D** : réécrit C++ « GPU-first », couche macro FLUDA (CUDA/HIP/SYCL), **zéro lib externe, tout sur device**, 73 Md points sur 97 % de Frontier ([AIAA 2024](https://ntrs.nasa.gov/api/citations/20240007678/downloads/2024_AIAA_Aviation_Exascale_Paper.pdf)). Solveur linéaire à **81 % du pic BW A100**. Jacobien stocké en **FP16**. **ITAR**.
- **NASA LAVA** : WMLES Voronoi sur GPU (HLPW5), ~10× moins d'énergie ; **reste FP64** car CLmax sensible à la précision ([NTRS 2024](https://ntrs.nasa.gov/api/citations/20240015729/downloads/LAVA_WMLES_Final.pdf)).
- **OpenFOAM** : pas de GPU natif. **SPUMA** (Cineca, 12/2025) : port complet CUDA/HIP, atomiques, 1 A100 ≈ 200–300 cœurs ; DrivAer DDES 2,6× moins cher que CPU ([arXiv 2512.22215](https://arxiv.org/html/2512.22215v1)). Offload du solveur linéaire seul = gains « non significatifs ».
- **SU2** : adjoint discret par AD (CoDiPack), **CPU uniquement** ; GPU = morceaux GSoC.
- **ExaWind (Nalu-Wind)** : Kokkos + hypre ; goulot = **assemblage + setup AMG**, pas le solve.
- **PyFR** (haut ordre, codegen multi-backend), **NekRS**, **JAX-Fluids 2.0** (différentiable mais cartésien), **ONERA SoNICS** (graphes CUDA/HIP, soucis AMD), **CERFACS AVBP** (OpenACC→OpenMP), **code_saturne** (CUDA incrémental).

### 1.3 Frontière numérique (workshops AIAA)

- **Vérification résolue** : sur cas bien défini, les codes RANS concordent à 0,005 de CL. **L'erreur restante = modèle + pratiques** ([HLPW-5](https://ntrs.nasa.gov/api/citations/20240014255/downloads/HLPW5_Summary.pdf)).
- Près de CLmax : dispersion RANS ~0,5 CL vs ~0,15 HRLES/WMLES ; aucun RANS ne capture le mécanisme post-décrochage ; convergence des résidus non atteinte sur configs complexes.
- Plus de dispersion **sans données expérimentales** → pas de consensus pour la prédiction aveugle.
- **Tremblement (buffet)** DPW-8 : dispersion « surprenante », réduite en figeant domaine/variante de modèle/convergence.
- **WMLES** : charLES à 0,05 de CLmax en < 1 jour ; trois codes à 2 % sur CRM-HL ; mais branches de solutions multiples, transition sur le bec non résolue, pas de guidelines.
- **Adaptation** : rend les équipes cohérentes, pas plus justes ; non démontrée pour WMLES ; adjoint instationnaire ouvert. **Aucune adaptation sur GPU.**
- **Vision 2030** : estimation d'erreur fiable, maillage automatique, UQ → jalons probablement manqués ([Malik 2024](https://www.cfd2030.com/Aviation24/CFD-2030-Aviation-2024-Malik-10-Years-Since-CFD-Vision-2030.pdf)). ML pour la turbulence : n'a pas encore aidé.

### 1.4 IA × CFD

- Surrogates sur DrivAerML : AB-UPT ~3,8 % L2 pression de surface ; R² Cd ≈ 0,985 (AB-UPT, Transolver-3) ; ~3,6 counts RMSE. **Généralisation hors famille paramétrique faible** (CarBench : erreur ×3,5).
- Solveurs accélérés : **warm start appris = gain le plus documenté (jusqu'à 16×)** ; AMG appris : +4 à 37 %, rien en production.
- Différentiable : **aucun code GPU non structuré différentiable de bout en bout**.
- Agents LLM (OpenFOAM) : 34–68 % physiquement corrects sur benchmarks indépendants ; l'interface texte d'OpenFOAM est une grosse part du problème.
- Confiance : score SimAI = distance au training set ; le **résidu du solveur** comme signal de confiance (AUROC ~0,9) est prometteur.

### 1.5 Marché

- Consolidation : Synopsys-Ansys (07/2025), Siemens-Altair (03/2025), Cadence-Hexagon (02/2026). Synopsys vise 10 %/an pour la simulation → aéro probablement moins prioritaire.
- Douleurs : préparation du modèle (44 %), coût des licences, **robustesse** = 1er souhait.
- Startups : Flow360 (GPU, RANS bas Mach, 3,2 min sur 8 A100), Luminary (cloud + Physics AI), PhysicsX (2,4 Md$), Emmi AI, Navier. **Toutes cloud-first** ; aucun solveur GPU européen on-prem.
- Trous : haute vitesse / réactif encore sur codes CPU ; risque export-control réel (mai 2025).

### 1.6 Stack logicielle

- **CUDA** : référence NVIDIA ; **HIP** porté mécaniquement n'est pas optimal sur AMD (wavefront 64, LDS, atomiques).
- **Kokkos 5 (C++20)** : proche du natif sur les deux vendeurs, gros écosystème solveurs (hypre, PETSc, Ginkgo, Kokkos-Kernels).
- **SYCL (AdaptiveCpp)** : bat CUDA sur MG-CFD (proxy Hydra) sur A100, 86 % BW — mais chaîne la plus fragile ([arXiv 2309.10075](https://arxiv.org/abs/2309.10075)).
- **OP2** (Rolls-Royce Hydra) et **PyFR** : générateurs de code → meilleure référence de design.
- **Rust** : CUDA Rust NVIDIA annoncé 09/2026, alpha, nightly, **pas d'AMD** → **hôte uniquement**.
- Inadaptés : Triton, Mojo, std::par, OpenACC, Julia pour le cœur production.

### 1.7 Solveurs linéaires

- **AmgX** (NVIDIA seul, BSR natif) : couplé OpenFOAM 260 M cellules → solve linéaire 10–20× vs CPU, l'assemblage devient le goulot.
- **hypre BoomerAMG** GPU : portable, mais **pas d'AMG systèmes sur GPU** (blocs 5×5).
- **PETSc GAMG/Kokkos** : nouveau chemin bloc (06/2026). **Ginkgo** : agrégation, CB-GMRES. **cuDSS** : grille grossière.
- Précision mixte : préconditionneur FP32 dans Krylov FP64 ≈ mêmes itérations ; FUN3D FP16 → +53 % solveur.
- FUN3D/NSU3D/Fluent : **multigrille d'agglomération + line-implicit** pour les couches limites.

### 1.8 Matériel et précision

- **Le FP64 disparaît chez NVIDIA** : B300 ≈ 1,2 TF FP64, Rubin 33 TF (vs ~37–40 B200), RTX PRO 6000 et GB10 très faibles. Émulation FP64 (Ozaki) = **GEMM dense uniquement**, inutile pour la CFD.
- Sur ces puces un kernel FP64 devient **limité par le calcul**.
- AMD MI430X (2027) : seule puce FP64 forte annoncée.
- Preuves FP32/FP16 : Fluent GPU en simple par défaut ; OpenSBLI FP16 travail/FP32 conservatif 2,16× ; FUN3D Jacobien FP16. FP64 nécessaire : accumulation de solution, normes, géométrie, niveaux grossiers AMG, bas Mach.

### 1.9 Techniques de kernels

- **Atomiques > coloration globale** depuis Volta (localité) ; **pré-agrégation warp** FUN3D : 2,5–6× sur MI100.
- Coloration hiérarchique + mémoire partagée (OP2) en repli / reproductibilité.
- Renumérotation : ~1,4×/kernel ; SoA/AoSoA jusqu'à 4,7× ; boucles cellule quand pas de données de face.
- **Fusion totale** : MFC (Gordon Bell 2025) 3,8–4,2 ns/cellule/pas sur GH200, FP16 stockage/FP32 calcul.
- CUDA Graphs 1,2–1,5× sur petits cas ; ≥ 2–8 M cellules/GPU.

---

## 2. Où battre l'état de l'art

| # | Opportunité | Pourquoi c'est un trou | Difficulté |
|---|---|---|---|
| 1 | **Un seul code GPU non structuré, ouvert et portable : densité + pression, RANS → DDES → WMLES** | Fluent GPU = pression seul ; STAR-CCM+ a le couplé sur GPU mais fermé, sans adjoint ni overset GPU ; Charles = LES seul ; FUN3D = ITAR ; LBM = cartésien | Élevée (cœur du projet) |
| 2 | **Adjoint discret sur GPU** | CPU partout (SU2, Fluent, STAR) | Élevée — mais quasi gratuit si les kernels sont générés et différentiés (AD source/Enzyme) |
| 3 | **Conçu pour le FP64 rare, avec invariance de CLmax prouvée** | Hardware NVIDIA sans FP64 ; LAVA bloqué en FP64 ; question ouverte (CFD 2030) | Moyenne — c'est de la V&V systématique |
| 4 | **Robustesse + « vérification par défaut »** (convergence profonde, conventions figées, résidus standardisés) | Dispersion des workshops en partie auto-infligée ; robustesse = 1er souhait des utilisateurs | Moyenne |
| 5 | **V&V publique en CI** (TMR, HLPW, DPW, DrivAer) avec chiffres reproductibles | Aucun éditeur ne publie de benchmark comparable | Faible-moyenne |
| 6 | **IA native** : usine de données, warm start par surrogate, résidu comme signal de confiance, API typée pour agents | Personne n'a la boucle fermée surrogate↔solveur | Moyenne |
| 7 | **Adaptation de maillage sur GPU** (puis métriques pour WMLES) | Inexistante | Très élevée — phase 2+ |
| 8 | **On-prem souverain, prix simple, portable NVIDIA + AMD** | Startups toutes cloud ; licences opaques ; enfermement vendeur | Faible (choix produit) |

Non recommandé de viser d'entrée : maillage CAO complet, multiphasique/combustion, haut ordre.

---

## 3. Spec révisée (deltas)

- **Cœur** : C++20. **Kernels ponctuels (face/cellule) décrits une fois** → générateur maison (style OP2/PyFR) émettant **CUDA + HIP natifs** (+ Kokkos en repli CPU), **et leurs versions tangente/adjointe** (AD source ou Enzyme). C'est le levier qui rend l'adjoint GPU et la fusion abordables.
- **Hôte** : C++ ; Rust possible pour outillage/IO/CLI via ABI C, pas pour le cœur.
- **Précision** : calcul FP32, stockage FP16/BF16 des Jacobiens/préconditionneurs, FP64 pour accumulation, normes, géométrie, niveaux grossiers ; pression de jauge en bas Mach. Précision = paramètre de template, invariance vérifiée en CI.
- **Kernels** : boucles face + atomiques avec pré-agrégation warp ; coloration hiérarchique en repli/déterministe ; RCM/Hilbert + faces triées ; SoA ; fusion gradient→limiteur→flux ; CUDA Graphs ; halos GPU-aware MPI recouverts.
- **Solveurs linéaires** : pression = CG/FCG + **AmgX** (NVIDIA) / **hypre** (portable) derrière une interface, préconditionneur FP32 ; **à construire** : BSR 5×5 + relaxation par blocs multicolore, **multigrille d'agglomération + line-implicit**, JFNK externe, assemblage fusionné au résidu ; harnais de benchmark des matrices réelles.
- **Matériel cible** : dev RTX PRO 6000 / GB10 ; prod B200/GB200 ou MI300X/MI355X ; Rubin / MI430X ensuite. Dimensionner sur Go et TB/s, pas sur FLOPS.
- **Objectifs mesurables** : ≥ 70 % du pic BW sur kernels chauds ; ns/cellule/itération publié vs Fluent GPU, SPUMA, Flow360 sur cas identiques ; TMR dans la tolérance.

---

## 4. Incertitudes

- Plusieurs chiffres viennent de résumés de recherche (budget de recherche web épuisé pour 4 agents) : MueLu, rocALUTION, PCHPDDM, specs H100/MI300, prix.
- Pas de comparaison publique AmgX vs hypre vs GAMG sur CFD non structurée.
- Pas d'étude atomiques vs coloration sur Hopper/Blackwell pour la CFD non structurée.
- DPW-8 (06/2026) : pas de résumé publié trouvé.
