---
source_hash: "d1ab26ee9ba4"
---

# Architecture

Autan est un cœur C++, un générateur de kernels et un SDK Python qui échangent des documents aux schémas fixés. Cette page montre comment les parties s'assemblent ; le [plan](../PLAN.md) donne les jalons, et le [README](../../README.md#done-and-planned) ce qui est fait.

## Composants

![Composants d'Autan](../assets/diagrams/architecture.svg)

| Répertoire | Rôle | Jalon |
|---|---|---|
| `io/` | Lecteurs de maillages Gmsh, Fluent, CGNS et OpenFOAM. | M0 |
| `core/` | Le [maillage](meshes.fr.md) par faces, la géométrie FP64, la qualité et la renumérotation. | M0 |
| `partition/` | Partitionnement par METIS ou Scotch, et échange de halos MPI. | M0 |
| `py/` | Le SDK et la CLI `autan` : schéma de cas, `plan`, `run`, résultats. | M0, M4 |
| `codegen/`, `kernels/` | Kernels écrits une fois, émis pour CUDA, HIP, CPU et Metal avec leur tangent et leur adjoint. | M1, M1b |
| `runtime/` | `autan-run` : lit une entrée du solveur, pilote une physique enregistrée, écrit le répertoire de calcul. | M2 |
| `physics/`, `solvers/` | Schémas de flux, modèles de turbulence, pas de temps, solveurs linéaires. Aujourd'hui : écoulements compressibles non visqueux sur CPU, en `fp64` et `fp32+acc64` ([discrétisation Euler](../design/euler.md), en anglais). | M2, M3 |
| `vv/`, `bench/`, `docker/` | Vérification par solutions manufacturées et données de référence ; image de développement et mesures de bande passante. | M0 |
| `ui/` | Interface web : éditeur de cas et visualisation des résultats. | M10 |

Les composants communiquent par quatre schémas, jamais par leurs internes : le [cas](../contracts/case-schema.md) (`autan.case/1`), le [MeshInfo](../contracts/mesh-info.md) (`autan.mesh-info/1`), l'[entrée du solveur](../contracts/solver-input.md) (`autan.solver-input/1`), qu'[`autan prepare`](../sdk/prepare.md) écrit à partir d'un cas planifié et qu'`autan-run` lit, et le [répertoire de calcul](../contracts/results.md) (`autan.results/1`).

## Comment Autan dérive ses kernels

![Tangent et adjoint du résidu](../assets/diagrams/ad-tangent-adjoint.svg)

Le résidu est une chaîne de boucles sur les cellules et les faces. Le générateur dérive le corps de chaque boucle et en déduit :

- le **tangent**, qui exécute les boucles dans le même ordre et donne le produit Jacobien-vecteur utilisé par le solveur implicite ;
- l'**adjoint**, qui exécute les boucles transposées en ordre inverse et donne le gradient d'une sortie, comme la traînée, par rapport à toutes les entrées, pour le coût d'environ une résolution de plus.

Dans l'adjoint, une boucle qui lisait les cellules voisines y écrit : le générateur recalcule donc quelles boucles entrent en conflit. Le test du produit scalaire vérifie que tangent et adjoint concordent. La conception et ses alternatives sont dans la [recherche sur le générateur de kernels](../research/kernel-generator.md#4-automatic-differentiation) (en anglais). Le tangent arrive avec M1, l'adjoint complet avec M6.

## Précision

Les kernels calculent en FP32 et accumulent en FP64 par défaut ; un cas peut demander du FP64 partout. [Schéma de cas : physique](../contracts/case-schema.md#physics) définit les deux modes, et chaque calcul enregistre celui qu'il a utilisé.
