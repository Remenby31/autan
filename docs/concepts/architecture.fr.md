---
source_hash: 93e777375c1e
---

# Architecture

Autan est un cœur C++, un générateur de kernels et un SDK Python qui échangent des documents aux schémas fixés. Cette page montre comment les parties s'assemblent ; le [plan](../PLAN.md) donne les jalons, et le [README](../../README.md#done-and-planned) ce qui est fait.

## Components

![Components d'Autan](../assets/diagrams/architecture.svg)

| Répertoire | Rôle | Jalon |
|---|---|---|
| `io/` | Lectures de maillages Gmsh, Fluent, CGNS et OpenFOAM. | M0 |
| `core/` | Le [maillage](meshes.fr.md) par faces, la géométrie FP64, le partitionnement et les halos. | M0 |
| `py/` | Le SDK et la CLI `autan` : schéma de cas, `plan`, `run`, résultats. | M0, M4 |
| `codegen/`, `kernels/` | Kernels écrits une fois, émis pour CUDA, HIP et CPU avec leur tangent et leur adjoint. | M1 |
| `physics/`, `solvers/` | Schémas de flux, modèles de turbulence, pas de temps, solveurs linéaires. | M2, M3 |
| `vv/`, `bench/` | Vérification par solutions manufacturées et données de référence ; measures de bande passante. | M0 |

Les components communiquent par trois schémas, jamais par leurs internes : le [cas](../contracts/case-schema.md) (`autan.case/1`), le [MeshInfo](../contracts/mesh-info.md) (`autan.mesh-info/1`) et le [répertoire de calcul](../contracts/results.md) (`autan.results/1`).

## Comment Autan dérive ses kernels

![Tangent et adjoint du résidu](../assets/diagrams/ad-tangent-adjoint.svg)

Le résidu est une chaîne de boucles sure les cellules et les faces. Le générateur dérive le corps de chaque boucle et en déduit :

- le **tangent**, qui exécute les boucles dans le même ordre et donne le produit Jacobien-vecteur utilisé par le solveur implicite ;
- l'**adjoint**, qui exécute les boucles transposées en ordre inverse et donne le gradient d'une sortie, comme la traînée, par rapport à toutes les entrées, pour le coût d'environ une résolution de plus.

Dans l'adjoint, une boucle qui lisait les cellules voisines y écrit : le générateur recalcule donc quelles boucles entrent en conflict. Le test du produit scalaire vérifie que tangent et adjoint concordent. La conception et ses alternatives sont dans la [recherche sure le générateur de kernels](../research/kernel-generator.md#4-automatic-differentiation) (en anglais). Le tangent arrive avec M1, l'adjoint complete avec M6.

## Précision

Les kernels calculent en FP32 et accumulent mises à jour, normes et géométrie en FP64 (`fp32+acc64`) ; `fp64` calcule tout en FP64. Chaque calcul enregistre sa précision.
