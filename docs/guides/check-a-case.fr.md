---
source_hash: acd508212e5e
---

# Vérifier un cas avant de le lancer

`autan plan` vérifie un cas face à son maillage avant que vous réserviez un GPU. Il affiche la condition de chaque zone et son origine, signale ce qui ne va pas avec un code et une correction, et estime la mémoire GPU. Ce guide planifie l'example de l'échangeur thermique, puis lit et corrige un diagnostic.

## Prérequis

- L'environment de [Prise en main du SDK Autan](../getting-started/sdk.fr.md).

## Planifier le cas

Le cas de l'échangeur a deux gaz et une plaque de cuivre. Planifiez-le avec le MeshInfo de son maillage :

```sh
uv run autan plan py/examples/heat_exchanger.py --mesh-info py/tests/fixtures/heat-exchanger.meshinfo.json
```

```text
boundaries  5/5 zones covered
  air-inlet       4 000 faces  inlet          code
  …
  casing         96 000 faces  wall           mesh
cell zones
  air        0.5 M cells  fluid(air)               mesh
  co2        0.5 M cells  fluid(co2)               code
  plate      0.2 M cells  solid(copper)            code
memory      1.5 GB total · 1.5 GB/GPU on 1 GPU · 7 equations (autan.memory/1)
plan OK
```

Lisez-le zone par zone :

- `casing` n'a pas de règle dans le cas : c'est une paroi parce que le fichier de maillage le dit (`mesh`).
- `co2` et `plate` prennent les matériaux que le cas leur donne (`code`).
- La ligne mémoire est une borne inférieure ; [autan plan](../sdk/plan.md#memory-model-autanmemory1) liste ce qu'elle compte.

## Lire un diagnostic

Chaque problème a un code. Un code en `E` arrête le plan avec le statut de sortie 1 ; `W` est un avertissement. Affichez ce qu'un code signifie et comment le corriger :

```sh
uv run autan explain E0413
```

```text
## E0413

**Mesh boundary type needs values** (error)

The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values only the case can provide.

Fix: Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.
```

Le maillage de l'échangeur type ses quatre ports en entrées et sorties, et le cas donne leurs pressions et températures. Supprimez la ligne `*-outlet` d'une copie du cas et `plan` signale `E0413` pour les deux sorties.

## Utiliser le plan dans un script

`--json` affiche le même plan en un document JSON, chaque diagnostic au [format des diagnostics](../contracts/case-schema.md#diagnostics) :

```sh
uv run autan plan py/examples/heat_exchanger.py --mesh-info py/tests/fixtures/heat-exchanger.meshinfo.json --json
```

Le statut de sortie vaut 0 quand le cas peut être lancé, 1 sinon.

## Et ensuite

- [Conditions aux limites](../concepts/boundary-conditions.fr.md) : les règles de résolution derrière le plan.
- [Codes de diagnostic](../sdk/errors.md) : chaque code, avec sa correction.
