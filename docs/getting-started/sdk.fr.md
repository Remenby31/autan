---
source_hash: "9de7ec952081"
---

# Prise en main du SDK Autan

Vérifiez un cas face à son maillage et écrivez un répertoire de calcul avec le SDK Python, depuis un clone du dépôt. Cette page utilise le backend `fake` du SDK, qui écrit des résultats synthétiques dans les vrais formats et ne demande aucune compilation. Le solveur, qu'appelle `autan run --backend cpu`, calcule des écoulements compressibles non visqueux sur CPU et se compile depuis les sources ([Compiler Autan depuis les sources](build.fr.md)) ; l'exemple ci-dessous utilise un modèle de turbulence, que le solveur ne calcule pas encore.

> Les résultats du backend `fake` sont synthétiques. Ne les utilisez jamais comme données de référence.

## Prérequis

- Un clone de [Remenby31/autan](https://github.com/Remenby31/autan).
- [uv](https://docs.astral.sh/uv/) et Python 3.12 ou plus récent.

Aucun GPU n'est nécessaire.

## Préparer l'environnement

À la racine du clone, installez le SDK et ses dépendances verrouillées :

```sh
uv sync --locked
```

## Vérifier un cas

Le dépôt fournit un cas pour la configuration hypersustentée CRM-HL de la NASA, `py/examples/crm_hl.py`, et un MeshInfo de test qui tient lieu de son maillage : les effectifs sont réalistes, le fichier de maillage n'existe pas. Planifiez-le :

```sh
uv run autan plan py/examples/crm_hl.py --mesh-info py/tests/fixtures/crm-hl.meshinfo.json
```

```text
boundaries  7/7 zones covered
  wing.slat      0.4 M faces  wall           code
  …
  fuselage       0.7 M faces  wall           mesh
  …
memory      650.2 GB per rank (450.5 host, 199.7 cuda) x 1 · 6 equations (autan.memory/2)
warning[W0205]: mesh '…/crm-hl.cgns' not found; the MeshInfo is taken as given
plan OK
```

Chaque zone frontière a une condition, et la dernière colonne indique d'où elle vient : votre code ou le fichier de maillage. L'avertissement est attendu : le fichier de maillage lui-même n'est pas dans le dépôt. Pour votre propre maillage, l'outil C++ `autan-mesh-info --json MESH` écrit ce document ([contrat de maillage : lecteurs](../contracts/mesh.md#readers)). [Conditions aux limites](../concepts/boundary-conditions.fr.md) explique la résolution.

## Le lancer sur le backend fake

```sh
uv run autan run py/examples/crm_hl.py --mesh-info py/tests/fixtures/crm-hl.meshinfo.json --runs "${TMPDIR:-/tmp}/autan-runs"
```

La commande affiche l'identifiant du calcul, son état final et son répertoire. Ce répertoire contient le cas tel qu'exécuté, les métadonnées, l'historique des résidus et une solution CGNS, comme le décrit [Calculs](../concepts/runs.fr.md).

## Lire les résultats

Ouvrez le calcul avec `autan.Results` :

```python
import os
from pathlib import Path

import autan

run_dir = max(Path(os.environ.get("TMPDIR", "/tmp"), "autan-runs").glob("run-*"))
results = autan.Results.open(run_dir)
print(results.run.status, results.run.solver.backend)
print(results.forces.cl[-1])
```

`results.run.solver.backend` vaut `fake` : le coefficient de portance est synthétique.

## Et ensuite

- [Vérifier un cas avant de le lancer](../guides/check-a-case.fr.md) : lire le plan et corriger ce qu'il signale.
- [Maillages](../concepts/meshes.fr.md) : comment Autan stocke cellules, faces et zones.
- [Schéma de cas](../contracts/case-schema.md) : tous les champs d'un cas (en anglais).
