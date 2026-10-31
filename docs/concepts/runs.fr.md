---
source_hash: 230b6d2222f1
---

# Calculus

Un calcul est une exécution d'un cas sure un maillage. Il écrit un répertoire qui contient tout ce qu'il faut pour le lire, le comparer et le reproduire. Sa structure est le [contrat de résultats](../contracts/results.md) ; `autan.Results` le lit.

## États d'un calcul

![États d'un calcul](../assets/diagrams/run-lifecycle.svg)

`autan plan` passe d'abord. S'il trouve une erreur, rien n'est écrit. Sinon le calcul démarre dans l'état `running` et se determine dans un seul état :

| État | Signification |
|---|---|
| `converged` | Chaque résidu a baissé du nombre d'ordres demandé. |
| `completed` | Le budget d'itérations ou de pas de temps s'est épuisé avant. |
| `stopped` | Vous, un signal ou une limit d'exécution l'avez arrêté. |
| `diverged` | La solution est devenue non finie ou non physique ; un diagnostic `D` nomme la cellule. |
| `failed` | Le backend a rencontré une erreur inattendue, `E0704`. |

## Le répertoire de calcul

| Fichier | Écrit |
|---|---|
| `run.json` | Au démarrage, puis à chaque changement d'état. |
| `case.toml` | Au démarrage : le cas tel qu'exécuté, toutes les valeurs par défaut remplies. |
| `mesh-info.json` | Au démarrage : le MeshInfo du maillage lu. |
| `history.jsonl` | Une ligne par itération enregistrée, pendant le calcul. |
| `solution.cgns` | À la fin, avant que l'état quitte `running`. |

Vous pouvez ouvrir un calcul en course (`running`) : l'historique est lisible, la solution ne l'est pas.

## Comment un calcul est identifié

Chaque calcul enregistre le hash du maillage, le hash du cas, le commit du solveur et la précision. Leur combinaison, avec le backend et les périphériques, est l'`identity` du calcul. En mode déterministe bit à bit, deux calculus de même identité donnent les mêmes bits ; [run.json](../contracts/results.md#runjson) définit chaque champ.
