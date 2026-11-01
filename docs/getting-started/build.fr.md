---
source_hash: "c61b411a9ae4"
---

# Compiler Autan depuis les sources

Compilez le runtime du solveur `autan-run` et l'outil de maillage `autan-mesh-info` depuis un clone du dépôt, puis lancez un cas compressible non visqueux sur CPU. Il n'existe pas encore de paquet binaire.

> Le solveur calcule des écoulements compressibles non visqueux sur CPU, en `fp64` et `fp32+acc64`. Toute autre combinaison de physique, modèle de turbulence, précision et périphérique acceptée par le schéma de cas se termine par `E0706`.

## Prérequis

- Linux, x86-64 ou arm64. Ces étapes sont vérifiées sur Ubuntu 24.04.
- Un compilateur C++20 (GCC 13 ou Clang 18), CMake 3.28 ou plus récent, Git et Python 3.12. Sur Ubuntu 24.04 : `sudo apt-get install build-essential cmake git python3`.
- [uv](https://docs.astral.sh/uv/) pour le SDK Python.
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) pour la bibliothèque CGNS : Autan lit et écrit du CGNS 4, que les paquets Ubuntu ne fournissent pas.

## Installer les dépendances

À la racine du clone, placez CGNS 4.5 et Ninja, depuis conda-forge, dans un préfixe dédié, et le SDK dans l'environnement du dépôt :

```sh
micromamba create --yes --prefix "$HOME/autan-deps" --channel conda-forge cgns=4.5.2 ninja=1.13.2
uv sync --locked
```

## Compiler

```sh
cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="$HOME/autan-deps/bin/ninja" -DCMAKE_PREFIX_PATH="$HOME/autan-deps" -DCMAKE_BUILD_TYPE=Release -DAUTAN_BUILD_TESTS=OFF
cmake --build build
```

La compilation génère d'abord les kernels d'écoulement depuis `kernels/` avec Python, puis les compile. Elle produit `build/runtime/autan-run` et `build/io/autan-mesh-info`.

## Lancer un cas sur CPU

Écrivez un cas pour le maillage de test hexaédrique de 27 cellules du dépôt : un écoulement à Mach 0,3 le long d'une paroi glissante, avec des conditions de champ lointain sur les faces ouvertes.

```sh
cat > euler_box.py <<'EOF'
import autan
from autan.units import Pa, deg

case = autan.Case("euler-box", mesh="tests/data/mesh/hex/polyMesh")
case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
case.freestream(mach=0.3, aoa=5 * deg, pressure=101325 * Pa)
case.time = autan.Steady(max_iterations=200)
case.boundary("ymin").slip()
for zone in ("xmin", "xmax", "ymax", "zmin"):
    case.boundary(zone).farfield()
EOF
```

`zmax` est un plan de symétrie dans le fichier de maillage : le cas ne le nomme pas. Résumez le maillage, puis lancez le cas avec le solveur :

```sh
build/io/autan-mesh-info --json tests/data/mesh/hex/polyMesh > euler_box.meshinfo.json
AUTAN_RUN="$PWD/build/runtime/autan-run" uv run autan run euler_box.py --mesh-info euler_box.meshinfo.json --backend cpu --runs "${TMPDIR:-/tmp}/autan-runs"
```

La commande affiche l'identifiant du calcul, son état final et son répertoire. `run.json` enregistre `"backend": "cpu"` ; [Calculs](../concepts/runs.fr.md) décrit le reste du répertoire.

## Et ensuite

- [autan run](../sdk/run.md) : backends, périphériques et statut de sortie (en anglais).
- [Schéma de cas](../contracts/case-schema.md) : tous les champs d'un cas (en anglais).
- [Discrétisation Euler](../design/euler.md) : le schéma que calcule le solveur (en anglais).
