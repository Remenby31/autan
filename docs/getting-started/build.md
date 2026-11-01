# Build Autan from source

Build the solver runtime `autan-run` and the mesh tool `autan-mesh-info` from a clone of the repository, then run an inviscid compressible case on the CPU. There is no binary package yet.

> The solver runs inviscid compressible flow on the CPU, in `fp64` and `fp32+acc64`. Another combination of physics, turbulence model, precision and device that the case schema accepts ends with `E0706`.

## Prerequisites

- Linux, x86-64 or arm64. These steps are checked on Ubuntu 24.04.
- A C++20 compiler (GCC 13 or Clang 18), CMake 3.28 or later, Git and Python 3.12. On Ubuntu 24.04: `sudo apt-get install build-essential cmake git python3`.
- [uv](https://docs.astral.sh/uv/) for the Python SDK.
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for the CGNS library: Autan reads and writes CGNS 4, which the Ubuntu packages do not provide.

## Install the dependencies

From the root of the clone, put CGNS 4.5 and Ninja from conda-forge in a prefix of their own, and the SDK in the repository's environment:

```sh
micromamba create --yes --prefix "$HOME/autan-deps" --channel conda-forge cgns=4.5.2 ninja=1.13.2
uv sync --locked
```

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="$HOME/autan-deps/bin/ninja" -DCMAKE_PREFIX_PATH="$HOME/autan-deps" -DCMAKE_BUILD_TYPE=Release -DAUTAN_BUILD_TESTS=OFF
cmake --build build
```

The build generates the flow kernels from `kernels/` with Python, then compiles them. It writes `build/runtime/autan-run` and `build/io/autan-mesh-info`.

## Run a case on the CPU

Write a case for the 27-cell hexahedral test mesh of the repository, in a directory of its own: flow at Mach 0.3 over a slip wall, with farfield conditions on the open sides.

```sh
mkdir -p "${TMPDIR:-/tmp}/euler-box"
cat > "${TMPDIR:-/tmp}/euler-box/case.py" <<EOF
import autan
from autan.units import Pa, deg

case = autan.Case("euler-box", mesh="$PWD/tests/data/mesh/hex/polyMesh")
case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision="fp64")
case.freestream(mach=0.3, aoa=5 * deg, pressure=101325 * Pa)
case.time = autan.Steady(max_iterations=200)
case.boundary("ymin").slip()
for zone in ("xmin", "xmax", "ymax", "zmin"):
    case.boundary(zone).farfield()
EOF
```

The unquoted `EOF` lets the shell write the absolute path of the mesh into the case. `zmax` is a symmetry plane in the mesh file, so the case does not name it. Summarise the mesh, then run the case with the solver:

```sh
build/io/autan-mesh-info --json tests/data/mesh/hex/polyMesh > "${TMPDIR:-/tmp}/euler-box/meshinfo.json"
AUTAN_RUN="$PWD/build/runtime/autan-run" uv run autan run "${TMPDIR:-/tmp}/euler-box/case.py" --mesh-info "${TMPDIR:-/tmp}/euler-box/meshinfo.json" --backend cpu --runs "${TMPDIR:-/tmp}/euler-box/runs"
```

The command prints the run id, its final state and its directory. `run.json` records `"backend": "cpu"`; [Runs](../concepts/runs.md) describes the rest of the directory.

## Next steps

- [autan run](../sdk/run.md): backends, devices and exit status.
- [Case schema](../contracts/case-schema.md): every field of a case.
- [Euler discretisation](../design/euler.md): the scheme the solver runs.
