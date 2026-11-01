#!/bin/sh
# Writes run directories with the real runtime and the core's test relaxation physics:
# converged, diverged at iteration 12 and, with --live, one slow running run.
# Needs: cmake --build build --target autan-run-relaxation autan-mesh-info
set -eu
runs=${1:?usage: real-runs.sh RUNS [--live]}
repo=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
mesh="$repo/tests/data/mesh/hybrid/polyMesh"
"$repo/build/io/autan-mesh-info" --json "$mesh" > "$work/hybrid.meshinfo.json"
cat > "$work/hybrid.py" <<CASE
import autan
from autan.units import K, Pa
case = autan.Case("hybrid", mesh="$mesh")
case.physics = autan.Compressible(turbulence=autan.Laminar())
case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)
for zone in ("xmin", "xmax", "ymax", "zmin"):
    case.boundary(zone).farfield()
CASE
export AUTAN_RUN="$repo/build/tests/runtime/autan-run-relaxation"
autan() { uv run --locked --project "$repo/ui/server" autan "$@" >/dev/null 2>&1; }
run="run $work/hybrid.py --mesh-info $work/hybrid.meshinfo.json --backend cpu --runs $runs"
autan $run
AUTAN_TEST_DIVERGE_AT=12 autan $run || [ $? -eq 1 ]
if [ "${2:-}" = "--live" ]; then
  AUTAN_TEST_PAUSE_MS=1500 autan $run
fi
rm -r "$work"
