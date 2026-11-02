#!/bin/sh
# Writes what the solver itself writes, for the golden-path e2e tests: the hybrid test mesh's
# MeshInfo and boundary surface from autan-mesh-info, and two runs of autan-run on CPU (an inviscid
# case at two angles of attack, one study). Needs AUTAN_MESH_INFO and AUTAN_RUN, the built tools.
set -eu
out=${1:?usage: golden-runs.sh DIR}
: "${AUTAN_MESH_INFO:?set AUTAN_MESH_INFO to the built autan-mesh-info}"
: "${AUTAN_RUN:?set AUTAN_RUN to the built autan-run}"
repo=$(cd "$(dirname "$0")/../.." && pwd)
mesh="$repo/tests/data/mesh/hybrid/polyMesh"
mkdir -p "$out/runs" "$out/meshes" "$out/cases"
"$AUTAN_MESH_INFO" --json "$mesh" > "$out/meshes/hybrid.meshinfo.json"
"$AUTAN_MESH_INFO" --surface "$out/meshes/hybrid.surface" "$mesh"
for aoa in 3 6; do
  cat > "$out/cases/golden-$aoa.py" <<PY
import autan
from autan.units import K, Pa, deg, m

case = autan.Case("golden", mesh="$mesh")
case.physics = autan.Compressible(turbulence=autan.Inviscid())
case.freestream(mach=0.3, aoa=$aoa * deg, sideslip=10 * deg, pressure=101325 * Pa, temperature=300 * K)
for zone in ("xmin", "xmax", "ymax", "zmin"):
    case.boundary(zone).farfield()
case.boundary("ymin").slip()
case.boundary("zmax").symmetry()
case.monitor(autan.forces("ymin", ref_area=1 * m**2), every=5)
case.time = autan.Steady(max_iterations=200)
PY
  uv run --locked --project "$repo/ui/server" autan run "$out/cases/golden-$aoa.py" \
    --mesh-info "$out/meshes/hybrid.meshinfo.json" --runs "$out/runs" --backend cpu >/dev/null
done
