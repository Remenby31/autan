#!/usr/bin/env bash
# Observed order of a manufactured solution through autan-run and autan-vv verify, on one mesh family.
# Usage: physics/mms/study.sh SOLUTION MESHES PRECISION OUT
#   physics/mms/study.sh navier_stokes meshes/hex fp64 /tmp/study
# MESHES holds a family from `autan-vv meshes` (gmsh: generate it on macOS or x86-64).
# Needs runtime/autan-run and io/autan-mesh-info in $AUTAN_BUILD (default build/) and the uv
# environments of py/ and vv/.
set -euo pipefail
solution=$1 meshes=$2 precision=$3 out=$4
top=$(git rev-parse --show-toplevel)
build=${AUTAN_BUILD:-$top/build}
export AUTAN_RUN=$build/runtime/autan-run
vv=(uv run --locked --project "$top/vv" autan-vv)
autan=(uv run --locked --project "$top" autan)
declare -A oracle=([euler]=mms-euler [navier_stokes]=mms-navier-stokes [rans_sa]=mms-rans-sa)
declare -A gate=([euler]=euler-mms-order [navier_stokes]=ns-mms-order [rans_sa]=rans-sa-mms-order)
declare -A model=([euler]=inviscid [navier_stokes]=laminar [rans_sa]=sa)
mkdir -p "$out"
"${vv[@]}" mms "$solution" --out "$out" >/dev/null
mms="$out/$solution.json"
sha=$(sha256sum "$mms" | cut -d' ' -f1)
params=$(python3 -c "import json,sys; p=json.load(open('$mms'))['parameters']; print(p['gas_constant'], p.get('viscosity', 1e-5), p.get('prandtl', 0.72))")
read -r rgas mu prandtl <<<"$params"
molar=$(python3 -c "print(8.314462618 / $rgas)")
runs=()
for msh in "$meshes"/*.msh; do
  name=$(basename "$msh" .msh)
  dir="$out/$name"
  mkdir -p "$dir"
  "$build/io/autan-mesh-info" --json "$msh" >"$dir/mesh-info.json"
  cat >"$dir/case.toml" <<TOML
schema = "autan.case/1"
name = "$solution-$name"
mesh = "$msh"

[physics]
kind = "compressible"
precision = "$precision"
turbulence = { model = "${model[$solution]}" }
gas = { kind = "ideal-gas", name = "mms", molar_mass = "$molar kg / mol", gamma = 1.4, prandtl = $prandtl, viscosity = { law = "constant", mu = "$mu Pa * s" } }

[freestream]
mach = 0.2
aoa = "0.0 deg"
temperature = "348.43 K"
pressure = "100000.0 Pa"

[time]
kind = "steady"
max_iterations = 2000
residual_orders = 10.0

[verification.manufactured]
path = "$mms"
sha256 = "$sha"
TOML
  run=$("${autan[@]}" run "$dir/case.toml" --mesh-info "$dir/mesh-info.json" --backend cpu --runs "$out/runs" --json | python3 -c "import json,sys; print(json.load(sys.stdin)['path'])" || true)
  echo "$name: $run"
  runs+=("$run")
done
"${vv[@]}" verify "${runs[@]}" --against "${oracle[$solution]}" --gate "${gate[$solution]}" --out "$out/verdict.json"
