#!/usr/bin/env bash
# The Couette gates (acceptance.md, M3; #596) on the couette family, through autan-run and
# autan-vv verify. A level counts once its run converged; the others are listed, not counted.
# Usage: vv/studies/couette.sh PRECISION OUT [LEVELS]
#   vv/studies/couette.sh fp64 /tmp/couette 8,16,32,64,128
# OUT must not exist. Needs runtime/autan-run and io/autan-mesh-info in $AUTAN_BUILD (default
# build/) and the uv environments of py/ and vv/. Writes OUT/<gate>.json, the verdict of each gate,
# and OUT/levels.tsv, each level's run, status and whether it counts.
set -euo pipefail
precision=$1 out=$2 levels=${3:-8,16,32,64,128}
top=$(git rev-parse --show-toplevel)
build=${AUTAN_BUILD:-$top/build}
export AUTAN_RUN=$build/runtime/autan-run
vv=(uv run --locked --project "$top/vv" autan-vv)
autan=(uv run --locked --project "$top" autan)
mkdir "$out"
out=$(cd "$out" && pwd)
"${vv[@]}" meshes couette --levels "$levels" --out "$out/mesh" >/dev/null
"${vv[@]}" couette-cases --family "$out/mesh" --precision "$precision" --out "$out/cases" >/dev/null
runs=()
printf 'level\trun\tstatus\tcounted\n' >"$out/levels.tsv"
mapfile -t level_lines < <(python3 -c "import json,pathlib,sys
for lv in json.load(open(sys.argv[1]))['levels']:
    print(lv['n'], pathlib.Path(lv['file']).stem)" "$out/mesh/manifest.json")
for line in "${level_lines[@]}"; do
  read -r n name <<<"$line"
  "$build/io/autan-mesh-info" --json "$out/mesh/$name.msh" >"$out/cases/$name.mesh-info.json"
  # The CLI's JSON is kept: on a refusal (E0711 and others) it holds the diagnostics.
  "${autan[@]}" run "$out/cases/$name.toml" --mesh-info "$out/cases/$name.mesh-info.json" \
    --backend cpu --runs "$out/runs" --json >"$out/cases/$name.run.json" || true
  run=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('path', ''))" \
    "$out/cases/$name.run.json" 2>/dev/null || true)
  status=refused
  if [[ -n $run ]]; then
    status=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$run/run.json")
  fi
  counted=no
  if [[ $status == converged ]]; then
    counted=yes
    runs+=("$run")
  fi
  printf '%s\t%s\t%s\t%s\n' "$n" "${run:--}" "$status" "$counted" >>"$out/levels.tsv"
done
cat "$out/levels.tsv"
if ((${#runs[@]} == 0)); then
  echo "no level converged" >&2
  exit 1
fi
status=0
for gate in couette-velocity-exact couette-temperature-order; do
  "${vv[@]}" verify "${runs[@]}" --against couette --gate "$gate" --out "$out/$gate.json" || status=1
done
exit "$status"
