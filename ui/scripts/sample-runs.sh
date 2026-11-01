#!/bin/sh
# Writes sample run directories with the SDK's fake backend: converged, stopped, diverged and,
# with --live, one run that keeps writing for a few minutes.
set -eu
runs=${1:?usage: sample-runs.sh RUNS [--live]}
repo=$(cd "$(dirname "$0")/../.." && pwd)
autan() { uv run --locked --project "$repo/ui/server" autan "$@" >/dev/null 2>&1; }
crm="$repo/py/examples/crm_hl.py --mesh-info $repo/py/tests/fixtures/crm-hl.meshinfo.json"
hx="$repo/py/examples/heat_exchanger.py --mesh-info $repo/py/tests/fixtures/heat-exchanger.meshinfo.json"
autan run $crm --runs "$runs"
autan run $crm --runs "$runs" --iterations 60
autan run $crm --runs "$runs" --diverge-at 80 || [ $? -eq 1 ]
autan run $hx --runs "$runs"
# An angle-of-attack sweep of the same case: a study for the Compare screen and its polar.
sweep=$(mktemp -d)
for aoa in 7.05 11.29 17.05 21.47; do
  # Forces every 30 iterations: the fake converges at 100, between two samples, as real runs do.
  sed "s/aoa=19.57/aoa=$aoa/; s/every=10/every=30/" "$repo/py/examples/crm_hl.py" > "$sweep/crm_hl_$aoa.py"
  autan run "$sweep/crm_hl_$aoa.py" --mesh-info "$repo/py/tests/fixtures/crm-hl.meshinfo.json" --runs "$runs"
done
rm -r "$sweep"
if [ "${2:-}" = "--live" ]; then
  autan run $crm --runs "$runs" --iterations 3000 --pace 0.1
fi
