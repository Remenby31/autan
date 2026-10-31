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
if [ "${2:-}" = "--live" ]; then
  autan run $crm --runs "$runs" --iterations 3000 --pace 0.1
fi
