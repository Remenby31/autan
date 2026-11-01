#!/usr/bin/env bash
# Prints the last commit of main that gpu-bench measured (a push or dispatch run on main whose
# bench job ran and passed; a pull request from main measures a merge tree, not its head), for
# bench/ci/changes.sh to compare a push with. Runs on main queue in one concurrency group, and
# GitHub cancels a pending run when a newer one arrives: comparing with the push's own `before`
# would let a measured change that only a cancelled run carried go unmeasured. Prints nothing
# when the API cannot tell, and changes.sh then measures.
# Usage: bench/ci/last-measured.sh   (GH_TOKEN, GITHUB_REPOSITORY set; needs actions: read)
set -uo pipefail
api() {
  local path=$1
  shift
  gh api "repos/$GITHUB_REPOSITORY/$path" "$@"
}

runs=$(api "actions/workflows/gpu-bench.yml/runs?branch=main&status=success&per_page=50" \
  --jq '.workflow_runs[] | select(.head_branch == "main" and (.event == "push" or .event == "workflow_dispatch")) | "\(.id) \(.head_sha)"') || {
  echo "::warning::cannot list gpu-bench runs; measuring" >&2
  exit 0
}
while read -r id sha; do
  [[ -n $id ]] || continue
  bench=$(api "actions/runs/$id/jobs" --jq '[.jobs[] | select(.name == "bench") | .conclusion] | first // ""') || {
    echo "::warning::cannot read the jobs of run $id; measuring" >&2
    exit 0
  }
  if [[ $bench == success ]]; then
    echo "last measured commit of main: $sha (run $id)" >&2
    echo "$sha"
    exit 0
  fi
done <<<"$runs"
echo "::warning::no measured commit of main among the last 50 successful runs; measuring" >&2
