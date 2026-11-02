#!/usr/bin/env bash
# Downloads the result artifacts of the gpu-bench runs that measured main (push or dispatch on
# main, bench job passed or failed after measuring), one directory per run, for history.py.
# Usage: bench/history/fetch.sh OUT_DIR [MAX_RUNS]   (GH_TOKEN, GITHUB_REPOSITORY; actions: read)
# Artifacts expire after the repository's retention period: older commits drop out.
set -euo pipefail
out=$1
max=${2:-300}
mkdir -p "$out"
# Listed in full before cutting: `head` on the live pipe would kill gh with SIGPIPE.
runs=$(gh api --paginate "repos/$GITHUB_REPOSITORY/actions/workflows/gpu-bench.yml/runs?branch=main&per_page=100" \
  --jq '.workflow_runs[] | select(.head_branch == "main" and (.event == "push" or .event == "workflow_dispatch") and .status == "completed") | .id')
for id in $(head -n "$max" <<<"$runs"); do
  name=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$id/artifacts" \
    --jq '.artifacts[] | select(.name | startswith("gpu-bench-")) | select(.expired | not) | .name' | head -1)
  [[ -n $name ]] || continue
  [[ -d $out/$id ]] || gh run download "$id" -R "$GITHUB_REPOSITORY" -n "$name" -D "$out/$id" || {
    echo "::warning::cannot download $name of run $id" >&2
    rm -rf "${out:?}/$id"
  }
done
