#!/usr/bin/env bash
# The bench's health precondition (bench/ci/health.py): STREAM built from the GB10 baseline's own
# commit, measured on this Spark before the pull request's benchmarks, and judged against the
# baseline. Prints state=healthy|degraded for $GITHUB_OUTPUT; exits 0 healthy, 3 degraded, other
# codes when the probe cannot be built or judged (never read as healthy).
# Usage: bench/ci/health.sh BASELINE OUT_DIR   (from the repository's root, in the bench job)
set -euo pipefail
baseline=$1 out=$2
commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["solver"]["commit"])' "$baseline")
[[ $commit =~ ^[0-9a-f]{40}$ ]] || { echo "health: no baseline commit in $baseline" >&2; exit 2; }
# Keyed by the commit: another baseline is another probe. An export (git archive), not a
# worktree: nothing is registered in the repository, so a cleaned RUNNER_TEMP leaves nothing stale.
ref=$out/reference-$commit
if [[ ! -x $ref/build/stream ]]; then
  git fetch -q --depth 1 origin "$commit"
  rm -rf "$ref"
  mkdir -p "$ref/src"
  git archive "$commit" bench | tar -x -C "$ref/src"
  # Under the GPU lock: on the bench Spark, nothing compiles beside a measurement. The export has
  # no .git: the commit is given, and the probe's result must record it (health.py checks).
  bench/lock/autan-gpu-lock -m "gpu-bench: build the health probe (STREAM at ${commit:0:8})" -t 600 -- \
    bash -c "cmake -S '$ref/src/bench' -B '$ref/build' -DAUTAN_COMMIT=$commit >/dev/null && cmake --build '$ref/build' >/dev/null"
fi
echo "probe: STREAM at $commit, binary $(sha256sum "$ref/build/stream" | cut -c1-16)…" >&2
# -n: autan-bench takes the lock itself; the wrapper bounds the whole probe (a hung STREAM is
# stopped at -t), as every GPU command on the Sparks.
bench/lock/autan-gpu-lock -n -m "gpu-bench: health probe (STREAM at ${commit:0:8})" -t 600 -- \
  uv run --project bench --locked autan-bench stream --binary "$ref/build/stream" --out "$out/health.json" >&2
status=0
python3 bench/ci/health.py "$out/health.json" "$baseline" --commit "$commit" >&2 || status=$?
case $status in
  0) echo "state=healthy" ;;
  3) echo "state=degraded" ;;
esac
exit "$status"
