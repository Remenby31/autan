#!/usr/bin/env bash
# The bench's health precondition (bench/ci/health.py): STREAM built from the GB10 baseline's own
# commit, measured on this Spark before the pull request's benchmarks, and judged against the
# baseline. Prints state=healthy|degraded for $GITHUB_OUTPUT; exits 0 healthy, 3 degraded.
# Usage: bench/ci/health.sh BASELINE OUT_DIR   (from the repository's root, in the bench job)
set -euo pipefail
baseline=$1 out=$2
commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["provenance"]["solver"]["commit"])' "$baseline")
ref=$out/reference
mkdir -p "$out"
if [[ ! -x $ref/build/stream ]]; then
  git fetch -q --depth 1 origin "$commit"
  rm -rf "$ref/src"
  git worktree add -q --detach "$ref/src" "$commit"
  # Under the GPU lock: on the bench Spark, nothing compiles beside a measurement.
  bench/lock/autan-gpu-lock -m "gpu-bench: build the health probe (STREAM at ${commit:0:8})" -t 600 -- \
    bash -c "cmake -S '$ref/src/bench' -B '$ref/build' >/dev/null && cmake --build '$ref/build' >/dev/null"
fi
echo "probe: STREAM at $commit, binary $(sha256sum "$ref/build/stream" | cut -c1-16)…" >&2
uv run --project bench --locked autan-bench stream --binary "$ref/build/stream" --out "$out/health.json" >&2
status=0
python3 bench/ci/health.py "$out/health.json" "$baseline" >&2 || status=$?
case $status in
  0) echo "state=healthy" ;;
  3) echo "state=degraded" ;;
  *) exit "$status" ;;
esac
exit "$status"
