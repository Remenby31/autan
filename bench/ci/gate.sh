#!/usr/bin/env bash
# Regression gate confirmed by a second, independent measurement.
# Usage: bench/ci/gate.sh RESULT BASELINE -- MEASURE...
# Compares RESULT with BASELINE. When a median is more than 5 % worse (compare exits 1 for a
# regression, 3 when inconclusive), runs MEASURE once (it must rewrite RESULT) and compares again:
# that second comparison decides. A regression there fails; an inconclusive verdict there is a
# warning, since no regression was shown. Errors (exit 2) fail at once. On the shared Spark, run
# medians drift by up to 4-6 % between runs, so one run alone is not evidence (docs/infra/bench.md).
set -uo pipefail
result=$1 baseline=$2
shift 2
[[ ${1:-} == -- ]] && shift
compare() { uv run --project bench --locked autan-bench compare "$result" "$baseline"; }

compare
status=$?
if [[ $status -eq 1 || $status -eq 3 ]]; then
  echo "::warning title=autan-bench::worse than $baseline in one run; measuring again to confirm"
  "$@" || exit $?
  compare
  status=$?
  if [[ $status -eq 3 ]]; then
    echo "::warning title=autan-bench::not confirmed by the second run (within noise)"
    status=0
  fi
fi
exit "$status"
