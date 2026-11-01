#!/usr/bin/env bash
# Regression gate with one automatic rerun.
# Usage: bench/ci/gate.sh RESULT BASELINE -- MEASURE...
# Compares RESULT with BASELINE. If the comparison is inconclusive (a median more than 5 % worse
# but within the noise, exit 3), runs MEASURE once (it must rewrite RESULT) and compares again.
# A second inconclusive comparison is reported as a warning, not a failure: no regression was
# shown. A regression (exit 1) or an error (exit 2) fails at once.
set -uo pipefail
result=$1 baseline=$2
shift 2
[[ ${1:-} == -- ]] && shift
compare() { uv run --project bench --locked autan-bench compare "$result" "$baseline"; }

compare
status=$?
if [[ $status -eq 3 ]]; then
  echo "::warning title=autan-bench::inconclusive against $baseline; measuring once more"
  "$@" || exit $?
  compare
  status=$?
  if [[ $status -eq 3 ]]; then
    echo "::warning title=autan-bench::still inconclusive after a rerun: no regression shown"
    status=0
  fi
fi
exit "$status"
