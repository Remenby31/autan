#!/usr/bin/env bash
# Compares the tracked loops of two bisection probes: exits 1 when NEW regressed from OLD under
# the gate's rule (autan-bench compare: more than 5 % worse with disjoint confidence intervals).
# Usage: bench/ci/compare-points.sh NEW_DIR OLD_DIR
set -uo pipefail
uv run --project bench --locked autan-bench compare \
  "$1/codegen-hex-n128-fp32.json" "$2/codegen-hex-n128-fp32.json"
