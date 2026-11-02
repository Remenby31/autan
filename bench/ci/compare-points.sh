#!/usr/bin/env bash
# Compares the tracked loops of two bisection probes: exits 1 when NEW regressed from OLD under
# the gate's rule (autan-bench compare: more than 5 % worse with disjoint confidence intervals),
# 0 when not, 2 when they cannot be compared. Only the benchmarks of the tracked baseline count:
# OLD is reduced to them, since compare judges every benchmark of its reference.
# Usage: bench/ci/compare-points.sh NEW_DIR OLD_DIR
set -uo pipefail
result=codegen-hex-n128-fp32.json
tracked=$(mktemp)
trap 'rm -f "$tracked"' EXIT
python3 - "$2/$result" bench/baselines/gb10-codegen-hex-n128-fp32.json "$tracked" <<'PY' || exit 2
import json, sys
old, baseline, out = sys.argv[1:]
names = {b["name"] for b in json.load(open(baseline))["benchmarks"]}
doc = json.load(open(old))
doc["benchmarks"] = [b for b in doc["benchmarks"] if b["name"] in names]
json.dump(doc, open(out, "w"))
PY
uv run --project bench --locked autan-bench compare "$1/$result" "$tracked"
