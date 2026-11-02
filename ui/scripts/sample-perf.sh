#!/usr/bin/env bash
# A perf-history directory as gpu-bench's history job writes it, built by bench/history/history.py
# from the two bench artifacts the bench tests record.
# Usage: scripts/sample-perf.sh OUT_DIR
set -euo pipefail
out=$1
bench=$(cd "$(dirname "$0")/../../bench" && pwd)
recorded=$bench/tests/data/history
order=$(mktemp)
trap 'rm -f "$order"' EXIT
for run in "$recorded"/*/; do
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["provenance"]["solver"]["commit"])' "$run/stream.json"
done >"$order"
python3 -I "$bench/history/history.py" "$recorded" --order "$order" --baselines "$bench/baselines" --out "$out" >/dev/null
