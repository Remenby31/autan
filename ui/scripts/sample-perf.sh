#!/usr/bin/env bash
# A perf-history directory as gpu-bench's history job writes it, built by bench/history/history.py
# from the two bench artifacts the bench tests record, between which main gains a commit that
# measured STREAM only, and whose newest run bisected a regression to its own commit.
# Usage: scripts/sample-perf.sh OUT_DIR
set -euo pipefail
out=$1
bench=$(cd "$(dirname "$0")/../../bench" && pwd)
recorded=$bench/tests/data/history
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$recorded" "$work/artifacts"
python3 -I - "$work" <<'EOF'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
old, new = sorted((work / "artifacts").iterdir())
commit = lambda run: json.loads((run / "stream.json").read_text())["provenance"]["solver"]["commit"]
between = "b" * 40
stream = json.loads((old / "stream.json").read_text())
stream["provenance"]["solver"]["commit"] = between
stream["created"] = json.loads((new / "stream.json").read_text())["created"]
(work / "artifacts" / "between").mkdir()
(work / "artifacts" / "between" / "stream.json").write_text(json.dumps(stream))
(new / "bisect").mkdir()
(new / "bisect" / "summary.json").write_text(
    json.dumps({"status": "culprit", "culprit": commit(new), "good": between, "bad": commit(new)})
)
(work / "order").write_text(f"{commit(old)}\n{between}\n{commit(new)}\n")
EOF
python3 -I "$bench/history/history.py" "$work/artifacts" --order "$work/order" \
  --baselines "$bench/baselines" --out "$out" >/dev/null
