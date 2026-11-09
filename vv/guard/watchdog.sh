#!/usr/bin/env bash
# Nightly guard (#611), run by autan-guard-watchdog.timer at 07:00 UTC: appends last night's
# outcome (green, red, void, or MISSING without a verdict) to $G/STATUS, and removes the build and
# sources of green nights older than 14 days. Red and void nights stay whole until an issue closes.
set -euo pipefail
G=${AUTAN_GUARD_ROOT:-$HOME/cfd/guard}
night=$(date -u -d yesterday +%F)
v=$G/$night/verdict.json
if [ -e "$v" ]; then
  line=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
fail = ' '.join(f\"{c['gate']}:{c.get('equation') or c.get('measure') or c.get('rule') or ''}\" for c in d.get('failing', []))
print(d.get('commit', '?')[:12], d.get('outcome', 'MISSING'), fail or d.get('reason') or d.get('infra') or '')" "$v")
else
  line="$(cat "$G/$night/COMMIT" 2>/dev/null | cut -c1-12) MISSING"
fi
echo "$night $line" >> "$G/STATUS"
find "$G" -maxdepth 1 -name '20*' -type d -mtime +14 | while read -r d; do
  grep -q '"outcome": "green"' "$d/verdict.json" 2>/dev/null && rm -rf "$d/src" "$d/build"
done
