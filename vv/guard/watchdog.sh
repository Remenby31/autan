#!/usr/bin/env bash
# Nightly guard (#611), run by autan-guard-watchdog.timer at 07:00 UTC (Persistent): appends the
# outcome of the night whose deadline the latest 07:00 slot closed (green, red, void, or MISSING) to
# $G/STATUS, once per night; trims green nights older than 14 days to their verdict, histories and
# run.json. Red and void nights stay whole.
set -uo pipefail
G=${AUTAN_GUARD_ROOT:-$HOME/cfd/guard}
if [ "$(date -u +%H%M)" -lt 0700 ]; then night=$(date -u -d "2 days ago" +%F); else night=$(date -u -d yesterday +%F); fi
grep -q "^$night " "$G/STATUS" 2>/dev/null && exit 0
v=$G/$night/verdict.json
line=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
fail = ' '.join(f\"{c['gate']}:{c.get('equation') or c.get('measure') or c.get('rule') or ''}\" for c in d.get('failing', []))
print(d.get('commit', '?')[:12], d.get('outcome', 'MISSING'), fail or d.get('reason') or d.get('infra') or '')" "$v" 2>/dev/null) \
  || line="$(cut -c1-12 "$G/$night/COMMIT" 2>/dev/null || echo '?') MISSING"
echo "$night $line" >> "$G/STATUS"
find "$G" -maxdepth 1 -name '20*' -type d -mtime +14 | while read -r d; do
  python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('outcome') != 'green')" "$d/verdict.json" 2>/dev/null || continue
  rm -rf "$d/src" "$d/build" "$d/venv"
  find "$d" -name '*.cgns' -delete
done
