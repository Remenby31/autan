#!/usr/bin/env bash
# Nightly guard (#611), run by autan-guard-watchdog.timer at 07:00 UTC (Persistent): appends the
# outcome of the night whose deadline the latest 07:00 slot closed (green, red, void, or MISSING) to
# $G/STATUS, once per night, flagging a second void in a row. A night whose job is still pending in
# the queue is withdrawn and void, and so is a drained night left without a verdict (DRAINED). Trims
# green nights older than 14 days to their verdict, histories and run.json; red and void nights stay
# whole.
set -uo pipefail
G=${AUTAN_GUARD_ROOT:-$HOME/cfd/vv-guard}
exec 9> "$G/.watchdog.lock"
flock 9
now=$(date -u +%s)
if [ "$(date -u -d "@$now" +%H%M)" -lt 0700 ]; then now=$((now - 86400)); fi
night=$(date -u -d "@$((now - 86400))" +%F)
grep -q "^$night " "$G/STATUS" 2>/dev/null && exit 0
v=$G/$night/verdict.json
name=05-guard-$night.job
if [ ! -e "$v" ] && mv "$HOME/cfd/queue/pending/$name" "$G/$night/withdrawn.job" 2>/dev/null; then
  printf '{"schema": "autan.vv-guard-night/1", "commit": "%s", "outcome": "void", "infra": "never started: pending since %s, withdrawn at %s"}\n' \
    "$(cat "$G/$night/COMMIT")" "$(cat "$G/$night/PUBLISHED" 2>/dev/null || echo unknown)" "$(date -u +%FT%TZ)" > "$v.tmp"
  mv "$v.tmp" "$v"
fi
if [ ! -e "$v" ] && [ -e "$G/$night/DRAINED" ]; then
  python3 -c "import json, sys; print(json.dumps({'schema': 'autan.vv-guard-night/1', 'commit': sys.argv[1], 'outcome': 'void', 'infra': 'drained, no verdict: ' + sys.argv[2]}))" \
    "$(cat "$G/$night/COMMIT" 2>/dev/null || echo unknown)" "$(tr '\n' ' ' < "$G/$night/DRAINED" | cut -c1-300)" > "$v.tmp" && mv "$v.tmp" "$v"
fi
line=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
fail = ' '.join(f\"{c['gate']}:{c.get('equation') or c.get('measure') or c.get('rule') or ''}\" for c in d.get('failing', []))
print(d.get('commit', '?')[:12], d.get('outcome', 'MISSING'), fail or d.get('reason') or d.get('infra') or '')" "$v" 2>/dev/null) \
  || line="$(cut -c1-12 "$G/$night/COMMIT" 2>/dev/null || echo '?') MISSING"
if [ "$(echo "$line" | awk '{print $2}')" = void ] && grep -q "^$(date -u -d "$night -1 day" +%F) [^ ]* void" "$G/STATUS" 2>/dev/null; then
  line="$line [second void in a row: open an issue]"
fi
echo "$night $line" >> "$G/STATUS"
find "$G" -maxdepth 1 -name '20*' -type d -mtime +14 | while read -r d; do
  python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('outcome') != 'green')" "$d/verdict.json" 2>/dev/null || continue
  rm -rf "$d/src" "$d/build" "$d/venv"
  find "$d" -name '*.cgns' -delete
done
