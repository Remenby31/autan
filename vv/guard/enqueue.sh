#!/usr/bin/env bash
# Nightly guard (#611), run by autan-guard-enqueue.timer on spark-5512 at 23:30 UTC (Persistent:
# a missed run catches up). Prepares the night's directory from tonight's origin/main and publishes
# its job file into the queue. Nothing is built or run on the host. A night is the date of the
# latest 23:30 UTC slot; a night already published is not prepared again; one prepared but not
# published is published. Refused inputs give the night a red verdict without a job.
set -euo pipefail
G=${AUTAN_GUARD_ROOT:-$HOME/cfd/vv-guard}
MESH_SHA256=ceb75abce31f12437e954107467e3e2ca4acbf66b1059b4a24d910b86ae7a59e
exec 9> "$G/.enqueue.lock"
flock -n 9 || exit 0
if [ "$(date -u +%H%M)" -lt 2330 ]; then night=$(date -u -d yesterday +%F); else night=$(date -u +%F); fi
dir=$G/$night
publish() {
  [ -e "$dir/ENQUEUED" ] && return 0
  local name="10-guard-$night.job" q=$HOME/cfd/queue
  if [ ! -e "$q/pending/$name" ] && [ ! -e "$q/running/$name" ] && [ ! -e "$q/done/$name" ]; then
    cp "$dir/job.file" "$q/pending/.$name.tmp" && mv "$q/pending/.$name.tmp" "$q/pending/$name"
  fi
  touch "$dir/ENQUEUED"
}
refuse() {
  python3 -c "import json,sys; print(json.dumps({'schema': 'autan.vv-guard-night/1', 'commit': sys.argv[1], 'outcome': 'red', 'reason': 'refused input: ' + sys.argv[2]}))" \
    "${sha:-unknown}" "$1" > "$tmp/verdict.json"
  mv "$tmp" "$dir"
  exit 0
}
if [ -e "$dir" ]; then [ -e "$dir/job.file" ] && publish; exit 0; fi
tmp=$(mktemp -d "$G/.prep-XXXXXX")
echo "$night" > "$tmp/NIGHT"
git -C "$G/repo" fetch -q origin main || refuse "git fetch failed"
sha=$(git -C "$G/repo" rev-parse origin/main) || refuse "rev-parse failed"
echo "$sha" > "$tmp/COMMIT"
mkdir "$tmp/src"
git -C "$G/repo" archive "$sha" | tar xf - -C "$tmp/src" || refuse "git archive failed"
printf 'COMMIT = "%s"\nDIRTY = False\n' "$sha" > "$tmp/src/py/autan/_build_info.py"
cp "$tmp/src/vv/guard/flatplate-case.toml" "$tmp/case.toml" || refuse "no guard case in the pinned commit"
cp "$G/inputs/flatplate-137x97.msh" "$tmp/" || refuse "no mesh in $G/inputs"
[ "$(sha256sum "$tmp/flatplate-137x97.msh" | cut -c1-64)" = "$MESH_SHA256" ] || refuse "mesh sha256 is not $MESH_SHA256"
pr=$(sed -n 's/^PR=\([0-9]*\)$/\1/p' "$tmp/src/vv/guard/has544.conf" 2>/dev/null || true)
if [ -n "$pr" ]; then
  hit=$(git -C "$G/repo" log --format=%H --grep="(#$pr)\$" "$sha") || refuse "git log failed"
  [ -n "$hit" ] && touch "$tmp/HAS_544"
fi
last=""
for v in "$G"/20*/verdict.json; do
  [ -e "$v" ] || continue
  if python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('outcome') != 'green')" "$v" 2>/dev/null; then last=$v; fi
done
if [ -n "$last" ]; then
  cp "$last" "$tmp/previous.json"
  h=$(ls -d "$(dirname "$last")"/run1/run-*/history.jsonl 2>/dev/null | head -1 || true)
  [ -n "$h" ] && cp "$h" "$tmp/previous-history.jsonl"
else
  cp "$tmp/src/vv/guard/reference.json" "$tmp/previous.json"
  [ -e "$G/inputs/reference-history.jsonl" ] && cp "$G/inputs/reference-history.jsonl" "$tmp/previous-history.jsonl"
fi
printf 'ID=guard-%s\nOWNER=CFD - Comparatif SU2 (V&V)\nDECL=https://github.com/Remenby31/autan/issues/611#issuecomment-6731459780\nCAP_GIB=6\nRESERVE_GIB=3\nEST_MIN=150\nTHREADS=1\nGPU=0\nWORKDIR=%s\nCMD=bash /work/src/vv/guard/job.sh\n' \
  "$night" "$dir" > "$tmp/job.file"
mv "$tmp" "$dir"
publish
