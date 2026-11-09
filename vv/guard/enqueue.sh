#!/usr/bin/env bash
# Nightly guard (#611), run by autan-guard-enqueue.timer on spark-5512 at 23:30 UTC: pins tonight's
# origin/main, prepares the night's directory and writes its job file into the queue. No build or
# run on the host. A night that already has a directory is not prepared again.
set -euo pipefail
G=${AUTAN_GUARD_ROOT:-$HOME/cfd/guard}
night=$(date -u +%F)
dir=$G/$night
[ -e "$dir" ] && exit 0
git -C "$G/repo" fetch -q origin main
sha=$(git -C "$G/repo" rev-parse origin/main)
tmp=$(mktemp -d "$G/.prep-XXXXXX")
mkdir "$tmp/src"
git -C "$G/repo" archive "$sha" | tar xf - -C "$tmp/src"
printf 'COMMIT = "%s"\nDIRTY = False\n' "$sha" > "$tmp/src/py/autan/_build_info.py"
echo "$sha" > "$tmp/COMMIT"
cp "$tmp/src/vv/guard/flatplate-case.toml" "$tmp/case.toml"
cp "$G/inputs/flatplate-137x97.msh" "$tmp/"
(cd "$tmp" && sha256sum -c --quiet "$G/inputs/mesh.sha256")
# #544 is in the night's commit when its contract sentence is in the night's case-schema.md.
if grep -q "the largest of the momentum components' residuals in the reference" \
    "$tmp/src/docs/contracts/case-schema.md"; then touch "$tmp/HAS_544"; fi
last=$(for v in "$G"/20*/verdict.json; do [ -e "$v" ] && grep -q '"outcome": "green"' "$v" && echo "$v"; done | sort | tail -1)
[ -n "$last" ] && cp "$last" "$tmp/previous.json"
mv "$tmp" "$dir"
job=$(mktemp "$G/.job-XXXXXX")
printf 'ID=guard-%s\nOWNER=CFD - Comparatif SU2 (V&V)\nDECL=https://github.com/Remenby31/autan/issues/611#issuecomment-6731459780\nCAP_GIB=6\nRESERVE_GIB=3\nEST_MIN=150\nTHREADS=1\nGPU=0\nWORKDIR=%s\nCMD=bash /work/src/vv/guard/job.sh\n' \
  "$night" "$dir" > "$job"
mv "$job" "$HOME/cfd/queue/pending/10-guard-$night.job"
