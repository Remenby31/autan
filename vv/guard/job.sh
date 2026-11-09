#!/usr/bin/env bash
# Nightly guard (#611), the queue job: build the pinned commit, run the guard case twice from one
# binary in fresh directories, then `autan-vv guard verdict`. The night's deadline, 06:30 UTC the
# day after NIGHT, stops the build or the runs (TERM, then KILL 300 s later). Infrastructure evidence
# (void) is the deadline firing before run 1 started (a late queue start) or a cgroup OOM kill that
# this job's own counter records; anything else that fails leaves the runs incomplete, and the
# verdict is red.
set -u
cd /work || exit 1
trap 'chown -R 1000:1000 /work' EXIT
export HOME=/tmp UV_CACHE_DIR=/tmp/uvc UV_PROJECT_ENVIRONMENT=/work/venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 AUTAN_RUN=/work/build/runtime/autan-run
log() { echo "$(date -u +%FT%TZ) $*" >> /work/driver.log; }
commit=$(cat COMMIT)
oom() { awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0; }
oom0=$(oom)
start=$(date -u +%FT%TZ)
started=0
deadline=$(date -u -d "$(cat NIGHT) + 1 day 06:30" +%s)
( sleep $(( deadline - $(date -u +%s) > 0 ? deadline - $(date -u +%s) : 0 )); touch /work/DEADLINE
  pkill -TERM -f "python /work/venv/bin/autan run"; pkill -TERM -x ninja; pkill -TERM -x cmake
  sleep 300; pkill -KILL -f "autan-run|python /work/venv/bin/autan run"; pkill -KILL -x ninja ) &
killer=$!
[ "$(date -u +%s)" -lt "$deadline" ] \
  && cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAUTAN_COMMIT="$commit" -DAUTAN_DIRTY=OFF > cmake.log 2>&1 \
  && nice -n 19 cmake --build build -j 1 --target autan-run autan-mesh-info > build.log 2>&1 \
  && (cd src/vv && uv sync --locked -q) \
  && build/io/autan-mesh-info --json flatplate-137x97.msh > mesh-info.json \
  || log "build or setup failed"
infra=""
rc=(99 99)
for k in 1 2; do
  [ -x venv/bin/autan ] && [ ! -e /work/DEADLINE ] || break
  rm -rf "run$k"; mkdir "run$k"
  log "run $k start"
  started=1
  venv/bin/autan run case.toml --mesh-info mesh-info.json --backend cpu --runs "run$k" > "run$k.log" 2>&1
  rc[$((k - 1))]=$?
  log "run $k exit ${rc[$((k - 1))]}"
done
[ "$(date -u +%s)" -ge "$deadline" ] && [ "$started" = 0 ] \
  && infra="deadline before run 1 started; queue start $start; pending since $(cat PUBLISHED 2>/dev/null || echo unknown)"
[ "$(oom)" -gt "$oom0" ] && infra="${infra:+$infra; }cgroup OOM kill (memory.events oom_kill $oom0 -> $(oom))"
kill "$killer" 2>/dev/null
pick() { local n; n=$(ls -d "$1"/run-* 2>/dev/null | wc -l); [ "$n" = 1 ] && ls -d "$1"/run-* || echo "/work/$1/not-one-run"; }
args=(guard verdict /work --commit "$commit" --run1 "$(pick run1)" --run2 "$(pick run2)" --rc1 "${rc[0]}" --rc2 "${rc[1]}")
[ -e HAS_544 ] && args+=(--has-544)
[ -e previous.json ] && args+=(--previous previous.json)
[ -e previous-history.jsonl ] && args+=(--previous-history previous-history.jsonl)
[ -n "$infra" ] && args+=(--infra "$infra")
if [ -x venv/bin/autan-vv ]; then
  venv/bin/autan-vv "${args[@]}" > verdict.log 2>&1
else
  printf '{"schema": "autan.vv-guard-night/1", "commit": "%s", "outcome": "%s", "reason": "build or setup failed"%s}\n' \
    "$commit" "$([ -n "$infra" ] && echo void || echo red)" "$([ -n "$infra" ] && echo ", \"infra\": \"$infra\"")" > verdict.json
fi
log "verdict $(tail -1 verdict.log 2>/dev/null)"
grep -q '"outcome": "green"' verdict.json && rm -rf venv build/CMakeFiles
