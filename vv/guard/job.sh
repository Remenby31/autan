#!/usr/bin/env bash
# Nightly guard (#611), the queue job: build the pinned commit, run the guard case twice at once
# from one binary in fresh directories, then `autan-vv guard verdict`. The night's deadline, 06:30
# UTC the day after NIGHT, stops the build or the runs (TERM, then KILL 300 s later). The case runs
# a fixed 20 000 steps, so missing the deadline measures the queue and the host's load, not the
# numerics: it is infrastructure evidence (void), as are a cgroup OOM kill that this job's own
# counter records and a drain that wrote DRAINED in the night's directory before stopping the runs.
# Anything else that fails leaves the runs incomplete, and the verdict is red.
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
deadline=$(date -u -d "$(cat NIGHT) + 1 day 06:30" +%s)
( sleep $(( deadline - $(date -u +%s) > 0 ? deadline - $(date -u +%s) : 0 )); touch /work/DEADLINE
  pkill -TERM -f "^/work/venv/bin/python /work/venv/bin/autan run"; pkill -TERM -x ninja; pkill -TERM -x cmake; pkill -TERM -x uv
  pkill -TERM -x autan-mesh-info
  sleep 300; pkill -KILL -f "autan-run|^/work/venv/bin/python /work/venv/bin/autan run"; pkill -KILL -x ninja; pkill -KILL -x uv
  pkill -KILL -x autan-mesh-info ) &
killer=$!
setup=1
[ "$(date -u +%s)" -lt "$deadline" ] \
  && cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAUTAN_COMMIT="$commit" -DAUTAN_DIRTY=OFF > cmake.log 2>&1 \
  && nice -n 19 cmake --build build -j 1 --target autan-run autan-mesh-info > build.log 2>&1 \
  && (cd src/vv && uv sync --locked -q) \
  && build/io/autan-mesh-info --json flatplate-137x97.msh > mesh-info.json \
  && setup=0 \
  || log "build or setup failed"
infra=""
rc=(99 99)
runs_start=""
end=""
if [ "$setup" = 0 ] && [ "$(date -u +%s)" -lt "$deadline" ]; then
  runs_start=$(date -u +%s)
  log "runs start"
  for k in 1 2; do
    rm -rf "run$k"; mkdir "run$k"
    /work/venv/bin/autan run case.toml --mesh-info mesh-info.json --backend cpu --runs "run$k" > "run$k.log" 2>&1 &
    pid[k]=$!
  done
  wait "${pid[1]}"; rc[0]=$?
  wait "${pid[2]}"; rc[1]=$?
  end=$(date -u +%s)
  log "runs exit ${rc[0]} ${rc[1]}"
fi
steps() { cat "$1"/run-*/history.jsonl 2>/dev/null | wc -l; }
if [ "${end:-$(date -u +%s)}" -ge "$deadline" ] && [ "${rc[*]}" != "0 0" ]; then
  queue="queue start $start; pending since $(cat PUBLISHED 2>/dev/null || echo unknown)"
  if [ -z "$runs_start" ]; then
    infra="deadline before the runs started; $queue"
  else
    n1=$(steps run1) n2=$(steps run2) min=$(( (deadline - runs_start) / 60 ))
    infra="deadline during the runs: steps $n1 and $n2 of 20000 in $min min ($(( n1 / (min > 0 ? min : 1) )) steps/min); $queue"
  fi
fi
[ "$(oom)" -gt "$oom0" ] && infra="${infra:+$infra; }cgroup OOM kill (memory.events oom_kill $oom0 -> $(oom))"
# A drain writes DRAINED (who, when, why) before its TERM: runs it stopped are infrastructure.
[ -e /work/DRAINED ] && [ "${rc[*]}" != "0 0" ] \
  && infra="${infra:+$infra; }drained: $(tr -d '"\\' < /work/DRAINED | tr '\n' ' ' | cut -c1-300)"
kill "$killer" 2>/dev/null
pick() { local n; n=$(ls -d "$1"/run-* 2>/dev/null | wc -l); [ "$n" = 1 ] && ls -d "$1"/run-* || echo "/work/$1/not-one-run"; }
args=(guard verdict /work --commit "$commit" --run1 "$(pick run1)" --run2 "$(pick run2)" --rc1 "${rc[0]}" --rc2 "${rc[1]}")
[ -e HAS_544 ] && args+=(--has-544)
[ -e previous.json ] && args+=(--previous previous.json)
[ -e previous-history.jsonl ] && args+=(--previous-history previous-history.jsonl)
[ -n "$infra" ] && args+=(--infra "$infra")
[ "$setup" = 0 ] && venv/bin/autan-vv "${args[@]}" > verdict.log 2>&1
if ! grep -q '"outcome": ' verdict.json 2>/dev/null; then
  rm -f verdict.json
  printf '{"schema": "autan.vv-guard-night/1", "commit": "%s", "outcome": "%s", "reason": "build, setup or verdict failed"%s}\n' \
    "$commit" "$([ -n "$infra" ] && echo void || echo red)" "$([ -n "$infra" ] && echo ", \"infra\": \"$infra\"")" > verdict.json.tmp
  mv verdict.json.tmp verdict.json
fi
log "verdict $(tail -1 verdict.log 2>/dev/null)"
grep -q '"outcome": "green"' verdict.json && rm -rf venv build/CMakeFiles
