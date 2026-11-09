#!/usr/bin/env bash
# Nightly guard (#611), the queue job: build the pinned commit, run the guard case twice from one
# binary, then `autan-vv guard verdict`. A cgroup OOM kill is the only infrastructure evidence it
# names (the night is void); every other failure leaves the runs to the verdict, which is red.
set -u
cd /work || exit 1
sdk=""
trap '[ -n "$sdk" ] && kill -TERM "$sdk" 2>/dev/null; wait; exit 143' TERM
trap 'chown -R 1000:1000 /work' EXIT
export HOME=/tmp UV_CACHE_DIR=/tmp/uvc UV_PROJECT_ENVIRONMENT=/work/venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 AUTAN_RUN=/work/build/runtime/autan-run
log() { echo "$(date -u +%FT%TZ) $*" >> /work/driver.log; }
commit=$(cat COMMIT)
cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAUTAN_COMMIT="$commit" -DAUTAN_DIRTY=OFF > cmake.log 2>&1 \
  && nice -n 19 cmake --build build -j 1 --target autan-run autan-mesh-info > build.log 2>&1 \
  && (cd src/vv && uv sync --locked -q) \
  && build/io/autan-mesh-info --json flatplate-137x97.msh > mesh-info.json \
  || log "build or setup failed"
deadline=$(date -u -d "$(date -u +%F) 06:30" +%s)
[ "$deadline" -gt "$(date -u +%s)" ] || deadline=$(date -u -d "tomorrow 06:30" +%s)
infra=""
for k in 1 2; do
  [ -x venv/bin/autan ] || break
  mkdir -p "run$k"
  log "run $k start"
  venv/bin/autan run case.toml --mesh-info mesh-info.json --backend cpu --runs "run$k" > "run$k.log" 2>&1 &
  sdk=$!
  ( sleep $(( deadline - $(date -u +%s) )); kill -TERM "$sdk" 2>/dev/null ) &
  timer=$!
  wait "$sdk"; rc=$?
  kill "$timer" 2>/dev/null
  sdk=""
  log "run $k exit $rc"
  oom=$(awk '$1 == "oom_kill" {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
  if [ "${oom:-0}" -gt 0 ]; then infra="cgroup OOM kill (memory.events oom_kill $oom)"; break; fi
done
args=(guard verdict /work --commit "$commit")
r1=$(ls -d run1/run-* 2>/dev/null | head -1); r2=$(ls -d run2/run-* 2>/dev/null | head -1)
args+=(--run1 "${r1:-/work/run1/none}" --run2 "${r2:-/work/run2/none}")
[ -e HAS_544 ] && args+=(--has-544)
[ -e previous.json ] && args+=(--previous previous.json)
[ -n "$infra" ] && args+=(--infra "$infra")
if [ -x venv/bin/autan-vv ]; then
  venv/bin/autan-vv "${args[@]}" > verdict.log 2>&1
else
  printf '{"schema": "autan.vv-guard-night/1", "commit": "%s", "outcome": "red", "reason": "build or setup failed"}\n' "$commit" > verdict.json
fi
log "verdict $(cat verdict.log 2>/dev/null | tail -1)"
rm -rf build/CMakeFiles venv
