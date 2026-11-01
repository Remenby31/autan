#!/usr/bin/env bash
# Gives a CI job priority on the Spark's GPU lock: holds ci.lock (shared) in the background, so
# that agents' autan-gpu-lock and autan-bench wait behind CI (docs/infra/bench.md).
# Usage: bench/ci/priority.sh start MINUTES   (before the job's first GPU step; at most MINUTES)
#        bench/ci/priority.sh stop            (last step, with if: always())
set -euo pipefail
lock=${AUTAN_GPU_LOCK:-/home/admin/autan-gpu/gpu.lock}
ci_lock=$(dirname "$lock")/ci.lock
pid_file=${RUNNER_TEMP:?}/autan-ci-priority.pid

case ${1:-} in
  start)
    [[ ${2:-} =~ ^[1-9][0-9]{0,3}$ ]] || { echo "usage: $0 start MINUTES" >&2; exit 2; }
    if [[ ! -f $ci_lock ]]; then
      echo "::error::$ci_lock is missing: run docker/runner/deploy.sh start on this Spark" >&2
      exit 1
    fi
    # setsid: the holder is its own process group, stopped as a whole; -o: sleep does not keep
    # the lock if flock is killed. Bounded by MINUTES if the stop step never runs.
    setsid flock -o -s "$ci_lock" sleep $(($2 * 60)) </dev/null >/dev/null 2>&1 &
    echo $! >"$pid_file"
    # Held once an exclusive attempt fails.
    for _ in $(seq 100); do
      flock -n -x "$ci_lock" true 2>/dev/null || exit 0
      sleep 0.05
    done
    echo "::error::could not hold $ci_lock" >&2
    exit 1 ;;
  stop)
    [[ -f $pid_file ]] || exit 0
    kill -TERM -- "-$(cat "$pid_file")" 2>/dev/null || true
    rm -f "$pid_file" ;;
  *) echo "usage: $0 start MINUTES | stop" >&2; exit 2 ;;
esac
