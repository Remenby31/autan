#!/usr/bin/env bash
# Admits a CI GPU step on a shared Spark, before it starts.
# Usage: bench/ci/gpu-admit.sh NEED_GIB [WAIT_S]
# Lists the GPU's tenants outside Autan's containers, as autan-gpu-guard last listed them, in
# the job's summary. On a host that runs timed benchmarks (bench/ci/timed-hosts), a foreign
# tenant is a refusal: a measurement beside it would not be evidence. Elsewhere it is not: the
# step waits, at most WAIT_S seconds (default 600), until the host's MemAvailable reaches the
# memory floor of autan-gpu-lock (24 GiB) plus the step's declared NEED_GIB (GB10's GPU
# allocates from host memory), and fails clearly if it never does: not the change's fault.
# The host is $AUTAN_BENCH_HOST (the runners set it), else the machine's name.
set -uo pipefail
need=${1:-} wait_s=${2:-600}
[[ $need =~ ^[0-9]+$ && $wait_s =~ ^[0-9]+$ ]] || {
  echo "::error::usage: bench/ci/gpu-admit.sh NEED_GIB [WAIT_S], whole numbers" >&2
  exit 2
}
here=$(cd "$(dirname "$0")" && pwd)
lock=${AUTAN_GPU_LOCK:-/home/admin/autan-gpu/gpu.lock}
meminfo=${AUTAN_MEMINFO:-/proc/meminfo}
host=${AUTAN_BENCH_HOST:-$(hostname)}
floor_gib=24
summary=${GITHUB_STEP_SUMMARY:-/dev/null}

available_gib() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' "$meminfo"; }

# The guard's list: a header line, then pid, container, MiB and process name, tab-separated. Its
# names hold no '#' or '[' (bench/lock/autan-gpu-guard), so they cannot be workflow commands.
tenants=""
if list=$(cat "${lock%/*}/foreign-gpu" 2>/dev/null); then
  tenants=$(awk -F'\t' 'NR > 1 {printf "%s%s (%s, %s)", sep, $4, $2, $3; sep = "; "}' <<<"$list")
  echo "GPU tenants outside Autan on $host: ${tenants:-none}" | tee -a "$summary"
else
  echo "GPU tenants outside Autan on $host: unknown (no list from autan-gpu-guard)" | tee -a "$summary"
fi

if grep -qxF "$host" "$here/timed-hosts" && [[ -n $tenants ]]; then
  echo "::error::GPU taken by a foreign tenant on $host, which runs timed benchmarks: $tenants"
  exit 1
fi

target=$((floor_gib + need))
deadline=$((SECONDS + wait_s))
announced=0
while :; do
  free=$(available_gib)
  if [[ -n $free ]] && ((free >= target)); then
    echo "Host memory available on $host: $free GiB, at least the floor $floor_gib + this step's $need GiB" | tee -a "$summary"
    exit 0
  fi
  if ((SECONDS >= deadline)); then
    echo "::error::WAITING FOR MEMORY timed out after ${wait_s} s on $host: ${free:-unknown} GiB available, the step needs the floor $floor_gib + $need GiB. Not this change's fault: rerun the job later."
    exit 1
  fi
  if ((announced == 0 || SECONDS - announced >= 60)); then
    echo "WAITING FOR MEMORY on $host: ${free:-unknown} GiB available, needs $target GiB (at most ${wait_s} s)"
    announced=$SECONDS
  fi
  sleep "${AUTAN_ADMIT_POLL_S:-10}"
done
