#!/usr/bin/env bash
# After a GPU step failed on a shared Spark: names the GPU's processes outside Autan's containers,
# as autan-gpu-guard last listed them (every 30 s), and the host memory still available (GB10's
# GPU allocates from it). A foreign tenant is an error annotation on a host that runs timed
# benchmarks (bench/ci/timed-hosts), where it refuses GPU steps; elsewhere a warning, since it
# is allowed there and may not be the cause (bench/ci/gpu-admit.sh). Nothing is retried.
# Usage: bench/ci/gpu-tenants.sh   (AUTAN_GPU_LOCK: the lock, default /home/admin/autan-gpu/gpu.lock)
set -uo pipefail
lock=${AUTAN_GPU_LOCK:-/home/admin/autan-gpu/gpu.lock}
host=${AUTAN_BENCH_HOST:-$(hostname)}
# The job can set the variable itself: a name outside [A-Za-z0-9._-] is never printed or trusted.
[[ $host =~ ^[A-Za-z0-9._-]{1,64}$ ]] || host=unnamed-host
level=warning
grep -qxF "$host" "$(cd "$(dirname "$0")" && pwd)/timed-hosts" && level=error
state=${lock%/*}/foreign-gpu
memory=$(awk '/^MemTotal/ {t = $2} /^MemAvailable/ {a = $2}
  END {printf "host memory available: %.1f of %.1f GiB", a / 1048576, t / 1048576}' /proc/meminfo)
# One read: the guard replaces the file atomically, so this is one consistent list.
if ! list=$(cat "$state" 2>/dev/null); then
  echo "::warning::GPU tenants unknown: no list from autan-gpu-guard ($state); $memory"
  exit 0
fi
header=$(head -1 <<<"$list")
stamp=${header:2:20}
age=$(($(date -u +%s) - $(date -u -d "$stamp" +%s 2>/dev/null || echo 0)))
# The guard's names hold no '#' or '[' and every row starts with a pid: no workflow command.
printf '%s\n' "$list"
if [[ $header == *"nvidia-smi failed"* ]]; then
  echo "::warning::GPU tenants unknown: nvidia-smi failed on the host at $stamp; $memory"
elif ((age > 120)); then
  echo "::warning::GPU tenants unknown: the guard's list is ${age} s old ($stamp); $memory"
fi
tenants=$(awk -F'\t' 'NR > 1 {printf "%s%s (%s, pid %s, %s)", sep, $4, $2, $1, $3; sep = "; "}' <<<"$list")
if [[ -n $tenants ]]; then
  echo "::$level::GPU shared with a foreign tenant on $host: $tenants; $memory (as of $stamp)"
elif [[ $header != *"nvidia-smi failed"* ]] && ((age <= 120)); then
  echo "No foreign process on the GPU (as of $stamp); $memory"
fi
