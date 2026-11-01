#!/usr/bin/env bash
# After a GPU step failed on a shared Spark: names the GPU's processes outside Autan's containers,
# as autan-gpu-guard last listed them (every 30 s), and the host memory still available (GB10's
# GPU allocates from it). A foreign tenant becomes an error annotation; nothing is retried.
# Usage: bench/ci/gpu-tenants.sh   (AUTAN_GPU_LOCK: the lock, default /home/admin/autan-gpu/gpu.lock)
set -uo pipefail
lock=${AUTAN_GPU_LOCK:-/home/admin/autan-gpu/gpu.lock}
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
  echo "::error::GPU taken by a foreign tenant: $tenants; $memory (as of $stamp)"
elif [[ $header != *"nvidia-smi failed"* ]] && ((age <= 120)); then
  echo "No foreign process on the GPU (as of $stamp); $memory"
fi
