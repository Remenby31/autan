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
if [[ ! -r $state ]]; then
  echo "::warning::no record of the GPU's other tenants ($state missing); $memory"
  exit 0
fi
cat "$state"
tenants=$(awk -F'\t' '!/^#/ {printf "%s%s (%s, pid %s, %s)", sep, $4, $2, $1, $3; sep = "; "}' "$state")
if [[ -n $tenants ]]; then
  echo "::error::GPU taken by a foreign tenant: $tenants; $memory (as of $(head -1 "$state" | cut -c3-22))"
else
  echo "No foreign process on the GPU ($(head -1 "$state" | cut -c3-)); $memory"
fi
