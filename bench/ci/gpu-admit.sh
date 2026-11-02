#!/usr/bin/env bash
# Admits a CI GPU step on a shared Spark, before it starts.
# Usage: bench/ci/gpu-admit.sh NEED_GIB [WAIT_S]
# Lists the GPU's tenants outside Autan's containers, as autan-gpu-guard last listed them, in
# the job's summary. On a host that runs timed benchmarks (bench/ci/timed-hosts), a foreign
# tenant is a refusal: a measurement beside it would not be evidence. Elsewhere it is not: the
# step waits, at most WAIT_S seconds (default 600), until the host's MemAvailable reaches the
# memory floor of autan-gpu-lock (24 GiB) plus the step's declared NEED_GIB (GB10's GPU
# allocates from host memory), and fails clearly if it never does: not the change's fault.
# A list the guard did not refresh within 120 s, or could not take, is unknown: said, and no
# refusal. The host is $AUTAN_BENCH_HOST (the runners set it), else the machine's name.
set -uo pipefail
need=${1:-} wait_s=${2:-600}
# Decimal, bounded: a leading zero must not read as octal, nor a huge value overflow.
if [[ ! $need =~ ^[0-9]{1,4}$ || ! $wait_s =~ ^[0-9]{1,5}$ ]]; then
  echo "::error::usage: bench/ci/gpu-admit.sh NEED_GIB [WAIT_S], whole numbers" >&2
  exit 2
fi
need=$((10#$need)) wait_s=$((10#$wait_s))
here=$(cd "$(dirname "$0")" && pwd)
lock=${AUTAN_GPU_LOCK:-/home/admin/autan-gpu/gpu.lock}
meminfo=${AUTAN_MEMINFO:-/proc/meminfo}
# The job can set the variable itself: a name outside [A-Za-z0-9._-] is never printed or trusted.
host=${AUTAN_BENCH_HOST:-$(hostname)}
[[ $host =~ ^[A-Za-z0-9._-]{1,64}$ ]] || host=unnamed-host
floor_gib=24
max_age_s=120
summary=${GITHUB_STEP_SUMMARY:-/dev/null}

# MemAvailable in KiB, and the target, compared exactly: no rounding at the edge.
available_kib() { awk '/^MemAvailable:/ {print $2}' "$meminfo" 2>/dev/null; }
gib() { awk -v k="$1" 'BEGIN {printf "%.1f", k / 1048576}'; }

# The guard's list: "# <UTC time> <n> foreign" or "# <UTC time> nvidia-smi failed", then pid,
# container, MiB and process name, tab-separated. Its names hold no '#' or '[' and no line
# starts with "::" (bench/lock/autan-gpu-guard), so they cannot be workflow commands.
tenants="" known=0
if list=$(cat "${lock%/*}/foreign-gpu" 2>/dev/null); then
  header=$(head -1 <<<"$list")
  stamp=${header:2:20}
  taken=$(date -u -d "$stamp" +%s 2>/dev/null || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$stamp" +%s 2>/dev/null || echo 0)
  age=$(($(date -u +%s) - taken))
  if [[ $header == *"nvidia-smi failed"* ]]; then
    state="unknown (the guard's nvidia-smi failed at $stamp)"
  elif ((age > max_age_s)); then
    state="unknown (the guard's list is ${age} s old)"
  else
    known=1
    tenants=$(awk -F'\t' 'NR > 1 {printf "%s%s (%s, %s)", sep, $4, $2, $3; sep = "; "}' <<<"$list")
    state=${tenants:-none}
  fi
else
  state="unknown (no list from autan-gpu-guard)"
fi
echo "GPU tenants outside Autan on $host: $state" | tee -a "$summary"

if ((known)) && [[ -n $tenants ]] && grep -qxF "$host" "$here/timed-hosts"; then
  echo "::error::GPU taken by a foreign tenant on $host, which runs timed benchmarks: $tenants"
  exit 1
fi

target_kib=$(((floor_gib + need) * 1048576))
deadline=$((SECONDS + wait_s))
announced=-60
poll=${AUTAN_ADMIT_POLL_S:-10}
while :; do
  free=$(available_kib)
  if [[ $free =~ ^[0-9]+$ ]] && ((free >= target_kib)); then
    echo "Host memory available on $host: $(gib "$free") GiB, at least the floor $floor_gib + this step's $need GiB" | tee -a "$summary"
    exit 0
  fi
  shown=$([[ $free =~ ^[0-9]+$ ]] && gib "$free" || echo unknown)
  remaining=$((deadline - SECONDS))
  if ((remaining <= 0)); then
    echo "::error::WAITING FOR MEMORY timed out after ${wait_s} s on $host: $shown GiB available, the step needs the floor $floor_gib + $need GiB. Not this change's fault: rerun the job later."
    exit 1
  fi
  if ((SECONDS - announced >= 60)); then
    echo "WAITING FOR MEMORY on $host: $shown GiB available, needs $((floor_gib + need)) GiB (at most ${wait_s} s)"
    announced=$SECONDS
  fi
  # Never past the deadline: the last check falls on it.
  if awk -v p="$poll" -v r="$remaining" 'BEGIN {exit !(p > r)}'; then poll_now=$remaining; else poll_now=$poll; fi
  sleep "$poll_now"
done
