#!/usr/bin/env bash
# #703 diagnosis (declared before any run, issue #703 comment 6818316068): the generator's slow GPU
# tests on a GB10 runner, inside the runner's own container. Proof branch only, never merged.
#   slow-gpu-703.sh LIB OUT ORDER     (run from codegen/; ORDER = forward | reverse)
# Holds, in order: the CI reproduction gate (shard 1 as in CI), shard 1 under three arms x two
# passes (forward: base, pinA, pinB; reverse: pinB, pinA, base), shards 2-4 base, then the
# fast-suite control (base, pinB). Each hold is exactly CI's `autan-gpu-lock --untimed -t 600`,
# one measurement per hold, 120 s between holds; nothing loops under the lock.
# Inside an arm hold every shard test runs in its own pytest process, its start and end appended
# (flushed) to OUT/tests.tsv before the next starts: a test the hold cuts keeps its start line.
set -uo pipefail
lib=$1 out=$2 order=$3
mkdir -p "$out"
export AUTAN_SPIKE_LIB=$lib
lock=../bench/lock/autan-gpu-lock
runner=${RUNNER_NAME:-unknown}
case $runner in
  spark-75d6) pinA=1-4 pinB=16-19 ;;
  spark-5512) pinA=10-12,15 pinB=16-19 ;;
  *) echo "unknown runner $runner"; exit 2 ;;
esac
{
  echo "runner $runner"; echo "nproc $(nproc)"; grep Cpus_allowed_list /proc/self/status
  echo "cpuset.cpus.effective $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)"
  env | grep -E '^(OMP|OPENBLAS|MKL|GOMP|KMP)_' || echo "no thread-pool variable set"
  free -m; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv
  [ "$runner" = spark-5512 ] && echo "note: spark-5512 runs with Qwen up (cores 13,14 excluded, ~20 GiB GPU memory)"
  git -C .. rev-parse HEAD
} > "$out/env.txt" 2>&1

uv run --locked pytest --backend gpu -m slow --collect-only -q | grep '::' > "$out/slow.txt"
for k in 0 1 2 3; do awk -v k="$k" 'NR % 4 == k' "$out/slow.txt" > "$out/shard$((k + 1)).txt"; done

# Telemetry every 1 s while a hold runs: per-CPU /proc/stat, CPU pressure, the allowed CPUs of
# every thread of every pytest process, and the GPU (no other container is inspected).
sample() {
  while :; do
    t=$(date -u +%FT%T.%3NZ)
    echo "T $t"
    grep '^cpu' /proc/stat
    echo "PSI $(head -1 /proc/pressure/cpu 2>/dev/null)"
    for p in $(pgrep -f 'pytest --backend gpu'); do
      for th in /proc/"$p"/task/*; do
        echo "THREAD $p ${th##*/} $(awk '/Cpus_allowed_list/{print $2}' "$th"/status 2>/dev/null) psr=$(awk '{print $39}' "$th"/stat 2>/dev/null)"
      done
    done
    echo "GPU $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,memory.used,power.draw --format=csv,noheader 2>/dev/null) apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
    sleep 1
  done
}

gap() { sleep 120; }

# One hold: TAG, then the command (run by bash -c under the lock). Records the lock's exit, the
# hold's wall time and the telemetry.
hold() {
  local tag=$1 cmd=$2 s0 s1 rc
  sample > "$out/samples-$tag.txt" 2>/dev/null &
  local sp=$!
  s0=$(date +%s.%N)
  "$lock" --untimed -t 600 -m "703 diagnosis $tag" -- bash -c "$cmd" > "$out/hold-$tag.log" 2>&1
  rc=$?
  s1=$(date +%s.%N)
  kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
  printf '%s\t%s\t%s\t%s\n' "$tag" "$rc" "$(python3 -c "print(round($s1 - $s0, 2))")" "$runner" >> "$out/holds.tsv"
}

# Per-test loop for an arm hold: PREFIX (taskset or empty), TAG, SHARD file.
per_test() {
  local prefix=$1 tag=$2 shard=$3
  cat <<EOF
while IFS= read -r id; do
  printf 'START\t%s\t%s\t%s\n' "$tag" "\$(date -u +%FT%T.%3NZ)" "\$id" >> "$out/tests.tsv"
  t0=\$(date +%s.%N)
  timeout 300 $prefix uv run --locked pytest --backend gpu -q -p no:cacheprovider "\$id" >> "$out/log-$tag.txt" 2>&1
  rc=\$?
  t1=\$(date +%s.%N)
  printf 'END\t%s\t%s\t%s\t%s\t%s\n' "$tag" "\$(date -u +%FT%T.%3NZ)" "\$id" "\$rc" "\$(python3 -c "print(round(\$t1 - \$t0, 3))")" >> "$out/tests.tsv"
done < "$shard"
EOF
}

prefix_of() { case $1 in base) echo "" ;; pinA) echo "taskset -c $pinA" ;; pinB) echo "taskset -c $pinB" ;; esac; }

# 1. The reproduction gate: shard 1 exactly as in CI (one pytest process).
hold gate "mapfile -t s < '$out/shard1.txt'; uv run --locked pytest --backend gpu -q \"\${s[@]}\" > '$out/gate-pytest.log' 2>&1"
# 2. Shard 1, three arms x two passes.
if [ "$order" = reverse ]; then arms="pinB pinA base"; else arms="base pinA pinB"; fi
for arm in $arms; do
  for pass in 1 2; do
    gap
    hold "s1-$arm-$pass" "$(per_test "$(prefix_of "$arm")" "s1-$arm-$pass" "$out/shard1.txt")"
  done
done
# 3. Shards 2-4, base, one pass.
for k in 2 3 4; do
  gap
  hold "s$k-base-1" "$(per_test "" "s$k-base-1" "$out/shard$k.txt")"
done
# 4. The fast-suite control, as in CI (one process), base then pinB.
for arm in base pinB; do
  gap
  hold "fast-$arm" "$(prefix_of "$arm") uv run --locked pytest --backend gpu -m 'not slow' -q > '$out/fast-$arm.log' 2>&1"
done
echo done >> "$out/holds.tsv"
