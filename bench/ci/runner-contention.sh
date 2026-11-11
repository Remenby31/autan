#!/usr/bin/env bash
# Runner core-contention experiment (#697, declared before any run): MPI test #265 under four arms,
# three interleaved rounds, inside the runner's own container. Proof branch only, never merged.
# Writes one JSON line per run to $OUT/runs.jsonl and the affinity samples to $OUT/affinity-*.txt.
set -uo pipefail
build=$1 out=$2
mkdir -p "$out"
test_re='^Navier-Stokes at ghost depth 3 repeats bit for bit at a fixed rank count$'
busy() { awk '/^cpu /{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
for round in 1 2 3; do
  for arm in baseline B C BC; do
    prefix=() env_yield=()
    case $arm in B | BC) prefix=(taskset -c 16-19) ;; esac
    case $arm in C | BC) env_yield=(OMPI_MCA_mpi_yield_when_idle=1) ;; esac
    tag=$arm-r$round
    # Affinity of every autan-run rank, every 2 s: pid, current CPU (psr), allowed CPUs.
    ( while :; do
        for p in $(pgrep -f autan-run); do
          printf '%s %s %s %s\n' "$(date -u +%T)" "$p" "$(ps -o psr= -p "$p" 2>/dev/null)" \
            "$(awk '/Cpus_allowed_list/{print $2}' /proc/"$p"/status 2>/dev/null)"
        done
        sleep 2
      done ) > "$out/affinity-$tag.txt" 2>/dev/null &
    sampler=$!
    read -r b0 t0 < <(busy)
    s0=$(date +%s.%N)
    env "${env_yield[@]}" "${prefix[@]}" ctest --test-dir "$build" -R "$test_re" --output-on-failure \
      > "$out/ctest-$tag.log" 2>&1
    rc=$?
    s1=$(date +%s.%N)
    read -r b1 t1 < <(busy)
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    wall=$(python3 -c "print(round($s1 - $s0, 2))")
    host_busy=$(python3 -c "print(round(100 * ($b1 - $b0) / max($t1 - $t0, 1), 1))")
    timeout=$(grep -c "Timeout" "$out/ctest-$tag.log")
    printf '{"runner":"%s","arm":"%s","round":%d,"ctest_exit":%d,"timeout":%d,"wall_s":%s,"host_busy_pct":%s,"ranks_cpus_allowed":"%s"}\n' \
      "${RUNNER_NAME:-unknown}" "$arm" "$round" "$rc" "$timeout" "$wall" "$host_busy" \
      "$(awk '{print $4}' "$out/affinity-$tag.txt" | sort -u | tr '\n' ' ')" >> "$out/runs.jsonl"
  done
done
