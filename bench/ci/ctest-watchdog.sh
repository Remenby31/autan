#!/usr/bin/env bash
# Runs a test command; when an MPI launcher it started lives longer than LIMIT_S seconds, dumps
# the stacks of every process under it (gdb, `thread apply all bt`) into DIR and the job log,
# then kills it, so the test fails at once with the evidence instead of hanging until its
# timeout. A diagnostic for intermittent hangs (#304), not a retry.
# One case is not a failure of the test: Open MPI's launcher (PRRTE 3.0.x) sometimes stays alive
# after every rank has ended (#304). A launcher that has had no living process below it for
# AUTAN_WATCHDOG_NET_S seconds (default 15) gets its stack dumped and is killed, with a warning
# naming the test, when its ranks were seen or all N left their exit marker (<parent of
# --runs>/exits/exit-<rank>, #380); the test then judges on the ranks' markers, not on the
# launcher's status. A launcher with a living rank past LIMIT_S is a hang, as above.
# Usage: bench/ci/ctest-watchdog.sh LIMIT_S DIR -- COMMAND...
# AUTAN_WATCHDOG_GDB overrides the debugger command (default: sudo -n gdb, else gdb). The
# debuggers run in parallel, each killed after AUTAN_WATCHDOG_GDB_TIMEOUT seconds (default 30), so
# a hang costs at most LIMIT_S + that + 10 s: well inside CTest's per-test timeout.
set -uo pipefail
limit=$1 dir=$2
shift 2
[[ ${1:-} == -- ]] && shift
mkdir -p "$dir"
gdb_timeout=${AUTAN_WATCHDOG_GDB_TIMEOUT:-30}
net_s=${AUTAN_WATCHDOG_NET_S:-15}
gdb_cmd=${AUTAN_WATCHDOG_GDB:-}
if [[ -z $gdb_cmd ]]; then
  if sudo -n true 2>/dev/null; then gdb_cmd="sudo -n gdb"; else gdb_cmd=gdb; fi
fi

descendants() {  # PID: itself and every process below it
  local pid=$1 child
  echo "$pid"
  for child in $(pgrep -P "$pid"); do descendants "$child"; done
}

running() {  # PID: alive and not a zombie (a hung launcher may never reap its ranks)
  local stat
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  stat=${stat##*) }
  [[ ${stat%% *} != Z ]]
}

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null; }

markers() {  # launcher PID: "FOUND/N" exit markers of its ranks, empty if N or --runs is unknown
  local -a args
  local i n="" runs="" found
  mapfile -d '' -t args <"/proc/$1/cmdline" 2>/dev/null || return 0
  for ((i = 0; i + 1 < ${#args[@]}; i++)); do
    case ${args[i]} in
      -n | -np | --np | -c) [[ -z $n ]] && n=${args[i + 1]} ;;
      --runs) runs=${args[i + 1]} ;;
    esac
  done
  [[ $n =~ ^[0-9]+$ && -n $runs ]] || return 0
  found=$(find "$(dirname "$runs")/exits" -maxdepth 1 -name 'exit-*' 2>/dev/null | wc -l)
  echo "$((found))/$n"
}

net_kill() {  # launcher PID whose ranks have all ended
  local launcher=$1 test marks
  test=$(cmdline "$(ps -o ppid= -p "$launcher" | tr -d ' ')")
  marks=$(markers "$launcher")
  {
    echo "# pid $launcher: $(cmdline "$launcher")"
    # shellcheck disable=SC2086
    timeout -k 5 "$gdb_timeout" $gdb_cmd -batch -p "$launcher" -ex "thread apply all bt" 2>&1
  } >"$dir/net-$launcher.txt"
  echo "::warning::watchdog: the MPI launcher outlived every rank by ${net_s} s and was killed (#304); test: ${test:-unknown}; rank exit markers: ${marks:-not found}"
  if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
    {
      echo "### MPI launcher killed after every rank ended (#304)"
      echo "\`$(cmdline "$launcher")\` (test \`${test:-unknown}\`, rank exit markers ${marks:-not found}); its stack is in the \`mpi-hang-stacks\` artifact."
    } >>"$GITHUB_STEP_SUMMARY"
  fi
  kill -TERM "$launcher" 2>/dev/null
  for _ in 1 2 3 4 5; do running "$launcher" || return 0; sleep 1; done
  kill -KILL "$launcher" 2>/dev/null
}

dump_and_kill() {  # launcher PID
  local launcher=$1 pid tree debuggers=()
  tree=$(descendants "$launcher")
  echo "::group::watchdog: $(tr '\0' ' ' <"/proc/$launcher/cmdline") ran over ${limit} s; stacks"
  if ! command -v "${gdb_cmd##* }" >/dev/null 2>&1; then
    echo "::warning::watchdog: no debugger ($gdb_cmd): stacks not collected"
  fi
  for pid in $tree; do
    {
      echo "# pid $pid: $(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)"
      # shellcheck disable=SC2086
      timeout -k 5 "$gdb_timeout" $gdb_cmd -batch -p "$pid" -ex "thread apply all bt" 2>&1
    } >"$dir/$launcher-$pid.txt" &
    debuggers+=($!)
  done
  wait "${debuggers[@]}"
  for pid in $tree; do cat "$dir/$launcher-$pid.txt"; done
  echo "::endgroup::"
  if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
    {
      echo "### MPI hang caught by the watchdog"
      echo "\`$(tr '\0' ' ' <"/proc/$launcher/cmdline")\` ran over ${limit} s; the stacks of its $(wc -w <<<"$tree") processes are in the job log and in the \`mpi-hang-stacks\` artifact."
    } >>"$GITHUB_STEP_SUMMARY"
  fi
  # shellcheck disable=SC2086
  kill -TERM $tree 2>/dev/null
  sleep 5
  # shellcheck disable=SC2086
  kill -KILL $tree 2>/dev/null
}

"$@" &
cmd=$!
caught=0
declare -A seen=() had_ranks=() quiet_since=()
while kill -0 "$cmd" 2>/dev/null; do
  for pid in $(descendants "$cmd"); do
    name=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case $name in mpiexec | mpirun | prterun) ;; *) continue ;; esac
    [[ -n ${seen[$pid]:-} ]] && continue
    running "$pid" || continue
    living=0
    for rank in $(descendants "$pid"); do
      [[ $rank == "$pid" ]] && continue
      had_ranks[$pid]=1
      running "$rank" && living=1
    done
    if ((living)); then
      unset "quiet_since[$pid]"
    else
      : "${quiet_since[$pid]:=$SECONDS}"
      if ((SECONDS - quiet_since[$pid] >= net_s)); then
        marks=$(markers "$pid")
        if [[ -n ${had_ranks[$pid]:-} || (-n $marks && ${marks%/*} -ge ${marks#*/}) ]]; then
          seen[$pid]=1
          net_kill "$pid"
          continue
        fi
      fi
    fi
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n $age && $age -gt $limit ]]; then
      seen[$pid]=1
      caught=1
      dump_and_kill "$pid"
    fi
  done
  sleep 2
done
wait "$cmd"
status=$?
if ((caught && status == 0)); then status=1; fi
exit "$status"
