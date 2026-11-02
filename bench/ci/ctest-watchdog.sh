#!/usr/bin/env bash
# Runs a test command; when an MPI launcher it started lives longer than LIMIT_S seconds, dumps
# the stacks of every process under it (gdb, `thread apply all bt`) into DIR and the job log,
# then kills it, so the test fails at once with the evidence instead of hanging until its
# timeout. A diagnostic for intermittent hangs (#304), not a retry.
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
gdb_cmd=${AUTAN_WATCHDOG_GDB:-}
if [[ -z $gdb_cmd ]]; then
  if sudo -n true 2>/dev/null; then gdb_cmd="sudo -n gdb"; else gdb_cmd=gdb; fi
fi

descendants() {  # PID: itself and every process below it
  local pid=$1 child
  echo "$pid"
  for child in $(pgrep -P "$pid"); do descendants "$child"; done
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
declare -A seen=()
while kill -0 "$cmd" 2>/dev/null; do
  for pid in $(descendants "$cmd"); do
    name=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case $name in mpiexec | mpirun | prterun) ;; *) continue ;; esac
    [[ -n ${seen[$pid]:-} ]] && continue
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
