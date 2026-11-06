#!/usr/bin/env bash
# Runs a test command; when an MPI launcher it started lives longer than LIMIT_S seconds, dumps
# the stacks of every process under it (gdb, `thread apply all bt`) into DIR and the job log,
# then kills it, so the test fails at once with the evidence instead of hanging until its
# timeout. A diagnostic for intermittent hangs (#304), not a retry.
# One case is not a failure of the test: Open MPI's launcher (PRRTE 3.0.x) sometimes stays alive
# after every rank has ended (#304). The proof that they ended is theirs (#380): N markers
# exit-0 .. exit-<N-1>, in the launcher's $AUTAN_TEST_EXIT_DIR or else <parent of --runs>/exits,
# each written since the launcher started and holding "<status> <pid>" of a process that no
# longer runs. With that proof and no living process below the launcher for
# AUTAN_WATCHDOG_NET_S seconds (default 5), its stack is dumped (job log and DIR) and it is
# killed, with a warning naming the test; the test then judges on the markers, not on the
# launcher's status. Without it, a launcher past LIMIT_S is a hang, as above.
# Usage: bench/ci/ctest-watchdog.sh LIMIT_S DIR -- COMMAND...
# AUTAN_WATCHDOG_GDB overrides the debugger command (default: sudo -n gdb, else gdb);
# AUTAN_WATCHDOG_PTRACER, below, lets it attach without sudo on the self-hosted runners. The
# debuggers run in parallel, each killed after AUTAN_WATCHDOG_GDB_TIMEOUT seconds (default 30), so
# a hang costs at most LIMIT_S + that + 10 s: well inside CTest's per-test timeout.
set -uo pipefail
limit=$1 dir=$2
shift 2
[[ ${1:-} == -- ]] && shift
mkdir -p "$dir"
gdb_timeout=${AUTAN_WATCHDOG_GDB_TIMEOUT:-30}
net_s=${AUTAN_WATCHDOG_NET_S:-5}
clk_tck=$(getconf CLK_TCK)
boot=$(awk '/^btime/{print $2}' /proc/stat)
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

gone() {  # PID LAUNCHER: the process no longer exists, as run_on_ranks requires (kill -0 gives
  # ESRCH), or is a zombie of the launcher, reaped as soon as the launcher is killed
  local stat
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 0
  stat=${stat##*) }
  local state ppid
  read -r state ppid _ <<<"$stat"
  [[ $state == Z && $ppid == "$2" ]]
}

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null; }

started() {  # PID: its start time, in clock ticks since boot (with the PID, a process's identity)
  local stat
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  stat=${stat##*) }
  awk '{print $20}' <<<"$stat"
}

markers() {  # launcher PID: "FOUND/N" proven-ended ranks; empty if N or the directory is unknown
  local -a args
  local i n="" runs="" exits="" found=0 since rank file content pid
  mapfile -d '' -t args <"/proc/$1/cmdline" 2>/dev/null || return 0
  for ((i = 0; i < ${#args[@]}; i++)); do
    case ${args[i]} in
      -n | -np | --np | -c) [[ -z $n ]] && n=${args[i + 1]:-} ;;
      -n=* | -np=* | --np=* | -c=*) [[ -z $n ]] && n=${args[i]#*=} ;;
      --runs) runs=${args[i + 1]:-} ;;
      --runs=*) runs=${args[i]#*=} ;;
    esac
  done
  exits=$(tr '\0' '\n' <"/proc/$1/environ" 2>/dev/null | sed -n 's/^AUTAN_TEST_EXIT_DIR=//p' | head -1)
  [[ -z $exits && -n $runs ]] && exits=$(dirname "$runs")/exits
  [[ $n =~ ^[1-9][0-9]*$ && -n $exits ]] || return 0
  # A relative path is the launcher's, not this script's.
  [[ $exits == /* ]] || exits=$(readlink "/proc/$1/cwd")/$exits
  since=$((boot + $(started "$1") / clk_tck))
  for ((rank = 0; rank < n; rank++)); do
    file=$exits/exit-$rank
    [[ -f $file && $(stat -c %Y "$file") -ge $since ]] || continue
    content=$(<"$file")
    [[ $content =~ ^[0-9]+\ ([0-9]+)$ ]] || continue
    pid=${BASH_REMATCH[1]}
    gone "$pid" "$1" || continue
    found=$((found + 1))
  done
  echo "$found/$n"
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
  echo "::group::watchdog: the launcher's stack (#304)"
  cat "$dir/net-$launcher.txt"
  echo "::endgroup::"
  echo "::warning::watchdog: the MPI launcher outlived every rank by ${net_s} s and was killed (#304); test: ${test:-unknown}; launcher: $(cmdline "$launcher"); ranks proven ended by their exit markers: ${marks:-not found}"
  if [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
    {
      echo "### MPI launcher killed after every rank ended (#304)"
      echo "\`$(cmdline "$launcher")\` (test \`${test:-unknown}\`, rank exit markers ${marks:-not found}); its stack is in the job log."
    } >>"$GITHUB_STEP_SUMMARY"
  fi
  # KILL, not TERM: a launcher that handles TERM by exiting would end normally, and its test would
  # judge its status instead of the markers.
  kill -KILL "$launcher" 2>/dev/null
}

ticks() {  # PID: "START STATE CPU_TICKS VOLUNTARY NONVOLUNTARY" (start time: the process's
  # identity with its PID; utime + stime of all its threads; the leader's context switches)
  local stat fields
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  read -ra fields <<<"${stat##*) }"  # after the command name, which may hold spaces
  awk -v id="${fields[19]}" -v s="${fields[0]}" -v t=$((fields[11] + fields[12])) \
    '/^voluntary_ctxt_switches/ {v = $2} /^nonvoluntary_ctxt_switches/ {n = $2}
     END {print id, s, t, v + 0, n + 0}' "/proc/$1/status" 2>/dev/null
}

activity() {  # PIDS...: what each does over about one second, from /proc alone (no ptrace).
  # Observations, not a diagnosis: ranks deadlocked while busy-polling also burn CPU. CPU sums
  # all threads (above 100 % for a multi-threaded rank); state and wait channel are the leader's.
  local pid i=0 before=() after=() t_before=() t_after=() id0 id1 s1 c0 c1 v0 v1 n0 n1 ms
  for pid in "$@"; do before+=("$(ticks "$pid")"); t_before+=("${EPOCHREALTIME/./}"); done
  sleep 1
  for pid in "$@"; do after+=("$(ticks "$pid")"); t_after+=("${EPOCHREALTIME/./}"); done
  for pid in "$@"; do
    read -r id0 _ c0 v0 n0 <<<"${before[$i]}"
    read -r id1 s1 c1 v1 n1 <<<"${after[$i]}"
    ms=$(((t_after[i] - t_before[i]) / 1000))
    if [[ -z ${before[$i]} || -z ${after[$i]} || $id0 != "$id1" ]] || ((ms <= 0)); then
      printf 'pid %s: gone during the sample, no activity measured\n' "$pid"
    else
      printf 'pid %s %s: state %s, CPU %d %% over %d ms, wchan %s, +%d voluntary +%d involuntary switches\n' \
        "$pid" "$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')" "$s1" \
        $(((c1 - c0) * 100000 / clk_tck / ms)) "$ms" "$(cat "/proc/$pid/wchan" 2>/dev/null || echo '?')" \
        $((v1 - v0)) $((n1 - n0))
    fi
    i=$((i + 1))
  done
}

dump_and_kill() {  # launcher PID
  local launcher=$1 pid tree debuggers=()
  tree=$(descendants "$launcher")
  echo "::group::watchdog: $(tr '\0' ' ' <"/proc/$launcher/cmdline") ran over ${limit} s; stacks"
  # shellcheck disable=SC2086
  activity $tree | tee "$dir/$launcher-activity.txt"
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

# AUTAN_WATCHDOG_PTRACER (set by the self-hosted runner's image) names a library preloaded into
# the command, and nothing else: its processes then let gdb, their sibling, attach to them where
# Yama's ptrace_scope 1 would refuse it (docker/runner/ptracer.c).
if [[ -n ${AUTAN_WATCHDOG_PTRACER:-} && -r $AUTAN_WATCHDOG_PTRACER ]]; then
  LD_PRELOAD="$AUTAN_WATCHDOG_PTRACER${LD_PRELOAD:+:$LD_PRELOAD}" "$@" &
else
  "$@" &
fi
cmd=$!
caught=0
declare -A seen=() quiet_since=()
while kill -0 "$cmd" 2>/dev/null; do
  for pid in $(descendants "$cmd"); do
    name=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case $name in mpiexec | mpirun | prterun) ;; *) continue ;; esac
    # A PID is reused: a process is its PID and its start time.
    key=$pid:$(started "$pid") || continue
    [[ -n ${seen[$key]:-} ]] && continue
    running "$pid" || continue
    living=0
    for rank in $(descendants "$pid"); do
      [[ $rank != "$pid" ]] && running "$rank" && living=1
    done
    marks=$(markers "$pid")
    if ((!living)) && [[ -n $marks && ${marks%/*} -eq ${marks#*/} ]]; then
      : "${quiet_since[$key]:=$SECONDS}"
      if ((SECONDS - quiet_since[$key] >= net_s)); then
        seen[$key]=1
        net_kill "$pid"
      fi
      # Proven ended: never a hang, even past LIMIT_S.
      continue
    fi
    unset "quiet_since[$key]"
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n $age && $age -gt $limit ]]; then
      seen[$key]=1
      caught=1
      dump_and_kill "$pid"
    fi
  done
  # Wake when the command ends, not at the end of the poll: a lock that stops the job at its
  # limit must never catch the watchdog asleep after a command that passed.
  sleep 2 &
  sleeper=$!
  which=
  wait -n -p which "$cmd" "$sleeper"
  waited=$?
  if [[ $which == "$cmd" ]]; then
    kill "$sleeper" 2>/dev/null
    wait "$sleeper" 2>/dev/null
    status=$waited
    break
  fi
done
[[ -n ${status:-} ]] || { wait "$cmd"; status=$?; }
if ((caught && status == 0)); then status=1; fi
exit "$status"
