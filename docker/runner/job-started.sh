#!/usr/bin/env bash
# Runner job-started hook (ACTIONS_RUNNER_HOOK_JOB_STARTED). It runs on the runner before any
# step and outside the workflow's control. It accepts only jobs of the repository named in the
# root-owned file `repository` next to this script (never from the environment, which a
# workflow controls), triggered by push, workflow_dispatch, schedule, or a pull_request whose
# head branch lives in that same repository. Fork pull requests are refused.
#
# A failing hook only fails its own step: the runner still evaluates the job's later steps,
# so `if: always()` steps would run. Refusing therefore kills the Runner.Worker process that
# runs this job, which ends the job before any of its steps.
set -euo pipefail

# The runner starts this script as `bash job-started.sh` directly under Runner.Worker, so the
# worker is our parent. Only the parent is killed: run any other way (tests, a job calling it)
# the script must not kill a worker it does not belong to.
kill_worker() {
  local parent comm=""
  parent=$(awk '/^PPid:/ {print $2}' "/proc/$$/status" 2>/dev/null || true)
  [[ -z $parent ]] || comm=$(cat "/proc/$parent/comm" 2>/dev/null || true)
  if [[ $comm == Runner.Worker ]]; then
    kill -KILL "$parent"
  else
    echo "::error title=autan-runner::not started by Runner.Worker; the job was not stopped" >&2
  fi
}

deny() {
  echo "::error title=autan-runner::job refused: $*" >&2
  kill_worker
  exit 1
}

allowed=$(<"$(dirname "${BASH_SOURCE[0]}")/repository")
[[ $allowed =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || deny "no valid repository configured"
[[ -r ${GITHUB_EVENT_PATH:-} ]] || deny "no event payload"
repository=$(jq -r '.repository.full_name // empty' "$GITHUB_EVENT_PATH")
[[ $repository == "$allowed" && ${GITHUB_REPOSITORY:-} == "$allowed" ]] ||
  deny "repository '${GITHUB_REPOSITORY:-}' (payload '$repository') is not $allowed"

event=${GITHUB_EVENT_NAME:-}
case $event in
  push | workflow_dispatch | schedule) ;;
  pull_request)
    head=$(jq -r '.pull_request.head.repo.full_name // empty' "$GITHUB_EVENT_PATH")
    [[ $head == "$allowed" ]] || deny "pull_request from '${head:-unknown}', not from a branch of $allowed"
    ;;
  *) deny "event '$event' is not allowed on GPU runners" ;;
esac

# Host memory, on a Spark shared with other teams: the job starts only once the host's available
# memory reaches the floor in the root-owned file min-available-gib (0 or absent: none). It waits
# at most 15 minutes (admit-wait-s overrides it), then is refused before its first step: the host
# is busy, not the change at fault. GB10's GPU allocates from host memory, so this also protects
# the GPU's other tenants. A meminfo file beside the hook replaces /proc/meminfo, for its tests.
here=$(dirname "${BASH_SOURCE[0]}")
min_gib=$(cat "$here/min-available-gib" 2>/dev/null || echo 0)
wait_s=$(cat "$here/admit-wait-s" 2>/dev/null || echo 900)
[[ $min_gib =~ ^[0-9]{1,3}$ && $wait_s =~ ^[0-9]{1,5}$ ]] || deny "bad memory admission settings"
meminfo=/proc/meminfo
[[ -e $here/meminfo ]] && meminfo=$here/meminfo
if ((10#$min_gib > 0)); then
  target_kib=$((10#$min_gib * 1048576))
  deadline=$((SECONDS + 10#$wait_s))
  announced=-60
  while :; do
    free=$(awk '/^MemAvailable:/ {print $2}' "$meminfo" 2>/dev/null || true)
    [[ $free =~ ^[0-9]+$ ]] && ((free >= target_kib)) && break
    shown=$([[ $free =~ ^[0-9]+$ ]] && awk -v k="$free" 'BEGIN {printf "%.1f", k / 1048576}' || echo unknown)
    ((SECONDS < deadline)) ||
      deny "host memory available $shown GiB, below this runner's floor of $min_gib GiB for ${wait_s} s: the host is busy, rerun later"
    if ((SECONDS - announced >= 60)); then
      echo "autan-runner: WAITING FOR MEMORY: $shown GiB available, the job needs $min_gib GiB (at most ${wait_s} s)"
      announced=$SECONDS
    fi
    sleep 1
  done
fi
echo "autan-runner: accepted $event job from $allowed"
