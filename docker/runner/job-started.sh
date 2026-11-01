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
echo "autan-runner: accepted $event job from $allowed"
