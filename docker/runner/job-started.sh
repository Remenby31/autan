#!/usr/bin/env bash
# Runner job-started hook (ACTIONS_RUNNER_HOOK_JOB_STARTED). It runs on the runner before any
# step and outside the workflow's control; a non-zero exit fails the job. It accepts only
# jobs of AUTAN_RUNNER_REPOSITORY triggered by push, workflow_dispatch, schedule, or a
# pull_request whose head branch lives in that same repository. Fork pull requests are refused.
set -euo pipefail

deny() {
  echo "::error title=autan-runner::job refused: $*" >&2
  exit 1
}

allowed=${AUTAN_RUNNER_REPOSITORY:?AUTAN_RUNNER_REPOSITORY is not set}
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
