#!/usr/bin/env bash
# Registers the runner once (state kept in /state) or starts it from that state.
# Usage: entrypoint.sh register NAME LABELS URL  (registration token on stdin) | entrypoint.sh run
set -euo pipefail
cd /opt/actions-runner
files=(.runner .credentials .credentials_rsaparams)

case ${1:-run} in
  register)
    # The runner reads the token from its environment, which keeps it out of the process list.
    read -r ACTIONS_RUNNER_INPUT_TOKEN
    export ACTIONS_RUNNER_INPUT_TOKEN
    ./config.sh --unattended --replace --name "$2" --labels "$3" --url "$4" --work /work
    unset ACTIONS_RUNNER_INPUT_TOKEN
    cp "${files[@]}" /state/
    ;;
  run)
    for f in "${files[@]}"; do
      [[ -f /state/$f ]] || { echo "error[RUNNER_NOT_REGISTERED]: /state/$f missing; run deploy.sh register" >&2; exit 1; }
      cp "/state/$f" .
    done
    exec ./run.sh
    ;;
  *) echo "usage: entrypoint.sh register NAME LABELS URL | run" >&2; exit 2 ;;
esac
