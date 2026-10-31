#!/usr/bin/env bash
# Registers the runner once (state kept in /state) or starts it from that state.
# Usage: entrypoint.sh register NAME LABELS URL  (registration token on stdin) | entrypoint.sh run
set -euo pipefail
cd /opt/actions-runner
files=(.runner .credentials .credentials_rsaparams)

case ${1:-run} in
  register)
    read -r token
    ./config.sh --unattended --replace --name "$2" --labels "$3" --url "$4" --token "$token" --work /work
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
