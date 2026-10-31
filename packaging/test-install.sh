#!/bin/sh
# Runs install.sh against the wheels in REPO/wheelhouse on a clean system, then checks the result.
# As root (a fresh container), it adds curl and an unprivileged user first, like a new workstation.
# REQUIRE_GPU=1 makes a GPU without native code, or no GPU at all, a failure.
#
# Usage: sh packaging/test-install.sh REPO
set -eu
repo=$1

if [ "$(id -u)" -eq 0 ]; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null
    apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null
  elif command -v dnf >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || dnf install -y -q curl >/dev/null
  fi
  useradd -m autan
  exec su autan -c "REQUIRE_GPU=${REQUIRE_GPU:-0} sh '$repo/packaging/test-install.sh' '$repo'"
fi

cd "$HOME"
AUTAN_FIND_LINKS="$repo/wheelhouse" sh "$repo/install.sh"

bin=$HOME/.local/bin
"$bin/autan" doctor --json >doctor.json
"$bin/autan" version

# The probe is not a dependency of autan yet: install it with the uv that install.sh fetched.
uv=$(ls "${XDG_DATA_HOME:-$HOME/.local/share}"/autan/uv-*/uv)
"$uv" tool install --quiet --python 3.13 --find-links "$repo/wheelhouse" autan-probe
if [ "${REQUIRE_GPU:-0}" = 1 ]; then
  "$bin/autan-probe" --require-gpu
else
  "$bin/autan-probe"
fi
