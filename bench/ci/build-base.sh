#!/usr/bin/env bash
# Build the base's benchmarks for the paired gate (#403), in a worktree of its own.
# Usage: bench/ci/build-base.sh DIR [REV]
# Checks REV (default HEAD^1: in a pull request's merge commit, main at merge time) out into
# DIR and builds there what the bench job builds from the change: bench/'s STREAM and the
# generated loops' autan_loop_bench, under the GPU lock as every build on the bench Spark is.
# A failure of the base's own build is not the change's: it ends in "BASE BUILD FAILED", never
# green and never a verdict on the change. A lock that could not be had is an ordinary error.
# The worktree is left for the measurement; `bench/ci/build-base.sh --remove DIR` removes it.
set -uo pipefail

if [[ ${1:-} == --remove ]]; then
  git worktree remove --force "$2" 2>/dev/null || rm -rf "$2"
  git worktree prune
  exit 0
fi
dir=$1 rev=${2:-HEAD^1}

base_failed() {
  local sha message
  sha=$(git rev-parse --short "$rev" 2> /dev/null) || sha=unknown
  message="BASE BUILD FAILED: $1 at $rev ($sha); no verdict on the change"
  echo "::error title=autan-bench::$message"
  [[ -n ${GITHUB_STEP_SUMMARY:-} ]] && echo "### $message" >> "$GITHUB_STEP_SUMMARY"
  exit 1
}

# A worktree left by a cancelled job would make the add fail.
git worktree remove --force "$dir" 2>/dev/null || rm -rf "$dir"
git worktree prune
git rev-parse --verify --quiet "$rev^{commit}" > /dev/null || base_failed "no such commit"
git worktree add --detach "$dir" "$rev" || base_failed "checkout"

# shellcheck disable=SC2016 # $1 expands in the inner bash, the worktree passed as its argument.
bench/lock/autan-gpu-lock -m "gpu-bench: build the base's benchmarks" -t 600 -- bash -c '
  cmake -S "$1/bench" -B "$1/build/bench" && cmake --build "$1/build/bench" &&
  cmake -S "$1" -B "$1/build/autan" -DAUTAN_ENABLE_CUDA=ON -DAUTAN_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 &&
  cmake --build "$1/build/autan" --target autan_loop_bench' build-base "$dir"
status=$?
case $status in
  0) exit 0 ;;
  # The lock wrapper's own failures (bench/lock/autan-gpu-lock): no lock, the time limit, a
  # leftover, the memory floor, a signal. Not a statement about the base's code.
  70 | 71 | 75 | 124 | 143) exit "$status" ;;
  *) base_failed "build (exit $status)" ;;
esac
