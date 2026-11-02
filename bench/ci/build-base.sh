#!/usr/bin/env bash
# Build the base's benchmarks for the paired gate (#403), in a worktree of its own.
# Usage: bench/ci/build-base.sh DIR [REV]
# Checks REV (default HEAD^1: in a pull request's merge commit, main at merge time) out into
# DIR and builds there what the paired gate measures: bench/'s STREAM and the generated loops'
# autan_loop_bench, under the GPU lock as every build on the bench Spark is.
# A failure of the base's own build is not the change's: it ends in "BASE BUILD FAILED", never
# green and never a verdict on the change. A lock that could not be had is an ordinary error.
# The lock is waited for at most $AUTAN_BENCH_LOCK_TIMEOUT seconds, as every wait of the job.
# The worktree is left for the measurement; `bench/ci/build-base.sh --remove DIR` removes it.
set -uo pipefail

# Removes the base's worktree: git's removal (twice forced: it is ours, even locked), or, for a
# directory git does not know, a plain removal. Anything left is an infrastructure error (exit 2),
# never the base's.
remove() {
  local dir=$1
  if git worktree list --porcelain | grep -qxF "worktree $(cd "$dir" 2> /dev/null && pwd -P)"; then
    git worktree remove -f -f "$dir" || { echo "::error title=autan-bench::cannot remove the worktree $dir"; return 2; }
  elif [[ -e $dir ]]; then
    rm -rf "$dir" || { echo "::error title=autan-bench::cannot remove $dir"; return 2; }
  fi
  git worktree prune
  [[ ! -e $dir ]] || { echo "::error title=autan-bench::$dir is still there"; return 2; }
}

if [[ ${1:-} == --remove ]]; then
  remove "$2"
  exit $?
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
remove "$dir" || exit 2
git rev-parse --verify --quiet "$rev^{commit}" > /dev/null || base_failed "no such commit"
# The commit exists: a checkout that fails is the machine's (disk, permissions), not the base's.
git worktree add --detach "$dir" "$rev" || {
  echo "::error title=autan-bench::cannot check the base out into $dir"
  exit 2
}

# The build writes its own status: a failure before it, of the lock wrapper (no lock in time, a
# missing lock file, a signal), leaves none and is never blamed on the base.
status_file="$dir/.autan-build-status"
# shellcheck disable=SC2016 # $1 expands in the inner bash, the worktree passed as its argument.
bench/lock/autan-gpu-lock -m "gpu-bench: build the base's benchmarks" \
  -w "${AUTAN_BENCH_LOCK_TIMEOUT:-3600}" -t 600 -- bash -c '
  cmake -S "$1/bench" -B "$1/build/bench" && cmake --build "$1/build/bench" &&
  cmake -S "$1" -B "$1/build/autan" -DAUTAN_ENABLE_CUDA=ON -DAUTAN_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 &&
  cmake --build "$1/build/autan" --target autan_loop_bench
  status=$?; echo "$status" > "$2"; exit "$status"' build-base "$dir" "$status_file"
wrapper=$?
if [[ ! -s $status_file ]]; then
  echo "::error title=autan-bench::the base's build did not start (GPU lock wrapper exit $wrapper)"
  exit $((wrapper == 0 ? 2 : wrapper))
fi
build=$(< "$status_file")
[[ $build == 0 && $wrapper == 0 ]] && exit 0
# The build ran: its failure is the base's. A build that passed but whose wrapper then failed
# (a time limit, a leftover) is the wrapper's.
[[ $build != 0 ]] && base_failed "build (exit $build)"
exit "$wrapper"
