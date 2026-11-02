#!/usr/bin/env bash
# Decides which gpu-bench jobs a change needs; prints bench=true|false and harness=true|false
# on stdout (for $GITHUB_OUTPUT), the reasons on stderr.
# Usage: bench/ci/changes.sh EVENT DRAFT MEASURED   (run in a checkout of the commit under test)
#        bench/ci/changes.sh --measured   (the measured-paths pattern, for bench/ci/check-measured.py)
# A pull request's checkout is its merge commit with the current base: its first parent is that
# base, so HEAD^1..HEAD is exactly the pull request's change, even after the branch merged main.
# A push compares with the last commit of main that was measured (MEASURED, from
# bench/ci/last-measured.sh). A dispatch, or a base that is missing or cannot be read, measures.
set -euo pipefail
# Measured on the reference Spark: the files the measured binaries (autan_loop_bench,
# autan_host_bench, stream) are built from, sources, included headers and generator inputs,
# exactly, listed in bench/ci/measured-files.txt: gpu-bench checks the list against the build
# graph (bench/ci/check-measured.py), so that a change to a file no benchmark is built from does
# not wait hours for a measurement of nothing it changed (#360). Then the image the benchmarks
# run in, and the measurement method itself (bench/src: statistics, timing, protocol; the
# harness's dependencies; the baselines; the gate), the build files, the mesh generator, which
# feeds the loop benchmark (what the mesh job's cache key hashes), and this filter, its list and
# its check, so that a change to any of them is checked against the real build graph.
files=${AUTAN_MEASURED_FILES:-$(dirname "$0")/measured-files.txt}
measured_paths='^docker/dev/|^bench/(src|cmake|baselines)/|^bench/(pyproject\.toml|uv\.lock|ci/gate\.sh)$|(^|/)CMakeLists\.txt$|\.cmake$|^vv/src/autan_vv/(meshes|oracles)/|^vv/src/autan_vv/(cli|errors|__init__)\.py$|^vv/(pyproject\.toml|uv\.lock|\.python-version)$|^bench/ci/(changes\.sh|check-measured\.py|measured-files\.txt)$|^\.github/workflows/gpu-bench\.yml$'
if [[ ${1:-} == --measured ]]; then
  echo "$measured_paths"; exit 0
fi
[[ -f $files ]] || { echo "error: $files is missing" >&2; exit 2; }
event=$1 draft=$2 measured=$3
say() { echo "$1=$2"; echo "$1: $2" >&2; }

if [[ $event == pull_request && $draft == true ]]; then
  say bench false; say harness false; echo "skipped: the pull request is a draft" >&2; exit 0
fi
case $event in
  pull_request) base=HEAD^1 ;;
  push) base=$measured ;;
  *) base= ;;
esac
if [[ -z $base ]] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
  say bench true; say harness true; echo "forced: $event, no readable base" >&2; exit 0
fi
changed=$(git diff --name-only "$base" HEAD --)
printf 'changed files:\n%s\n' "$changed" >&2
# The harness's own GPU tests and the image smoke test: whenever the harness or an image changed.
# They run on the bench Spark too: the tests runner's GPU is shared with another team's process.
harness='^(bench|docker)/'
# A measured file: an exact line of the list (fixed strings, whole lines), or a measured path.
if grep -q -x -F -f "$files" <<<"$changed" || grep -q -E "$measured_paths" <<<"$changed"; then
  say bench true
else
  say bench false
fi
grep -q -E "$harness" <<<"$changed" && say harness true || say harness false
