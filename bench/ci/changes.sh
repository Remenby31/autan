#!/usr/bin/env bash
# Decides which gpu-bench jobs a change needs; prints bench=true|false and harness=true|false
# on stdout (for $GITHUB_OUTPUT), the reasons on stderr.
# Usage: bench/ci/changes.sh EVENT DRAFT MEASURED   (run in a checkout of the commit under test)
# A pull request's checkout is its merge commit with the current base: its first parent is that
# base, so HEAD^1..HEAD is exactly the pull request's change, even after the branch merged main.
# A push compares with the last commit of main that was measured (MEASURED, from
# bench/ci/last-measured.sh). A dispatch, or a base that is missing or cannot be read, measures.
set -euo pipefail
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
# Measured on the reference Spark: code that is timed or builds what is timed (readers in io/
# fix the face and cell order of the measured mesh; the mesh generator feeds the loop
# benchmark), the image it runs in, and the measurement method itself (bench/src: statistics,
# timing, protocol; the harness's dependencies; the STREAM binary; the baselines; the gate).
measured_paths='^(runtime|codegen|kernels|physics|solvers|core|io)/|^docker/dev/|^bench/(src|stream|cmake|baselines)/|^bench/(pyproject\.toml|uv\.lock|ci/gate\.sh)$|(^|/)CMakeLists\.txt$|\.cmake$|^vv/src/autan_vv/meshes/|^\.github/workflows/gpu-bench\.yml$'
# The harness's own GPU tests and the image smoke test: whenever the harness or an image changed.
# Without a measured change they run on the tests runner, keeping the bench Spark for measurements.
harness='^(bench|docker)/'
grep -q -E "$measured_paths" <<<"$changed" && say bench true || say bench false
grep -q -E "$harness" <<<"$changed" && say harness true || say harness false
