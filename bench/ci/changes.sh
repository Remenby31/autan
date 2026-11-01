#!/usr/bin/env bash
# Decides which gpu-bench jobs a change needs; prints bench=true|false and harness=true|false
# on stdout (for $GITHUB_OUTPUT), the reasons on stderr.
# Usage: bench/ci/changes.sh EVENT DRAFT BEFORE   (run in a checkout of the commit under test)
# A pull request's checkout is its merge commit with the current base: its first parent is that
# base, so HEAD^1..HEAD is exactly the pull request's change, even after the branch merged main.
# A push compares with the commit it replaced (BEFORE). A dispatch, or a base that cannot be read,
# runs everything.
set -euo pipefail
event=$1 draft=$2 before=$3
say() { echo "$1=$2"; echo "$1: $2" >&2; }

if [[ $event == pull_request && $draft == true ]]; then
  say bench false; say harness false; echo "skipped: the pull request is a draft" >&2; exit 0
fi
case $event in
  pull_request) base=HEAD^1 ;;
  push) base=$before ;;
  *) base= ;;
esac
if [[ -z $base ]] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
  say bench true; say harness true; echo "forced: $event, no readable base" >&2; exit 0
fi
changed=$(git diff --name-only "$base" HEAD --)
printf 'changed files:\n%s\n' "$changed" >&2
# Code that is timed or builds what is timed; readers (io/) fix the face and cell order of the
# measured mesh; the mesh generator feeds the loop benchmark.
measured='^(runtime|codegen|kernels|physics|solvers|core|io|bench|docker)/|(^|/)CMakeLists\.txt$|\.cmake$|^vv/src/autan_vv/meshes/|^\.github/workflows/gpu-bench\.yml$'
# The harness's own GPU tests and the image smoke test take the GPU lock: only when they changed.
harness='^(bench|docker)/'
grep -q -E "$measured" <<<"$changed" && say bench true || say bench false
grep -q -E "$harness" <<<"$changed" && say harness true || say harness false
