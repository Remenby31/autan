#!/usr/bin/env bash
# Decides which gpu-bench jobs a change needs; prints bench=true|false and harness=true|false
# on stdout (for $GITHUB_OUTPUT), the reasons on stderr.
# Usage: bench/ci/changes.sh EVENT DRAFT MEASURED   (run in a checkout of the commit under test)
#        bench/ci/changes.sh --measured   (the measured directories, then the measured-paths
#                                          pattern, for bench/ci/check-measured.py)
# A pull request's checkout is its merge commit with the current base: its first parent is that
# base, so HEAD^1..HEAD is exactly the pull request's change, even after the branch merged main.
# A push compares with the last commit of main that was measured (MEASURED, from
# bench/ci/last-measured.sh). A dispatch, or a base that is missing or cannot be read, measures.
set -euo pipefail
# Measured on the reference Spark: the top-level directories the measured binaries
# (autan_loop_bench, stream) are built from, sources and included headers, exactly: gpu-bench
# checks them against the build graph (bench/ci/check-measured.py), so that a change elsewhere
# (runtime/, physics/, solvers/ on 2026-11-02) does not wait hours for a measurement of nothing
# it changed (#360). Then the image the benchmarks run in, and the measurement method itself
# (bench/src: statistics, timing, protocol; the harness's dependencies; the STREAM binary; the
# baselines; the gate), the build files, the mesh generator, which feeds the loop benchmark (what
# the mesh job's cache key hashes), and this filter and its check, so that a change to either
# is checked against the real build graph.
measured_dirs=(codegen core io kernels partition)
measured_paths="^($(IFS='|'; echo "${measured_dirs[*]}"))/"'|^docker/dev/|^bench/(src|stream|cmake|baselines)/|^bench/(pyproject\.toml|uv\.lock|ci/gate\.sh)$|(^|/)CMakeLists\.txt$|\.cmake$|^bench/ci/(measured-mesh-files\.txt|mesh-modules\.py|mesh-key\.sh)$|^vv/(pyproject\.toml|uv\.lock|\.python-version)$|^bench/ci/(changes\.sh|check-measured\.py)$|^\.github/workflows/gpu-bench\.yml$'
if [[ ${1:-} == --measured ]]; then
  echo "${measured_dirs[*]}"; echo "$measured_paths"; exit 0
fi
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
# The mesh's generator: exactly the repository modules it loads (bench/ci/mesh-modules.py checks
# the list when the mesh job generates), as fixed strings on whole lines.
mesh_files=$(dirname "$0")/measured-mesh-files.txt
[[ -f $mesh_files ]] || { echo "error: $mesh_files is missing" >&2; exit 2; }
mesh_list=$(grep -v -E '^(#|$)' "$mesh_files")
# An entry with a listed module's name added beside it (a package families/, an extension
# families.*.so) changes what Python imports without touching a listed file: measured too.
shadows=$(while read -r f; do stem=${f%.py}; echo "$stem/"; echo "$stem."; done <<<"$mesh_list")
moved=$(git diff --name-status "$base" HEAD -- | awk '$1 !~ /^M/ {for (i = 2; i <= NF; i++) print $i}')
# The prefixes through the environment: awk -v cannot carry newlines everywhere (BSD awk).
shadowed() { SHADOWS=$shadows awk 'BEGIN {n = split(ENVIRON["SHADOWS"], s, "\n")} {for (i = 1; i <= n; i++) if (s[i] != "" && index($0, s[i]) == 1) found = 1} END {exit !found}' <<<"$moved"; }
if grep -q -E "$measured_paths" <<<"$changed" || grep -q -x -F -f <(echo "$mesh_list") <<<"$changed" || shadowed; then
  say bench true
else
  say bench false
fi
grep -q -E "$harness" <<<"$changed" && say harness true || say harness false
