#!/usr/bin/env bash
# The benchmark mesh's cache key, for gpu-bench's mesh job and mesh-cache.yml alike: the files the
# generator loads (bench/ci/measured-mesh-files.txt), the list itself, the generator's locked
# dependencies and the script that generates and checks. Prints key=... for $GITHUB_OUTPUT.
set -euo pipefail
list=bench/ci/measured-mesh-files.txt
mapfile -t files < <(grep -v -E '^(#|$)' "$list")
# The names of the entries beside the listed modules too: one added there can change what is
# imported (a package families/ shadowing families.py), so it must regenerate the mesh, and so
# re-check the list.
mapfile -t dirs < <(for f in "${files[@]}"; do echo "${f%/*}"; done | sort -u)
names=$(find "${dirs[@]}" -mindepth 1 -maxdepth 1 ! -name __pycache__ | LC_ALL=C sort)
hash=$({ sha256sum "$list" "${files[@]}" vv/pyproject.toml vv/uv.lock vv/.python-version \
  bench/ci/mesh-modules.py; echo "$names"; } | sha256sum | cut -c1-32)
echo "key=mesh-hex-n128-$hash"
