#!/usr/bin/env bash
# The benchmark mesh's cache key, for gpu-bench's mesh job and mesh-cache.yml alike: the files the
# generator loads (bench/ci/measured-mesh-files.txt), the list itself, the generator's locked
# dependencies and the script that generates and checks. Prints key=... for $GITHUB_OUTPUT.
set -euo pipefail
list=bench/ci/measured-mesh-files.txt
mapfile -t files < <(grep -v -E '^(#|$)' "$list")
hash=$(sha256sum "$list" "${files[@]}" vv/pyproject.toml vv/uv.lock vv/.python-version \
  bench/ci/mesh-modules.py | sha256sum | cut -c1-32)
echo "key=mesh-hex-n128-$hash"
