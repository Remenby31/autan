#!/usr/bin/env bash
# Regenerates every mesh in tests/data/mesh. Needs uv and Docker.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
data=$(dirname "$here")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

uv run --project "$here" --locked python "$here/generate.py" "$work/gmsh"
cp "$here/convert.sh" "$work/"
docker run --rm -v "$work:/work" opencfd/openfoam-default:2506 \
  bash -l /work/convert.sh /work/gmsh /work/out
for dir in "$work"/out/*/; do
  name=$(basename "$dir")
  uv run --project "$here" --locked python "$here/fluent_binary.py" \
    "$dir/$name.fluent.msh" "$dir/$name.fluent-binary.msh"
  rm -rf "${data:?}/$name"
  cp -r "$dir" "$data/$name"
done
