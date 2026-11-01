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
# Gmsh-only meshes (no OpenFOAM or Fluent form) are copied as Gmsh wrote them.
for name in periodic-box; do
  rm -rf "${data:?}/$name"
  mkdir -p "$data/$name"
  cp "$work/gmsh/$name.ascii.msh" "$work/gmsh/$name.binary.msh" "$data/$name/"
done
for dir in "$work"/out/*/; do
  name=$(basename "$dir")
  uv run --project "$here" --locked python "$here/fluent_binary.py" \
    "$dir/$name.fluent.msh" "$dir/$name.fluent-binary.msh"
  rm -rf "${data:?}/$name"
  cp -r "$dir" "$data/$name"
done
