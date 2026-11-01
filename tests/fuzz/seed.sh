#!/usr/bin/env bash
# Builds the seed corpora of the fuzzers from tests/data/mesh into $1:
# the small test meshes in every format a reader takes.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
data=$here/../data/mesh
out=$1
mkdir -p "$out"/{gmsh,fluent,polymesh,cgns}
for name in hex tet pyramid hanging periodic-box; do
  dir=$data/$name
  for f in "$dir/$name.ascii.msh" "$dir/$name.binary.msh"; do
    [ -f "$f" ] && cp "$f" "$out/gmsh/"
  done
  for f in "$dir/$name.fluent.msh" "$dir/$name.fluent-binary.msh"; do
    [ -f "$f" ] && cp "$f" "$out/fluent/"
  done
  [ -f "$dir/$name.cgns" ] && cp "$dir/$name.cgns" "$out/cgns/"
  # A polyMesh seed is its five files joined by NUL bytes (fuzz_reader.cpp).
  [ -d "$dir/polyMesh" ] || continue
  for part in points faces owner neighbour boundary; do
    cat "$dir/polyMesh/$part"
    printf '\0'
  done > "$out/polymesh/$name"
done
