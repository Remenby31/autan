#!/bin/sh
# Writes the server's surface test data with the producer itself: the MeshInfo and the
# autan.surface/1 boundary of tests/data/mesh/hybrid in four formats. Needs autan-mesh-info
# built at build/io (cmake --build build --target autan-mesh-info), e.g. through docker/spark.sh.
set -eu
repo=$(cd "$(dirname "$0")/../.." && pwd)
out="$repo/ui/server/tests/data/surface"
tool="$repo/build/io/autan-mesh-info"
mkdir -p "$out"
for mesh in hybrid.ascii.msh hybrid.cgns hybrid.fluent.msh polyMesh; do
  name=$(echo "$mesh" | tr . -)
  "$tool" --surface "$out/$name.surface" "$repo/tests/data/mesh/hybrid/$mesh"
  "$tool" --json "$repo/tests/data/mesh/hybrid/$mesh" > "$out/$name.meshinfo.json"
done
