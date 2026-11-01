#!/bin/sh
# Writes the server's surface test data with the producer itself: the MeshInfo and the
# autan.surface/1 boundary of tests/data/mesh/hybrid in four formats, and of its polyMesh with
# overlapping zone groups (mesh-info.md#zone-groups) as `grouped`. Needs autan-mesh-info built at
# build/io (cmake --build build --target autan-mesh-info), e.g. through docker/spark.sh.
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
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$repo/tests/data/mesh/hybrid/polyMesh" "$work/grouped"
sha=$(sed -n 's/.*"sha256": *"\([0-9a-f]*\)".*/\1/p' "$out/polyMesh.meshinfo.json" | head -1)
cat > "$work/grouped.zones.json" <<JSON
{"schema": "autan.zone-groups/1", "mesh_sha256": "$sha", "provenance": {"note": "ui/scripts/surface-fixtures.sh"}, "groups": {"plates": ["ymin", "ymax"], "box.x": ["xmin", "xmax"], "walls": ["ymin"]}}
JSON
cd "$work"
"$tool" --surface "$out/grouped.surface" grouped
"$tool" --json grouped > "$out/grouped.meshinfo.json"
