#!/usr/bin/env bash
# Derives the OpenFOAM and Fluent test meshes from the Gmsh ones with
# OpenFOAM's own converters. Runs inside opencfd/openfoam-default:2506:
#   docker run --rm -v "$WORK:/work" opencfd/openfoam-default:2506 \
#     bash -l /work/convert.sh /work/gmsh /work/out
# $1 holds the output of generate.py, $2 receives one directory per mesh.
set -euo pipefail
src=$1
out=$2
mkdir -p "$out"

new_case() {
  rm -rf "$1" && mkdir -p "$1/system" && cd "$1"
  cat > system/controlDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application none; startFrom startTime; startTime 0; stopAt endTime; endTime 1;
deltaT 1; writeControl timeStep; writeInterval 1;
writeFormat ascii; writePrecision 17; writeCompression off;
EOF
  cat > system/fvSchemes <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {}
interpolationSchemes {} snGradSchemes {}
EOF
  cat > system/fvSolution <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
EOF
}

# Boundary types that exercise the readers: one wall, one symmetry plane.
set_types() {
  foamDictionary constant/polyMesh/boundary -entry entry0/ymin/type -set wall > /dev/null
  foamDictionary constant/polyMesh/boundary -entry entry0/zmax/type -set symmetryPlane > /dev/null
}

export_case() {
  local name=$1
  checkMesh > log.checkMesh 2>&1 || { cat log.checkMesh; exit 1; }
  grep -q "^Mesh OK." log.checkMesh || { cat log.checkMesh; exit 1; }
  foamMeshToFluent > log.foamMeshToFluent 2>&1
  mkdir -p "$out/$name/polyMesh"
  for f in points faces owner neighbour boundary cellZones; do
    [ -f "constant/polyMesh/$f" ] && cp "constant/polyMesh/$f" "$out/$name/polyMesh/"
  done
  cp fluentInterface/*.msh "$out/$name/$name.fluent.msh"
}

# Binary copy of the current case's polyMesh, written by OpenFOAM itself.
export_binary() {
  local name=$1
  sed -i 's/writeFormat ascii;/writeFormat binary;/' system/controlDict
  foamFormatConvert -constant -noFunctionObjects > log.foamFormatConvert 2>&1
  mkdir -p "$out/$name/polyMesh-binary"
  for f in points faces owner neighbour boundary cellZones; do
    [ -f "constant/polyMesh/$f" ] && cp "constant/polyMesh/$f" "$out/$name/polyMesh-binary/"
  done
  grep -q "format *binary" "$out/$name/polyMesh-binary/points"
}

for name in hex prism pyramid tet hybrid tet-gmsh; do
  new_case "/tmp/cases/$name"
  gmshToFoam "$src/$name.v22.msh" > log.gmshToFoam 2>&1
  set_types
  export_case "$name"
  [ "$name" = hybrid ] && export_binary "$name"
  cp "$src/$name.ascii.msh" "$src/$name.binary.msh" "$out/$name/"
done

# Polyhedral dual of the unstructured tetrahedral mesh; feature angle 80
# keeps the cube's edges and corners.
new_case /tmp/cases/poly
gmshToFoam "$src/tet-gmsh.v22.msh" > log.gmshToFoam 2>&1
set_types
polyDualMesh 80 -overwrite > log.polyDualMesh 2>&1
# polyDualMesh leaves the tetrahedral cellZones empty; drop them.
rm -f constant/polyMesh/cellZones
export_case poly
export_binary poly

new_case /tmp/cases/hanging
mkdir -p constant && cp -r "$src/hanging/constant/polyMesh" constant/
export_case hanging
