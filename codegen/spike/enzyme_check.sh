#!/bin/sh
# Builds and runs enzyme_check.cpp with clang and the Enzyme plugin of the same LLVM.
#   enzyme_check.sh CLANGXX ENZYME_PLUGIN GENERATED_DIR [extra clang flags]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
clangxx=$1 plugin=$2 gen=$3
shift 3
out=$(mktemp -d)
"$clangxx" -std=c++20 -O2 -fplugin="$plugin" -I"$here/../include" -I"$gen" "$@" \
  "$here/enzyme_check.cpp" -o "$out/enzyme_check"
"$out/enzyme_check"
