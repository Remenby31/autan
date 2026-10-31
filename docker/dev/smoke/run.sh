#!/bin/sh
# Configures, builds and runs the image smoke test inside the dev image.
# Usage: run.sh BUILD_DIR. Runs smoke_cuda only when a GPU is visible.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
build=$1
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-121}"
cmake --build "$build"
cd "$build"
mpirun -n 2 --oversubscribe ./smoke
status=0
mpirun -n 2 --oversubscribe ./smoke_cuda || status=$?
if [ "$status" -eq 77 ] && [ -z "${REQUIRE_GPU:-}" ]; then exit 0; fi
exit "$status"
