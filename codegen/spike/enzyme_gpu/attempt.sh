#!/bin/bash
# One pinned attempt at Enzyme on the DGX Spark (research section 4.3 item 5, spike day 8):
# LLVM 22 from apt.llvm.org, Enzyme v0.0.303 built against it, clang CUDA for sm_121 with the
# CUDA 13.0 toolkit of the dev image. Run as root inside a throwaway autan-dev container.
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq wget gnupg lsb-release software-properties-common >/dev/null
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && bash /tmp/llvm.sh 22 >/dev/null 2>&1
apt-get install -y -qq llvm-22-dev llvm-22-tools libclang-22-dev clang-22 libzstd-dev zlib1g-dev >/dev/null
clang++-22 --version
cd /tmp && wget -q https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v0.0.303.tar.gz && tar xzf v0.0.303.tar.gz
cmake -S /tmp/Enzyme-0.0.303/enzyme -B /tmp/enzyme-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-22/build/utils/lit/lit.py 2>&1 | tail -5
cmake --build /tmp/enzyme-build --target ClangEnzyme-22 2>&1 | tail -3
plugin=$(find /tmp/enzyme-build -name 'ClangEnzyme-22.so' | head -1)
clang++-22 -O2 -x cuda --cuda-gpu-arch=sm_121 --cuda-path=/usr/local/cuda -fplugin="$plugin" \
  "$1/square.cu" -o /tmp/square -L/usr/local/cuda/lib64 -lcudart 2>&1 | tail -20
/tmp/square
echo "exit=$?"
