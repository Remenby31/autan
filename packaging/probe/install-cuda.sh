#!/bin/sh
# Installs nvcc and the CUDA runtime headers in a manylinux_2_28 (AlmaLinux 8) build container.
# Only the compiler and static runtime are needed: the wheel links cudart statically and loads
# libcuda.so.1 from the user's driver.
set -eu
CUDA_VERSION="${AUTAN_CUDA_VERSION:-13-0}"
case "$(uname -m)" in
  x86_64) repo=x86_64 ;;
  aarch64) repo=sbsa ;;
  *) echo "install-cuda.sh: unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${repo}/cuda-rhel8.repo"

# Distribution mirrors sometimes serve broken metadata ("repomd.xml parser error"). Each attempt
# drops the cached metadata so the mirror list picks again; dnf retries each download too.
attempt=1
until dnf install -y --setopt=install_weak_deps=False --setopt=retries=5 --setopt=timeout=60 \
  "cuda-nvcc-${CUDA_VERSION}" "cuda-cudart-devel-${CUDA_VERSION}" "cuda-cuobjdump-${CUDA_VERSION}"; do
  if [ "$attempt" -ge 4 ]; then
    echo "install-cuda.sh: dnf failed $attempt times" >&2
    exit 1
  fi
  echo "install-cuda.sh: dnf failed (attempt $attempt of 4); refreshing metadata and retrying" >&2
  dnf clean metadata
  sleep $((attempt * 20))
  attempt=$((attempt + 1))
done
ln -sfn "/usr/local/cuda-$(echo "$CUDA_VERSION" | tr - .)" /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
