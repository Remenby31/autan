#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "autan/core/wall_distance_device.hpp"

namespace autan {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
T* upload(const T* host, std::size_t n) {
  if (n == 0) return nullptr;
  T* device = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device), n * sizeof(T)), "cudaMalloc");
  const cudaError_t copied = cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice);
  if (copied != cudaSuccess) {
    (void)cudaFree(device);
    check(copied, "cudaMemcpy");
  }
  return device;
}

__global__ void distances(wall::View surface, const double* x, const double* y, const double* z,
                          std::int64_t n, double* out) {
  const std::int64_t i = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = wall::distance(surface, {x[i], y[i], z[i]});
}

}  // namespace

DeviceWallSurface::DeviceWallSurface(const WallSurface& host) {
  const wall::View h = host.view();
  const auto triangles = host.triangles();
  try {
    view_.nodes = upload(h.nodes, host.n_nodes());
    view_.n_nodes = h.n_nodes;
    view_.a = upload(h.a, triangles);
    view_.b = upload(h.b, triangles);
    view_.c = upload(h.c, triangles);
    view_.directions = upload(h.directions, host.n_directions());
    view_.reach = upload(h.reach, host.n_directions());
    view_.n_directions = h.n_directions;
    // From pageable memory a copy may return before its transfer ends; a
    // non-blocking stream would not wait for it.
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  } catch (...) {
    for (const void* p :
         {static_cast<const void*>(view_.nodes), static_cast<const void*>(view_.a),
          static_cast<const void*>(view_.b), static_cast<const void*>(view_.c),
          static_cast<const void*>(view_.directions), static_cast<const void*>(view_.reach)})
      (void)cudaFree(const_cast<void*>(p));
    throw;
  }
}

DeviceWallSurface::~DeviceWallSurface() {
  for (const void* p :
       {static_cast<const void*>(view_.nodes), static_cast<const void*>(view_.a),
        static_cast<const void*>(view_.b), static_cast<const void*>(view_.c),
        static_cast<const void*>(view_.directions), static_cast<const void*>(view_.reach)})
    (void)cudaFree(const_cast<void*>(p));
}

void DeviceWallSurface::distance(const double* x, const double* y, const double* z, std::int64_t n,
                                 double* out, void* stream) const {
  if (n <= 0) return;
  constexpr int block = 128;
  const std::int64_t blocks = (n + block - 1) / block;
  distances<<<static_cast<unsigned>(blocks), block, 0, static_cast<cudaStream_t>(stream)>>>(
      view_, x, y, z, n, out);
  check(cudaGetLastError(), "wall distances");
}

}  // namespace autan
