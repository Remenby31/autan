#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "autan/core/error.hpp"
#include "autan/partition/device_self.hpp"

namespace autan {
namespace {

constexpr int max_components = 16;

template <class T>
struct Components {
  T* data[max_components];
  int n;
};

template <class T>
__global__ void copy_images(Components<T> c, const index_t* pairs, std::int64_t n_pairs) {
  const std::int64_t p = std::int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (p >= n_pairs) return;
  const index_t owner = pairs[2 * p], ghost = pairs[2 * p + 1];
  for (int k = 0; k < c.n; ++k) c.data[k][ghost] = c.data[k][owner];
}

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw Error(ErrorCode::halo_device_unsupported,
                std::string(what) + ": " + cudaGetErrorString(e));
  }
}

template <class T>
void run(std::span<T* const> components, const index_t* pairs, std::int64_t n_pairs, void* stream) {
  if (components.size() > static_cast<std::size_t>(max_components)) {
    throw Error(ErrorCode::partition_invalid_request,
                "at most 16 components per forward, not " + std::to_string(components.size()));
  }
  if (n_pairs == 0 || components.empty()) return;
  Components<T> c{};
  c.n = static_cast<int>(components.size());
  for (std::size_t k = 0; k < components.size(); ++k) c.data[k] = components[k];
  constexpr int block = 256;
  const std::int64_t blocks = (n_pairs + block - 1) / block;
  copy_images<<<static_cast<unsigned>(blocks), block, 0, static_cast<cudaStream_t>(stream)>>>(
      c, pairs, n_pairs);
  check(cudaGetLastError(), "copy_images");
}

}  // namespace

DeviceSelfForward::DeviceSelfForward(const HaloPattern& halo, index_t n_owned_cells) {
  std::vector<index_t> pairs;
  for (std::size_t i = 0; i < halo.neighbours.size(); ++i) {
    if (halo.neighbours[i] != halo.rank) continue;
    // As forward_self: the ghosts of block i come from send_cells of block i, in order.
    const index_t count = halo.recv_offsets[i + 1] - halo.recv_offsets[i];
    for (index_t k = 0; k < count; ++k) {
      pairs.push_back(halo.send_cells[static_cast<std::size_t>(halo.send_offsets[i] + k)]);
      pairs.push_back(n_owned_cells + halo.recv_offsets[i] + k);
    }
  }
  if (pairs.empty()) return;
  index_t* device = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device), pairs.size() * sizeof(index_t)),
        "cudaMalloc");
  // From pageable memory a copy may return before its transfer ends, and the
  // caller's stream, possibly non-blocking, would not wait for it.
  const cudaError_t copied =
      cudaMemcpy(device, pairs.data(), pairs.size() * sizeof(index_t), cudaMemcpyHostToDevice);
  const cudaError_t done = copied == cudaSuccess ? cudaDeviceSynchronize() : copied;
  if (done != cudaSuccess) {
    (void)cudaFree(device);
    check(done, "uploading the image pairs");
  }
  pairs_ = device;
  n_pairs_ = static_cast<std::int64_t>(pairs.size() / 2);
}

DeviceSelfForward::~DeviceSelfForward() {
  if (pairs_ != nullptr) (void)cudaFree(pairs_);
}

void DeviceSelfForward::forward(std::span<float* const> components, void* stream) {
  run(components, pairs_, n_pairs_, stream);
}

void DeviceSelfForward::forward(std::span<double* const> components, void* stream) {
  run(components, pairs_, n_pairs_, stream);
}

}  // namespace autan
