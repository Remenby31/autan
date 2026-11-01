#pragma once

#include <span>

#include "autan/partition/partition.hpp"

namespace autan {

/// forward_self on device memory: copies each owned cell's values into the
/// ghosts that are periodic images of it, the pairs held on the device, on
/// `stream` (a cudaStream_t, null for the default stream). Components hold
/// n_cells values each, at most HaloExchange's 16 per call. Throws E0145 in
/// a build without CUDA, and E0140 for more than 16 components.
class DeviceSelfForward {
 public:
  DeviceSelfForward(const HaloPattern& halo, index_t n_owned_cells);
  DeviceSelfForward(const DeviceSelfForward&) = delete;
  DeviceSelfForward& operator=(const DeviceSelfForward&) = delete;
  ~DeviceSelfForward();

  void forward(std::span<float* const> components, void* stream);
  void forward(std::span<double* const> components, void* stream);

 private:
  index_t* pairs_ = nullptr;  // device: owner, ghost, owner, ghost...
  index_t n_pairs_ = 0;
};

}  // namespace autan
