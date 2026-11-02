#include "autan/core/error.hpp"
#include "autan/partition/device_self.hpp"

namespace autan {

DeviceSelfForward::DeviceSelfForward(const HaloPattern& /*halo*/, index_t /*n_owned_cells*/) {
  throw Error(ErrorCode::halo_device_unsupported,
              "this build has no CUDA: periodic images cannot be filled in device memory");
}

DeviceSelfForward::~DeviceSelfForward() = default;

void DeviceSelfForward::forward(std::span<float* const> /*components*/, void* /*stream*/) {}

void DeviceSelfForward::forward(std::span<double* const> /*components*/, void* /*stream*/) {}

}  // namespace autan
