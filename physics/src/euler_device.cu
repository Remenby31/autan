#include <memory>

#include "autan/physics/euler_device.hpp"
#include "device_stepper.cuh"

namespace autan::physics {

template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const Euler<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys,
                                                      bool jacobian) {
  return std::make_unique<device::CudaStepper<P>>(mesh, geometry, host, gamma, second_order, ranks,
                                                  ghosts, stream, keys, jacobian);
}

template std::unique_ptr<DeviceStepper<kernels::fp64>> make_device_stepper(
    const Mesh&, const Geometry&, const Euler<kernels::fp64>&, double, bool, runtime::Ranks&,
    runtime::GhostExchange&, void*, const kernels::GlobalKeys*, bool);
template std::unique_ptr<DeviceStepper<kernels::fp32_acc64>> make_device_stepper(
    const Mesh&, const Geometry&, const Euler<kernels::fp32_acc64>&, double, bool, runtime::Ranks&,
    runtime::GhostExchange&, void*, const kernels::GlobalKeys*, bool);

}  // namespace autan::physics
