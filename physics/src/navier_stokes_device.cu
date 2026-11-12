#include <memory>
#include <stdexcept>

#include "navier_stokes_device.cuh"

namespace autan::physics {

using device::NavierStokesStepper;
using device::SpalartAllmarasDevice;

template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const NavierStokes<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys) {
  return std::make_unique<NavierStokesStepper<P>>(mesh, geometry, host, gamma, second_order, ranks,
                                                  ghosts, stream, keys);
}

template <class P>
std::unique_ptr<DeviceSpalartAllmaras<P>> make_device_spalart_allmaras(
    const SpalartAllmaras<P>& host, DeviceStepper<P>& flow) {
  auto* ns = dynamic_cast<NavierStokesStepper<P>*>(&flow);
  if (ns == nullptr || !ns->copies(host.data().mu_t)) {
    throw std::invalid_argument(
        "Spalart-Allmaras on a device needs the stepper of the flow it is coupled to");
  }
  return std::make_unique<SpalartAllmarasDevice<P>>(host, *ns);
}

template std::unique_ptr<DeviceSpalartAllmaras<kernels::fp64>> make_device_spalart_allmaras(
    const SpalartAllmaras<kernels::fp64>&, DeviceStepper<kernels::fp64>&);
template std::unique_ptr<DeviceSpalartAllmaras<kernels::fp32_acc64>> make_device_spalart_allmaras(
    const SpalartAllmaras<kernels::fp32_acc64>&, DeviceStepper<kernels::fp32_acc64>&);

template std::unique_ptr<DeviceStepper<kernels::fp64>> make_device_stepper(
    const Mesh&, const Geometry&, const NavierStokes<kernels::fp64>&, double, bool, runtime::Ranks&,
    runtime::GhostExchange&, void*, const kernels::GlobalKeys*);
template std::unique_ptr<DeviceStepper<kernels::fp32_acc64>> make_device_stepper(
    const Mesh&, const Geometry&, const NavierStokes<kernels::fp32_acc64>&, double, bool,
    runtime::Ranks&, runtime::GhostExchange&, void*, const kernels::GlobalKeys*);

}  // namespace autan::physics
