#pragma once

#include <memory>
#include <span>

#include "autan/physics/euler_device.hpp"
#include "autan/physics/spalart_allmaras.hpp"

namespace autan::physics {

// SpalartAllmaras<P> on the CUDA device of a Navier-Stokes stepper (make_device_stepper of a
// NavierStokes<P>): its loops read the flow's arrays of that stepper's last residual, and the
// eddy viscosity goes into that stepper for its next one. rho nt is FP64 in, stored as the
// host's callers store it (P::state).
template <class P>
class DeviceSpalartAllmaras {
 public:
  virtual ~DeviceSpalartAllmaras() = default;
  // SpalartAllmaras<P>::residual of rho nt (n_cells) into `res` (n_cells).
  virtual void residual(std::span<const double> rnu, std::span<double> res) = 0;
  // SpalartAllmaras<P>::eddy_viscosity of rho nt, per cell and per boundary face.
  virtual void eddy_viscosity(std::span<const double> rnu) = 0;
};

// The model of `host` (its wall distance, boundary values, source and parameters when this is
// called) on the device of `flow`, which must come from make_device_stepper of the NavierStokes<P>
// `host` is coupled to (std::invalid_argument otherwise).
template <class P>
std::unique_ptr<DeviceSpalartAllmaras<P>> make_device_spalart_allmaras(
    const SpalartAllmaras<P>& host, DeviceStepper<P>& flow);

}  // namespace autan::physics
