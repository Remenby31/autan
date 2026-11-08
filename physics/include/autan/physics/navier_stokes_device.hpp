#pragma once

#include <memory>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/physics/euler_device.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/runtime/physics.hpp"
#include "autan/runtime/ranks.hpp"

namespace autan::physics {

// The stepper of `host`'s residual, NavierStokes<P>::residual, on the current CUDA device: its
// Euler residual as make_device_stepper for Euler<P> gives it, then the viscous fluxes of
// kernels/navier_stokes.py from the boundary fit and eddy viscosity `host` holds when this is
// called. One rank without ghost cells, as NavierStokes<P>. With `keys`, those `host` was built
// with, every face loop gathers in the host's global order.
template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const NavierStokes<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys = nullptr);

}  // namespace autan::physics
