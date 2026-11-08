#pragma once

#include "autan/runtime/physics.hpp"

namespace autan::physics {

/// Adds the compressible inviscid physics (Euler equations, ideal gas) to `registry`, in
/// `fp64` and `fp32+acc64` on the CPU. Steady cases march in pseudo-time with local time steps;
/// unsteady ones with a global time step. Every boundary kind of the solver input that applies
/// to inviscid flow is supported, `manufactured` included.
void register_euler(runtime::PhysicsRegistry& registry);

/// Adds the compressible laminar physics (Navier-Stokes equations, ideal gas, Sutherland or
/// constant viscosity, docs/design/ns-sa.md) to `registry`, with the same precisions, devices and
/// time stepping as register_euler; `wall` boundaries are no-slip, adiabatic or isothermal.
void register_navier_stokes(runtime::PhysicsRegistry& registry);

}  // namespace autan::physics
