#pragma once

#include "autan/runtime/physics.hpp"

namespace autan::physics {

/// Adds the compressible inviscid physics (Euler equations, ideal gas) to `registry`, in
/// `fp64` and `fp32+acc64` on the CPU. Steady cases march in pseudo-time with local time steps;
/// unsteady ones with a global time step. Every boundary kind of the solver input that applies
/// to inviscid flow is supported, `manufactured` included.
void register_euler(runtime::PhysicsRegistry& registry);

}  // namespace autan::physics
