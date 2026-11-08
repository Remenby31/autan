#pragma once

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/runtime/physics.hpp"

namespace autan::physics {

template <class P>
class Euler;

/// Adds the compressible inviscid physics (Euler equations, ideal gas) to `registry`, in
/// `fp64` and `fp32+acc64` on the CPU. Steady cases march in pseudo-time with local time steps;
/// unsteady ones with a global time step. Every boundary kind of the solver input that applies
/// to inviscid flow is supported, `manufactured` included.
void register_euler(runtime::PhysicsRegistry& registry);

/// Adds the compressible laminar physics (Navier-Stokes equations, ideal gas, Sutherland or
/// constant viscosity, docs/design/ns-sa.md) to `registry`, with the same precisions, devices and
/// time stepping as register_euler; `wall` boundaries are no-slip, adiabatic or isothermal.
void register_navier_stokes(runtime::PhysicsRegistry& registry);

/// G = max over cells of sum_e |w_e| max_f |x_f - x_c|: how much a boundary value moves the face
/// values the cells reconstruct, over every gradient stencil `flow`'s residual reads (#586,
/// criterion 2(b)). After set_boundary.
template <class P>
double device_stencil_gain(const Euler<P>& flow, const Mesh& mesh, const Geometry& geometry);

/// The largest G at which a laminar device run's agreement with the host on farfield and inlet
/// faces was shown tight in fp32+acc64 (#586, run gpunssa-d4-2b32); a laminar device run in that
/// precision with such faces and a larger G warns W0717.
inline constexpr double device_covered_gain = 4.37325;

}  // namespace autan::physics
