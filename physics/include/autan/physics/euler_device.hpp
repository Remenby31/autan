#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/physics/euler.hpp"
#include "autan/runtime/ranks.hpp"

namespace autan::physics {

// Explicit SSP-RK3 steps of the Euler residual on a CUDA device. The conserved variables stay on
// the device between steps, in FP64 like the host's, and every update repeats the host's
// arithmetic operation for operation; the residual is the generated loops' (kernels/euler.py).
template <class P>
class DeviceStepper {
 public:
  struct Result {
    std::vector<double> residuals;   // RMS over the owned cells of R at the step's start state
    double time_step = 0.0;          // the global time step; the smallest local one if local
    std::optional<index_t> invalid;  // first_invalid_cell of the host, after the step
  };
  virtual ~DeviceStepper() = default;
  // One step from the device state: time step `dt` if given, else min CFL V / lambda over the
  // cells, or with `local` CFL V / lambda per cell (pseudo-time).
  virtual Result step(double cfl, std::optional<double> dt, bool local) = 0;
  // The host's first_invalid_cell on the device state, and the RMS residual of that state.
  virtual Result evaluate() = 0;
  // R of the stored state of `u` (5 x n_cells, FP64) into `res` (5 x n_cells), on the device.
  virtual void residual(std::span<const double> u, std::span<double> res) = 0;
  virtual void upload(std::span<const double> u) = 0;
  virtual void download(std::span<double> u) = 0;
};

// The stepper of `host`'s residual (its boundary conditions, source, gradient weights and
// scheme, as set when this is called; the linear reconstruction if `second_order`) on the
// current CUDA device, its work on `stream`.
template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const Euler<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      void* stream);

}  // namespace autan::physics
