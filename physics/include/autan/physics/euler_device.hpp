#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/physics/euler.hpp"
#include "autan/runtime/physics.hpp"
#include "autan/runtime/ranks.hpp"

namespace autan::physics {

// The blocks of the implicit step's matrix as the device assembled them, in the host's terms:
// `diagonal` 25 per cell, row-major; for cell c, its incidences offsets[c] .. offsets[c + 1] in
// the device's order, each with its face and the coupling block to the cell across it, row-major,
// zero on a boundary face (`off_diagonal`, 25 per incidence).
struct DeviceStepperBlocks {
  std::vector<double> diagonal;
  std::vector<kernels::offset_t> offsets;
  std::vector<index_t> faces;
  std::vector<double> off_diagonal;
};

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
  // Switches the linear reconstruction on or off, as Euler<P>::set_second_order.
  virtual void set_second_order(bool on) = 0;
  virtual void upload(std::span<const double> u) = 0;
  virtual void download(std::span<double> u) = 0;
  // The implicit step's matrix at the device state, kept on the device: as
  // Euler<P>::residual_and_jacobian then Euler<P>::assemble(shift), operation for operation.
  // Needs the face moments (a host that is not planar). Throws std::logic_error for a physics
  // whose terms have no device Jacobian.
  virtual void assemble(std::span<const double> shift) = 0;
  // The blocks of the last assemble.
  virtual DeviceStepperBlocks jacobian_blocks() = 0;
  // At the state of the last assemble: dres = Euler<P>::gradient_tangent(s, State(du), ., Faces::
  // boundary) from zero, the term EulerPhysics::add_boundary_gradient_term adds to the operator.
  // Throws std::logic_error before assemble, or for a physics without it.
  virtual void boundary_gradient_tangent(std::span<const double> du, std::span<double> dres) = 0;
};

// The stepper of `host`'s residual (its boundary conditions, source, gradient weights and
// scheme, as set when this is called; the linear reconstruction if `second_order`) on the
// current CUDA device, its work on `stream`. Ghost cells (periodic images, other ranks' cells)
// take the primitives, gradients and limiter through `ghosts`, as on the host. With `keys`, those
// `host` was built with, the face loops gather each cell's faces in the host's global order and
// orientation, and the geometry is packed with the same keys.
template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const Euler<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys = nullptr);

}  // namespace autan::physics
