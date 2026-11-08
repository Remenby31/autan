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

namespace autan::solvers {
class LineSmoother;
}  // namespace autan::solvers

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
  // One SSP-RK3 step from the device state: time step `dt` if given, else min CFL V / lambda
  // over the cells, or with `local` CFL V / lambda per cell (pseudo-time: the viscous physics'
  // steady step, ADR 0005, until its device Jacobian lands, #163 PRs 6 and 7).
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
  // The implicit steady step on the device (EulerPhysics::steady_step with linear_solver =
  // "lines"), its control left to the caller. Each returns at most one value to the host.
  struct LinearSolve {
    int cycles = 0;
    double residual_drop = 1.0;  // NaN when not finite
    bool singular = false;       // a singular diagonal block: the solve failed
  };
  // Euler<P>::residual_and_jacobian at the device state, then the spectral radius; the RMS
  // residuals over the owned cells, as Result::residuals.
  virtual std::vector<double> evaluate_jacobian() = 0;
  // Assembles the blocks of that evaluation with shift_c = lambda_c / cfl.
  virtual void assemble_blocks(double cfl) = 0;
  // The fine level's line smoother of the host's `lines`, copied to the device.
  virtual void set_lines(const solvers::LineSmoother& lines) = 0;
  // du from A du = -R by GCR preconditioned by `sweeps` symmetric line sweeps, with the
  // boundary gradient term in the operator when `gradient_term`, as EulerPhysics::steady_step.
  virtual LinearSolve solve(int sweeps, int max_cycles, double tolerance, bool gradient_term) = 0;
  // EulerPhysics::admissible_step on the device: the largest alpha in (0, 1], halved from 1,
  // keeping rho and p above (1 - max_decrease) of their values in every owned cell.
  virtual double admissible_step(double gamma_minus_one, double max_decrease, double min_step) = 0;
  // u += alpha du.
  virtual void apply_step(double alpha) = 0;
  // The host's first_invalid_cell on the device state and the residual of its last evaluation.
  virtual std::optional<index_t> first_invalid() = 0;
  // The last du (5 x n_cells, FP64), for the diagnostics of a diverged run.
  virtual void download_step(std::span<double> du) = 0;
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
// orientation, and the geometry is packed with the same keys. With `jacobian`, the face moments
// are kept for the Jacobian loops (assemble, the implicit steady step) even under planar loops.
template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const Euler<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys = nullptr,
                                                      bool jacobian = false);

}  // namespace autan::physics
