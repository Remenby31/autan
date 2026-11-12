#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

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

  // The implicit SA step (ADR 0005, B), on one rank without ghosts under identity keys, as the
  // host's SpalartAllmaras<P> takes it; the arrays stay on the device between the calls.
  // residual_and_jacobian: SpalartAllmaras<P>::residual_and_jacobian of rho nt into `res`.
  virtual void residual_and_jacobian(std::span<const double> rnu, std::span<double> res) = 0;
  // The Jacobian blocks of the last residual_and_jacobian, laid out as the host's data()
  // (nt_cell_jac n_cells, nt_face_jac 2 n_faces, nt_bnd_jac n_faces), converted to FP64.
  virtual void jacobians(std::span<double> cell, std::span<double> face,
                         std::span<double> boundary) = 0;
  // SpalartAllmaras<P>::assemble of the last Jacobian blocks with `shift` (n_cells).
  virtual void assemble(std::span<const double> shift, std::span<double> diag,
                        std::span<double> upper, std::span<double> lower) = 0;
  // SpalartAllmaras<P>::solve of the given system, `dx` the initial guess in and the result out:
  // one launch per colour, colours 0 ... n-1 then n-1 ... 0, `sweeps` times.
  virtual void solve(std::span<const double> diag, std::span<const double> upper,
                     std::span<const double> lower, std::span<const double> b, std::span<double> dx,
                     int sweeps, const std::vector<std::vector<std::size_t>>& colours) = 0;
  // One implicit SA step of rho nt (`rnu`, updated in place): residual_and_jacobian, assemble with
  // `shift`, solve from dx = 0 with b = -R, then update_rho_nt with the flow's updated density
  // `rho`, nt_inf and alpha.
  virtual void step(std::span<double> rnu, std::span<const double> shift,
                    std::span<const double> rho, double nt_inf, double alpha, int sweeps,
                    const std::vector<std::vector<std::size_t>>& colours) = 0;
};

// The model of `host` (its wall distance, boundary values, source and parameters when this is
// called) on the device of `flow`, which must come from make_device_stepper of the NavierStokes<P>
// `host` is coupled to (std::invalid_argument otherwise) and outlive the model. The implicit step's
// parts refuse (std::invalid_argument) a flow with ghost cells or without identity keys.
template <class P>
std::unique_ptr<DeviceSpalartAllmaras<P>> make_device_spalart_allmaras(
    const SpalartAllmaras<P>& host, DeviceStepper<P>& flow);

}  // namespace autan::physics
