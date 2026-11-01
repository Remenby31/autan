#pragma once

#include <memory>
#include <span>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/physics/navier_stokes.hpp"

namespace autan::physics {

/// Spalart-Allmaras, negative variant (SA-neg), loosely coupled to a NavierStokes<P> residual
/// (kernels/spalart_allmaras.py, docs/design/ns-sa.md). The state is rho nt per cell (FP64 in,
/// the policy's state type stored), nt the working variable nu-tilde. Each evaluation reads the
/// primitives and gradients of the flow's last residual evaluation, so the flow is evaluated first.
template <class P>
class SpalartAllmaras {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  /// `ft2`: false gives SA-noft2 (manufactured solutions of that variant only).
  SpalartAllmaras(const Mesh& mesh, const Geometry& geometry, NavierStokes<P>& flow,
                  const Transport& transport, bool ft2 = true);
  ~SpalartAllmaras();
  SpalartAllmaras(const SpalartAllmaras&) = delete;
  SpalartAllmaras& operator=(const SpalartAllmaras&) = delete;

  /// nt at every boundary face whose condition gives it (manufactured, farfield, inlet; the
  /// others are ignored): n_boundary_faces values, m^2/s.
  void set_boundary(std::span<const double> nt);
  /// Distance of every cell centroid to the nearest wall (m); a manufactured solution's own.
  void set_wall_distance(std::span<const double> d);
  /// Volume source added to the right-hand side, per unit volume; zero by default.
  void set_source(std::span<const double> source);

  /// R(s) into res, both n_cells, after the flow's residual; the eddy viscosity of s is then
  /// written to the flow for its next evaluation.
  void residual(std::span<const State> s, std::span<Res> res);
  /// The same, with the scalar Jacobian blocks of the implicit step.
  void residual_and_jacobian(std::span<const State> s, std::span<Res> res);
  /// After residual_and_jacobian: solves (diag(shift) + J) dx = b with `sweeps` symmetric
  /// Gauss-Seidel sweeps in cell order (deterministic), J the Jacobian with the negative part of
  /// each cell's source derivative dropped so that J stays an M-matrix-like operator.
  void solve(std::span<const double> shift, std::span<const double> b, std::span<double> dx,
             int sweeps) const;
  /// Writes the eddy viscosity of s to the flow, per cell and per boundary face.
  void eddy_viscosity(std::span<const State> s);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::physics
