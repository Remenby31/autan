#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/physics/navier_stokes.hpp"

namespace autan::kernels::spalart_allmaras {
template <class P>
struct Data;
}  // namespace autan::kernels::spalart_allmaras

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

  /// `ft2`: false gives SA-noft2 (manufactured solutions of that variant only). With `keys` (the
  /// flow's), every sum follows the global numbering, as NavierStokes<P>'s (runtime.md, Ranks).
  SpalartAllmaras(const Mesh& mesh, const Geometry& geometry, NavierStokes<P>& flow,
                  const Transport& transport, bool ft2 = true,
                  const kernels::GlobalKeys* keys = nullptr);
  ~SpalartAllmaras();
  SpalartAllmaras(const SpalartAllmaras&) = delete;
  SpalartAllmaras& operator=(const SpalartAllmaras&) = delete;

  /// Per boundary face, n_boundary_faces values: nt (m^2/s) where the condition gives it
  /// (manufactured, farfield); at inlets the ratio nt / nu, nu the laminar viscosity of the
  /// inlet's boundary state at each evaluation; ignored elsewhere.
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
  /// After residual_and_jacobian: the implicit operator diag(shift) + J, J the Jacobian with the
  /// negative part of each cell's source derivative dropped (production does not lower the
  /// diagonal): `diag` per cell, `upper` (row owner, column neighbour) and `lower` (row
  /// neighbour, column owner) per interior face, in mesh face order, as BlockMatrix.
  /// Under the flow's global keys, each diagonal entry sums its cell's faces in global order and
  /// orientation.
  void assemble(std::span<const double> shift, std::span<double> diag, std::span<double> upper,
                std::span<double> lower) const;
  /// Solves the assembled system A dx = b from the initial guess in `dx` with `sweeps` symmetric
  /// Gauss-Seidel sweeps. With `colours` (the owned cells by global colour, colour_groups): colour
  /// by colour, forward then backward, `exchange` filling dx's ghosts after each colour, the same
  /// sweeps on any number of ranks and in any cell order (ADR 0005, option B). Without: cell by
  /// cell, 0 ... n - 1 then back, each row's interior faces in increasing index, for one rank
  /// without ghosts.
  void solve(std::span<const double> diag, std::span<const double> upper,
             std::span<const double> lower, std::span<const double> b, std::span<double> dx,
             int sweeps, const std::vector<std::vector<std::size_t>>* colours = nullptr,
             const std::function<void(std::span<double>)>& exchange = {}) const;
  /// The ghost exchange: nt's gradient, computed on the owned cells, is filled into the ghosts.
  void set_ghosts(GhostFill* ghosts) noexcept;
  /// Writes the eddy viscosity of s to the flow, per cell and per boundary face.
  void eddy_viscosity(std::span<const State> s);
  /// For a device copy: the generated loops' data in host memory (geometry, wall distance,
  /// boundary values, source, parameters). Under the flow's global keys the face loops gather.
  [[nodiscard]] const kernels::spalart_allmaras::Data<P>& data() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// The owned cells (the first `owned` of `mesh`) grouped by colour, in increasing index, for
/// SpalartAllmaras<P>::solve. `colours` gives every cell of `mesh`, owned and ghosts, a colour in
/// [0, n_colours), and no interior face joins two cells of one colour (the runtime's distance-1
/// colouring, PhysicsContext::cell_colours). Refused otherwise (std::invalid_argument, E0720).
std::vector<std::vector<std::size_t>> colour_groups(const Mesh& mesh,
                                                    std::span<const std::int32_t> colours,
                                                    std::int32_t n_colours, std::size_t owned);

}  // namespace autan::physics
