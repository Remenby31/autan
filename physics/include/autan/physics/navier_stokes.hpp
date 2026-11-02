#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <utility>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"

namespace autan::physics {

/// Molecular transport of an ideal gas.
struct Transport {
  /// Sutherland's law mu = mu_ref (T / t_ref)^(3/2) (t_ref + s) / (T + s); constant mu_ref
  /// otherwise.
  bool sutherland = true;
  double mu_ref = 1.716e-5;  ///< Pa s
  double t_ref = 273.15;     ///< K
  double s = 110.4;          ///< K
  double prandtl = 0.72;
  double prandtl_turbulent = 0.9;
};

/// Compressible Navier-Stokes residual: the Euler residual of `Euler<P>` plus the viscous fluxes of
/// kernels/navier_stokes.py, with an eddy viscosity set by a turbulence model (zero by default).
/// Same conventions and call sequence as Euler<P>; `wall` boundary faces are no-slip.
template <class P>
class NavierStokes {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  NavierStokes(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
               Transport transport);
  ~NavierStokes();
  NavierStokes(const NavierStokes&) = delete;
  NavierStokes& operator=(const NavierStokes&) = delete;

  [[nodiscard]] index_t n_cells() const noexcept;
  [[nodiscard]] index_t n_boundary_faces() const noexcept;

  /// As Euler<P>'s; it also builds the boundary fit of the viscous fluxes, which the residual
  /// needs (std::logic_error before it) and which refuses a cell whose stencil leaves the fit
  /// undetermined (std::invalid_argument).
  void set_boundary(std::span<const BoundaryKind> kind, std::span<const double> data);
  /// As Euler<P>'s; the fit depends on the conditions only, so it is kept.
  void set_boundary_data(std::span<const double> data);
  /// Euler<P>'s ghost exchange. The viscous terms fill no ghosts of their own: a mesh with
  /// ghosts (periodic pairs, partitions) is outside this physics' scope.
  void set_ghosts(GhostFill* ghosts) noexcept;
  void set_second_order(bool on);
  void set_source(std::span<const double> source);
  /// Eddy viscosity mu_t (Pa s) per cell and per boundary face; zero until set.
  void set_eddy_viscosity(std::span<const double> cells, std::span<const double> boundary_faces);

  void state(std::span<const double> u, std::span<State> s) const;
  void residual(std::span<const State> s, std::span<Res> res);
  /// R(s) and the approximate Jacobian: Euler<P>'s, plus dF_v/dW of every face with the cell
  /// gradients held fixed.
  void residual_and_jacobian(std::span<const State> s, std::span<Res> res);
  /// The arrays of the last evaluation and the eddy viscosity per cell and per face, which a
  /// turbulence model reads and writes (docs/design/ns-sa.md).
  struct Arrays {
    typename Euler<P>::Arrays flow;
    State* mu_t;
    State* mu_t_b;
    double p_ref;  ///< Pa: the primitive pressure is p - p_ref
    double gas_constant;
  };
  [[nodiscard]] Arrays arrays() noexcept;
  /// For the implicit step's matrix-free product: Euler<P>::freeze at s (its limiter and
  /// reconstruction branches held), and a workspace of the viscous terms' own; the viscous
  /// gradients are not held. Leaves what residual() last evaluated untouched.
  void freeze(std::span<const State> s);
  /// R(s) with the branches of freeze()'s state, in that workspace: equal to residual(s) bit for
  /// bit at that state; the eddy viscosity is the one last set. Throws std::logic_error before
  /// freeze().
  void residual_frozen(std::span<const State> s, std::span<Res> res);
  /// The same, with the eddy viscosity a turbulence model writes at s: `eddy` runs once the
  /// primitives of s are in frozen_arrays(), and writes its mu_t and mu_t_b there before the
  /// viscous terms read them.
  void residual_frozen(std::span<const State> s, std::span<Res> res,
                       const std::function<void()>& eddy);
  /// After residual_frozen: how many face sides took the other branch of a reconstruction's
  /// positivity fallback (the cell value where the reconstructed density or pressure would be
  /// non-positive) than at freeze()'s state, for the convective (limited) reconstruction, whose
  /// branch residual_frozen holds, or for the unlimited one of the viscous terms and the
  /// turbulence model's face states, whose branch it does not. Nonzero: the evaluation is not the
  /// smooth continuation of the residual at freeze()'s state. Throws std::logic_error before
  /// freeze().
  [[nodiscard]] std::size_t frozen_switches() const;
  /// During and after residual_frozen: arrays() of its state, in freeze()'s workspace.
  [[nodiscard]] Arrays frozen_arrays();
  /// Frees what freeze() allocated.
  void release_frozen();
  /// A = diag(shift) + J~ into `a`: Euler<P>'s blocks, the viscous face Jacobians and, through
  /// the boundary fit, the viscous fluxes' dependence on the fitted cell and the face's other cell.
  void assemble(std::span<const double> shift, BlockMatrix& a) const;
  /// After assemble: Frobenius norms of the boundary fit's Jacobian blocks it added and of those
  /// outside the block pattern (stencil entries that are not the face's cells), which it leaves
  /// out.
  [[nodiscard]] std::pair<double, double> fit_jacobian_norms() const;
  /// After residual(s): per cell, the convective spectral radius plus 4 times the viscous one
  /// (Blazek 2015, eq. 6.20), so that CFL V / lambda is the cell's time step.
  void spectral_radius(std::span<State> lambda);
  void boundary_pressure(std::span<double> p, std::span<double> force = {}) const;
  /// After residual(s): force per unit area the fluid exerts on every boundary face through the
  /// viscous stress, -tau n (3 x n_boundary_faces, Pa, n out of the fluid); zero on slip faces.
  void viscous_traction(std::span<double> t) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::physics
