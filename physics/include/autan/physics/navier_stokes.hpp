#pragma once

#include <memory>
#include <optional>
#include <span>
#include <utility>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"

namespace autan::kernels::navier_stokes {
template <class P>
struct Data;
}  // namespace autan::kernels::navier_stokes

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
/// The ghost stencil a partition holds for the Navier-Stokes physics (laminar or SA). Under
/// reproducibility "ranks", vertex4: each rank fits its owned cells and their face neighbours,
/// whose fits reach three vertex rings of them, from its own ghosts, the one-rank bits. Otherwise
/// vertex3: each rank fits its owned cells only, and receives its ghost face neighbours' fits from
/// their owners (#651).
GhostStencil navier_stokes_ghost_stencil(bool ranks_reproducible) noexcept;

template <class P>
class NavierStokes {
 public:
  using State = typename P::state;
  using Res = typename P::residual;
  using Faces = typename Euler<P>::Faces;

  /// With `keys`, as Euler<P>'s: its Euler<P> takes them, the viscous face loops gather each cell's
  /// faces in global order and orientation, the boundary fit lists its entries in ascending global
  /// id, and assemble() sums and places its blocks in that order and orientation. On a partition,
  /// `ghosts` must hold vertex3 at least (E0148 otherwise); below vertex4, the ghosts' fits come
  /// from their owners, through the GhostFill of set_ghosts, which must carry rows.
  NavierStokes(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
               Transport transport, const kernels::GlobalKeys* keys = nullptr,
               std::optional<GhostStencil> ghosts = std::nullopt);
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
  /// The ghost exchange, Euler<P>'s too. The viscous terms fill the gradient they read at a
  /// neighbour (`grad_v`) and compute the boundary fits of the owned cells and of their face
  /// neighbours from values every rank holds alike.
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
  /// A = diag(shift) + J~ into `a`: Euler<P>'s blocks, the viscous face Jacobians and, through
  /// the boundary fit, the viscous fluxes' dependence on the fitted cell and the face's other cell.
  void assemble(std::span<const double> shift, BlockMatrix& a) const;
  /// After residual_and_jacobian(s): adds to dres the chosen faces' derivative through the
  /// gradients that `assemble` leaves out, convective (Euler<P>::gradient_tangent) and viscous
  /// (the cell gradients, and the boundary fit's stencil entries outside the block pattern), the
  /// limiter, the eddy viscosity and the fits' boundary values held fixed.
  void gradient_tangent(std::span<const State> s, std::span<const State> du, std::span<Res> dres,
                        typename Euler<P>::Faces faces);
  /// After assemble: Frobenius norms of the boundary fit's Jacobian blocks it added and of those
  /// outside the block pattern (stencil entries that are not the face's cells), which it leaves
  /// out, over the owned cells' rows: on a partition the ranks' squares sum to the whole mesh's.
  /// Below vertex4, a received row's entries this rank does not hold are not counted.
  [[nodiscard]] std::pair<double, double> fit_jacobian_norms() const;
  /// The convective reconstruction's quadratic gradient (Euler<P>::quadratic_gradient).
  [[nodiscard]] typename Euler<P>::QuadraticGradient quadratic_gradient() const;
  /// The cells that keep the linear gradient (Euler<P>::linear_cells).
  [[nodiscard]] typename Euler<P>::LinearCells linear_cells() const noexcept;
  /// After residual(s): per cell, the convective spectral radius plus 4 times the viscous one
  /// (Blazek 2015, eq. 6.20), so that CFL V / lambda is the cell's time step.
  void spectral_radius(std::span<State> lambda);
  void boundary_pressure(std::span<double> p, std::span<double> force = {}) const;
  /// After residual(s): force per unit area the fluid exerts on every boundary face through the
  /// viscous stress, -tau n (3 x n_boundary_faces, Pa, n out of the fluid); zero on slip faces.
  void viscous_traction(std::span<double> t) const;
  /// After residual(s): the wall state of the viscous wall flux on every boundary face, its
  /// absolute pressure p_w (Pa, the zero-gradient wall pressure of kernels/euler.py wall_state) and
  /// its temperature T_w (K: the condition's on an isothermal wall, the cell's extrapolated to the
  /// face otherwise, kernels/navier_stokes.py boundary_temperature), n_boundary_faces each.
  void wall_state(std::span<double> pressure, std::span<double> temperature) const;

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

  /// For a device copy of the residual, after set_boundary: the Euler residual it adds to, and
  /// the viscous loops' data in host memory (geometry, the boundary fit's stencil and weights,
  /// eddy viscosity, parameters).
  [[nodiscard]] const Euler<P>& euler() const noexcept;
  [[nodiscard]] const kernels::navier_stokes::Data<P>& data() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Dynamic viscosity of `transport` at temperature `t` (K), in Pa s.
[[nodiscard]] double viscosity(const Transport& transport, double t);

/// results.md, WallSolution YPlus: y1 sqrt(|tau_w| rho_w) / mu(T_w), rho_w = p_w / (R T_w), for a
/// face whose first-cell height is `y1` (m), wall shear `tau` (Pa) and wall state (p_w, T_w).
[[nodiscard]] double wall_y_plus(double y1, double tau, double p_wall, double t_wall,
                                 double gas_constant, const Transport& transport);

}  // namespace autan::physics
