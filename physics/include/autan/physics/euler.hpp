#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/block_matrix.hpp"

namespace autan::kernels::euler {
template <class P>
struct Data;
}  // namespace autan::kernels::euler

namespace autan::physics {

/// Numerical flux of the Euler residual.
enum class Flux : std::uint8_t { roe, hllc, ausm_up };

/// Condition of a boundary face, with the meaning of its five data values.
enum class BoundaryKind : std::uint8_t {
  state,     ///< boundary state W = (rho, u, v, w, p) given
  farfield,  ///< freestream W; Riemann invariants normal to the face
  slip,      ///< slip wall or symmetry plane; data unused
  inlet,     ///< subsonic, flow normal to the face: (p0, T0, -, -, -)
  outlet,    ///< subsonic: (-, -, -, -, p); supersonic outflow extrapolates
  wall,      ///< no-slip wall at rest: (T_wall, -, -, -, -), T_wall = 0 when adiabatic
};

struct Gas {
  double gamma = 1.4;
  double gas_constant = 287.0;  ///< J/(kg K)
};

/// Cells whose values enter a cell's least-squares gradient, beyond its face neighbours
/// (autan::kernels::StencilKind): `face2` adds the face neighbours of face neighbours, `vertex`
/// every cell sharing a node.
enum class GradientStencil : std::uint8_t { face2, vertex };

/// Reconstruction gradient (docs/design/euler.md, Quadratic gradient): the linear least squares
/// over the GradientStencil, or the quadratic fit of quadratic_gradient in every cell. Chosen by
/// the physics: the Euler physics keeps the linear one, Navier-Stokes takes the quadratic one.
enum class GradientFit : std::uint8_t { linear, quadratic };

struct Scheme {
  Flux flux = Flux::roe;
  GradientStencil gradient = GradientStencil::face2;
  GradientFit gradient_fit = GradientFit::linear;
  /// Linear reconstruction with least-squares gradients; first order otherwise.
  bool second_order = true;
  /// Venkatakrishnan limiter with threshold eps_k^2 = (K h / length)^3 scale_k^2, h^3 = V.
  bool limiter = true;
  double venkat_k = 10.0;
  double length = 1.0;                                   ///< m
  std::array<double, 5> scale{1.0, 1.0, 1.0, 1.0, 1.0};  ///< W units: rho, u, v, w, p
  /// Cut-off Mach number of AUSM+-up's low-Mach scaling (the freestream Mach number).
  double mach_ref = 0.3;
  /// Reference conserved state (the freestream): cells store U - U_ref and fluxes use the gauge
  /// pressure p - p_ref, so that FP32 keeps the resolution of the variations. Density > 0.
  std::array<double, 5> reference{1.0, 0.0, 0.0, 0.0, 2.5};
};

/// Fills the ghost cells of a partitioned mesh from their owners: each pointer is one component
/// of n_cells values, owned cells then ghosts (at most 16 per call). The runtime's GhostExchange
/// behind it; a mesh without ghosts needs none.
class GhostFill {
 public:
  virtual ~GhostFill() = default;
  virtual void forward(std::span<float* const> components) = 0;
  virtual void forward(std::span<double* const> components) = 0;
};

/// Euler residual R(U) = sum_f F . S - s V of a mesh, in precision policy P
/// (autan::kernels::fp64 or fp32_acc64), on the host. Arrays are structure of arrays: component
/// c of cell i at c * n_cells + i; boundary arrays are indexed by boundary face b, mesh face
/// n_interior_faces + b.
template <class P>
class Euler {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  /// With `keys` (a partition's global numbering, kernels::GlobalKeys), every cell sums its faces
  /// and stencil entries in global order and every face is evaluated in its global orientation,
  /// so that the residual and the Jacobian of each owned cell have the same bits on any number
  /// of ranks and in any cell order (solver-input.md, reproducibility "ranks"). The spans must
  /// outlive the constructor only.
  Euler(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
        const kernels::GlobalKeys* keys = nullptr);
  ~Euler();
  Euler(const Euler&) = delete;
  Euler& operator=(const Euler&) = delete;

  [[nodiscard]] index_t n_cells() const noexcept;
  [[nodiscard]] index_t n_boundary_faces() const noexcept;

  /// On a partition: the residual fills the ghosts' primitives, gradients and limiter through
  /// `ghosts` between its loops, so that the faces of owned cells read their neighbours' values.
  /// Cell loops cover the owned cells and face loops the owned faces. Null: no ghosts.
  void set_ghosts(GhostFill* ghosts) noexcept;
  /// Sets each boundary face's condition and its data (5 x n_boundary_faces).
  void set_boundary(std::span<const BoundaryKind> kind, std::span<const double> data);
  /// Replaces the data of the conditions set_boundary set (5 x n_boundary_faces), keeping them.
  void set_boundary_data(std::span<const double> data);
  /// Every face moment of the mesh is an exact zero (kernels::PackedGeometry::all_planar).
  [[nodiscard]] bool all_planar() const noexcept;
  /// Evaluates the residual with the loops without the warped-face terms (kernels/euler.py,
  /// `*_planar`): where every moment is zero those terms are signed zeros, so the residual has
  /// the same values wherever the full loops' arithmetic stays finite (a state so large that
  /// the warp term overflows makes the full residual NaN and the planar one finite). Throws
  /// std::invalid_argument unless all_planar().
  /// On a partition every rank must make the same choice (EulerPhysics takes the minimum
  /// over the ranks). The Jacobian and tangent loops keep their full form.
  void set_planar(bool on);
  [[nodiscard]] bool planar() const noexcept;
  /// Switches the linear reconstruction on or off (Scheme::second_order).
  void set_second_order(bool on);
  /// The least-squares weights of the gradient over its stencil (layout of
  /// kernels::CellStencil::lsq_weight). They depend on set_boundary: boundary faces take part
  /// where their condition gives the state there, and in the cells whose interior stencil alone
  /// cannot determine a gradient.
  [[nodiscard]] std::vector<double> gradient_weights() const;
  /// With GradientFit::quadratic, the quadratic gradient (kernels/euler.py, band_gradients) of
  /// every cell (empty with linear): per cell 1 where it replaces the linear one, else 0; its
  /// stencil (n_cells + 1 offsets, entries as kernels::Topology::stencil) and weights (3 x
  /// entries, the layout of gradient_weights). They depend on set_boundary. A cell whose two
  /// vertex rings cannot determine the fit keeps the linear gradient (0 there).
  struct QuadraticGradient {
    std::vector<double> fitted;
    std::vector<std::int64_t> offsets;
    std::vector<index_t> entries;
    std::vector<double> weights;
  };
  [[nodiscard]] QuadraticGradient quadratic_gradient() const;
  /// With GradientFit::quadratic, the cells that keep the linear gradient: how many, and the
  /// first (-1 if none). They depend on set_boundary.
  struct LinearCells {
    std::size_t count = 0;
    index_t first = -1;
  };
  [[nodiscard]] LinearCells linear_cells() const noexcept;
  /// After a second-order residual: the linear least-squares gradient of its primitives into
  /// `out` (15 x n_cells, the reconstruction's layout), the reconstruction's own under
  /// GradientFit::linear.
  void linear_gradient(std::span<typename P::grad> out);
  /// Sets the volume source s (5 x n_cells), per unit volume; zero by default.
  void set_source(std::span<const double> source);

  /// The stored state of conserved variables u (5 x n_cells, FP64): u - U_ref.
  void state(std::span<const double> u, std::span<State> s) const;
  /// R(s) into res, s a stored state; both 5 x n_cells.
  void residual(std::span<const State> s, std::span<Res> res);
  /// R(s) into res, and the approximate Jacobian's blocks: dF/dW of every face with the
  /// reconstruction's gradients and limiter held fixed, and dW/dU of every cell.
  void residual_and_jacobian(std::span<const State> s, std::span<Res> res);
  /// After residual_and_jacobian: A = diag(shift) + dR/dU into `a`, shift_c per cell (V / dt).
  void assemble(std::span<const double> shift, BlockMatrix& a) const;
  /// First-order residual's tangent dR/dU du (no reconstruction): the preconditioning operator.
  void first_order_tangent(std::span<const State> u, std::span<const State> du,
                           std::span<Res> dres);
  /// The faces a gradient tangent covers.
  enum class Faces { boundary, interior };
  /// After residual(s) or residual_and_jacobian(s): adds to dres the derivative of the chosen
  /// faces' fluxes through the reconstruction's gradients, (dF/d grad W)(d grad W/dU) du, the
  /// limiter held fixed and the boundary values the gradient fit keeps following their owner cell
  /// (docs/design/euler.md, Boundary values): the term the approximate Jacobian drops (Time
  /// stepping). Nothing for a first-order scheme.
  void gradient_tangent(std::span<const State> s, std::span<const State> du, std::span<Res> dres,
                        Faces faces);
  /// The cells whose linear gradient fit keeps every boundary face, in increasing order:
  /// gradient_tangent adds the tangent of their boundary values (#581). Listed on the first call
  /// after set_boundary, then kept.
  [[nodiscard]] std::span<const std::size_t> kept_cells() const;

  /// After residual(u): sum over each cell's faces of (|v . n| + c) A (n_cells), so that
  /// CFL V / lambda is the cell's convective time step.
  void spectral_radius(std::span<State> lambda);
  /// Arrays of the last evaluation, for physics that add terms to this residual (viscous fluxes,
  /// docs/design/ns-sa.md): primitives W - (0, 0, 0, 0, p_ref) and their gradients per cell,
  /// boundary states, codes and data per face, dW/dU per cell after residual_and_jacobian.
  /// Layouts as the datasets of kernels/euler.py.
  struct Arrays {
    State* prim;
    typename P::grad* grad;
    State* prim_b;
    typename P::geom* bc_kind;
    State* bc;
    const typename P::jac* dwdu;
    kernels::Topology topo;  ///< with the gradient stencil
    const typename P::geom_value* stencil_lsq_weight;
  };
  [[nodiscard]] Arrays arrays() noexcept;

  /// After residual(s): absolute pressure of the boundary flux's face state at every boundary face
  /// (n_boundary_faces, FP64) and, unless `force` is empty, the pressure force on each face as
  /// the residual integrates it, warped-face correction included (3 x n_boundary_faces).
  void boundary_pressure(std::span<double> p, std::span<double> force = {}) const;

  /// For a device copy of the residual: the generated loops' data in host memory (geometry,
  /// boundary conditions, source, parameters), the gradient stencil and the scheme.
  [[nodiscard]] const kernels::euler::Data<P>& data() const;
  [[nodiscard]] const kernels::CellStencil& stencil() const;
  /// The twin of each entry of the gradient stencil (kernels::CellStencil::twin: the entry of
  /// cell c in the stencil of the entry's cell, -1 for a boundary face), for the adjoint's reverse
  /// reductions. No forward loop reads it, so it is built on the first call, not at construction;
  /// stencil().twin stays empty.
  [[nodiscard]] std::span<const kernels::offset_t> stencil_twin();
  [[nodiscard]] const Scheme& scheme() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::physics
