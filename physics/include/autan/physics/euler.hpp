#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/physics/block_matrix.hpp"

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
};

struct Gas {
  double gamma = 1.4;
  double gas_constant = 287.0;  ///< J/(kg K)
};

/// Cells whose values enter a cell's least-squares gradient, beyond its face neighbours
/// (autan::kernels::StencilKind): `face2` adds the face neighbours of face neighbours, `vertex`
/// every cell sharing a node.
enum class GradientStencil : std::uint8_t { face2, vertex };

struct Scheme {
  Flux flux = Flux::roe;
  GradientStencil gradient = GradientStencil::face2;
  /// Linear reconstruction with least-squares gradients; first order otherwise.
  bool second_order = true;
  /// Venkatakrishnan limiter with threshold eps_k^2 = (K h / length)^3 scale_k^2, h^3 = V.
  bool limiter = true;
  double venkat_k = 5.0;
  double length = 1.0;                                   ///< m
  std::array<double, 5> scale{1.0, 1.0, 1.0, 1.0, 1.0};  ///< W units: rho, u, v, w, p
  /// Cut-off Mach number of AUSM+-up's low-Mach scaling (the freestream Mach number).
  double mach_ref = 0.3;
  /// Reference conserved state (the freestream): cells store U - U_ref and fluxes use the gauge
  /// pressure p - p_ref, so that FP32 keeps the resolution of the variations. Density > 0.
  std::array<double, 5> reference{1.0, 0.0, 0.0, 0.0, 2.5};
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

  Euler(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme);
  ~Euler();
  Euler(const Euler&) = delete;
  Euler& operator=(const Euler&) = delete;

  [[nodiscard]] index_t n_cells() const noexcept;
  [[nodiscard]] index_t n_boundary_faces() const noexcept;

  /// Sets each boundary face's condition and its data (5 x n_boundary_faces).
  void set_boundary(std::span<const BoundaryKind> kind, std::span<const double> data);
  /// Switches the linear reconstruction on or off (Scheme::second_order).
  void set_second_order(bool on);
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

  /// After residual(u): sum over each cell's faces of (|v . n| + c) A (n_cells), so that
  /// CFL V / lambda is the cell's convective time step.
  void spectral_radius(std::span<State> lambda);
  /// After residual(s): absolute pressure of the boundary flux's face state at every boundary face
  /// (n_boundary_faces, FP64) and, unless `force` is empty, the pressure force on each face as
  /// the residual integrates it, warped-face correction included (3 x n_boundary_faces).
  void boundary_pressure(std::span<double> p, std::span<double> force = {}) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::physics
