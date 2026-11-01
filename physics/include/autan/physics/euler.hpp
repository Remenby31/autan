#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

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

struct Scheme {
  Flux flux = Flux::roe;
  /// Linear reconstruction with least-squares gradients; first order otherwise.
  bool second_order = true;
  /// Venkatakrishnan limiter with threshold eps_k^2 = (K h / length)^3 scale_k^2, h^3 = V.
  bool limiter = true;
  double venkat_k = 5.0;
  double length = 1.0;                                   ///< m
  std::array<double, 5> scale{1.0, 1.0, 1.0, 1.0, 1.0};  ///< W units: rho, u, v, w, p
  /// Cut-off Mach number of AUSM+-up's low-Mach scaling (the freestream Mach number).
  double mach_ref = 0.3;
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
  /// Sets the volume source s (5 x n_cells), per unit volume; zero by default.
  void set_source(std::span<const double> source);

  /// R(u) into res; both 5 x n_cells.
  void residual(std::span<const State> u, std::span<Res> res);
  /// First-order residual's tangent dR/dU du (no reconstruction): the preconditioning operator.
  void first_order_tangent(std::span<const State> u, std::span<const State> du,
                           std::span<Res> dres);

  /// After residual(u): sum over each cell's faces of (|v . n| + c) A (n_cells), so that
  /// CFL V / lambda is the cell's convective time step.
  void spectral_radius(std::span<State> lambda);
  /// After residual(u): pressure of the reconstruction at every boundary face's centroid, from
  /// its owner cell (n_boundary_faces, FP64).
  void boundary_pressure(std::span<double> p) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace autan::physics
