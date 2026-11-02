#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"

namespace autan::physics {

/// The bounds of rho nt's change in one steady step: at most `decrease` below and `increase`
/// above max(|rho nt|, rho nt_inf), relative because SA-neg allows rho nt < 0.
struct NtBounds {
  double decrease = 0.5;
  double increase = 2.0;
};

/// The largest fraction alpha <= 1 of the update dnt that keeps every cell's change of rho nt
/// within `bounds` (docs/design/coupled-sa.md): the whole step is shortened by it, never clipped
/// cell by cell. 0 when an update is NaN. `rho`, `rnu`, `dnt` per cell; `nt_inf` in m^2/s.
[[nodiscard]] double nt_step_fraction(std::span<const double> rho, std::span<const double> rnu,
                                      std::span<const double> dnt, double nt_inf, NtBounds bounds);

/// The matrix-free product of the steady step that solves the flow and rho nt together
/// (docs/design/coupled-sa.md): y = shift v + [R(X + eps v) - R(X)] / eps over the six unknowns
/// of each cell, equation-major (rho, rho u, rho v, rho w, rho E, rho nt), R the coupled
/// residual with its switches held at X by NavierStokes::freeze and SpalartAllmaras::freeze.
/// eps follows Knoll and Keyes (2004) on S^-1-scaled unknowns, halved until the probe is
/// admissible (density and pressure within 10 %, rho nt within its step bounds) and
/// representable in the stored state; after 10 halvings every later product is NaN. Equations
/// left out have zero rows and their entries of v are ignored. One rank: every cell is owned.
template <class P>
class CoupledProduct {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  /// `flow_scale`: the scales of the five conserved unknowns (the freestream's); rho nt's is
  /// the root mean square of rho nt at each start().
  CoupledProduct(NavierStokes<P>& flow, SpalartAllmaras<P>& model, double gamma,
                 std::array<double, 5> flow_scale, NtBounds bounds);

  /// Holds the switches at X = (u, rnu) and evaluates R(X). `u`: conserved, 5 n; `state`: the
  /// flow's stored state of u; `rnu`: n; `shift`: V/dt per cell. Call after the flow's
  /// residual at `state` and the model's residual at rnu, as their freeze() requires.
  void start(std::span<const double> u, std::span<const State> state, std::span<const double> rnu,
             double nt_inf, std::span<const double> shift, std::array<bool, 6> left_out);
  void apply(std::span<const double> v, std::span<double> y);

  /// The inner product and largest entry of residuals scaled by S^-1, for the GCR.
  [[nodiscard]] double dot(std::span<const double> a, std::span<const double> b) const;
  [[nodiscard]] double max_abs(std::span<const double> a) const;

  /// R(X) of start(), 6 n.
  [[nodiscard]] std::span<const double> residual() const { return base_; }
  [[nodiscard]] const std::array<double, 6>& scale() const { return scale_; }
  /// The largest number of halvings of eps, and of |S^-1 (stored - eps v)| / |S^-1 eps v|, over
  /// the products since start().
  [[nodiscard]] int halvings() const { return halvings_; }
  [[nodiscard]] double representation() const { return representation_; }

 private:
  [[nodiscard]] double scaled_rms(std::span<const double> x, bool skip_left_out) const;

  NavierStokes<P>& flow_;
  SpalartAllmaras<P>& model_;
  double gamma_;
  NtBounds bounds_;
  std::size_t n_;
  double eps_storage_;
  std::array<double, 6> scale_{};
  std::array<bool, 6> left_out_{};
  double nt_inf_ = 0.0;
  std::vector<double> current_, probe_, base_, shift_;
  std::vector<State> state_, probe_state_, nt_, nt_current_;
  std::vector<Res> res_, res_nt_;
  int halvings_ = 0;
  double representation_ = 0.0;
  bool failed_ = false;
};

}  // namespace autan::physics
