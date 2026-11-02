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

/// The matrix-free product of the viscous steady step (docs/design/matrix-free-step.md; with SA,
/// coupled-sa.md): y = shift v + [R(X + eps v) - R(X)] / eps over the flow's five unknowns of
/// each cell, or six with SA (rho nt last), equation-major, R the residual with its switches
/// held at X (NavierStokes::freeze, SpalartAllmaras::freeze).
/// eps = sqrt(epsilon of the stored state) (1 + rms(S^-1 X)) / rms(S^-1 v) (Knoll and Keyes
/// 2004), each rms over the rows it counts, halved at most 10 times until the probe is
/// admissible (density and pressure within 10 %, rho nt within its step bounds), switches no
/// face state's positivity fallback (NavierStokes::frozen_switches) and gives a finite
/// residual. A probe whose stored state does not receive the perturbation (|S^-1 (stored -
/// eps v)| > 1e-2 |S^-1 eps v|) fails at once, a smaller step only rounding more away; after a
/// failure every product is NaN. One rank: every cell is owned.
template <class P>
class MatrixFreeProduct {
 public:
  using State = typename P::state;
  using Res = typename P::residual;

  /// `model`: SA, or null for the flow alone. `flow_scale`: the scales of the five conserved
  /// unknowns (the freestream's); rho nt's is the root mean square of rho nt at each start().
  MatrixFreeProduct(NavierStokes<P>& flow, SpalartAllmaras<P>* model, double gamma,
                    std::array<double, 5> flow_scale, NtBounds bounds);

  /// Holds the switches at X = (u, rnu) and evaluates R(X). `u`: conserved, 5 n; `state`: the
  /// flow's stored state of u; `rnu`: n with a model, empty without; `shift`: V/dt per cell.
  /// Call after the flow's residual at `state` (and the model's at rnu), as freeze() requires.
  /// The rows of `left_out` equations are zero and their entries of v ignored.
  void start(std::span<const double> u, std::span<const State> state, std::span<const double> rnu,
             double nt_inf, std::span<const double> shift, std::array<bool, 6> left_out);
  void apply(std::span<const double> v, std::span<double> y);

  /// The inner product and largest entry of residuals scaled by S^-1, for the GCR.
  [[nodiscard]] double dot(std::span<const double> a, std::span<const double> b) const;
  [[nodiscard]] double max_abs(std::span<const double> a) const;

  [[nodiscard]] std::size_t equations() const { return m_; }
  /// R(X) of start(), equations() n.
  [[nodiscard]] std::span<const double> residual() const { return base_; }
  [[nodiscard]] const std::array<double, 6>& scale() const { return scale_; }
  /// The largest number of halvings of eps, and of |S^-1 (stored - eps v)| / |S^-1 eps v|, over
  /// the products since start().
  [[nodiscard]] int halvings() const { return halvings_; }
  [[nodiscard]] double representation() const { return representation_; }

 private:
  [[nodiscard]] double scaled_rms(std::span<const double> x, bool skip_left_out) const;

  NavierStokes<P>& flow_;
  SpalartAllmaras<P>* model_;
  double gamma_;
  NtBounds bounds_;
  std::size_t n_;
  std::size_t m_;
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
