#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "autan/runtime/round_off.hpp"

namespace autan::physics {

/// euler.md#time-stepping: a steady Euler run starts at first order and solves second order from
/// the step after the one whose first-order residual has dropped `drop_to` below its start, or
/// that is the 201st: at most 201 steps are provisional (case-schema.md#time). In between, an
/// `fp64` run also ends the start once its drop has stalled (runtime::Plateau); an `fp32+acc64`
/// run instead once every watched equation's first-order residual level (the median of its last
/// `window` values) is at most `band` times its round-off floor (#611). The level exists only from
/// the `window`-th step, so this is no adaptive length: where the levels are within the band by
/// then (every NACA 0012 case #259 measured), the start lasts in practice 100 first-order steps,
/// set by the window; where the residual rises, 201 (the cap).
class FirstOrderStart {
 public:
  static constexpr double drop_to = 1e-3;
  static constexpr int max_steps = 200;
  static constexpr std::size_t equations = 5;
  static constexpr std::size_t window = runtime::plateau_window;
  static constexpr double band = 10.0;

  /// `levels`: the fp32+acc64 rule (level crossing) instead of the stall.
  explicit FirstOrderStart(bool levels = false) : levels_(levels) {
    if (levels) samples_.assign(equations * window, 0.0);
  }

  /// The relative drop of the step just taken and, for the level rule, each equation's
  /// first-order residual and floor (a floor of 0: the equation is not watched); true when the
  /// start ends with that step.
  bool ends_after(double drop, std::span<const double> residuals = {},
                  std::span<const double> floors = {}) {
    if (!levels_) {
      const bool flat = plateau_.add(drop);
      if (drop < drop_to) {
        end_ = End::drop;
      } else if (flat) {
        end_ = End::stall;
      } else if (++steps_ > max_steps) {
        end_ = End::cap;
      } else {
        return false;
      }
      return true;
    }
    for (std::size_t k = 0; k < equations; ++k)
      samples_[k * window + count_ % window] = k < residuals.size() ? residuals[k] : 0.0;
    ++count_;
    if (drop < drop_to) {
      end_ = End::drop;
    } else if (crossed(floors)) {
      end_ = End::level;
    } else if (++steps_ > max_steps) {
      end_ = End::cap;
    } else {
      return false;
    }
    return true;
  }

  enum class End { none, drop, stall, cap, level };
  [[nodiscard]] End end() const noexcept { return end_; }
  [[nodiscard]] static const char* name(End e) noexcept {
    switch (e) {
      case End::drop:
        return "drop";
      case End::stall:
        return "stall";
      case End::cap:
        return "cap";
      case End::level:
        return "level";
      case End::none:
        break;
    }
    return "none";
  }

  /// What a continued run carries over: the steps and the stall's plateau in the physics'
  /// RestartState; for the level rule, `window_state()` in its SolverState.
  struct State {
    int steps = 0;
    runtime::Plateau::State plateau;
  };
  [[nodiscard]] State state() const { return {steps_, plateau_.state()}; }
  void restore(const State& s) {
    steps_ = s.steps;
    plateau_.restore(s.plateau);
  }

  /// The level rule's window: the number of samples, then each equation's `window` slots.
  static constexpr std::size_t window_size = 1 + equations * window;
  [[nodiscard]] std::vector<double> window_state() const {
    std::vector<double> v{static_cast<double>(count_)};
    v.insert(v.end(), samples_.begin(), samples_.end());
    return v;
  }
  void restore_window(std::span<const double> v) {
    count_ = static_cast<std::size_t>(v[0]);
    samples_.assign(v.begin() + 1, v.end());
  }

 private:
  // Every watched equation's level at most `band` floors, with a full window; none watched:
  // never.
  [[nodiscard]] bool crossed(std::span<const double> floors) const {
    if (count_ < window) return false;
    bool watched = false;
    for (std::size_t k = 0; k < equations && k < floors.size(); ++k) {
      if (!(floors[k] > 0.0)) continue;
      watched = true;
      // The median as runtime::ResidualPlateau takes it: the mean of the two middle values.
      std::array<double, window> sorted{};
      std::copy_n(samples_.begin() + static_cast<std::ptrdiff_t>(k * window), window,
                  sorted.begin());
      const auto upper = sorted.begin() + window / 2;
      std::nth_element(sorted.begin(), upper, sorted.end());
      const double level = (*std::max_element(sorted.begin(), upper) + *upper) / 2;
      if (!(level <= band * floors[k])) return false;
    }
    return watched;
  }

  bool levels_ = false;
  runtime::Plateau plateau_;
  int steps_ = 0;
  End end_ = End::none;
  std::size_t count_ = 0;
  std::vector<double> samples_;  // the level rule's window, equation by equation
};

}  // namespace autan::physics
