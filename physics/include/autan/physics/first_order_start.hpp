#pragma once

#include "autan/runtime/round_off.hpp"

namespace autan::physics {

/// euler.md#time-stepping: a steady Euler run starts at first order and solves second order from
/// the step after the one whose first-order residual has dropped `drop_to` below its start, whose
/// drop has stalled (runtime::Plateau), or that is the 201st: at most 201 steps are provisional
/// (case-schema.md#time).
class FirstOrderStart {
 public:
  static constexpr double drop_to = 1e-3;
  static constexpr int max_steps = 200;

  /// The relative drop of the step just taken; true when the start ends with that step.
  bool ends_after(double drop) {
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

  /// What ended the start: the drop to `drop_to`, the stall, or the step cap.
  enum class End { none, drop, stall, cap };
  [[nodiscard]] End end() const noexcept { return end_; }
  [[nodiscard]] static const char* name(End e) noexcept {
    switch (e) {
      case End::drop:
        return "drop";
      case End::stall:
        return "stall";
      case End::cap:
        return "cap";
      case End::none:
        break;
    }
    return "none";
  }

  /// What a continued run carries over, in the physics' RestartState.
  struct State {
    int steps = 0;
    runtime::Plateau::State plateau;
  };
  [[nodiscard]] State state() const { return {steps_, plateau_.state()}; }
  void restore(const State& s) {
    steps_ = s.steps;
    plateau_.restore(s.plateau);
  }

 private:
  runtime::Plateau plateau_;
  int steps_ = 0;
  End end_ = End::none;
};

}  // namespace autan::physics
