#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace autan::runtime {

/// case-schema.md#time: an equation whose residual at iteration 1 of the original run is below
/// 1e-12 of the largest residual there is identically zero (the spanwise momentum of a
/// one-cell-thick slab); it is exempt while it stays at or below that floor, and once above it
/// must drop like the others. The runtime's convergence decision and a physics' pseudo-time
/// step use this one rule: fed the same reported residuals, they decide the same.
class RoundOffExemption {
 public:
  /// `names` are the equations; `woken`, those that left their floor earlier in a continued
  /// chain (solver-input.md, `start.woken`).
  RoundOffExemption(std::vector<std::string> names, std::vector<std::string> woken)
      : names_(std::move(names)), woken_(std::move(woken)) {}

  /// Sets the floor from the residuals at iteration 1 of the original run (`start.
  /// residual_reference` in a continuation), unless already set.
  void reference(std::span<const double> first) {
    if (!exempt_.empty()) return;
    floor_ = 1e-12 * *std::ranges::max_element(first);
    for (std::size_t e = 0; e < first.size(); ++e) {
      exempt_.push_back(first[e] < floor_ && std::ranges::find(woken_, names_[e]) == woken_.end());
    }
  }

  /// This iteration's residuals: the equations that leave their floor now, no longer exempt.
  std::vector<std::size_t> update(std::span<const double> residuals) {
    reference(residuals);
    std::vector<std::size_t> woke;
    for (std::size_t e = 0; e < residuals.size(); ++e) {
      if (exempt_[e] && residuals[e] > floor_) {
        exempt_[e] = false;
        woke.push_back(e);
      }
    }
    return woke;
  }

  [[nodiscard]] bool exempt(std::size_t e) const { return !exempt_.empty() && exempt_[e]; }
  [[nodiscard]] double floor() const { return floor_; }
  [[nodiscard]] const std::string& name(std::size_t e) const { return names_[e]; }

 private:
  std::vector<std::string> names_, woken_;
  std::vector<bool> exempt_;
  double floor_ = 0.0;
};

/// case-schema.md#time: the reference of a sequence is its first value, then each value below
/// plateau_ratio times the reference before it; the sequence is on a plateau once plateau_steps
/// values have passed without a new reference. Converging fp64 NACA 0012 runs (#473) go at most
/// 49 steps without one, so 100 does not take a converging run for a stationary one.
inline constexpr std::size_t plateau_steps = 100;
inline constexpr double plateau_ratio = 0.9;

/// The plateau rule over one sequence (a residual, or a physics' relative drop), fed one value
/// per step.
class Plateau {
 public:
  /// Adds this step's value; true once the sequence is on a plateau.
  bool add(double value) {
    ++count_;
    if (value < plateau_ratio * reference_) {
      reference_ = value;
      since_ = count_;
    }
    return count_ - since_ >= plateau_steps;
  }

  /// Forgets every value: the next add() starts a new sequence.
  void reset() { *this = Plateau{}; }

 private:
  double reference_ = std::numeric_limits<double>::infinity();
  std::size_t count_ = 0, since_ = 0;
};

}  // namespace autan::runtime
