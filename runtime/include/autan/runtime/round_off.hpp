#pragma once

#include <algorithm>
#include <array>
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
/// must drop like the others from a reference of its kind (woken_reference()). The runtime's
/// convergence decision and a physics' pseudo-time step use this one rule: fed the same reported
/// residuals, they decide the same.
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
    first_.assign(first.begin(), first.end());
    floor_ = 1e-12 * *std::ranges::max_element(first);
    for (std::size_t e = 0; e < first.size(); ++e) {
      zero_.push_back(first[e] < floor_);
      exempt_.push_back(zero_[e] && std::ranges::find(woken_, names_[e]) == woken_.end());
    }
  }

  /// Whether equation `e` was identically zero at iteration 1 and has left its floor.
  [[nodiscard]] bool woken(std::size_t e) const { return zero_[e] && !exempt_[e]; }

  /// The value a woken equation drops from: the largest residual at iteration 1 of the momentum
  /// components for a momentum component, which share units, of every equation otherwise.
  [[nodiscard]] double woken_reference(std::size_t e) const {
    const bool momentum = names_[e].starts_with("momentum-");
    double largest = 0.0;
    for (std::size_t k = 0; k < first_.size(); ++k) {
      if (!momentum || names_[k].starts_with("momentum-")) largest = std::max(largest, first_[k]);
    }
    return largest;
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
  std::vector<bool> exempt_, zero_;
  std::vector<double> first_;
  double floor_ = 0.0;
};

/// A plateau's reference is the first value, then each value below plateau_ratio times the
/// reference before it; the sequence is on a plateau once plateau_steps values have passed
/// without a new reference.
inline constexpr std::size_t plateau_steps = 100;
inline constexpr double plateau_ratio = 0.9;

/// The plateau rule over the values themselves: euler.md#time-stepping's stall of the first-order
/// start, fed its relative drop once per step.
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

  /// What add() has accumulated, so that a continued run goes on as the uninterrupted one.
  struct State {
    double reference = std::numeric_limits<double>::infinity();
    std::size_t count = 0, since = 0;
  };
  [[nodiscard]] State state() const { return {reference_, count_, since_}; }
  void restore(const State& s) {
    reference_ = s.reference;
    count_ = s.count;
    since_ = s.since;
  }

 private:
  double reference_ = std::numeric_limits<double>::infinity();
  std::size_t count_ = 0, since_ = 0;
};

/// case-schema.md#time: the round-off plateau of a residual, over its level, the median of its
/// last plateau_window counted values (the mean of the two middle ones). On stationary round-off
/// a level's relative spread is at most 0.022 (#557), so plateau_ratio is 4.5 of them, where one
/// value's spread (up to 0.20) made the reference a draw of the noise's lower tail.
inline constexpr std::size_t plateau_window = 100;

class ResidualPlateau {
 public:
  /// Adds this counted step's residual; true once the residual is on a plateau.
  bool add(double value) {
    window_[count_ % plateau_window] = value;
    ++count_;
    if (count_ >= plateau_window) {
      std::array<double, plateau_window> sorted = window_;
      const auto upper = sorted.begin() + plateau_window / 2;
      std::nth_element(sorted.begin(), upper, sorted.end());
      const double level = (*std::max_element(sorted.begin(), upper) + *upper) / 2;
      if (level < plateau_ratio * reference_) {
        reference_ = level;
        since_ = count_;
      }
    }
    return since_ > 0 && count_ - since_ >= plateau_steps;
  }

  /// Forgets every value: the next add() starts a new sequence.
  void reset() { *this = ResidualPlateau{}; }

 private:
  std::array<double, plateau_window> window_{};
  double reference_ = std::numeric_limits<double>::infinity();
  std::size_t count_ = 0, since_ = 0;
};

}  // namespace autan::runtime
