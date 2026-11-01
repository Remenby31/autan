#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "autan/runtime/exact_sum.hpp"

namespace autan::runtime {

/// The processes a run spans, one part of the mesh each (docs/contracts/runtime.md,
/// Ranks). Every member is collective: all ranks call it, in the same order.
class Ranks {
 public:
  virtual ~Ranks() = default;
  [[nodiscard]] virtual int rank() const noexcept = 0;
  [[nodiscard]] virtual int size() const noexcept = 0;
  /// Replaces each sum by the sum of that sum over all ranks. As ExactSum is
  /// exact, value() then gives the same bits on any number of ranks.
  virtual void sum(std::span<ExactSum> sums) = 0;
  /// The least of `value` over all ranks.
  [[nodiscard]] virtual double min(double value) = 0;
  /// The greatest of `value` over all ranks.
  [[nodiscard]] virtual std::int64_t max(std::int64_t value) = 0;
  [[nodiscard]] virtual double max(double value) = 0;
  /// The least `value` over all ranks and the `id` (>= 0) that goes with it,
  /// the least id among equal values: the same pair on any number of ranks.
  /// NaN counts as +infinity, so it wins only when every value is NaN.
  [[nodiscard]] std::pair<double, std::int64_t> argmin(double value, std::int64_t id) {
    constexpr std::int64_t top = std::numeric_limits<std::int64_t>::max();
    const double key = std::isnan(value) ? std::numeric_limits<double>::infinity() : value;
    const double least = min(key);
    // The least id as the greatest top - id: no overflow for ids >= 0.
    const std::int64_t mine = key == least ? top - id : std::numeric_limits<std::int64_t>::min();
    return {least, top - max(mine)};
  }
};

/// A run on one rank.
class OneRank final : public Ranks {
 public:
  [[nodiscard]] int rank() const noexcept override { return 0; }
  [[nodiscard]] int size() const noexcept override { return 1; }
  void sum(std::span<ExactSum> /*sums*/) override {}
  [[nodiscard]] double min(double value) override { return value; }
  [[nodiscard]] std::int64_t max(std::int64_t value) override { return value; }
  [[nodiscard]] double max(double value) override { return value; }
};

}  // namespace autan::runtime
