#pragma once

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
  /// The least finite `value` over all ranks and the `id` that goes with it,
  /// the least id among equal values: the same pair on any number of ranks.
  [[nodiscard]] std::pair<double, std::int64_t> argmin(double value, std::int64_t id) {
    const double least = min(value);
    const std::int64_t mine = value == least ? -id : std::numeric_limits<std::int64_t>::min();
    return {least, -max(mine)};
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
