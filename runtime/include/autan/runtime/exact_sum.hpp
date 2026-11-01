#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace autan::runtime {

/// The exact sum of FP64 terms, rounded once to the nearest double (ties to
/// even) by value(). The result does not depend on the order of the terms nor
/// on how they are grouped into partial sums merged with merge(), so a sum
/// over cells split across ranks gives the bits of the sum over one rank.
/// +inf, -inf and NaN terms make value() +inf, -inf or NaN as IEEE 754 would;
/// a finite exact sum beyond the double range rounds to ±inf.
class ExactSum {
 public:
  void add(double term) noexcept;
  void add(std::span<const double> terms) noexcept {
    for (const double t : terms) add(t);
  }
  void merge(const ExactSum& other) noexcept;
  [[nodiscard]] double value() const noexcept;

  /// Fixed-point limbs of 32 bits each in int64 words (limb i weighs
  /// 2^(32 i - offset)), then the counts of +inf, -inf and NaN terms: the
  /// words an MPI reduction adds element-wise after normalize().
  static constexpr int limbs = 68;
  static constexpr int words = limbs + 3;
  void normalize() noexcept;
  [[nodiscard]] std::span<std::int64_t, words> data() noexcept { return words_; }

 private:
  std::array<std::int64_t, words> words_{};
  std::int64_t pending_ = 0;  // additions since the last normalize()
};

}  // namespace autan::runtime
