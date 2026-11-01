#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/runtime/exact_sum.hpp"
#include "exact_sum_device.hpp"

using autan::runtime::ExactSum;

namespace {

double host_sum(const std::vector<double>& terms) {
  ExactSum s;
  s.add(terms);
  return s.value();
}

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

}  // namespace

TEST_CASE("The device exact sum has the host ExactSum's bits on any launch shape",
          "[runtime][exact-sum][gpu]") {
  // Terms spread over the whole exponent range, of both signs, subnormals
  // included, and a non-multiple of the warp and block sizes; the second sum
  // cancels to a value far below its terms.
  const std::size_t n = GENERATE(std::size_t{1}, std::size_t{1000}, std::size_t{300001});
  const int blocks = GENERATE(0, 1, 7);
  CAPTURE(n, blocks);
  std::mt19937_64 random(n);
  std::uniform_real_distribution<double> exponent(-1070.0, 1000.0);
  std::vector<double> a(n), b(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = (i % 3 == 0 ? -1.0 : 1.0) * std::exp2(exponent(random));
    b[i] = (i % 2 == 0 ? 1.0 : -1.0) * std::ldexp(1.0, 600);
  }
  b[0] += std::numeric_limits<double>::denorm_min();
  const auto sums = autan_test::device_sums(a, b, blocks);
  CHECK(same_bits(sums[0].value(), host_sum(a)));
  CHECK(same_bits(sums[1].value(), host_sum(b)));

  // Merged with a host partial sum, as Ranks::sum merges ranks.
  ExactSum merged = sums[0];
  merged.merge(sums[1]);
  std::vector<double> both = a;
  both.insert(both.end(), b.begin(), b.end());
  CHECK(same_bits(merged.value(), host_sum(both)));
}

TEST_CASE("The device exact sum counts non-finite terms as the host does",
          "[runtime][exact-sum][gpu]") {
  constexpr double inf = std::numeric_limits<double>::infinity();
  std::vector<double> a(100, 1.0), b(100, 1.0);
  a[37] = inf;
  b[5] = inf;
  b[90] = -inf;
  const auto sums = autan_test::device_sums(a, b, 0);
  CHECK(sums[0].value() == inf);
  CHECK(std::isnan(sums[1].value()));
}
