#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "autan/runtime/exact_sum.hpp"

using autan::runtime::ExactSum;

namespace {

double sum(std::initializer_list<double> terms) {
  ExactSum s;
  for (const double t : terms) s.add(t);
  return s.value();
}

bool same_bits(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

}  // namespace

TEST_CASE("ExactSum rounds the exact sum once, to nearest, ties to even", "[runtime][exact-sum]") {
  constexpr double max = std::numeric_limits<double>::max();
  constexpr double tiny = std::numeric_limits<double>::denorm_min();
  constexpr double inf = std::numeric_limits<double>::infinity();
  const double ulp1 = std::ldexp(1.0, -52);  // spacing of doubles in [1, 2)

  CHECK(sum({}) == 0.0);
  CHECK(sum({std::ldexp(1.0, 100), 1.0, -std::ldexp(1.0, 100)}) == 1.0);
  CHECK(sum({1.0, -3.0}) == -2.0);
  // Intermediate sums beyond the double range, exact result within it.
  CHECK(sum({max, max, -max}) == max);
  CHECK(sum({max, max}) == inf);
  CHECK(sum({-max, -max}) == -inf);
  // Halfway between 1 and 1 + ulp: to even, unless any lower bit is set.
  CHECK(sum({1.0, ulp1 / 2}) == 1.0);
  CHECK(sum({1.0, ulp1 / 2, std::ldexp(1.0, -1000)}) == 1.0 + ulp1);
  CHECK(sum({1.0 + ulp1, ulp1 / 2}) == 1.0 + 2 * ulp1);
  CHECK(sum({-std::ldexp(1.0, 100), 1.0}) == -std::ldexp(1.0, 100));
  // Subnormals are exact.
  CHECK(sum({tiny, tiny, tiny}) == 3 * tiny);
  CHECK(sum({tiny, -tiny}) == 0.0);
  CHECK(sum({std::numeric_limits<double>::min(), -tiny}) ==
        std::nextafter(std::numeric_limits<double>::min(), 0.0));
  // Non-finite terms, as IEEE 754 addition treats them.
  CHECK(sum({inf, 1.0, max}) == inf);
  CHECK(sum({-inf, 1.0}) == -inf);
  CHECK(std::isnan(sum({inf, -inf})));
  CHECK(std::isnan(sum({1.0, std::numeric_limits<double>::quiet_NaN()})));
}

TEST_CASE("ExactSum matches integer arithmetic whatever the order and grouping",
          "[runtime][exact-sum]") {
  // Terms k 2^-20 with |k| < 2^40: their exact sum is an int64 times 2^-20,
  // and converting that int64 to double rounds to nearest, ties to even.
  std::mt19937_64 random(20261101);
  std::uniform_int_distribution<std::int64_t> k(-(std::int64_t{1} << 40), std::int64_t{1} << 40);
  std::vector<std::int64_t> ks(5000);
  std::vector<double> terms;
  std::int64_t exact = 0;
  for (auto& v : ks) {
    v = k(random);
    exact += v;
    terms.push_back(std::ldexp(static_cast<double>(v), -20));
  }
  const double expected = std::ldexp(static_cast<double>(exact), -20);

  ExactSum forward;
  forward.add(terms);
  CHECK(same_bits(forward.value(), expected));
  std::ranges::shuffle(terms, random);
  ExactSum shuffled;
  shuffled.add(terms);
  CHECK(same_bits(shuffled.value(), expected));
  // Four partial sums merged, as four ranks would.
  std::array<ExactSum, 4> parts;
  for (std::size_t i = 0; i < terms.size(); ++i) parts[i % 4].add(terms[i]);
  ExactSum merged;
  for (const auto& p : parts) merged.merge(p);
  CHECK(same_bits(merged.value(), expected));
  // Partial sums that do not round to themselves: {2^100, 1} and {-2^100}.
  ExactSum left, right;
  left.add(std::ldexp(1.0, 100));
  left.add(1.0);
  right.add(-std::ldexp(1.0, 100));
  right.merge(left);
  CHECK(right.value() == 1.0);

  // Widely spread magnitudes: no integer oracle, but every order agrees.
  std::uniform_real_distribution<double> exponent(-1000.0, 1000.0);
  std::vector<double> wide;
  for (int i = 0; i < 2000; ++i)
    wide.push_back((i % 2 == 0 ? 1.0 : -1.0) * std::exp2(exponent(random)));
  ExactSum a, b;
  a.add(wide);
  std::ranges::reverse(wide);
  b.add(wide);
  CHECK(same_bits(a.value(), b.value()));
}

TEST_CASE("ExactSum carries past its lazy limb headroom", "[runtime][exact-sum]") {
  // 2^29 + 3 terms: one normalization during add, the rest in value().
  constexpr std::int64_t n = (std::int64_t{1} << 29) + 3;
  ExactSum s;
  for (std::int64_t i = 0; i < n; ++i) s.add(std::ldexp(1.0, 900));
  CHECK(s.value() == std::ldexp(static_cast<double>(n), 900));
}
