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
  // `a`: terms over the whole exponent range, of both signs. `b`: pairs
  // +-2^600 that cancel, then one or two least subnormals, whose exact sum
  // the low limbs alone carry. Sizes are not multiples of a warp or a block.
  const std::size_t n = GENERATE(std::size_t{1}, std::size_t{1000}, std::size_t{300001});
  const int blocks = GENERATE(0, 1, 7);
  CAPTURE(n, blocks);
  constexpr double tiny = std::numeric_limits<double>::denorm_min();
  std::mt19937_64 random(n);
  std::uniform_real_distribution<double> exponent(-1070.0, 1000.0);
  std::vector<double> a(n), b(n, tiny);
  for (std::size_t i = 0; i < n; ++i)
    a[i] = (i % 3 == 0 ? -1.0 : 1.0) * std::exp2(exponent(random));
  const std::size_t pairs = (n - 1) / 2;
  for (std::size_t i = 0; i < 2 * pairs; ++i)
    b[i] = (i % 2 == 0 ? 1.0 : -1.0) * std::ldexp(1.0, 600);
  const auto sums = autan_test::device_sums(a, b, blocks);
  CHECK(same_bits(sums[0].value(), host_sum(a)));
  CHECK(same_bits(sums[1].value(), static_cast<double>(n - 2 * pairs) * tiny));
  CHECK(same_bits(sums[1].value(), host_sum(b)));
}

TEST_CASE("A device exact sum merges with host sums exactly", "[runtime][exact-sum][gpu]") {
  // {2^100, 1} on the device and {-2^100} on the host: 1, which a device sum
  // rounded before the merge would lose.
  const auto sums = autan_test::device_sums({std::ldexp(1.0, 100), 1.0}, {0.0, 0.0}, 0);
  ExactSum host;
  host.add(-std::ldexp(1.0, 100));
  host.merge(sums[0]);
  CHECK(host.value() == 1.0);
}

TEST_CASE("The device exact sum counts non-finite terms as the host does",
          "[runtime][exact-sum][gpu]") {
  // Specials spread over many blocks: three +inf in `a`; in `b`, two -inf
  // and one NaN. The counts are words of ExactSum (data()), after the limbs.
  constexpr double inf = std::numeric_limits<double>::infinity();
  const std::size_t n = 300001;
  std::vector<double> a(n, 1.0), b(n, 1.0);
  for (const std::size_t i : {std::size_t{3}, std::size_t{150000}, n - 1}) a[i] = inf;
  b[10] = -inf;
  b[200000] = -inf;
  b[299999] = std::numeric_limits<double>::quiet_NaN();
  auto sums = autan_test::device_sums(a, b, 7);
  const auto words_a = sums[0].data();
  const auto words_b = sums[1].data();
  constexpr auto limbs = static_cast<std::size_t>(ExactSum::limbs);
  CHECK(words_a[limbs] == 3);      // +inf
  CHECK(words_a[limbs + 1] == 0);  // -inf
  CHECK(words_a[limbs + 2] == 0);  // NaN
  CHECK(words_b[limbs] == 0);
  CHECK(words_b[limbs + 1] == 2);
  CHECK(words_b[limbs + 2] == 1);
  CHECK(sums[0].value() == inf);
  CHECK(std::isnan(sums[1].value()));
  // -inf alone stays -inf.
  b[299999] = 1.0;
  CHECK(autan_test::device_sums(a, b, 0)[1].value() == -inf);
}
