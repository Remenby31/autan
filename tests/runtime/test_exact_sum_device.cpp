#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
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
  std::array<double, 2> on_device{};
  const auto sums = autan_test::device_sums(a, b, blocks, &on_device);
  CHECK(same_bits(sums[0].value(), host_sum(a)));
  // Rounded on the device: the host's bits.
  CHECK(same_bits(on_device[0], sums[0].value()));
  CHECK(same_bits(on_device[1], sums[1].value()));
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
  std::array<double, 2> on_device{};
  auto sums = autan_test::device_sums(a, b, 7, &on_device);
  CHECK(std::isinf(on_device[0]));
  CHECK(on_device[0] > 0);
  CHECK(std::isnan(on_device[1]));
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

TEST_CASE("A device exact sum rounds on the device as ExactSum::value()",
          "[runtime][exact-sum][gpu]") {
  // Each sum against IEEE rounding of its exact value: a tie to even down
  // (2^53 + 1) and up (2^53 + 3), beyond the double range (2 DBL_MAX: +inf,
  // and -inf), an exact zero, a negative sum and the least subnormal.
  constexpr double big = std::numeric_limits<double>::max();
  const double p53 = std::ldexp(1.0, 53);
  const std::vector<std::pair<std::vector<double>, double>> cases{
      {{p53, 1.0}, p53},
      {{p53, 3.0}, p53 + 4.0},
      {{big, big}, std::numeric_limits<double>::infinity()},
      {{-big, -big}, -std::numeric_limits<double>::infinity()},
      {{1.5, -1.5}, 0.0},
      {{-3.0, 0.25}, -2.75},
      {{std::numeric_limits<double>::denorm_min(), 0.0}, std::numeric_limits<double>::denorm_min()},
  };
  const double p54 = std::ldexp(1.0, 54);
  const std::vector<std::pair<std::vector<double>, double>> more{
      // Sticky bits below the 64 leading ones, in the same limbs and far below:
      // the tie is broken upwards.
      {{p53, 1.0, std::ldexp(1.0, -11)}, p53 + 2.0},
      {{p53, 1.0, std::ldexp(1.0, -1000)}, p53 + 2.0},
      // Rounding that carries into the exponent, and past the largest double.
      {{p54, -1.0}, p54},
      {{big, std::ldexp(1.0, 970)}, std::numeric_limits<double>::infinity()},
  };
  for (const auto& [terms, expected] : more) {
    CAPTURE(terms, expected);
    ExactSum s;
    s.add(terms);
    CHECK(same_bits(s.value(), expected));
    CHECK(same_bits(autan_test::device_value(s), expected));
  }
  // Non-finite terms, bit for bit with the host (NaN payload included).
  constexpr double inf = std::numeric_limits<double>::infinity();
  for (const std::vector<double>& terms :
       {std::vector<double>{-inf, 1.0}, std::vector<double>{inf, -inf},
        std::vector<double>{std::numeric_limits<double>::quiet_NaN(), inf}}) {
    CAPTURE(terms);
    ExactSum s;
    s.add(terms);
    CHECK(same_bits(autan_test::device_value(s), s.value()));
  }
  // A top limb beyond 32 bits, which no sum of doubles reaches before
  // overflowing: signed infinities.
  for (const std::int64_t top : {std::int64_t{1} << 32, -(std::int64_t{1} << 32)}) {
    CAPTURE(top);
    ExactSum s;
    s.data()[static_cast<std::size_t>(ExactSum::limbs - 1)] = top;
    CHECK(same_bits(s.value(), top > 0 ? inf : -inf));
    CHECK(same_bits(autan_test::device_value(s), s.value()));
  }
  for (const auto& [terms, expected] : cases) {
    CAPTURE(terms, expected);
    std::array<double, 2> on_device{};
    const auto sums =
        autan_test::device_sums(terms, std::vector<double>(terms.size(), 0.0), 0, &on_device);
    CHECK(same_bits(sums[0].value(), expected));
    CHECK(same_bits(on_device[0], expected));
  }
}

TEST_CASE("A device predicate skips the exact sums or does them unchanged",
          "[runtime][exact-sum][gpu]") {
  // Clear: the sums keep their earlier words and no term is evaluated. Set:
  // every term once each, and the bits of the unconditional sums.
  const std::size_t n = 300001;
  std::vector<double> a(n), b(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = std::ldexp(1.0 + static_cast<double>(i % 7), static_cast<int>(i % 200) - 100);
    b[i] = (i % 2 == 0 ? 1.0 : -1.0) * static_cast<double>(i);
  }
  unsigned long long evaluated = 0;
  auto skipped = autan_test::device_sums_if(a, b, 0, evaluated);
  CHECK(evaluated == 0);
  for (auto& s : skipped) {
    for (const std::int64_t w : s.data()) CHECK(w == 0x5a5a5a5a5a5a5a5aLL);
  }
  const auto done = autan_test::device_sums_if(a, b, 1, evaluated);
  CHECK(evaluated == 2 * n);
  const auto always = autan_test::device_sums(a, b, 0);
  for (std::size_t k = 0; k < 2; ++k) {
    auto x = done[k], y = always[k];
    CHECK(std::ranges::equal(x.data(), y.data()));
  }
}
