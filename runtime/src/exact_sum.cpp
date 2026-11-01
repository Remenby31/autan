#include "autan/runtime/exact_sum.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>

namespace autan::runtime {
namespace {

// Limb i weighs 2^(32 i - offset): bit 0 of limb 0 lies below 2^-1074, the
// least double, and the top limb holds 2^1056 and beyond, with the sign.
constexpr int offset = 1088;
constexpr int pos_inf = ExactSum::limbs, neg_inf = pos_inf + 1, nan_count = pos_inf + 2;
constexpr std::uint64_t low32 = 0xffffffffULL;
// A term adds less than 2^33 to a limb, so int64 limbs absorb 2^29 terms
// before their carries must move up.
constexpr std::int64_t normalize_every = std::int64_t{1} << 29;

}  // namespace

void ExactSum::add(double term) noexcept {
  const auto bits = std::bit_cast<std::uint64_t>(term);
  const auto biased = static_cast<int>((bits >> 52) & 0x7ff);
  const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52) - 1);
  const bool negative = (bits >> 63) != 0;
  if (biased == 0x7ff) {
    ++words_[fraction != 0 ? nan_count : (negative ? neg_inf : pos_inf)];
    return;
  }
  if (biased == 0 && fraction == 0) return;
  // term = ±m 2^e with m an integer of at most 53 bits and e >= -1074.
  const std::uint64_t m = biased == 0 ? fraction : (fraction | (std::uint64_t{1} << 52));
  const int e = biased == 0 ? -1074 : biased - 1075;
  const int p = e + offset;
  const auto i = static_cast<std::size_t>(p >> 5);
  const int s = p & 31;
  const std::uint64_t lo = (m & low32) << s, hi = (m >> 32) << s;
  const std::int64_t sign = negative ? -1 : 1;
  words_[i] += sign * static_cast<std::int64_t>(lo & low32);
  words_[i + 1] += sign * static_cast<std::int64_t>((lo >> 32) + (hi & low32));
  words_[i + 2] += sign * static_cast<std::int64_t>(hi >> 32);
  if (++pending_ == normalize_every) normalize();
}

void ExactSum::normalize() noexcept {
  for (std::size_t i = 0; i + 1 < static_cast<std::size_t>(limbs); ++i) {
    const std::int64_t carry = words_[i] >> 32;  // floor division by 2^32
    words_[i] -= carry * (std::int64_t{1} << 32);
    words_[i + 1] += carry;
  }
  pending_ = 0;
}

void ExactSum::merge(const ExactSum& other) noexcept {
  ExactSum b = other;
  b.normalize();
  normalize();
  for (std::size_t i = 0; i < words_.size(); ++i) words_[i] += b.words_[i];
}

double ExactSum::value() const noexcept {
  constexpr double inf = std::numeric_limits<double>::infinity();
  if (words_[nan_count] > 0 || (words_[pos_inf] > 0 && words_[neg_inf] > 0))
    return std::numeric_limits<double>::quiet_NaN();
  if (words_[pos_inf] > 0) return inf;
  if (words_[neg_inf] > 0) return -inf;
  ExactSum a = *this;
  a.normalize();
  auto& w = a.words_;
  const bool negative = w[limbs - 1] < 0;
  if (negative) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(limbs); ++i) w[i] = -w[i];
    a.normalize();
  }
  // Every limb now lies in [0, 2^32) but the top one, which is >= 0.
  int t = limbs - 1;
  while (t >= 0 && w[static_cast<std::size_t>(t)] == 0) --t;
  if (t < 0) return 0.0;
  const auto limb = [&](int k) {
    return k < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(w[static_cast<std::size_t>(k)]);
  };
  const std::uint64_t top = limb(t);
  if (top > low32) return negative ? -inf : inf;  // >= 2^1056
  // The 64 leading bits of the sum, and whether any bit below them is set.
  const int b = std::bit_width(top);  // 1..32
  const std::uint64_t below = (limb(t - 1) << 32) | limb(t - 2);
  std::uint64_t lead = (top << (64 - b)) | (below >> b);
  bool sticky = (below & ((std::uint64_t{1} << b) - 1)) != 0;
  for (int k = t - 3; k >= 0 && !sticky; --k) sticky = limb(k) != 0;
  // Round the 64 bits to 53, to nearest, ties to even. Bits below 2^-1074
  // are zero (every term is a multiple of it), so a subnormal result is exact.
  std::uint64_t mantissa = lead >> 11;
  const std::uint64_t rest = lead & 0x7ff;
  if (rest > 0x400 || (rest == 0x400 && (sticky || (mantissa & 1) != 0))) ++mantissa;
  int exponent = 32 * (t - 2) - offset + b + 11;
  if (mantissa == (std::uint64_t{1} << 53)) {
    mantissa >>= 1;
    ++exponent;
  }
  const double magnitude = std::ldexp(static_cast<double>(mantissa), exponent);
  return negative ? -magnitude : magnitude;
}

}  // namespace autan::runtime
