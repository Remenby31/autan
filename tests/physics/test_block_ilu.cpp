#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

extern "C" {
int autan_mms_block_ilu0(std::int64_t n, const std::int64_t* indptr, const std::int32_t* indices,
                         double* blocks, double* inverse, char* err, std::size_t err_len);
void autan_mms_block_ilu0_solve(std::int64_t n, const std::int64_t* indptr,
                                const std::int32_t* indices, const double* blocks,
                                const double* inverse, const double* b, double* x);
}

namespace {

using Block = std::array<double, 25>;
constexpr double eps = std::numeric_limits<double>::epsilon();

// A block CSR matrix from (row, column) -> block, rows and columns sorted.
struct BlockCsr {
  std::int64_t n = 0;
  std::vector<std::int64_t> indptr;
  std::vector<std::int32_t> indices;
  std::vector<double> blocks;

  explicit BlockCsr(std::int64_t rows, const std::map<std::pair<int, int>, Block>& entries)
      : n(rows), indptr{0} {
    for (std::int64_t i = 0; i < n; ++i) {
      for (const auto& [at, b] : entries) {
        if (at.first != i) continue;
        indices.push_back(at.second);
        blocks.insert(blocks.end(), b.begin(), b.end());
      }
      indptr.push_back(static_cast<std::int64_t>(indices.size()));
    }
  }
};

Block random_block(std::mt19937_64& rng, double diagonal) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Block b{};
  for (double& v : b) v = u(rng);
  for (int k = 0; k < 5; ++k) b[static_cast<std::size_t>(6 * k)] += diagonal;
  return b;
}

Block product(const Block& a, const Block& b) {
  Block c{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        c[static_cast<std::size_t>(5 * i + j)] +=
            a[static_cast<std::size_t>(5 * i + k)] * b[static_cast<std::size_t>(5 * k + j)];
      }
    }
  }
  return c;
}

}  // namespace

TEST_CASE("block ILU(0) of a block-tridiagonal matrix is its exact LU", "[block-ilu]") {
  // Gaussian elimination of a block-tridiagonal matrix creates no block outside its pattern, so
  // ILU(0) drops nothing and (LU)^-1 A is the identity.
  std::mt19937_64 rng(3);
  const int n = 12;
  std::map<std::pair<int, int>, Block> entries;
  for (int i = 0; i < n; ++i) {
    entries[{i, i}] = random_block(rng, 8.0);
    if (i > 0) {
      entries[{i, i - 1}] = random_block(rng, 0.0);
      entries[{i - 1, i}] = random_block(rng, 0.0);
    }
  }
  const BlockCsr a(n, entries);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(5 * n), av(5 * n, 0.0), x(5 * n);
  for (double& value : v) value = u(rng);
  for (const auto& [at, b] : entries) {
    for (int r = 0; r < 5; ++r) {
      for (int c = 0; c < 5; ++c) {
        av[static_cast<std::size_t>(5 * at.first + r)] +=
            b[static_cast<std::size_t>(5 * r + c)] * v[static_cast<std::size_t>(5 * at.second + c)];
      }
    }
  }
  auto blocks = a.blocks;
  std::vector<double> inverse(25 * n);
  std::array<char, 256> err{};
  REQUIRE(autan_mms_block_ilu0(a.n, a.indptr.data(), a.indices.data(), blocks.data(),
                               inverse.data(), err.data(), err.size()) == 0);
  autan_mms_block_ilu0_solve(a.n, a.indptr.data(), a.indices.data(), blocks.data(), inverse.data(),
                             av.data(), x.data());
  REQUIRE(std::ranges::all_of(x, [](double value) { return std::isfinite(value); }));
  double worst = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::max(worst, std::abs(x[i] - v[i]));
  CHECK(worst <= 1000.0 * eps);
}

TEST_CASE("block ILU(0) reproduces the matrix on its block pattern", "[block-ilu]") {
  // The defining property of ILU(0) (Saad 2003, section 10.3.2): (LU)_ij = A_ij for every (i, j)
  // in the pattern of A. Here the pattern is a 4 x 4 grid of cells, each joined to its 4 neighbours
  // and to one diagonal neighbour: its elimination fills blocks outside the pattern, and its
  // triangles make the elimination update off-diagonal blocks the pattern keeps.
  std::mt19937_64 rng(7);
  const int side = 4, n = side * side;
  std::map<std::pair<int, int>, Block> entries;
  for (int i = 0; i < n; ++i) {
    entries[{i, i}] = random_block(rng, 8.0);
    const int x = i % side, y = i / side;
    if (x + 1 < side) entries[{i, i + 1}] = random_block(rng, 0.0);
    if (x > 0) entries[{i, i - 1}] = random_block(rng, 0.0);
    if (y + 1 < side) entries[{i, i + side}] = random_block(rng, 0.0);
    if (y > 0) entries[{i, i - side}] = random_block(rng, 0.0);
    if (x + 1 < side && y + 1 < side) entries[{i, i + side + 1}] = random_block(rng, 0.0);
    if (x > 0 && y > 0) entries[{i, i - side - 1}] = random_block(rng, 0.0);
  }
  const BlockCsr a(n, entries);
  auto blocks = a.blocks;
  std::vector<double> inverse(25 * n);
  std::array<char, 256> err{};
  REQUIRE(autan_mms_block_ilu0(a.n, a.indptr.data(), a.indices.data(), blocks.data(),
                               inverse.data(), err.data(), err.size()) == 0);
  REQUIRE(std::ranges::all_of(blocks, [](double value) { return std::isfinite(value); }));
  REQUIRE(std::ranges::all_of(inverse, [](double value) { return std::isfinite(value); }));
  // The factors as returned: L strictly lower with identity diagonal blocks, U the upper blocks
  // including the diagonal; each returned inverse times its pivot block is the identity.
  std::map<std::pair<int, int>, Block> lower, upper;
  for (int i = 0; i < n; ++i) {
    for (auto p = a.indptr[static_cast<std::size_t>(i)];
         p < a.indptr[static_cast<std::size_t>(i) + 1]; ++p) {
      const int j = a.indices[static_cast<std::size_t>(p)];
      Block b{};
      std::copy_n(blocks.begin() + 25 * p, 25, b.begin());
      if (j < i) {
        lower[{i, j}] = b;
      } else {
        upper[{i, j}] = b;
      }
    }
    Block identity{};
    for (std::size_t k = 0; k < 5; ++k) identity[6 * k] = 1.0;
    lower[{i, i}] = identity;
    Block pivot_inverse{};
    std::copy_n(inverse.begin() + 25 * i, 25, pivot_inverse.begin());
    const Block one = product(upper.at({i, i}), pivot_inverse);
    for (std::size_t e = 0; e < 25; ++e) CHECK(std::abs(one[e] - identity[e]) <= 1000.0 * eps);
  }
  double worst = 0.0;
  for (const auto& [at, expected] : entries) {
    Block lu{};
    for (int k = 0; k <= std::min(at.first, at.second); ++k) {
      const auto l = lower.find({at.first, k});
      const auto u = upper.find({k, at.second});
      if (l == lower.end() || u == upper.end()) continue;
      const Block term = product(l->second, u->second);
      for (std::size_t e = 0; e < 25; ++e) lu[e] += term[e];
    }
    for (std::size_t e = 0; e < 25; ++e) worst = std::max(worst, std::abs(lu[e] - expected[e]));
  }
  CHECK(worst <= 1000.0 * eps);
}

TEST_CASE("block ILU(0) refuses a singular pivot block", "[block-ilu]") {
  std::map<std::pair<int, int>, Block> entries;
  entries[{0, 0}] = Block{};
  entries[{1, 1}] = Block{};
  for (int k = 0; k < 5; ++k) entries[{1, 1}][static_cast<std::size_t>(6 * k)] = 1.0;
  const BlockCsr a(2, entries);
  auto blocks = a.blocks;
  std::vector<double> inverse(50);
  std::array<char, 256> err{};
  CHECK(autan_mms_block_ilu0(a.n, a.indptr.data(), a.indices.data(), blocks.data(), inverse.data(),
                             err.data(), err.size()) != 0);
  CHECK(std::string(err.data()).starts_with("E1100"));
}
