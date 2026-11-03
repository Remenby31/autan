#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
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

// Inverse by Gauss-Jordan elimination: the test's own, independent of the library's.
Block inverse_of(const Block& a) {
  std::array<std::array<double, 10>, 5> m{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) m[i][j] = a[static_cast<std::size_t>(5 * i + j)];
    m[i][5 + i] = 1.0;
  }
  for (int p = 0; p < 5; ++p) {
    int best = p;
    for (int r = p + 1; r < 5; ++r) {
      if (std::abs(m[r][p]) > std::abs(m[best][p])) best = r;
    }
    std::swap(m[p], m[best]);
    const double d = m[p][p];
    for (double& v : m[p]) v /= d;
    for (int r = 0; r < 5; ++r) {
      if (r == p) continue;
      const double f = m[r][p];
      for (int j = 0; j < 10; ++j) m[r][j] -= f * m[p][j];
    }
  }
  Block out{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) out[static_cast<std::size_t>(5 * i + j)] = m[i][5 + j];
  }
  return out;
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
  double worst = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::max(worst, std::abs(x[i] - v[i]));
  CHECK(worst <= 1000.0 * eps);
}

TEST_CASE("block ILU(0) reproduces the matrix on its block pattern", "[block-ilu]") {
  // The defining property of ILU(0) (Saad 2003, section 10.3.2): (LU)_ij = A_ij for every (i, j)
  // in the pattern of A, here on the 5-point pattern of a 4 x 4 grid of cells, whose elimination
  // fills blocks outside the pattern.
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
  }
  const BlockCsr a(n, entries);
  auto blocks = a.blocks;
  std::vector<double> inverse(25 * n);
  std::array<char, 256> err{};
  REQUIRE(autan_mms_block_ilu0(a.n, a.indptr.data(), a.indices.data(), blocks.data(),
                               inverse.data(), err.data(), err.size()) == 0);
  // The factors: L strictly lower with identity diagonal blocks; U upper, its diagonal blocks the
  // inverses of the returned pivot inverses.
  std::map<std::pair<int, int>, Block> lower, upper;
  for (int i = 0; i < n; ++i) {
    for (auto p = a.indptr[static_cast<std::size_t>(i)];
         p < a.indptr[static_cast<std::size_t>(i) + 1]; ++p) {
      const int j = a.indices[static_cast<std::size_t>(p)];
      Block b{};
      std::copy_n(blocks.begin() + 25 * p, 25, b.begin());
      if (j < i) {
        lower[{i, j}] = b;
      } else if (j > i) {
        upper[{i, j}] = b;
      }
    }
    Block pivot{};
    std::copy_n(inverse.begin() + 25 * i, 25, pivot.begin());
    upper[{i, i}] = inverse_of(pivot);
    Block identity{};
    for (int k = 0; k < 5; ++k) identity[static_cast<std::size_t>(6 * k)] = 1.0;
    lower[{i, i}] = identity;
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
