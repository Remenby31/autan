#include "autan/solvers/dense.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace autan::solvers {

void factor_dense(std::span<double> a, std::size_t m, std::span<std::size_t> pivot) {
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < m; ++i) {
      if (std::abs(a[i * m + k]) > std::abs(a[p * m + k])) p = i;
    }
    if (a[p * m + k] == 0.0 || !std::isfinite(a[p * m + k])) {
      throw std::domain_error("singular coarsest operator");
    }
    pivot[k] = p;
    if (p != k) std::swap_ranges(&a[k * m], &a[k * m] + m, &a[p * m]);
    for (std::size_t i = k + 1; i < m; ++i) {
      const double f = a[i * m + k] /= a[k * m + k];
      if (f == 0.0) continue;
      for (std::size_t j = k + 1; j < m; ++j) a[i * m + j] -= f * a[k * m + j];
    }
  }
}

void solve_dense(std::span<const double> lu, std::span<const std::size_t> pivot,
                 std::span<double> x) {
  const std::size_t m = pivot.size();
  for (std::size_t k = 0; k < m; ++k) std::swap(x[k], x[pivot[k]]);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < i; ++j) x[i] -= lu[i * m + j] * x[j];
  }
  for (std::size_t i = m; i-- > 0;) {
    for (std::size_t j = i + 1; j < m; ++j) x[i] -= lu[i * m + j] * x[j];
    x[i] /= lu[i * m + i];
  }
}

}  // namespace autan::solvers
