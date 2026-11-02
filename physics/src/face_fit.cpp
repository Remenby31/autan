#include "autan/physics/face_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace autan::physics {

namespace {

// Eigen-decomposition of a symmetric 3 x 3 matrix by cyclic Jacobi rotations (Golub and Van
// Loan, 4th ed., algorithm 8.5.3): a = v diag(l) v^T, v's columns the eigenvectors.
void symmetric_eigen(std::array<double, 9> a, std::array<double, 3>& l, std::array<double, 9>& v) {
  v = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int sweep = 0; sweep < 50; ++sweep) {
    const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    const double diag = a[0] * a[0] + a[4] * a[4] + a[8] * a[8];
    if (off <= 1e-30 * diag) break;
    for (std::size_t p = 0; p < 2; ++p) {
      for (std::size_t q = p + 1; q < 3; ++q) {
        const double apq = a[3 * p + q];
        if (apq == 0.0) continue;
        const double theta = (a[3 * q + q] - a[3 * p + p]) / (2.0 * apq);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < 3; ++k) {
          const double akp = a[3 * k + p], akq = a[3 * k + q];
          a[3 * k + p] = c * akp - s * akq;
          a[3 * k + q] = s * akp + c * akq;
        }
        for (std::size_t k = 0; k < 3; ++k) {
          const double apk = a[3 * p + k], aqk = a[3 * q + k];
          a[3 * p + k] = c * apk - s * aqk;
          a[3 * q + k] = s * apk + c * aqk;
        }
        for (std::size_t k = 0; k < 3; ++k) {
          const double vkp = v[3 * k + p], vkq = v[3 * k + q];
          v[3 * k + p] = c * vkp - s * vkq;
          v[3 * k + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  l = {a[0], a[4], a[8]};
}

}  // namespace

std::vector<std::array<double, 3>> face_fit_weights(
    std::span<const std::array<double, 3>> offsets) {
  std::vector<std::array<double, 3>> w(offsets.size(), {0.0, 0.0, 0.0});
  std::vector<std::array<double, 3>> e(offsets.size(), {0.0, 0.0, 0.0});
  std::vector<double> length(offsets.size(), 0.0);
  std::array<double, 9> b{};
  for (std::size_t j = 0; j < offsets.size(); ++j) {
    const auto& d = offsets[j];
    length[j] = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (length[j] == 0.0) continue;
    for (std::size_t u = 0; u < 3; ++u) e[j][u] = d[u] / length[j];
    for (std::size_t u = 0; u < 3; ++u) {
      for (std::size_t v = 0; v < 3; ++v) b[3 * u + v] += e[j][u] * e[j][v];
    }
  }
  std::array<double, 3> l{};
  std::array<double, 9> v{};
  symmetric_eigen(b, l, v);
  const double largest = std::max({l[0], l[1], l[2]});
  if (!(largest > 0.0)) return w;
  // B+ = sum over kept eigenpairs of v_k v_k^T / l_k.
  std::array<double, 9> pinv{};
  for (std::size_t k = 0; k < 3; ++k) {
    if (!(l[k] > 1e-10 * largest)) continue;
    for (std::size_t u = 0; u < 3; ++u) {
      for (std::size_t s = 0; s < 3; ++s) pinv[3 * u + s] += v[3 * u + k] * v[3 * s + k] / l[k];
    }
  }
  for (std::size_t j = 0; j < offsets.size(); ++j) {
    if (length[j] == 0.0) continue;
    for (std::size_t u = 0; u < 3; ++u) {
      double s = 0.0;
      for (std::size_t q = 0; q < 3; ++q) s += pinv[3 * u + q] * e[j][q];
      w[j][u] = s / length[j];
    }
  }
  return w;
}

}  // namespace autan::physics
