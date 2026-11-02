#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/solvers/dense.hpp"
#include "autan/solvers/multigrid_gpu.hpp"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;

constexpr int b2 = 25;
constexpr int block = 128;

unsigned int grid(std::int64_t threads) {
  return static_cast<unsigned int>((threads + block - 1) / block);
}

kg::stream_t as_stream(void* s) { return static_cast<kg::stream_t>(s); }

// One thread per coarse diagonal block: its 25 entries summed from +0.0 in list order, a fine
// diagonal (entry >= 0) or a fine off-diagonal block (entry -(slot + 1)). __dadd_rn keeps the
// compiler from reassociating or contracting the sums.
template <class Off>
__global__ void __launch_bounds__(block) diagonal_kernel(
    std::int64_t n_coarse, std::int64_t n_fine, std::int64_t n_fine_slots, const offset_t* offsets,
    const std::int64_t* terms, const double* fine_diag, const Off* fine_off, double* coarse_diag) {
  const std::int64_t J = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (J >= n_coarse) return;
  double s[b2];
  for (int q = 0; q < b2; ++q) s[q] = 0.0;
  for (offset_t k = offsets[J]; k < offsets[J + 1]; ++k) {
    const std::int64_t t = terms[k];
    if (t >= 0) {
      for (int q = 0; q < b2; ++q) s[q] = __dadd_rn(s[q], fine_diag[q * n_fine + t]);
    } else {
      const std::int64_t slot = -(t + 1);
      for (int q = 0; q < b2; ++q) {
        s[q] = __dadd_rn(s[q], static_cast<double>(fine_off[q * n_fine_slots + slot]));
      }
    }
  }
  for (int q = 0; q < b2; ++q) coarse_diag[q * n_coarse + J] = s[q];
}

// One thread per coarse slot: the fine off-diagonal blocks of its list, summed from +0.0 in list
// order. Slots no coarse edge uses get a zero block.
template <class Off>
__global__ void __launch_bounds__(block) off_kernel(std::int64_t n_coarse_slots,
                                                    std::int64_t n_fine_slots,
                                                    const offset_t* offsets, const offset_t* terms,
                                                    const Off* fine_off, double* coarse_off) {
  const std::int64_t s = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= n_coarse_slots) return;
  double a[b2];
  for (int q = 0; q < b2; ++q) a[q] = 0.0;
  for (offset_t k = offsets[s]; k < offsets[s + 1]; ++k) {
    const offset_t slot = terms[k];
    for (int q = 0; q < b2; ++q) {
      a[q] = __dadd_rn(a[q], static_cast<double>(fine_off[q * n_fine_slots + slot]));
    }
  }
  for (int q = 0; q < b2; ++q) coarse_off[q * n_coarse_slots + s] = a[q];
}

// x = A⁻¹ b with A⁻¹ dense row-major (m x m): one warp per row, lane l summing columns
// l, l + 32, ... in increasing order, then a fixed butterfly. The result depends only on the
// data and m, not on the launch.
__global__ void dense_product_kernel(std::int64_t m, const double* inverse, const double* b,
                                     double* x) {
  const std::int64_t warp =
      (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kg::warp_size;
  const int lane = static_cast<int>(threadIdx.x) % kg::warp_size;
  if (warp >= m) return;
  const double* row = inverse + warp * m;
  double s = 0.0;
  for (std::int64_t k = lane; k < m; k += kg::warp_size) {
    s = __dadd_rn(s, __dmul_rn(row[k], b[k]));
  }
  for (int d = kg::warp_size / 2; d > 0; d /= 2) s = __dadd_rn(s, kg::shfl_down(s, d));
  if (lane == 0) x[warp] = s;
}

// x = A⁻¹ b by factor_dense's LU (host numbering, row-major m x m) in one block: the host's
// pivot swaps in order, forward substitution by columns (each y_i receives L_ij y_j in increasing
// j, the host's order), back substitution by columns (each x_i receives U_ij x_j in decreasing j,
// then the division). `host_of[kd]` is the host unknown of device unknown kd.
__global__ void triangular_kernel(std::int64_t m, const double* lu, const std::int64_t* pivot,
                                  const std::int64_t* host_of, const double* b, double* x) {
  extern __shared__ double y[];
  for (std::int64_t kd = threadIdx.x; kd < m; kd += blockDim.x) y[host_of[kd]] = b[kd];
  __syncthreads();
  if (threadIdx.x == 0) {
    for (std::int64_t k = 0; k < m; ++k) {
      const double t = y[k];
      y[k] = y[pivot[k]];
      y[pivot[k]] = t;
    }
  }
  __syncthreads();
  for (std::int64_t j = 0; j < m; ++j) {
    const double yj = y[j];
    for (std::int64_t i = j + 1 + threadIdx.x; i < m; i += blockDim.x) {
      y[i] = __dadd_rn(y[i], -__dmul_rn(lu[i * m + j], yj));
    }
    __syncthreads();
  }
  for (std::int64_t j = m - 1; j >= 0; --j) {
    if (threadIdx.x == 0) y[j] = __ddiv_rn(y[j], lu[j * m + j]);
    __syncthreads();
    const double xj = y[j];
    for (std::int64_t i = threadIdx.x; i < j; i += blockDim.x) {
      y[i] = __dadd_rn(y[i], -__dmul_rn(lu[i * m + j], xj));
    }
    __syncthreads();
  }
  for (std::int64_t kd = threadIdx.x; kd < m; kd += blockDim.x) x[kd] = y[host_of[kd]];
}

// The device product's result for one right-hand side, computed on the host in the kernel's
// order (lane sums over columns l, l + 32, ..., then the butterfly), without fused multiply-add.
std::vector<double> emulate_product(const std::vector<double>& inverse, std::size_t m,
                                    const std::vector<double>& b) {
  std::vector<double> x(m);
  constexpr int lanes = 32;
  for (std::size_t r = 0; r < m; ++r) {
    double s[lanes];
    for (int l = 0; l < lanes; ++l) {
      volatile double acc = 0.0;
      for (std::size_t k = static_cast<std::size_t>(l); k < m; k += lanes) {
        const volatile double p = inverse[r * m + k] * b[k];
        acc = acc + p;
      }
      s[l] = acc;
    }
    for (int d = lanes / 2; d > 0; d /= 2) {
      for (int l = 0; l < lanes - d; ++l) {
        volatile double v = s[l] + s[l + d];
        s[l] = v;
      }
    }
    x[r] = s[0];
  }
  return x;
}

}  // namespace

struct Galerkin::Impl {
  std::int64_t n_fine = 0, n_fine_slots = 0, n_coarse = 0, n_coarse_slots = 0;
  kg::DeviceArray<offset_t> diag_offsets, off_offsets, off;
  kg::DeviceArray<std::int64_t> diag;

  template <class Off>
  void apply(const double* fine_diag, const Off* fine_off, double* coarse_diag, double* coarse_off,
             void* stream) const {
    if (n_coarse > 0) {
      diagonal_kernel<Off><<<grid(n_coarse), block, 0, as_stream(stream)>>>(
          n_coarse, n_fine, n_fine_slots, diag_offsets.get(), diag.get(), fine_diag, fine_off,
          coarse_diag);
      kg::check(kg::last_error(), "Galerkin diagonal");
    }
    if (n_coarse_slots > 0) {
      off_kernel<Off><<<grid(n_coarse_slots), block, 0, as_stream(stream)>>>(
          n_coarse_slots, n_fine_slots, off_offsets.get(), off.get(), fine_off, coarse_off);
      kg::check(kg::last_error(), "Galerkin off-diagonal");
    }
  }
};

Galerkin::Galerkin(const GalerkinLists& lists, std::size_t n_fine_nodes, std::size_t n_fine_slots)
    : impl_(std::make_unique<Impl>()) {
  if (lists.diag_offsets.empty() || lists.off_offsets.empty()) {
    throw std::invalid_argument("Galerkin: empty lists");
  }
  for (const std::int64_t t : lists.diag) {
    const bool ok = t >= 0 ? static_cast<std::size_t>(t) < n_fine_nodes
                           : static_cast<std::size_t>(-(t + 1)) < n_fine_slots;
    if (!ok) throw std::invalid_argument("Galerkin: diagonal list entry out of range");
  }
  for (const offset_t s : lists.off) {
    if (s < 0 || static_cast<std::size_t>(s) >= n_fine_slots) {
      throw std::invalid_argument("Galerkin: off-diagonal list entry out of range");
    }
  }
  impl_->n_fine = static_cast<std::int64_t>(n_fine_nodes);
  impl_->n_fine_slots = static_cast<std::int64_t>(n_fine_slots);
  impl_->n_coarse = static_cast<std::int64_t>(lists.diag_offsets.size() - 1);
  impl_->n_coarse_slots = static_cast<std::int64_t>(lists.off_offsets.size() - 1);
  impl_->diag_offsets = kg::DeviceArray<offset_t>(std::span<const offset_t>(lists.diag_offsets));
  impl_->diag = kg::DeviceArray<std::int64_t>(std::span<const std::int64_t>(lists.diag));
  impl_->off_offsets = kg::DeviceArray<offset_t>(std::span<const offset_t>(lists.off_offsets));
  impl_->off = kg::DeviceArray<offset_t>(std::span<const offset_t>(lists.off));
}

Galerkin::~Galerkin() = default;
Galerkin::Galerkin(Galerkin&&) noexcept = default;
Galerkin& Galerkin::operator=(Galerkin&&) noexcept = default;

std::size_t Galerkin::n_coarse_nodes() const { return static_cast<std::size_t>(impl_->n_coarse); }
std::size_t Galerkin::n_coarse_slots() const {
  return static_cast<std::size_t>(impl_->n_coarse_slots);
}

void Galerkin::apply(const double* fine_diag, const double* fine_off, double* coarse_diag,
                     double* coarse_off, void* stream) const {
  impl_->apply(fine_diag, fine_off, coarse_diag, coarse_off, stream);
}

void Galerkin::apply(const double* fine_diag, const float* fine_off, double* coarse_diag,
                     double* coarse_off, void* stream) const {
  impl_->apply(fine_diag, fine_off, coarse_diag, coarse_off, stream);
}

struct CoarsestSolve::Impl {
  std::int64_t m = 0;
  bool explicit_inverse = true;
  std::vector<double> inverse;
  kg::DeviceArray<double> device;  // the inverse, or the LU factors
  kg::DeviceArray<std::int64_t> pivot, host_of;
};

CoarsestSolve::CoarsestSolve(const Level& level, std::span<const index_t> order,
                             std::span<const double> diag, std::span<const double> upper,
                             std::span<const double> lower)
    : impl_(std::make_unique<Impl>()) {
  const std::size_t n = level.n_nodes, m = 5 * n, ne = level.owner.size();
  if (order.size() != n || diag.size() != 25 * n || upper.size() != 25 * ne ||
      lower.size() != 25 * ne) {
    throw std::invalid_argument("CoarsestSolve: sizes do not match the level");
  }
  if (m > 4096) {
    throw std::invalid_argument("CoarsestSolve: " + std::to_string(m) +
                                " unknowns; the dense coarsest solve takes at most 4096");
  }
  // The host multigrid's dense matrix (Multigrid::Smoother::factor_dense): unknown c of node h at
  // c * n + h, each block added once to zero (one edge per pair of nodes on a coarse level).
  std::vector<double> a(m * m, 0.0);
  auto put = [&](std::size_t i, std::size_t j, const double* b) {
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) a[(p * n + i) * m + q * n + j] += b[5 * p + q];
    }
  };
  for (std::size_t i = 0; i < n; ++i) put(i, i, &diag[25 * i]);
  for (std::size_t e = 0; e < ne; ++e) {
    const auto o = static_cast<std::size_t>(level.owner[e]);
    const auto b = static_cast<std::size_t>(level.neighbour[e]);
    put(o, b, &upper[25 * e]);
    put(b, o, &lower[25 * e]);
  }
  std::vector<std::size_t> pivot(m);
  factor_dense(a, m, pivot);
  // Column k of A⁻¹ (host numbering) is the solve of e_k; stored row-major in device numbering:
  // unknown c of device node j is host unknown c * n + order[j].
  std::vector<std::size_t> host_of(m);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j)
      host_of[c * n + j] = c * n + static_cast<std::size_t>(order[j]);
  }
  impl_->inverse.assign(m * m, 0.0);
  std::vector<double> column(m);
  for (std::size_t kd = 0; kd < m; ++kd) {
    std::fill(column.begin(), column.end(), 0.0);
    column[host_of[kd]] = 1.0;
    solve_dense(a, pivot, column);
    for (std::size_t rd = 0; rd < m; ++rd) impl_->inverse[rd * m + kd] = column[host_of[rd]];
  }
  impl_->m = static_cast<std::int64_t>(m);
  // Declared rule (multigrid-gpu.md, Coarsest level): the inverse is kept only if, on three fixed
  // right-hand sides, its product (in the device's operation order) lies within 1e-10 of the
  // host's LU solve; otherwise the device solves by the LU factors.
  for (int probe = 0; probe < 3 && impl_->explicit_inverse; ++probe) {
    std::vector<double> b_device(m), b_host(m);
    for (std::size_t kd = 0; kd < m; ++kd) {
      const auto k = static_cast<double>(kd);
      b_device[kd] = probe == 0 ? std::sin(k + 1.0) : probe == 1 ? 1.0 / (k + 1.0) : 1.0;
      b_host[host_of[kd]] = b_device[kd];
    }
    const std::vector<double> x = emulate_product(impl_->inverse, m, b_device);
    solve_dense(a, pivot, b_host);
    double scale = 0.0, error = 0.0;
    for (std::size_t kd = 0; kd < m; ++kd) {
      scale = std::max(scale, std::abs(b_host[host_of[kd]]));
      error = std::max(error, std::abs(x[kd] - b_host[host_of[kd]]));
    }
    if (!(error <= 1e-10 * scale)) impl_->explicit_inverse = false;
  }
  if (impl_->explicit_inverse) {
    impl_->device = kg::DeviceArray<double>(std::span<const double>(impl_->inverse));
  } else {
    impl_->device = kg::DeviceArray<double>(std::span<const double>(a));
    std::vector<std::int64_t> p(m), h(m);
    for (std::size_t k = 0; k < m; ++k) {
      p[k] = static_cast<std::int64_t>(pivot[k]);
      h[k] = static_cast<std::int64_t>(host_of[k]);
    }
    impl_->pivot = kg::DeviceArray<std::int64_t>(std::span<const std::int64_t>(p));
    impl_->host_of = kg::DeviceArray<std::int64_t>(std::span<const std::int64_t>(h));
  }
}

CoarsestSolve::~CoarsestSolve() = default;
CoarsestSolve::CoarsestSolve(CoarsestSolve&&) noexcept = default;
CoarsestSolve& CoarsestSolve::operator=(CoarsestSolve&&) noexcept = default;

std::size_t CoarsestSolve::size() const { return static_cast<std::size_t>(impl_->m); }
bool CoarsestSolve::explicit_inverse() const { return impl_->explicit_inverse; }
const std::vector<double>& CoarsestSolve::inverse() const { return impl_->inverse; }

void CoarsestSolve::solve(const double* b, double* x, void* stream) const {
  if (impl_->m == 0) return;
  if (impl_->explicit_inverse) {
    const std::int64_t threads = impl_->m * kg::warp_size;
    dense_product_kernel<<<grid(threads), block, 0, as_stream(stream)>>>(impl_->m,
                                                                         impl_->device.get(), b, x);
  } else {
    triangular_kernel<<<1, 1024, static_cast<std::size_t>(impl_->m) * sizeof(double),
                        as_stream(stream)>>>(impl_->m, impl_->device.get(), impl_->pivot.get(),
                                             impl_->host_of.get(), b, x);
  }
  kg::check(kg::last_error(), "coarsest solve");
}

}  // namespace autan::solvers::gpu
