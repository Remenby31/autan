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

// x = A⁻¹ b by factor_dense's LU (host numbering, row-major m x m), in solve_dense's operation
// order, so the result is the host's bit for bit: its pivot swaps in order; forward substitution
// by columns, all rows at once (row i still subtracts L_ij y_j in increasing j, the host's
// order); back substitution in one thread, row by row as the host does. `host_of[kd]` is the
// host unknown of device unknown kd.
__global__ void lu_solve_kernel(std::int64_t m, const double* lu, const std::int64_t* pivot,
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
  if (threadIdx.x == 0) {
    for (std::int64_t i = m - 1; i >= 0; --i) {
      double v = y[i];
      for (std::int64_t j = i + 1; j < m; ++j) v = __dadd_rn(v, -__dmul_rn(lu[i * m + j], y[j]));
      y[i] = __ddiv_rn(v, lu[i * m + i]);
    }
  }
  __syncthreads();
  for (std::int64_t kd = threadIdx.x; kd < m; kd += blockDim.x) x[kd] = y[host_of[kd]];
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
  kg::DeviceArray<double> lu;
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
  // Unknown c of device node j is host unknown c * n + order[j].
  std::vector<std::int64_t> p(m), host_of(m);
  for (std::size_t k = 0; k < m; ++k) p[k] = static_cast<std::int64_t>(pivot[k]);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t j = 0; j < n; ++j) {
      host_of[c * n + j] = static_cast<std::int64_t>(c * n + static_cast<std::size_t>(order[j]));
    }
  }
  impl_->m = static_cast<std::int64_t>(m);
  impl_->lu = kg::DeviceArray<double>(std::span<const double>(a));
  impl_->pivot = kg::DeviceArray<std::int64_t>(std::span<const std::int64_t>(p));
  impl_->host_of = kg::DeviceArray<std::int64_t>(std::span<const std::int64_t>(host_of));
}

CoarsestSolve::~CoarsestSolve() = default;
CoarsestSolve::CoarsestSolve(CoarsestSolve&&) noexcept = default;
CoarsestSolve& CoarsestSolve::operator=(CoarsestSolve&&) noexcept = default;

std::size_t CoarsestSolve::size() const { return static_cast<std::size_t>(impl_->m); }

void CoarsestSolve::solve(const double* b, double* x, void* stream) const {
  if (impl_->m == 0) return;
  lu_solve_kernel<<<1, 256, static_cast<std::size_t>(impl_->m) * sizeof(double),
                    as_stream(stream)>>>(impl_->m, impl_->lu.get(), impl_->pivot.get(),
                                         impl_->host_of.get(), b, x);
  kg::check(kg::last_error(), "coarsest solve");
}

}  // namespace autan::solvers::gpu
