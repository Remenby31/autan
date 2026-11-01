#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/solvers/multigrid_gpu.hpp"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;

constexpr int components = 5;
constexpr int block = 256;

// One thread per coarse node and component. Plain sums in list order: __dadd_rn keeps the
// compiler from reassociating or contracting them.
__global__ void restrict_kernel(std::int64_t n_fine, std::int64_t n_coarse, const offset_t* offsets,
                                const index_t* children, const double* r, double* b_coarse) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= components * n_coarse) return;
  const std::int64_t c = t / n_coarse;
  const std::int64_t node = t - c * n_coarse;
  const double* rc = r + c * n_fine;
  double s = 0.0;
  for (offset_t k = offsets[node]; k < offsets[node + 1]; ++k) s = __dadd_rn(s, rc[children[k]]);
  b_coarse[t] = s;
}

__global__ void prolong_kernel(std::int64_t n_fine, std::int64_t n_coarse,
                               const index_t* coarse_node, const double* x_coarse, double* e) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= components * n_fine) return;
  const std::int64_t c = t / n_fine;
  e[t] = x_coarse[c * n_coarse + coarse_node[t - c * n_fine]];
}

__global__ void correct_kernel(std::int64_t n_fine, std::int64_t n_coarse,
                               const index_t* coarse_node, const double* alpha,
                               const double* x_coarse, double* x) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= components * n_fine) return;
  const std::int64_t c = t / n_fine;
  const double e = x_coarse[c * n_coarse + coarse_node[t - c * n_fine]];
  x[t] = __dadd_rn(x[t], __dmul_rn(*alpha, e));
}

unsigned int grid(std::int64_t threads) {
  return static_cast<unsigned int>((threads + block - 1) / block);
}

kg::stream_t as_stream(void* s) { return static_cast<kg::stream_t>(s); }

}  // namespace

struct Transfers::Impl {
  struct Pair {
    std::int64_t n_fine = 0, n_coarse = 0;
    kg::DeviceArray<offset_t> offsets;     // coarse node -> its children, CSR
    kg::DeviceArray<index_t> children;     // in increasing fine index
    kg::DeviceArray<index_t> coarse_node;  // fine node -> coarse node
  };
  std::vector<Pair> pairs;

  const Pair& at(std::size_t l) const {
    if (l >= pairs.size()) {
      throw std::out_of_range("multigrid transfers: level " + std::to_string(l) + " of " +
                              std::to_string(pairs.size() + 1) + " has no coarser level");
    }
    return pairs[l];
  }
};

Transfers::Transfers(std::span<const Level> levels) : impl_(std::make_unique<Impl>()) {
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    const Level& f = levels[l];
    const std::size_t nc = levels[l + 1].n_nodes;
    std::vector<offset_t> offsets(nc + 1, 0);
    for (const index_t c : f.coarse_node) ++offsets[static_cast<std::size_t>(c) + 1];
    for (std::size_t c = 0; c < nc; ++c) offsets[c + 1] += offsets[c];
    std::vector<index_t> children(f.n_nodes);
    std::vector<offset_t> fill(offsets.begin(), offsets.end() - 1);
    for (std::size_t i = 0; i < f.n_nodes; ++i) {
      children[static_cast<std::size_t>(fill[static_cast<std::size_t>(f.coarse_node[i])]++)] =
          static_cast<index_t>(i);
    }
    Impl::Pair p;
    p.n_fine = static_cast<std::int64_t>(f.n_nodes);
    p.n_coarse = static_cast<std::int64_t>(nc);
    p.offsets = kg::DeviceArray<offset_t>(std::span<const offset_t>(offsets));
    p.children = kg::DeviceArray<index_t>(std::span<const index_t>(children));
    p.coarse_node = kg::DeviceArray<index_t>(std::span<const index_t>(f.coarse_node));
    impl_->pairs.push_back(std::move(p));
  }
}

Transfers::~Transfers() = default;
Transfers::Transfers(Transfers&&) noexcept = default;
Transfers& Transfers::operator=(Transfers&&) noexcept = default;

void Transfers::restrict_residual(std::size_t l, const double* r, double* b_coarse,
                                  void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  restrict_kernel<<<grid(components * p.n_coarse), block, 0, as_stream(stream)>>>(
      p.n_fine, p.n_coarse, p.offsets.get(), p.children.get(), r, b_coarse);
  kg::check(kg::last_error(), "multigrid restriction");
}

void Transfers::prolong(std::size_t l, const double* x_coarse, double* e, void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  prolong_kernel<<<grid(components * p.n_fine), block, 0, as_stream(stream)>>>(
      p.n_fine, p.n_coarse, p.coarse_node.get(), x_coarse, e);
  kg::check(kg::last_error(), "multigrid prolongation");
}

void Transfers::correct(std::size_t l, const double* alpha, const double* x_coarse, double* x,
                        void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  correct_kernel<<<grid(components * p.n_fine), block, 0, as_stream(stream)>>>(
      p.n_fine, p.n_coarse, p.coarse_node.get(), alpha, x_coarse, x);
  kg::check(kg::last_error(), "multigrid correction");
}

}  // namespace autan::solvers::gpu
