#include <algorithm>
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
// Children of an isotropic agglomerate (at most 8 for 2 x 2 x 2 hexahedra) restricted in one burst.
constexpr int unrolled = 8;

// One thread per node, its 5 components in turn, so each index list is read once. Plain sums in
// list order: __dadd_rn keeps the compiler from reassociating or contracting them.
__global__ void restrict_kernel(std::int64_t n_fine, std::int64_t n_coarse, const offset_t* offsets,
                                const index_t* children, const double* r, double* b_coarse) {
  const std::int64_t node = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= n_coarse) return;
  const offset_t first = offsets[node];
  const auto m = static_cast<int>(offsets[node + 1] - first);
  if (m <= unrolled) {
    // Every load issued before the first addition, so a thread keeps its children in flight.
    std::int64_t child[unrolled];
#pragma unroll
    for (int k = 0; k < unrolled; ++k) child[k] = k < m ? children[first + k] : 0;
#pragma unroll
    for (int c = 0; c < components; ++c) {
      double v[unrolled];
#pragma unroll
      for (int k = 0; k < unrolled; ++k) v[k] = k < m ? r[c * n_fine + child[k]] : 0.0;
      double s = 0.0;
#pragma unroll
      for (int k = 0; k < unrolled; ++k) {
        if (k < m) s = __dadd_rn(s, v[k]);
      }
      b_coarse[c * n_coarse + node] = s;
    }
    return;
  }
  double s[components] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (offset_t k = first; k < first + m; ++k) {
    const std::int64_t child = children[k];
    for (int c = 0; c < components; ++c) s[c] = __dadd_rn(s[c], r[c * n_fine + child]);
  }
  for (int c = 0; c < components; ++c) b_coarse[c * n_coarse + node] = s[c];
}

__global__ void prolong_kernel(std::int64_t n_fine, std::int64_t n_coarse,
                               const index_t* coarse_node, const double* x_coarse, double* e) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_fine) return;
  const std::int64_t parent = coarse_node[i];
  for (int c = 0; c < components; ++c) e[c * n_fine + i] = x_coarse[c * n_coarse + parent];
}

__global__ void correct_kernel(std::int64_t n_fine, std::int64_t n_coarse,
                               const index_t* coarse_node, const double* alpha,
                               const double* x_coarse, double* x) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_fine) return;
  const std::int64_t parent = coarse_node[i];
  const double a = *alpha;
  for (int c = 0; c < components; ++c) {
    double& xi = x[c * n_fine + i];
    xi = __dadd_rn(xi, __dmul_rn(a, x_coarse[c * n_coarse + parent]));
  }
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
    kg::DeviceArray<index_t> children;     // in increasing host fine index
    kg::DeviceArray<index_t> coarse_node;  // fine node -> coarse node
  };
  std::vector<Pair> pairs;
  std::vector<std::vector<index_t>> order;  // per level: host index of each device node

  const Pair& at(std::size_t l) const {
    if (l >= pairs.size()) {
      throw std::out_of_range("multigrid transfers: level " + std::to_string(l) + " of " +
                              std::to_string(pairs.size() + 1) + " has no coarser level");
    }
    return pairs[l];
  }
};

// Device numbering: level 0 keeps the mesh's; each coarse node takes the rank of its least child
// in the finer level's device numbering, so that consecutive coarse nodes gather nearby fine
// nodes. Lists keep the host's summation order (children by increasing host index), so every
// value is the host's, only stored at another index.
Transfers::Transfers(std::span<const Level> levels) : impl_(std::make_unique<Impl>()) {
  if (levels.empty()) return;
  std::vector<index_t> fine_order(levels[0].n_nodes), fine_rank(levels[0].n_nodes);
  for (std::size_t i = 0; i < fine_order.size(); ++i) {
    fine_order[i] = fine_rank[i] = static_cast<index_t>(i);
  }
  impl_->order.push_back(fine_order);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    const Level& f = levels[l];
    const std::size_t n = f.n_nodes, nc = levels[l + 1].n_nodes;
    std::vector<index_t> least(nc, static_cast<index_t>(n));
    for (std::size_t h = 0; h < n; ++h) {
      index_t& m = least[static_cast<std::size_t>(f.coarse_node[h])];
      m = std::min(m, fine_rank[h]);
    }
    std::vector<index_t> coarse_order(nc), coarse_rank(nc);
    for (std::size_t H = 0; H < nc; ++H) coarse_order[H] = static_cast<index_t>(H);
    std::sort(coarse_order.begin(), coarse_order.end(), [&](index_t a, index_t b) {
      return least[static_cast<std::size_t>(a)] < least[static_cast<std::size_t>(b)];
    });
    for (std::size_t J = 0; J < nc; ++J) {
      coarse_rank[static_cast<std::size_t>(coarse_order[J])] = static_cast<index_t>(J);
    }
    std::vector<offset_t> offsets(nc + 1, 0);
    for (const index_t H : f.coarse_node) {
      ++offsets[static_cast<std::size_t>(coarse_rank[static_cast<std::size_t>(H)]) + 1];
    }
    for (std::size_t J = 0; J < nc; ++J) offsets[J + 1] += offsets[J];
    std::vector<index_t> children(n), coarse_node(n);
    std::vector<offset_t> fill(offsets.begin(), offsets.end() - 1);
    for (std::size_t h = 0; h < n; ++h) {
      const index_t J = coarse_rank[static_cast<std::size_t>(f.coarse_node[h])];
      children[static_cast<std::size_t>(fill[static_cast<std::size_t>(J)]++)] = fine_rank[h];
      coarse_node[static_cast<std::size_t>(fine_rank[h])] = J;
    }
    Impl::Pair p;
    p.n_fine = static_cast<std::int64_t>(n);
    p.n_coarse = static_cast<std::int64_t>(nc);
    p.offsets = kg::DeviceArray<offset_t>(std::span<const offset_t>(offsets));
    p.children = kg::DeviceArray<index_t>(std::span<const index_t>(children));
    p.coarse_node = kg::DeviceArray<index_t>(std::span<const index_t>(coarse_node));
    impl_->pairs.push_back(std::move(p));
    impl_->order.push_back(coarse_order);
    fine_rank = std::move(coarse_rank);
  }
}

std::span<const index_t> Transfers::order(std::size_t l) const {
  if (l >= impl_->order.size()) {
    throw std::out_of_range("multigrid transfers: no level " + std::to_string(l));
  }
  return impl_->order[l];
}

Transfers::~Transfers() = default;
Transfers::Transfers(Transfers&&) noexcept = default;
Transfers& Transfers::operator=(Transfers&&) noexcept = default;

void Transfers::restrict_residual(std::size_t l, const double* r, double* b_coarse,
                                  void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  restrict_kernel<<<grid(p.n_coarse), block, 0, as_stream(stream)>>>(
      p.n_fine, p.n_coarse, p.offsets.get(), p.children.get(), r, b_coarse);
  kg::check(kg::last_error(), "multigrid restriction");
}

void Transfers::prolong(std::size_t l, const double* x_coarse, double* e, void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  prolong_kernel<<<grid(p.n_fine), block, 0, as_stream(stream)>>>(p.n_fine, p.n_coarse,
                                                                  p.coarse_node.get(), x_coarse, e);
  kg::check(kg::last_error(), "multigrid prolongation");
}

void Transfers::correct(std::size_t l, const double* alpha, const double* x_coarse, double* x,
                        void* stream) const {
  const Impl::Pair& p = impl_->at(l);
  correct_kernel<<<grid(p.n_fine), block, 0, as_stream(stream)>>>(
      p.n_fine, p.n_coarse, p.coarse_node.get(), alpha, x_coarse, x);
  kg::check(kg::last_error(), "multigrid correction");
}

}  // namespace autan::solvers::gpu
