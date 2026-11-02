#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/solvers/device_order.hpp"
#include "autan/solvers/multigrid_gpu.hpp"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;

constexpr int components = 5;
constexpr int block = 256;
// Restriction: coarse nodes per block, and list entries a block stages in shared memory (12 KB;
// about 8 children per agglomerate leaves twice the room).
constexpr int staged_nodes = 64;
constexpr int staged_entries = 1024;

// A block restricts `staged_nodes` consecutive coarse nodes: its threads gather their children's
// values, one list entry each, into shared memory, then one thread per coarse node sums its
// entries there in list order. The gather keeps every thread's loads independent whatever the
// agglomerate sizes; the sum is the host's, from +0.0 and without reassociation (__dadd_rn).
// Lists longer than the shared buffer are summed from global memory in the same order.
__global__ void __launch_bounds__(block) restrict_kernel(std::int64_t n_fine, std::int64_t n_coarse,
                                                         const offset_t* offsets,
                                                         const index_t* children, const double* r,
                                                         double* b_coarse) {
  __shared__ index_t child[staged_entries];
  __shared__ double value[staged_entries];
  const std::int64_t first = static_cast<std::int64_t>(blockIdx.x) * staged_nodes;
  const std::int64_t last = first + staged_nodes < n_coarse ? first + staged_nodes : n_coarse;
  const offset_t base = offsets[first];
  const offset_t length = offsets[last] - base;
  const bool staged = length <= staged_entries;
  if (staged) {
    for (offset_t k = threadIdx.x; k < length; k += blockDim.x) child[k] = children[base + k];
  }
  __syncthreads();
  const std::int64_t node = first + threadIdx.x;
  const bool owner = threadIdx.x < last - first;
  const offset_t begin = owner ? offsets[node] - base : 0;
  const offset_t end = owner ? offsets[node + 1] - base : 0;
  for (int c = 0; c < components; ++c) {
    const double* rc = r + c * n_fine;
    if (staged) {
      for (offset_t k = threadIdx.x; k < length; k += blockDim.x) value[k] = rc[child[k]];
    }
    __syncthreads();
    if (owner) {
      double s = 0.0;
      for (offset_t k = begin; k < end; ++k) {
        s = __dadd_rn(s, staged ? value[k] : rc[children[base + k]]);
      }
      b_coarse[c * n_coarse + node] = s;
    }
    __syncthreads();
  }
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

// Lists keep the host's summation order (children by increasing host index) in the device
// numbering of device_orders, so every value is the host's, only stored at another index.
Transfers::Transfers(std::span<const Level> levels) : impl_(std::make_unique<Impl>()) {
  impl_->order = device_orders(levels);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    const Level& f = levels[l];
    const std::size_t n = f.n_nodes, nc = levels[l + 1].n_nodes;
    const std::vector<index_t>& fine_order = impl_->order[l];
    std::vector<index_t> fine_rank(n), coarse_rank(nc);
    for (std::size_t j = 0; j < n; ++j)
      fine_rank[static_cast<std::size_t>(fine_order[j])] = static_cast<index_t>(j);
    for (std::size_t J = 0; J < nc; ++J) {
      coarse_rank[static_cast<std::size_t>(impl_->order[l + 1][J])] = static_cast<index_t>(J);
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
  restrict_kernel<<<static_cast<unsigned int>((p.n_coarse + staged_nodes - 1) / staged_nodes),
                    block, 0, as_stream(stream)>>>(p.n_fine, p.n_coarse, p.offsets.get(),
                                                   p.children.get(), r, b_coarse);
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
