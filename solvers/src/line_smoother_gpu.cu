#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/solvers/multigrid_gpu.hpp"

namespace autan::solvers::gpu {

namespace {

namespace kg = autan::kernels::gpu;

constexpr int b2 = 25;
constexpr int block = 64;

unsigned int grid(std::int64_t threads) {
  return static_cast<unsigned int>((threads + block - 1) / block);
}

kg::stream_t as_stream(void* s) { return static_cast<kg::stream_t>(s); }

// Gauss-Jordan inverse of a 5 x 5 row-major block with partial pivoting, in the operation order
// of the host's invert5 (solvers/src/blocks.hpp). Returns false on a zero or non-finite pivot.
__device__ bool invert5(const double* a, double* inv) {
  double m[50];
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      m[10 * i + j] = a[5 * i + j];
      m[10 * i + 5 + j] = i == j ? 1.0 : 0.0;
    }
  }
  for (int k = 0; k < 5; ++k) {
    int p = k;
    for (int i = k + 1; i < 5; ++i) {
      if (fabs(m[10 * i + k]) > fabs(m[10 * p + k])) p = i;
    }
    if (m[10 * p + k] == 0.0 || !isfinite(m[10 * p + k])) return false;
    if (p != k) {
      for (int j = 0; j < 10; ++j) {
        const double t = m[10 * k + j];
        m[10 * k + j] = m[10 * p + j];
        m[10 * p + j] = t;
      }
    }
    const double d = __ddiv_rn(1.0, m[10 * k + k]);
    for (int j = 0; j < 10; ++j) m[10 * k + j] = __dmul_rn(m[10 * k + j], d);
    for (int i = 0; i < 5; ++i) {
      if (i == k) continue;
      const double f = m[10 * i + k];
      for (int j = 0; j < 10; ++j)
        m[10 * i + j] = __dadd_rn(m[10 * i + j], -__dmul_rn(f, m[10 * k + j]));
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) inv[5 * i + j] = m[10 * i + 5 + j];
  }
  return true;
}

// c = a b, 5 x 5 row-major, the host's multiply5 order.
__device__ void multiply5(const double* a, const double* b, double* c) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      double s = 0.0;
      for (int k = 0; k < 5; ++k) s = __dadd_rn(s, __dmul_rn(a[5 * i + k], b[5 * k + j]));
      c[5 * i + j] = s;
    }
  }
}

// Structure-of-arrays block access: entry q of block i of an array of n blocks.
template <class T>
__device__ void load_block(const T* soa, std::int64_t n, std::int64_t i, double* out) {
  for (int q = 0; q < b2; ++q) out[q] = static_cast<double>(soa[q * n + i]);
}

__device__ void store_block(double* soa, std::int64_t n, std::int64_t i, const double* in) {
  for (int q = 0; q < b2; ++q) soa[q * n + i] = in[q];
}

// The coupling of a line node to its previous or next node: the blocks of every edge joining
// them, summed from +0.0 in the node's edge order (Operator::coupling).
template <class Off>
__device__ void coupling(const offset_t* offsets, const offset_t* slots, std::int64_t link,
                         const Off* off, std::int64_t n_slots, double* out) {
  for (int q = 0; q < b2; ++q) out[q] = 0.0;
  for (offset_t k = offsets[link]; k < offsets[link + 1]; ++k) {
    for (int q = 0; q < b2; ++q) {
      out[q] = __dadd_rn(out[q], static_cast<double>(off[q * n_slots + slots[k]]));
    }
  }
}

struct Lines {
  std::int64_t n_lines, n_line_nodes, n_nodes, n_slots;
  const offset_t* line_offsets;  // line -> its entries in line_nodes
  const index_t* line_nodes;     // device node of each line entry, in sweep order
  // Per line entry: coupling to the previous and to the next node (slots, CSR).
  const offset_t* prev_offsets;
  const offset_t* prev_slots;
  const offset_t* next_offsets;
  const offset_t* next_slots;
  // Per line entry: its node's edges in host edge order (slot, other device node, CSR), the
  // previous and next nodes left out (off_residual's skip).
  const offset_t* edge_offsets;
  const offset_t* edge_slots;
  const index_t* edge_nodes;
};

// Block LU of every line (Smoother::factor): D'_0 = D_0,
// D'_k = D_k − A(k, k−1) D'_{k−1}⁻¹ A(k−1, k); per entry D'⁻¹, A(k, k−1) and D'⁻¹ A(k, k+1).
template <class Off>
__global__ void __launch_bounds__(block) factor_kernel(Lines l, const double* diag, const Off* off,
                                                       double* dinv, double* lower, double* gain,
                                                       int* singular) {
  const std::int64_t line = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (line >= l.n_lines) return;
  double d[b2], t[b2], u[b2], inv[b2], g[b2];
  for (offset_t k = l.line_offsets[line]; k < l.line_offsets[line + 1]; ++k) {
    load_block(diag, l.n_nodes, l.line_nodes[k], d);
    if (k > l.line_offsets[line]) {
      double lo[b2], gp[b2];
      coupling(l.prev_offsets, l.prev_slots, k, off, l.n_slots, lo);
      store_block(lower, l.n_line_nodes, k, lo);
      load_block(gain, l.n_line_nodes, k - 1, gp);
      multiply5(lo, gp, t);
      for (int q = 0; q < b2; ++q) d[q] = __dadd_rn(d[q], -t[q]);
    }
    if (!invert5(d, inv)) {
      *singular = 1;
      for (int q = 0; q < b2; ++q) inv[q] = __longlong_as_double(0x7ff8000000000000LL);
    }
    store_block(dinv, l.n_line_nodes, k, inv);
    if (k + 1 < l.line_offsets[line + 1]) {
      coupling(l.next_offsets, l.next_slots, k, off, l.n_slots, u);
      multiply5(inv, u, g);
      store_block(gain, l.n_line_nodes, k, g);
    }
  }
}

// One colour's lines (Smoother::relax_line): the right-hand sides of the whole line from the
// current x, then forward elimination and back substitution, x written last. `work` holds 5
// values per line entry.
template <class Off>
__global__ void __launch_bounds__(block) relax_kernel(Lines l, std::int64_t n_colour_lines,
                                                      const index_t* colour_lines, const Off* off,
                                                      const double* dinv, const double* lower,
                                                      const double* gain, const double* b,
                                                      double* x, double* work) {
  const std::int64_t t = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_colour_lines) return;
  const std::int64_t line = colour_lines[t];
  const offset_t first = l.line_offsets[line], last = l.line_offsets[line + 1];
  const std::int64_t n = l.n_nodes, ne = l.n_line_nodes;
  for (offset_t k = first; k < last; ++k) {
    const std::int64_t i = l.line_nodes[k];
    double r[5];
    for (int c = 0; c < 5; ++c) r[c] = b[c * n + i];
    for (offset_t e = l.edge_offsets[k]; e < l.edge_offsets[k + 1]; ++e) {
      const offset_t s = l.edge_slots[e];
      const std::int64_t j = l.edge_nodes[e];
      for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 5; ++q) {
          r[p] = __dadd_rn(r[p], -__dmul_rn(static_cast<double>(off[(5 * p + q) * l.n_slots + s]),
                                            x[q * n + j]));
        }
      }
    }
    for (int c = 0; c < 5; ++c) work[c * ne + k] = r[c];
  }
  // Forward elimination y_k = D'_k⁻¹ (r_k − A(k, k−1) y_{k−1}).
  for (offset_t k = first; k < last; ++k) {
    double r[5], s[5];
    for (int c = 0; c < 5; ++c) r[c] = work[c * ne + k];
    if (k > first) {
      for (int a = 0; a < 5; ++a) {
        for (int c = 0; c < 5; ++c) {
          r[a] = __dadd_rn(r[a], -__dmul_rn(lower[(5 * a + c) * ne + k], work[c * ne + k - 1]));
        }
      }
    }
    for (int a = 0; a < 5; ++a) {
      s[a] = 0.0;
      for (int c = 0; c < 5; ++c)
        s[a] = __dadd_rn(s[a], __dmul_rn(dinv[(5 * a + c) * ne + k], r[c]));
    }
    for (int c = 0; c < 5; ++c) work[c * ne + k] = s[c];
  }
  // Back substitution x_k = y_k − D'_k⁻¹ A(k, k+1) x_{k+1}.
  for (offset_t k = last - 1; k >= first; --k) {
    double y[5];
    for (int c = 0; c < 5; ++c) y[c] = work[c * ne + k];
    if (k + 1 < last) {
      for (int a = 0; a < 5; ++a) {
        for (int c = 0; c < 5; ++c) {
          y[a] = __dadd_rn(y[a], -__dmul_rn(gain[(5 * a + c) * ne + k], work[c * ne + k + 1]));
        }
      }
    }
    for (int c = 0; c < 5; ++c) {
      work[c * ne + k] = y[c];
      x[c * n + l.line_nodes[k]] = y[c];
    }
  }
}

}  // namespace

struct LineSmoother::Impl {
  std::int64_t n_lines = 0, n_line_nodes = 0, n_nodes = 0, n_slots = 0;
  kg::DeviceArray<offset_t> line_offsets, prev_offsets, prev_slots, next_offsets, next_slots,
      edge_offsets, edge_slots;
  kg::DeviceArray<index_t> line_nodes, edge_nodes;
  std::vector<std::int64_t> colour_offsets;
  kg::DeviceArray<index_t> colour_lines;
  kg::DeviceArray<double> dinv, lower, gain, work;
  kg::DeviceArray<int> singular;
  bool factored = false, single = false;

  [[nodiscard]] Lines view() const {
    return {n_lines,
            n_line_nodes,
            n_nodes,
            n_slots,
            line_offsets.get(),
            line_nodes.get(),
            prev_offsets.get(),
            prev_slots.get(),
            next_offsets.get(),
            next_slots.get(),
            edge_offsets.get(),
            edge_slots.get(),
            edge_nodes.get()};
  }

  template <class Off>
  void factor(const double* diag, const Off* off, void* stream) {
    if (n_lines > 0) {
      kg::check(kg::detail::zero(singular.get(), sizeof(int), as_stream(stream)), "reset");
      factor_kernel<Off><<<grid(n_lines), block, 0, as_stream(stream)>>>(
          view(), diag, off, dinv.get(), lower.get(), gain.get(), singular.get());
      kg::check(kg::last_error(), "line factorisation");
      int flag = 0;
      kg::check(kg::detail::to_host(&flag, singular.get(), sizeof(int), as_stream(stream)),
                "line factorisation");
      kg::check(kg::detail::synchronize(as_stream(stream)), "line factorisation");
      if (flag != 0) {
        factored = false;
        throw std::domain_error("singular diagonal block");
      }
    }
    factored = true;
    single = std::is_same_v<Off, float>;
  }

  template <class Off>
  void relax(std::size_t colour, const Off* off, const double* b, double* x, void* stream) const {
    const std::int64_t first = colour_offsets[colour], count = colour_offsets[colour + 1] - first;
    if (count == 0) return;
    relax_kernel<Off><<<grid(count), block, 0, as_stream(stream)>>>(
        view(), count, colour_lines.get() + first, off, dinv.get(), lower.get(), gain.get(), b, x,
        work.get());
    kg::check(kg::last_error(), "line relaxation");
  }
};

LineSmoother::LineSmoother(const Level& level, std::span<const index_t> order,
                           const EdgeSlots& slots, std::size_t n_slots)
    : impl_(std::make_unique<Impl>()) {
  const std::size_t n = level.n_nodes, ne = level.owner.size();
  if (order.size() != n || slots.owner.size() != ne || slots.neighbour.size() != ne) {
    throw std::invalid_argument("LineSmoother: sizes do not match the level");
  }
  std::vector<index_t> rank(n, -1);
  for (std::size_t j = 0; j < n; ++j) {
    const auto h = static_cast<std::size_t>(order[j]);
    if (h >= n || rank[h] != -1)
      throw std::invalid_argument("LineSmoother: order is not a permutation");
    rank[h] = static_cast<index_t>(j);
  }
  for (std::size_t e = 0; e < ne; ++e) {
    for (const offset_t s : {slots.owner[e], slots.neighbour[e]}) {
      if (s < 0 || static_cast<std::size_t>(s) >= n_slots) {
        throw std::invalid_argument("LineSmoother: slot out of range");
      }
    }
  }
  // Each node's edges in increasing edge index (Operator::node_edges).
  std::vector<std::vector<std::size_t>> edges(n);
  for (std::size_t e = 0; e < ne; ++e) {
    edges[static_cast<std::size_t>(level.owner[e])].push_back(e);
    edges[static_cast<std::size_t>(level.neighbour[e])].push_back(e);
  }
  auto slot_of = [&](std::size_t e, std::size_t i) {
    return static_cast<std::size_t>(level.owner[e]) == i ? slots.owner[e] : slots.neighbour[e];
  };
  auto other = [&](std::size_t e, std::size_t i) {
    return static_cast<std::size_t>(
        static_cast<std::size_t>(level.owner[e]) == i ? level.neighbour[e] : level.owner[e]);
  };
  // Lines of several nodes; line index = position among them.
  std::vector<std::size_t> host_line_of(level.n_lines(), SIZE_MAX);
  std::vector<offset_t> line_offsets{0}, prev_offsets{0}, next_offsets{0}, edge_offsets{0};
  std::vector<offset_t> prev_slots, next_slots, edge_slots;
  std::vector<index_t> line_nodes, edge_nodes;
  for (std::size_t li = 0; li < level.n_lines(); ++li) {
    const auto first = static_cast<std::size_t>(level.line_offsets[li]);
    const auto last = static_cast<std::size_t>(level.line_offsets[li + 1]);
    if (last - first < 2) continue;
    host_line_of[li] = line_offsets.size() - 1;
    for (std::size_t k = first; k < last; ++k) {
      const auto i = static_cast<std::size_t>(level.line_nodes[k]);
      const std::size_t p =
          k > first ? static_cast<std::size_t>(level.line_nodes[k - 1]) : SIZE_MAX;
      const std::size_t q =
          k + 1 < last ? static_cast<std::size_t>(level.line_nodes[k + 1]) : SIZE_MAX;
      line_nodes.push_back(rank[i]);
      for (const std::size_t e : edges[i]) {
        const std::size_t j = other(e, i);
        if (j == p) prev_slots.push_back(slot_of(e, i));
        if (j == q) next_slots.push_back(slot_of(e, i));
        if (j != p && j != q) {
          edge_slots.push_back(slot_of(e, i));
          edge_nodes.push_back(rank[j]);
        }
      }
      prev_offsets.push_back(static_cast<offset_t>(prev_slots.size()));
      next_offsets.push_back(static_cast<offset_t>(next_slots.size()));
      edge_offsets.push_back(static_cast<offset_t>(edge_slots.size()));
    }
    line_offsets.push_back(static_cast<offset_t>(line_nodes.size()));
  }
  // The host's colours (line_colours), lines of one node left out.
  std::vector<index_t> colour_lines;
  impl_->colour_offsets.push_back(0);
  for (const auto& colour : line_colours(level)) {
    for (const std::size_t li : colour) {
      if (host_line_of[li] != SIZE_MAX)
        colour_lines.push_back(static_cast<index_t>(host_line_of[li]));
    }
    impl_->colour_offsets.push_back(static_cast<std::int64_t>(colour_lines.size()));
  }
  impl_->n_lines = static_cast<std::int64_t>(line_offsets.size() - 1);
  impl_->n_line_nodes = static_cast<std::int64_t>(line_nodes.size());
  impl_->n_nodes = static_cast<std::int64_t>(n);
  impl_->n_slots = static_cast<std::int64_t>(n_slots);
  auto up = [](const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    return kg::DeviceArray<T>(std::span<const T>(v));
  };
  impl_->line_offsets = up(line_offsets);
  impl_->line_nodes = up(line_nodes);
  impl_->prev_offsets = up(prev_offsets);
  impl_->prev_slots = up(prev_slots);
  impl_->next_offsets = up(next_offsets);
  impl_->next_slots = up(next_slots);
  impl_->edge_offsets = up(edge_offsets);
  impl_->edge_slots = up(edge_slots);
  impl_->edge_nodes = up(edge_nodes);
  impl_->colour_lines = up(colour_lines);
  const auto entries = static_cast<std::size_t>(b2) * line_nodes.size();
  impl_->dinv = kg::DeviceArray<double>(entries);
  impl_->lower = kg::DeviceArray<double>(entries);
  impl_->gain = kg::DeviceArray<double>(entries);
  impl_->work = kg::DeviceArray<double>(5 * line_nodes.size());
  impl_->singular = kg::DeviceArray<int>(1);
}

LineSmoother::~LineSmoother() = default;
LineSmoother::LineSmoother(LineSmoother&&) noexcept = default;
LineSmoother& LineSmoother::operator=(LineSmoother&&) noexcept = default;

std::size_t LineSmoother::n_lines() const { return static_cast<std::size_t>(impl_->n_lines); }
std::size_t LineSmoother::n_colours() const { return impl_->colour_offsets.size() - 1; }

void LineSmoother::factor(const double* diag, const double* off, void* stream) {
  impl_->factor(diag, off, stream);
}
void LineSmoother::factor(const double* diag, const float* off, void* stream) {
  impl_->factor(diag, off, stream);
}

void LineSmoother::sweep(const double* off, const double* b, double* x, int sweeps,
                         void* stream) const {
  if (!impl_->factored || impl_->single)
    throw std::logic_error("LineSmoother: factor with FP64 blocks first");
  for (int s = 0; s < sweeps; ++s) {
    for (std::size_t c = 0; c < n_colours(); ++c) impl_->relax(c, off, b, x, stream);
    for (std::size_t c = n_colours(); c-- > 0;) impl_->relax(c, off, b, x, stream);
  }
}

void LineSmoother::sweep(const float* off, const double* b, double* x, int sweeps,
                         void* stream) const {
  if (!impl_->factored || !impl_->single)
    throw std::logic_error("LineSmoother: factor with FP32 blocks first");
  for (int s = 0; s < sweeps; ++s) {
    for (std::size_t c = 0; c < n_colours(); ++c) impl_->relax(c, off, b, x, stream);
    for (std::size_t c = n_colours(); c-- > 0;) impl_->relax(c, off, b, x, stream);
  }
}

}  // namespace autan::solvers::gpu
