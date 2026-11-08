#pragma once

// Device pieces shared by the line smoothers of solvers/ (the multigrid's, line_smoother_gpu.cu,
// and the physics' LineSmoother, line_sweep_gpu.cu): 5 x 5 blocks in the host's operation order
// (blocks.hpp) and the block LU of lines.
#include <cstdint>

#include "autan/kernels/device.cuh"
#include "autan/solvers/multigrid_gpu.hpp"

namespace autan::solvers::gpu::detail {

constexpr int b2 = 25;
constexpr int block = 64;

inline unsigned int grid(std::int64_t threads) {
  return static_cast<unsigned int>((threads + block - 1) / block);
}

// Gauss-Jordan inverse of a 5 x 5 row-major block with partial pivoting, in the operation order
// of the host's invert5 (solvers/src/blocks.hpp). Returns false on a zero or non-finite pivot.
__device__ inline bool invert5(const double* a, double* inv) {
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
__device__ inline void multiply5(const double* a, const double* b, double* c) {
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

__device__ inline void store_block(double* soa, std::int64_t n, std::int64_t i, const double* in) {
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

}  // namespace autan::solvers::gpu::detail
