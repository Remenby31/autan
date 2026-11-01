#pragma once

// Euclidean norms on the device in a fixed summation order: the result depends on the data and
// its length only, never on the launch shape, so the deterministic mode reproduces it bitwise.
#include <cstdint>

#include "autan/kernels/gpu.hpp"

namespace autan::kernels::gpu {

inline constexpr int norm_block = 256;
inline constexpr int norm_per_thread = 16;
inline constexpr std::int64_t norm_chunk = std::int64_t{norm_block} * norm_per_thread;

/// Partial sums the norm needs: one per chunk of norm_chunk entries.
inline std::int64_t norm_partials(std::int64_t n) { return (n + norm_chunk - 1) / norm_chunk; }

__device__ inline void tree_sum(double* s) {
  for (int w = norm_block / 2; w > 0; w /= 2) {
    if (static_cast<int>(threadIdx.x) < w) {
      s[threadIdx.x] += s[threadIdx.x + w];
    }
    __syncthreads();
  }
}

template <class T>
__global__ void __launch_bounds__(norm_block)
    squares_by_chunk(const T* __restrict__ x, std::int64_t n, double* __restrict__ partial) {
  __shared__ double s[norm_block];
  const std::int64_t base = std::int64_t{blockIdx.x} * norm_chunk + threadIdx.x;
  double acc = 0.0;
  for (int j = 0; j < norm_per_thread; ++j) {
    const std::int64_t i = base + std::int64_t{j} * norm_block;
    if (i < n) {
      const double v = static_cast<double>(x[i]);
      acc += v * v;
    }
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  tree_sum(s);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = s[0];
  }
}

__global__ void __launch_bounds__(norm_block)
    sum_partials(const double* __restrict__ partial, std::int64_t m, double* __restrict__ out) {
  __shared__ double s[norm_block];
  double acc = 0.0;
  for (std::int64_t i = threadIdx.x; i < m; i += norm_block) {
    acc += partial[i];
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  tree_sum(s);
  if (threadIdx.x == 0) {
    *out = s[0];
  }
}

/// Writes sum_i x[i]^2, accumulated in FP64 in a fixed order, to the device scalar `out`.
/// `partials` is device storage for norm_partials(n) doubles.
template <class T>
void sum_of_squares(const T* x, std::int64_t n, double* partials, double* out, stream_t stream) {
  const std::int64_t m = norm_partials(n);
  if (m > 0) {
    squares_by_chunk<T><<<static_cast<unsigned>(m), norm_block, 0, stream>>>(x, n, partials);
  }
  sum_partials<<<1, norm_block, 0, stream>>>(partials, m, out);
  check(last_error(), "sum_of_squares");
}

}  // namespace autan::kernels::gpu
