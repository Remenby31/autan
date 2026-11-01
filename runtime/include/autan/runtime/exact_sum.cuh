#pragma once

// ExactSum on the device: the exact sum of FP64 terms that kernels produce,
// with the bits of the host ExactSum over the same terms whatever the launch
// shape, the block scheduling or the number of ranks. Compiled by nvcc or
// hipcc.

#if defined(__HIPCC__)
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "autan/runtime/exact_sum.hpp"

namespace autan::runtime::gpu {

#if defined(__HIPCC__)
using stream_t = hipStream_t;
using lane_mask_t = std::uint64_t;
#else
using stream_t = cudaStream_t;
using lane_mask_t = unsigned int;
#endif

/// The words of an ExactSum (ExactSum::data()): fixed-point limbs, then the
/// counts of +inf, -inf and NaN terms.
struct ExactWords {
  long long w[ExactSum::words];
};

namespace detail {

inline constexpr int offset = 1088;  // limb i weighs 2^(32 i - offset), as on the host
inline constexpr int pos_inf = ExactSum::limbs, neg_inf = pos_inf + 1, nan_count = pos_inf + 2;
inline constexpr int block_size = 256;
using ull = unsigned long long;

__device__ inline long long warp_sum(long long v) {
  for (int o = warpSize / 2; o > 0; o /= 2) v += __shfl_xor_sync(~lane_mask_t{0}, v, o);
  return v;
}

__device__ inline int warp_min(int v) {
  for (int o = warpSize / 2; o > 0; o /= 2) v = min(v, __shfl_xor_sync(~lane_mask_t{0}, v, o));
  return v;
}

__device__ inline int warp_max(int v) {
  for (int o = warpSize / 2; o > 0; o /= 2) v = max(v, __shfl_xor_sync(~lane_mask_t{0}, v, o));
  return v;
}

}  // namespace detail

/// Adds each lane's `term` to `acc` (shared memory, one per warp), exactly.
/// Warp-collective: every lane of the warp calls it; a lane without a term
/// passes active = false. Integer additions commute, so the atomics give the
/// same words in any order. One call adds less than 2^38 to a limb: 2^25
/// calls fit before ExactSum's int64 headroom.
__device__ inline void exact_add_warp(ExactWords& acc, double term, bool active) {
  using detail::ull;
  const auto bits = static_cast<ull>(__double_as_longlong(term));
  const int biased = static_cast<int>((bits >> 52) & 0x7ff);
  const ull fraction = bits & ((ull{1} << 52) - 1);
  const bool negative = (bits >> 63) != 0;
  if (active && biased == 0x7ff) {
    atomicAdd(reinterpret_cast<ull*>(
                  &acc.w[fraction != 0 ? detail::nan_count
                                       : (negative ? detail::neg_inf : detail::pos_inf)]),
              ull{1});
  }
  const bool finite = active && biased != 0x7ff && !(biased == 0 && fraction == 0);
  // The term as three limbs from limb i, as ExactSum::add splits it.
  int i = 1 << 30;
  long long part[3] = {0, 0, 0};
  if (finite) {
    const ull m = biased == 0 ? fraction : (fraction | (ull{1} << 52));
    const int p = (biased == 0 ? -1074 : biased - 1075) + detail::offset;
    i = p >> 5;
    const int s = p & 31;
    const ull lo = (m & 0xffffffffULL) << s, hi = (m >> 32) << s;
    part[0] = static_cast<long long>(lo & 0xffffffffULL);
    part[1] = static_cast<long long>((lo >> 32) + (hi & 0xffffffffULL));
    part[2] = static_cast<long long>(hi >> 32);
    if (negative) {
      for (long long& v : part) v = -v;
    }
  }
  const int first = detail::warp_min(i);
  const int last = detail::warp_max(finite ? i : -1);
  const bool leader = threadIdx.x % warpSize == 0;
  for (int j = first; j <= last + 2; ++j) {
    const int d = j - i;
    const long long v = detail::warp_sum(finite && d >= 0 && d < 3 ? part[d] : 0);
    if (leader && v != 0) atomicAdd(reinterpret_cast<ull*>(&acc.w[j]), static_cast<ull>(v));
  }
}

namespace detail {

// K exact sums over n items, term(i, k) the k-th term of item i; each block
// adds its normalized sums into `out` (zeroed beforehand).
template <int K, class Term>
__global__ void __launch_bounds__(block_size) exact_sums_kernel(std::int64_t n, Term term,
                                                                ExactWords* out) {
  constexpr int max_warps = block_size / 32;
  __shared__ ExactWords acc[max_warps][K];
  const int warps = static_cast<int>(blockDim.x) / warpSize;
  for (int t = static_cast<int>(threadIdx.x); t < warps * K * ExactSum::words;
       t += static_cast<int>(blockDim.x)) {
    acc[t / (K * ExactSum::words)][(t / ExactSum::words) % K].w[t % ExactSum::words] = 0;
  }
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const std::int64_t stride = std::int64_t{gridDim.x} * blockDim.x;
  // Whole warps step together, so every lane reaches each collective call.
  for (std::int64_t base = std::int64_t{blockIdx.x} * blockDim.x + warp * warpSize; base < n;
       base += stride) {
    const std::int64_t item = base + lane;
    const bool active = item < n;
    for (int k = 0; k < K; ++k) {
      exact_add_warp(acc[warp][k], active ? term(item, k) : 0.0, active);
    }
  }
  __syncthreads();
  // Carry each warp's limbs into [0, 2^32), merge the warps into warp 0's,
  // carry again, then add them to `out`: a block adds less than 2^32 to a
  // limb.
  const auto carry = [](long long* a) {
    for (int l = 0; l + 1 < ExactSum::limbs; ++l) {
      const long long c = a[l] >> 32;
      a[l] -= c * (1LL << 32);
      a[l + 1] += c;
    }
  };
  if (static_cast<int>(threadIdx.x) < warps * K) carry(acc[threadIdx.x / K][threadIdx.x % K].w);
  __syncthreads();
  for (int t = static_cast<int>(threadIdx.x); t < K * ExactSum::words;
       t += static_cast<int>(blockDim.x)) {
    long long s = 0;
    for (int q = 0; q < warps; ++q) s += acc[q][t / ExactSum::words].w[t % ExactSum::words];
    acc[0][t / ExactSum::words].w[t % ExactSum::words] = s;
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < K) carry(acc[0][threadIdx.x].w);
  __syncthreads();
  for (int t = static_cast<int>(threadIdx.x); t < K * ExactSum::words;
       t += static_cast<int>(blockDim.x)) {
    const long long v = acc[0][t / ExactSum::words].w[t % ExactSum::words];
    if (v != 0)
      atomicAdd(reinterpret_cast<ull*>(&out[t / ExactSum::words].w[t % ExactSum::words]),
                static_cast<ull>(v));
  }
}

inline void check(
#if defined(__HIPCC__)
    hipError_t e,
#else
    cudaError_t e,
#endif
    const char* what) {
#if defined(__HIPCC__)
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
#else
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
#endif
}

}  // namespace detail

/// K exact sums over n items: `out[k]` (device memory, K entries, zeroed
/// here) receives the sum over i of term(i, k), a __device__ functor
/// returning double. Asynchronous on `stream`; `blocks` 0 picks 4 per
/// multiprocessor. Up to 2^30 items per block and 2^31 blocks.
template <int K, class Term>
void exact_sums(std::int64_t n, Term term, ExactWords* out, stream_t stream, int blocks = 0) {
  static_assert(K >= 1 && K * ExactSum::words * 8 * (detail::block_size / 32) <= 48 * 1024,
                "K sums per warp must fit in 48 KB of shared memory");
#if defined(__HIPCC__)
  detail::check(hipMemsetAsync(out, 0, K * sizeof(ExactWords), stream), "hipMemsetAsync");
#else
  detail::check(cudaMemsetAsync(out, 0, K * sizeof(ExactWords), stream), "cudaMemsetAsync");
#endif
  if (blocks <= 0) {
    int device = 0, units = 0;
#if defined(__HIPCC__)
    detail::check(hipGetDevice(&device), "hipGetDevice");
    detail::check(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device),
                  "hipDeviceGetAttribute");
#else
    detail::check(cudaGetDevice(&device), "cudaGetDevice");
    detail::check(cudaDeviceGetAttribute(&units, cudaDevAttrMultiProcessorCount, device),
                  "cudaDeviceGetAttribute");
#endif
    blocks = 4 * units;
  }
  detail::exact_sums_kernel<K><<<blocks, detail::block_size, 0, stream>>>(n, term, out);
#if defined(__HIPCC__)
  detail::check(hipGetLastError(), "exact_sums");
#else
  detail::check(cudaGetLastError(), "exact_sums");
#endif
}

/// The sum the words hold, rounded once to the nearest double (ties to even),
/// on the device: the bits of ExactSum::value() on the host over the same
/// words. One thread; +inf, -inf and NaN terms as on the host.
__device__ inline double exact_value(const ExactWords& words) {
  using detail::ull;
  constexpr int limbs = ExactSum::limbs;
  const double inf = __longlong_as_double(0x7ff0000000000000LL);
  const long long* in = words.w;
  if (in[detail::nan_count] > 0 || (in[detail::pos_inf] > 0 && in[detail::neg_inf] > 0))
    return __longlong_as_double(0x7ff8000000000000LL);
  if (in[detail::pos_inf] > 0) return inf;
  if (in[detail::neg_inf] > 0) return -inf;
  // As ExactSum::value(): carry every limb into [0, 2^32), take the
  // magnitude, then round its 64 leading bits to 53.
  long long w[limbs];
  for (int i = 0; i < limbs; ++i) w[i] = in[i];
  const auto normalize = [&w] {
    for (int i = 0; i + 1 < limbs; ++i) {
      const long long carry = w[i] >> 32;  // floor division by 2^32
      w[i] -= carry * (1LL << 32);
      w[i + 1] += carry;
    }
  };
  normalize();
  const bool negative = w[limbs - 1] < 0;
  if (negative) {
    for (int i = 0; i < limbs; ++i) w[i] = -w[i];
    normalize();
  }
  int t = limbs - 1;
  while (t >= 0 && w[t] == 0) --t;
  if (t < 0) return 0.0;
  const auto limb = [&w](int k) { return k < 0 ? ull{0} : static_cast<ull>(w[k]); };
  const ull top = limb(t);
  if (top > 0xffffffffULL) return negative ? -inf : inf;    // >= 2^1056
  const int b = 64 - __clzll(static_cast<long long>(top));  // 1..32
  const ull below = (limb(t - 1) << 32) | limb(t - 2);
  const ull lead = (top << (64 - b)) | (below >> b);
  bool sticky = (below & ((ull{1} << b) - 1)) != 0;
  for (int k = t - 3; k >= 0 && !sticky; --k) sticky = limb(k) != 0;
  ull mantissa = lead >> 11;
  const ull rest = lead & 0x7ff;
  if (rest > 0x400 || (rest == 0x400 && (sticky || (mantissa & 1) != 0))) ++mantissa;
  int exponent = 32 * (t - 2) - detail::offset + b + 11;
  if (mantissa == (ull{1} << 53)) {
    mantissa >>= 1;
    ++exponent;
  }
  const double magnitude = ldexp(static_cast<double>(mantissa), exponent);
  return negative ? -magnitude : magnitude;
}

/// The K sums of `out` on the host, after the work queued on `stream`, ready
/// for Ranks::sum.
template <int K>
std::array<ExactSum, K> to_host(const ExactWords* out, stream_t stream) {
  std::array<ExactWords, K> words{};
#if defined(__HIPCC__)
  detail::check(hipMemcpyAsync(words.data(), out, sizeof(words), hipMemcpyDeviceToHost, stream),
                "hipMemcpyAsync");
  detail::check(hipStreamSynchronize(stream), "hipStreamSynchronize");
#else
  detail::check(cudaMemcpyAsync(words.data(), out, sizeof(words), cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync");
  detail::check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
#endif
  std::array<ExactSum, K> sums;
  for (int k = 0; k < K; ++k) {
    const auto d = sums[static_cast<std::size_t>(k)].data();
    for (int j = 0; j < ExactSum::words; ++j) d[static_cast<std::size_t>(j)] = words[k].w[j];
  }
  return sums;
}

}  // namespace autan::runtime::gpu
