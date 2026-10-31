#pragma once

// One spelling of the GPU runtime for the generated CUDA and HIP loops, so that both backends
// compile the same emitted text.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "autan/kernels/runtime.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define AUTAN_GPU_HIP 1
#else
#include <cuda_runtime.h>
#define AUTAN_GPU_CUDA 1
#endif

namespace autan::kernels::gpu {

#if defined(AUTAN_GPU_HIP)
using stream_t = hipStream_t;
using error_t = hipError_t;
inline constexpr error_t success = hipSuccess;
inline const char* error_string(error_t e) { return hipGetErrorString(e); }
inline error_t last_error() { return hipGetLastError(); }
using lane_mask_t = std::uint64_t;
// Wave size of the compiled target: 64 on CDNA (gfx9), 32 on RDNA.
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
inline constexpr int warp_size = 64;
#else
inline constexpr int warp_size = 32;
#endif
#else
using stream_t = cudaStream_t;
using error_t = cudaError_t;
inline constexpr error_t success = cudaSuccess;
inline const char* error_string(error_t e) { return cudaGetErrorString(e); }
inline error_t last_error() { return cudaGetLastError(); }
using lane_mask_t = unsigned int;
inline constexpr int warp_size = 32;
#endif

inline constexpr lane_mask_t full_mask = ~lane_mask_t{0};

inline void check(error_t e, const char* what) {
  if (e != success) {
    throw std::runtime_error(std::string("E1101: ") + what + ": " + error_string(e));
  }
}

template <class T>
__device__ inline T shfl_down(T v, int delta) {
  return __shfl_down_sync(full_mask, v, delta);
}

template <class T>
__device__ inline T shfl_up(T v, int delta) {
  return __shfl_up_sync(full_mask, v, delta);
}

/// Index of the contiguous run of `head` flags this lane belongs to, within its warp: lanes with
/// equal ids form one segment. Every lane of the warp must call it.
__device__ inline int segment_id(bool head) {
  const int lane = static_cast<int>(threadIdx.x) % warp_size;
#if defined(AUTAN_GPU_HIP)
  const lane_mask_t heads = __ballot(head);
  const lane_mask_t upto = full_mask >> (warp_size - 1 - lane);
  return __popcll(heads & upto);
#else
  const lane_mask_t heads = __ballot_sync(full_mask, head);
  const lane_mask_t upto = full_mask >> (warp_size - 1 - lane);
  return __popc(heads & upto);
#endif
}

/// Face loop race strategies (docs/research/kernel-generator.md section 6).
enum class Strategy : std::uint8_t {
  atomic,            ///< one thread per face, atomic increments
  warp_aggregated,   ///< owner-sorted faces: one atomic per owner segment of a warp
  coloured,          ///< one launch per face colour, plain increments
  store_gather,      ///< face fluxes stored, then summed per cell in incidence order
  gather_recompute,  ///< each cell evaluates the fluxes of its own faces
};

inline const char* name(Strategy s) {
  switch (s) {
    case Strategy::atomic:
      return "atomic";
    case Strategy::warp_aggregated:
      return "warp-aggregated";
    case Strategy::coloured:
      return "coloured";
    case Strategy::store_gather:
      return "store-gather";
    case Strategy::gather_recompute:
      return "gather-recompute";
  }
  return "?";
}

/// True for strategies whose sums run in an order fixed by the mesh numbering.
inline bool deterministic(Strategy s) {
  return s == Strategy::coloured || s == Strategy::store_gather ||
         s == Strategy::gather_recompute;
}

/// Faces of one range grouped in colours: no two faces of a colour touch the same cell.
/// `faces` is a device array; `offsets` (host) delimits each colour in it.
struct Colouring {
  const index_t* faces = nullptr;
  std::vector<index_t> offsets;
};

/// Everything a strategy needs besides the data: launch shape, colourings and scratch.
struct Schedule {
  int block = 256;
  Colouring interior;
  Colouring boundary;
  void* face_scratch = nullptr;  ///< store-gather: n_outputs x n_faces accumulation values
  stream_t stream = nullptr;
};

}  // namespace autan::kernels::gpu
