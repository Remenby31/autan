#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define AUTAN_HD __host__ __device__
#else
#define AUTAN_HD
#endif

namespace autan::kernels {

using index_t = std::int32_t;
using offset_t = std::int64_t;

/// Topology arrays that generated loops read; host or device pointers. Filled from
/// autan::MeshView (docs/contracts/mesh.md). Incidence k is entry k of the cell -> face CSR.
struct Topology {
  index_t n_cells;
  index_t n_owned_cells;
  index_t n_faces;
  index_t n_interior_faces;
  offset_t n_incidences;
  const index_t* owner;
  const index_t* neighbour;
  const offset_t* cell_face_offsets;
  const index_t* cell_faces;
};

/// Precision policies (docs/research/kernel-generator.md section 7): storage type of each role,
/// compute type of bodies, accumulation type of Inc targets and reductions.
struct fp64 {
  using state = double;
  using grad = double;
  using residual = double;
  using geom = double;
  using param = double;
  using compute = double;
  using accum = double;
  static constexpr const char* name = "fp64";
};

struct fp32_acc64 {
  using state = float;
  using grad = float;
  using residual = double;
  using geom = double;
  using param = double;
  using compute = float;
  using accum = double;
  static constexpr const char* name = "fp32+acc64";
};

struct fp32 {
  using state = float;
  using grad = float;
  using residual = float;
  using geom = float;
  using param = float;
  using compute = float;
  using accum = float;
  static constexpr const char* name = "fp32";
};

/// Complex-step policy: compute in std::complex<double> on the host (tests only).
struct complex_step {
  using state = std::complex<double>;
  using grad = std::complex<double>;
  using residual = std::complex<double>;
  using geom = double;
  using param = double;
  using compute = std::complex<double>;
  using accum = std::complex<double>;
  static constexpr const char* name = "complex-step";
};

// Comparisons go through these so that the complex-step policy compares real parts
// (Martins, Sturdza and Alonso 2003, section 4).
template <class T>
AUTAN_HD inline bool lt(T a, T b) {
  return a < b;
}
template <class T>
AUTAN_HD inline bool le(T a, T b) {
  return a <= b;
}
template <class T>
AUTAN_HD inline bool gt(T a, T b) {
  return a > b;
}
template <class T>
AUTAN_HD inline bool ge(T a, T b) {
  return a >= b;
}
template <class T>
inline bool lt(std::complex<T> a, std::complex<T> b) {
  return a.real() < b.real();
}
template <class T>
inline bool le(std::complex<T> a, std::complex<T> b) {
  return a.real() <= b.real();
}
template <class T>
inline bool gt(std::complex<T> a, std::complex<T> b) {
  return a.real() > b.real();
}
template <class T>
inline bool ge(std::complex<T> a, std::complex<T> b) {
  return a.real() >= b.real();
}

template <class T>
AUTAN_HD inline T infinity() {
  if constexpr (sizeof(T) == sizeof(float)) {
    return T(__builtin_huge_valf());
  } else {
    return T(__builtin_huge_val());
  }
}
template <>
inline std::complex<double> infinity<std::complex<double>>() {
  return {__builtin_huge_val(), 0.0};
}

}  // namespace autan::kernels
