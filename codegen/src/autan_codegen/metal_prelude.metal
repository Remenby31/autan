// Prelude of every generated Metal library: the Metal spelling of autan/kernels/runtime.hpp,
// the double-float accumulation type and the argument blocks of the generated kernels. Metal
// has no FP64, so the fp32+acc64 policy keeps its FP64 roles (residual, accumulation, geometry,
// parameters) as unevaluated sums of two floats.
#include <metal_stdlib>

namespace autan {
namespace kernels {

using metal::cos;
using metal::exp;
using metal::fma;
using metal::log;
using metal::sin;
using metal::sqrt;

typedef int index_t;
typedef long offset_t;

// Same layout as autan::kernels::Topology on the host, which fills it with GPU addresses.
struct Topology {
  index_t n_cells;
  index_t n_owned_cells;
  index_t n_faces;
  index_t n_interior_faces;
  offset_t n_incidences;
  device const index_t* owner;
  device const index_t* neighbour;
  device const offset_t* cell_face_offsets;
  device const index_t* cell_faces;
};

// Double-float: x = hi + lo with |lo| <= ulp(hi) / 2, about 48 significant bits.
// TwoSum: Knuth, TAOCP vol. 2, 4.2.2, theorem B; Fast2Sum: Dekker (1971), needs |a| >= |b|.
// Sum: Joldes, Muller and Popescu (2017), "Tight and rigorous error bounds for basic building
// blocks of double-word arithmetic", algorithm 6 (AccurateDWPlusDW), relative error <= 3u^2;
// algorithm 4 (DWPlusFP), relative error <= 2u^2, when y is a float (y.lo == 0): accumulations
// of body outputs, half the operations.
// Product: Dekker (1971), the exact error of hi * hi from an FMA, lo * lo dropped.
// Exact only without reassociation: the library is compiled with MTLMathModeSafe.
struct df64 {
  float hi;
  float lo;
  df64() = default;
  df64(float x) : hi(x), lo(0.0f) {}
  df64(int x) : hi(float(x)), lo(0.0f) {}
  df64(float h, float l) : hi(h), lo(l) {}
  explicit operator float() const { return hi + lo; }
};

inline df64 two_sum(float a, float b) {
  const float s = a + b;
  const float bb = s - a;
  return df64(s, (a - (s - bb)) + (b - bb));
}

inline df64 fast_two_sum(float a, float b) {
  const float s = a + b;
  return df64(s, b - (s - a));
}

inline df64 operator+(df64 x, df64 y) {
  if (y.lo == 0.0f) {
    const df64 s = two_sum(x.hi, y.hi);
    return fast_two_sum(s.hi, x.lo + s.lo);
  }
  const df64 s = two_sum(x.hi, y.hi);
  const df64 t = two_sum(x.lo, y.lo);
  const df64 v = fast_two_sum(s.hi, s.lo + t.hi);
  return fast_two_sum(v.hi, t.lo + v.lo);
}

inline df64 operator-(df64 x) { return df64(-x.hi, -x.lo); }
inline df64 operator-(df64 x, df64 y) { return x + (-y); }

inline df64 operator*(df64 x, df64 y) {
  const float p = x.hi * y.hi;
  const float e = fma(x.hi, y.hi, -p);
  return fast_two_sum(p, fma(x.hi, y.lo, fma(x.lo, y.hi, e)));
}

inline thread df64& operator+=(thread df64& a, df64 b) {
  a = a + b;
  return a;
}
inline device df64& operator+=(device df64& a, df64 b) {
  a = a + b;
  return a;
}
inline device df64& operator-=(device df64& a, df64 b) {
  a = a - b;
  return a;
}

inline bool operator<(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
inline bool operator>(df64 a, df64 b) { return b < a; }
inline bool operator<=(df64 a, df64 b) { return !(b < a); }
inline bool operator>=(df64 a, df64 b) { return !(a < b); }

template <class T>
inline bool lt(T a, T b) {
  return a < b;
}
template <class T>
inline bool le(T a, T b) {
  return a <= b;
}
template <class T>
inline bool gt(T a, T b) {
  return a > b;
}
template <class T>
inline bool ge(T a, T b) {
  return a >= b;
}

template <class T>
inline T infinity();
template <>
inline float infinity<float>() {
  return INFINITY;
}
template <>
inline df64 infinity<df64>() {
  return df64(INFINITY, 0.0f);
}

// Device-scope relaxed float atomics (MSL 3.0, section 6.15.2). There is no 64-bit
// floating-point atomic, so the double-float policy has none.
inline void atomicAdd(device float* p, float v) {
  metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic_float*>(p), v,
                                   metal::memory_order_relaxed);
}

// Precision policies: autan::kernels::metal::fp32 and fp32_acc64 on the host.
struct fp32 {
  typedef float state;
  typedef float grad;
  typedef float residual;
  typedef float geom;
  typedef float param;
  typedef float compute;
  typedef float accum;
};

struct fp32_acc64 {
  typedef float state;
  typedef float grad;
  typedef df64 residual;
  typedef df64 geom;
  typedef df64 param;
  typedef float compute;
  typedef df64 accum;
};

// Argument blocks of the generated kernels (buffer 1); host mirrors in autan/kernels/metal.hpp.
struct RangeArgs {
  index_t lo;
  index_t hi;
};

struct ColourArgs {
  device const index_t* faces;
  index_t count;
};

template <class A>
struct StoreArgs {
  index_t lo;
  index_t hi;
  device A* scratch;
};

template <class A>
struct GatherArgs {
  device const A* scratch;
};

constant int warp_size = 32;  // SIMD-group width of every Apple GPU family

/// Index of the contiguous run of `head` flags lane `lane` belongs to within its SIMD group:
/// lanes with equal ids form one segment. Every lane of the SIMD group must call it.
inline int segment_id(bool head, int lane) {
  const ulong heads = static_cast<ulong>(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(head)));
  const ulong upto = lane == 63 ? ~0ul : (2ul << lane) - 1ul;
  return metal::popcount(heads & upto);
}

}  // namespace kernels
}  // namespace autan
