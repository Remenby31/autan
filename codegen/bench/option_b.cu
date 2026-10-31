// Option B of research section 3 measured against option A on the Roe flux: the flux and its two
// 5x5 face Jacobians dF/dU_L and dF/dU_R, (B) hand-written as a templated __host__ __device__ body
// instantiated on 10-direction dual numbers (FUN3D style), (A) the generated tangent body called
// for the 10 unit directions. Prints one JSON object: times per face, the largest difference
// between the two Jacobians, and the line counts are taken from the sources by the report.
//
//   option_b MESH [--reps N]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/gpu.hpp"
#include "roe.hpp"

namespace {

namespace gpu = autan::kernels::gpu;
using autan::kernels::index_t;

// ---------------------------------------------------------------- option B, written by hand

template <int N>
struct Dual {
  static constexpr int n = N;
  float v;
  float d[N];
};

template <int N>
__host__ __device__ inline Dual<N> constant(float x) {
  Dual<N> r;
  r.v = x;
  for (int i = 0; i < N; ++i) r.d[i] = 0.0f;
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v + b.v;
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v - b.v;
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator-(const Dual<N>& a) {
  Dual<N> r;
  r.v = -a.v;
  for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v * b.v;
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r;
  r.v = a.v / b.v;
  for (int i = 0; i < N; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) / b.v;
  return r;
}
template <int N>
__host__ __device__ inline Dual<N> operator*(float a, const Dual<N>& b) {
  return constant<N>(a) * b;
}
template <int N>
__host__ __device__ inline Dual<N> sqrt(const Dual<N>& a) {
  Dual<N> r;
  r.v = sqrtf(a.v);
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] / (2.0f * r.v);
  return r;
}
template <int N>
__host__ __device__ inline bool operator<(const Dual<N>& a, const Dual<N>& b) {
  return a.v < b.v;
}
template <int N>
__host__ __device__ inline Dual<N> abs(const Dual<N>& a) {
  return a.v >= 0.0f ? a : -a;
}

template <class T>
__host__ __device__ inline T harten(const T& lam, const T& delta) {
  T a = abs(lam);
  if (a < delta) a = (lam * lam + delta * delta) / (2.0f * delta);
  return a;
}

template <class T>
__host__ __device__ inline void physical_flux(const T* u, const float* n, T* f) {
  const T gm1 = constant<T::n>(0.4f);
  const T vx = u[1] / u[0], vy = u[2] / u[0], vz = u[3] / u[0];
  const T p = gm1 * (u[4] - 0.5f * u[0] * (vx * vx + vy * vy + vz * vz));
  const T vn = vx * constant<T::n>(n[0]) + vy * constant<T::n>(n[1]) + vz * constant<T::n>(n[2]);
  f[0] = u[0] * vn;
  for (int k = 0; k < 3; ++k) f[1 + k] = u[1 + k] * vn + p * constant<T::n>(n[k]);
  f[4] = (u[4] + p) * vn;
}

// Roe (1981) with Harten's fix, the same equations as kernels/roe.py.
template <class T>
__host__ __device__ inline void roe_flux(const T* ul, const T* ur, const float* nf, float area,
                                         T* out) {
  constexpr int N = T::n;
  auto c_ = [](float x) { return constant<N>(x); };
  const T gm1 = c_(0.4f);
  const T rl = ul[0], rr = ur[0];
  T vl[3], vr[3], n[3];
  for (int k = 0; k < 3; ++k) {
    vl[k] = ul[1 + k] / rl;
    vr[k] = ur[1 + k] / rr;
    n[k] = c_(nf[k]);
  }
  const T pl = gm1 * (ul[4] - 0.5f * rl * (vl[0] * vl[0] + vl[1] * vl[1] + vl[2] * vl[2]));
  const T pr = gm1 * (ur[4] - 0.5f * rr * (vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]));
  const T hl = (ul[4] + pl) / rl, hr = (ur[4] + pr) / rr;
  const T s = sqrt(rr / rl);
  const T rho = s * rl;
  const T one_s = c_(1.0f) + s;
  T v[3], dv[3];
  for (int k = 0; k < 3; ++k) {
    v[k] = (vl[k] + s * vr[k]) / one_s;
    dv[k] = vr[k] - vl[k];
  }
  const T h = (hl + s * hr) / one_s;
  const T vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
  const T q2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const T c = sqrt(gm1 * (h - 0.5f * q2));
  const T dp = pr - pl, drho = rr - rl;
  const T dvn = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
  const T delta = 0.1f * c;
  const T l1 = harten(vn - c, delta), l2 = harten(vn, delta), l5 = harten(vn + c, delta);
  const T a1 = l1 * (dp - rho * c * dvn) / (2.0f * c * c);
  const T a5 = l5 * (dp + rho * c * dvn) / (2.0f * c * c);
  const T a2 = l2 * (drho - dp / (c * c));
  T fl[5], fr[5], d[5];
  physical_flux(ul, nf, fl);
  physical_flux(ur, nf, fr);
  d[0] = a1 + a2 + a5;
  for (int k = 0; k < 3; ++k) {
    d[1 + k] = a1 * (v[k] - c * n[k]) + a2 * v[k] + l2 * rho * (dv[k] - dvn * n[k]) +
               a5 * (v[k] + c * n[k]);
  }
  d[4] = a1 * (h - c * vn) + a2 * 0.5f * q2 +
         l2 * rho * (v[0] * dv[0] + v[1] * dv[1] + v[2] * dv[2] - vn * dvn) + a5 * (h + c * vn);
  for (int k = 0; k < 5; ++k) out[k] = c_(area) * (0.5f * (fl[k] + fr[k]) - 0.5f * d[k]);
}

__global__ void jacobian_option_b(index_t n_faces, const index_t* owner, const index_t* neighbour,
                                  index_t n_cells, const float* u, const double* normal,
                                  const double* area, index_t nf, float* jac) {
  const index_t f = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (f >= n_faces) return;
  const index_t l = owner[f], r = neighbour[f];
  Dual<10> ul[5], ur[5], out[5];
  for (int k = 0; k < 5; ++k) {
    ul[k] = constant<10>(u[k * n_cells + l]);
    ur[k] = constant<10>(u[k * n_cells + r]);
    ul[k].d[k] = 1.0f;
    ur[k].d[5 + k] = 1.0f;
  }
  const float n[3] = {float(normal[f]), float(normal[nf + f]), float(normal[2 * nf + f])};
  roe_flux(ul, ur, n, float(area[f]), out);
  for (int k = 0; k < 5; ++k) {
    jac[k * n_faces + f] = out[k].v;
    for (int j = 0; j < 10; ++j) jac[(5 + 10 * k + j) * n_faces + f] = out[k].d[j];
  }
}

// ---------------------------------------------------------------- option A, generated

__global__ void jacobian_option_a(index_t n_faces, const index_t* owner, const index_t* neighbour,
                                  index_t n_cells, const float* u, const double* normal,
                                  const double* area, index_t nf, float* jac) {
  namespace rf = autan::kernels::roe;
  using P = autan::kernels::fp32_acc64;
  const index_t f = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (f >= n_faces) return;
  const index_t l = owner[f], r = neighbour[f];
  float ul[5], ur[5];
  for (int k = 0; k < 5; ++k) {
    ul[k] = u[k * n_cells + l];
    ur[k] = u[k * n_cells + r];
  }
  const double n0 = normal[f], n1 = normal[nf + f], n2 = normal[2 * nf + f], a = area[f];
  float out[5];
  rf::roe_flux<P>(ul[0], ul[1], ul[2], ul[3], ul[4], ur[0], ur[1], ur[2], ur[3], ur[4], n0, n1, n2,
                  a, out);
  for (int k = 0; k < 5; ++k) jac[k * n_faces + f] = out[k];
#pragma unroll
  for (int j = 0; j < 10; ++j) {
    float e[10] = {};
    e[j] = 1.0f;
    rf::flux_tangent_body<P>(ul[0], ul[1], ul[2], ul[3], ul[4], ur[0], ur[1], ur[2], ur[3], ur[4],
                             n0, n1, n2, a, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8],
                             e[9], out);
    for (int k = 0; k < 5; ++k) jac[(5 + 10 * k + j) * n_faces + f] = out[k];
  }
}

template <class K>
double time_kernel(K launch, int reps) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  launch();
  gpu::check(cudaDeviceSynchronize(), "warm-up");
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2] * 1e-3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const int reps = argc > 3 ? std::stoi(argv[3]) : 30;
  auto mesh = autan::io::read_mesh(argv[1]);
  mesh = autan::renumber_cells(mesh, autan::rcm_order(mesh));
  const auto geom = autan::compute_geometry(mesh);
  const auto packed = autan::kernels::pack_geometry(mesh, geom);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const index_t nfi = mesh.n_interior_faces();
  const index_t nf = mesh.n_faces();
  std::mt19937_64 rng(1);
  std::normal_distribution<double> g;
  std::vector<float> u(5 * nc);
  for (std::size_t c = 0; c < nc; ++c) {
    const double rho = 1.0 + 0.1 * std::abs(g(rng)), vx = 0.3 * g(rng), vy = 0.3 * g(rng),
                 vz = 0.3 * g(rng);
    u[c] = float(rho);
    u[nc + c] = float(rho * vx);
    u[2 * nc + c] = float(rho * vy);
    u[3 * nc + c] = float(rho * vz);
    u[4 * nc + c] = float(2.5 + 0.5 * rho * (vx * vx + vy * vy + vz * vz));
  }
  auto up = [](const auto& v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T* p = nullptr;
    gpu::check(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
    gpu::check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
    return p;
  };
  const std::vector<index_t> owner(mesh.owner().begin(), mesh.owner().end());
  const std::vector<index_t> neighbour(mesh.neighbour().begin(), mesh.neighbour().end());
  index_t* d_owner = up(owner);
  index_t* d_neigh = up(neighbour);
  float* d_u = up(u);
  double* d_n = up(packed.face_normal);
  double* d_a = up(packed.face_area_magnitude);
  float *ja = nullptr, *jb = nullptr;
  const std::size_t nj = 55 * static_cast<std::size_t>(nfi);
  gpu::check(cudaMalloc(&ja, nj * sizeof(float)), "malloc");
  gpu::check(cudaMalloc(&jb, nj * sizeof(float)), "malloc");
  const unsigned block = 128, grid = static_cast<unsigned>((nfi + block - 1) / block);
  const index_t ncell = mesh.n_cells();
  const double ta = time_kernel(
      [&] {
        jacobian_option_a<<<grid, block>>>(nfi, d_owner, d_neigh, ncell, d_u, d_n, d_a, nf, ja);
      },
      reps);
  const double tb = time_kernel(
      [&] {
        jacobian_option_b<<<grid, block>>>(nfi, d_owner, d_neigh, ncell, d_u, d_n, d_a, nf, jb);
      },
      reps);
  std::vector<float> ha(nj), hb(nj);
  cudaMemcpy(ha.data(), ja, nj * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), jb, nj * sizeof(float), cudaMemcpyDeviceToHost);
  double diff = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < nj; ++i) {
    diff = std::max(diff, double(std::abs(ha[i] - hb[i])));
    scale = std::max(scale, double(std::abs(hb[i])));
  }
  std::printf(
      "{\"faces\":%d,\"generated_seconds\":%.6e,\"handwritten_seconds\":%.6e,"
      "\"generated_ns_per_face\":%.3f,\"handwritten_ns_per_face\":%.3f,\"max_rel_diff\":%.3e}\n",
      nfi, ta, tb, ta * 1e9 / nfi, tb * 1e9 / nfi, diff / scale);
  return 0;
}
