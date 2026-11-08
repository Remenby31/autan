#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/physics/euler_device.hpp"
#include "autan/runtime/exact_sum.cuh"
#include "euler_gpu.cuh"

namespace autan::physics::device {

namespace ek = autan::kernels::euler;
namespace gpu = autan::kernels::gpu;
using kernels::index_t;

constexpr int block = 256;

inline int blocks(std::size_t n) { return static_cast<int>((n + block - 1) / block); }

template <class Ptr>
using element = std::remove_cvref_t<std::remove_pointer_t<Ptr>>;

// Correctly rounded operations, never contracted into FMAs: the host is built with
// -ffp-contract=off, and these kernels repeat its arithmetic operation for operation.
template <class T>
__device__ T add(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    return __fadd_rn(a, b);
  } else {
    return __dadd_rn(a, b);
  }
}
template <class T>
__device__ T sub(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    return __fsub_rn(a, b);
  } else {
    return __dsub_rn(a, b);
  }
}
template <class T>
__device__ T mul(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    return __fmul_rn(a, b);
  } else {
    return __dmul_rn(a, b);
  }
}
template <class T>
__device__ T div(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    return __fdiv_rn(a, b);
  } else {
    return __ddiv_rn(a, b);
  }
}

struct Five {
  double v[5];
};

// Euler<P>::state: s = State(u - U_ref).
template <class S>
__global__ void store_state(index_t n, const double* u, Five r, S* s) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  for (int k = 0; k < 5; ++k) {
    const auto i = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + c;
    s[i] = static_cast<S>(sub(u[i], r.v[k]));
  }
}

// EulerPhysics::unsteady_step: dt_c = CFL V / lambda.
template <class S>
__global__ void local_time_step(index_t n, double cfl, const double* volume, const S* lambda,
                                double* dt) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  dt[c] = div(mul(cfl, volume[c]), static_cast<double>(lambda[c]));
}

// One SSP-RK3 stage: next = stage - dt / V R; stage = a u + b next.
template <class R>
__global__ void rk_stage(index_t n, double a, double b, const double* u, const R* res,
                         const double* volume, const double* dt, double dt_global, double* stage) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  const double h = dt != nullptr ? dt[c] : dt_global;
  for (int k = 0; k < 5; ++k) {
    const auto i = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + c;
    const double next = sub(stage[i], mul(div(h, volume[c]), static_cast<double>(res[i])));
    stage[i] = add(mul(a, u[i]), mul(b, next));
  }
}

// EulerPhysics::first_invalid_cell, its first pass: the stored state as the kernels read it,
// in P::compute.
template <class P>
__global__ void invalid_states(index_t count, index_t n, const double* u, Five r, double gamma,
                               double p_ref, std::uint8_t* flag) {
  using C = typename P::compute;
  using S = typename P::state;
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= count) return;
  const C g = static_cast<C>(gamma);
  const C r0 = static_cast<C>(r.v[0]);
  C dq[5];
  for (int k = 0; k < 5; ++k) {
    const auto i = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + c;
    dq[k] = static_cast<C>(static_cast<S>(sub(u[i], r.v[k])));
  }
  const C rho = add(r0, dq[0]);
  C dm_m = 0, mr_mr = 0;
  for (int k = 0; k < 3; ++k) {
    const C mr = static_cast<C>(r.v[k + 1]);
    dm_m = add(dm_m, mul(dq[k + 1], add(mul(C(2), mr), dq[k + 1])));
    mr_mr = add(mr_mr, mul(mr, mr));
  }
  const C dk = div(sub(mul(dm_m, r0), mul(mr_mr, dq[0])), mul(rho, r0));
  const C p = add(mul(sub(g, C(1)), sub(dq[4], mul(C(0.5), dk))), static_cast<C>(p_ref));
  flag[c] = !isfinite(static_cast<double>(rho)) || !isfinite(static_cast<double>(p)) ||
            !(rho > C(0)) || !(p > C(0));
}

// Its second pass: a residual that is not finite.
template <class R>
__global__ void invalid_residuals(index_t count, index_t n, const R* res, std::uint8_t* flag) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= count) return;
  bool bad = false;
  for (int k = 0; k < 5; ++k) {
    bad = bad || !isfinite(static_cast<double>(
                     res[static_cast<std::size_t>(k) * static_cast<std::size_t>(n) + c]));
  }
  flag[c] = bad;
}

// Of the flagged cells, the least global id (the cell index without ids): exact, so the same
// whatever the launch and the numbering.
static __global__ void least_flagged(index_t n, const std::uint8_t* flag, const std::int64_t* ids,
                                     unsigned long long* best) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n || flag[c] == 0) return;
  const auto id = static_cast<unsigned long long>(ids != nullptr ? ids[c] : c);
  atomicMin(best, id);
}

// The cell whose id is `best` (ids are unique).
static __global__ void cell_of(index_t n, const std::uint8_t* flag, const std::int64_t* ids,
                               const unsigned long long* best, index_t* cell) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n || flag[c] == 0) return;
  const auto id = static_cast<unsigned long long>(ids != nullptr ? ids[c] : c);
  if (id == *best) *cell = c;
}

// Terms of the residual norms: R_k^2 for k < 5, and 1 (the cell count) for k = 5.
template <class R>
struct SquareTerm {
  const R* res;
  std::int64_t n;
  __device__ double operator()(std::int64_t i, int k) const {
    if (k == 5) return 1.0;
    const double r = static_cast<double>(res[k * n + i]);
    return r * r;
  }
};

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// The device's built-in geometry. With the planar Euler loops and no other loop reading them
// (`moments` false), no face moments are uploaded (36 B per face in FP32).
inline kernels::PackedGeometry device_geometry(const Mesh& mesh, const Geometry& geometry,
                                               bool planar, bool moments,
                                               const kernels::GlobalKeys* keys) {
  auto packed = kernels::pack_geometry(mesh, geometry, keys);
  if (planar && !packed.all_planar) {
    throw std::invalid_argument("planar loops on a mesh with a face not proven planar");
  }
  if (planar && !moments) decltype(packed.face_moment)().swap(packed.face_moment);
  return packed;
}

// The Euler residual and its explicit steps on the device. A physics that adds terms to the
// residual derives from it: add_terms() runs after the convective fluxes, add_spectral_radius()
// after the convective spectral radius; the derived constructor binds its own Data to mesh_ and
// then calls mesh_.release_host() (`release` false here).
template <class P>
class CudaStepper : public DeviceStepper<P> {
 protected:
  using D = ek::Data<P>;
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  using Result = typename DeviceStepper<P>::Result;

 public:
  CudaStepper(const Mesh& mesh, const Geometry& geometry, const Euler<P>& host, double gamma,
              bool second_order, runtime::Ranks& ranks, runtime::GhostExchange& ghosts,
              void* stream, const kernels::GlobalKeys* keys, bool moments = false,
              bool release = true)
      : stream_(static_cast<cudaStream_t>(stream)),
        ranks_(ranks),
        ghosts_(ghosts),
        scheme_(host.scheme()),
        second_order_(second_order),
        planar_(host.planar()),
        n_(static_cast<std::size_t>(mesh.n_cells())),
        owned_(static_cast<std::size_t>(mesh.n_owned_cells())),
        nf_(static_cast<std::size_t>(mesh.n_faces())),
        gamma_(gamma),
        mesh_(mesh, device_geometry(mesh, geometry, planar_, moments, keys), &host.stencil(),
              {.block = block,
               .face_scratch_width = 5,
               .incidence_scratch_width = 0,
               .stream = stream_,
               .global = keys}),
        u_(5 * n_, stream_),
        stage_(5 * n_, stream_),
        volume_(std::span<const double>(geometry.cell_volume), stream_),
        dt_(n_, stream_),
        state_(5 * n_, stream_),
        prim_(5 * n_, stream_),
        prim_b_(5 * nf_, stream_),
        grad_(15 * n_, stream_),
        wmin_(5 * n_, stream_),
        wmax_(5 * n_, stream_),
        phi_(5 * n_, stream_),
        lam_(n_, stream_),
        res_(5 * n_, stream_),
        source_(5 * n_, stream_),
        bc_kind_(nf_, stream_),
        bc_(5 * nf_, stream_),
        flag_(n_, stream_),
        ids_(mesh.cell_global_id(), stream_),
        best_(1, stream_),
        cell_(1, stream_),
        sums_(6, stream_) {
    const D& h = host.data();
    source_.upload(std::span(h.source, 5 * n_), stream_);
    bc_kind_.upload(std::span(h.bc_kind, nf_), stream_);
    bc_.upload(std::span(h.bc, 5 * nf_), stream_);
    if (second_order_ && !scheme_.limiter) {
      phi_.upload(std::vector<S>(5 * n_, S(1)), stream_);
    }
    mesh_.set_stencil_weights<D>(host.gradient_weights());
    if (release) {
      mesh_.bind_and_release(d_);
    } else {
      mesh_.bind(d_);
    }
    d_.u = state_.get();
    d_.prim = prim_.get();
    d_.prim_b = prim_b_.get();
    d_.grad = grad_.get();
    d_.wmin = wmin_.get();
    d_.wmax = wmax_.get();
    d_.phi = phi_.get();
    d_.lam = lam_.get();
    d_.res = res_.get();
    d_.source = source_.get();
    d_.bc_kind = bc_kind_.get();
    d_.bc = bc_.get();
    d_.gamma[0] = h.gamma[0];
    d_.gas_constant[0] = h.gas_constant[0];
    d_.mach_ref[0] = h.mach_ref[0];
    d_.p_ref[0] = h.p_ref[0];
    for (int k = 0; k < 5; ++k) {
      d_.u_ref[k] = h.u_ref[k];
      d_.limiter_scale[k] = h.limiter_scale[k];
    }
    for (int k = 0; k < 5; ++k) ref_.v[k] = scheme_.reference[static_cast<std::size_t>(k)];
    // The host's value (computed in host code built with -ffp-contract=off), not recomputed
    // here: nvcc may contract this expression.
    p_ref_ = static_cast<double>(h.p_ref[0]);
    if (scheme_.gradient_fit == GradientFit::quadratic) {
      // Euler<P>::Impl::band_gradients: the quadratic fit's own stencil in place of the gradient
      // stencil, weights converted to the dataset's type as the host converts them.
      const auto q = host.quadratic_gradient();
      using B = element<decltype(D::band)>;
      using W = element<decltype(D::band_weight)>;
      const auto band = kernels::converted<B>(q.fitted);
      const std::vector<kernels::offset_t> offsets(q.offsets.begin(), q.offsets.end());
      const auto weights = kernels::converted<W>(q.weights);
      band_ = gpu::DeviceArray<B>(std::span(band), stream_);
      band_offsets_ = gpu::DeviceArray<kernels::offset_t>(std::span(offsets), stream_);
      band_entries_ = gpu::DeviceArray<index_t>(std::span(q.entries), stream_);
      band_weight_ = gpu::DeviceArray<W>(std::span(weights), stream_);
      grad_band_ = gpu::DeviceArray<element<decltype(D::grad_band)>>(15 * n_, stream_);
      d_.band = band_.get();
      d_.grad_band = grad_band_.get();
      d_.band_weight = band_weight_.get();
      band_topo_ = d_.topo;
      band_topo_.n_stencil = static_cast<kernels::offset_t>(q.entries.size());
      band_topo_.stencil_offsets = band_offsets_.get();
      band_topo_.stencil = band_entries_.get();
      band_topo_.stencil_twin = nullptr;
      band_topo_.stencil_stride = 1;
    }
  }

  Result step(double cfl, std::optional<double> dt, bool local) override {
    Result out;
    residual(u_.get());
    out.residuals = norms();
    double h = dt.value_or(0.0);
    const double* local_dt = nullptr;
    if (!dt) {
      ek::gpu_loops::spectral_radius(d_, mesh_.schedule());
      add_spectral_radius();
      local_time_step<<<blocks(n_), block, 0, stream_>>>(index(n_), cfl, volume_.get(), lam_.get(),
                                                         dt_.get());
      check(cudaGetLastError(), "local_time_step");
      h = ranks_.min(gpu::min(index(owned_), dt_.get(), stream_));
      if (local) local_dt = dt_.get();
    }
    // Shu and Osher (1988), J. Comput. Phys. 77:439-471, eq. 2.18.
    constexpr std::array<std::array<double, 2>, 3> weights{
        {{0.0, 1.0}, {0.75, 0.25}, {1.0 / 3.0, 2.0 / 3.0}}};
    check(cudaMemcpyAsync(stage_.get(), u_.get(), 5 * n_ * sizeof(double), cudaMemcpyDeviceToDevice,
                          stream_),
          "stage");
    for (std::size_t s = 0; s < 3; ++s) {
      if (s > 0) residual(stage_.get());
      rk_stage<<<blocks(n_), block, 0, stream_>>>(index(n_), weights[s][0], weights[s][1], u_.get(),
                                                  res_.get(), volume_.get(), local_dt, h,
                                                  stage_.get());
      check(cudaGetLastError(), "rk_stage");
    }
    std::swap(u_, stage_);
    out.time_step = h;
    out.invalid = invalid();
    return out;
  }

  Result evaluate() override {
    Result out;
    residual(u_.get());
    out.residuals = norms();
    out.invalid = invalid();
    return out;
  }

  void residual(std::span<const double> u, std::span<double> res) override {
    stage_.upload(u, stream_);
    residual(stage_.get());
    const auto r = res_.download(stream_);
    for (std::size_t i = 0; i < r.size(); ++i) res[i] = static_cast<double>(r[i]);
  }

  void set_second_order(bool on) override {
    second_order_ = on;
    // As the host: without reconstruction gradients and limiter are zero; unlimited, phi is 1.
    check(cudaMemsetAsync(grad_.get(), 0, grad_.size() * sizeof(*grad_.get()), stream_), "grad");
    if (on && !scheme_.limiter) {
      phi_.upload(std::vector<S>(5 * n_, S(1)), stream_);
    } else {
      check(cudaMemsetAsync(phi_.get(), 0, phi_.size() * sizeof(*phi_.get()), stream_), "phi");
    }
  }

  void upload(std::span<const double> u) override { u_.upload(u, stream_); }
  void download(std::span<double> u) override { u_.download(u, stream_); }

 protected:
  virtual void add_terms() {}
  virtual void add_spectral_radius() {}

  static index_t index(std::size_t n) { return static_cast<index_t>(n); }

  // Euler<P>::residual of the stored state of `u`.
  void residual(const double* u) {
    store_state<<<blocks(n_), block, 0, stream_>>>(index(n_), u, ref_, state_.get());
    check(cudaGetLastError(), "store_state");
    const auto& sched = mesh_.schedule();
    // As Euler<P>::Impl::cell_loops: each loop's output into the ghosts before a later loop
    // reads it there.
    ek::gpu_loops::primitives(d_, sched);
    fill_ghosts(prim_.get(), 5);
    ek::gpu_loops::boundary_states(d_, sched);
    if (second_order_) {
      if (scheme_.gradient_fit == GradientFit::quadratic) {
        auto b = d_;
        b.topo = band_topo_;
        ek::gpu_loops::band_gradients(b, sched);
        ek::gpu_loops::quadratic_gradients(d_, sched);
      } else {
        ek::gpu_loops::gradients(d_, sched);
      }
      fill_ghosts(grad_.get(), 15);
      if (scheme_.limiter) {
        ek::gpu_loops::limiter(d_, sched);
        fill_ghosts(phi_.get(), 5);
      }
    }
    ek::gpu_loops::sources(d_, sched);
    constexpr auto deterministic = gpu::Strategy::store_gather;
    if (planar_) {
      switch (scheme_.flux) {
        case Flux::roe:
          ek::gpu_loops::roe_faces_planar(d_, deterministic, sched);
          ek::gpu_loops::roe_boundaries_planar(d_, deterministic, sched);
          break;
        case Flux::hllc:
          ek::gpu_loops::hllc_faces_planar(d_, deterministic, sched);
          ek::gpu_loops::hllc_boundaries_planar(d_, deterministic, sched);
          break;
        case Flux::ausm_up:
          ek::gpu_loops::ausm_faces_planar(d_, deterministic, sched);
          ek::gpu_loops::ausm_boundaries_planar(d_, deterministic, sched);
          break;
      }
      add_terms();
      return;
    }
    switch (scheme_.flux) {
      case Flux::roe:
        ek::gpu_loops::roe_faces(d_, deterministic, sched);
        ek::gpu_loops::roe_boundaries(d_, deterministic, sched);
        break;
      case Flux::hllc:
        ek::gpu_loops::hllc_faces(d_, deterministic, sched);
        ek::gpu_loops::hllc_boundaries(d_, deterministic, sched);
        break;
      case Flux::ausm_up:
        ek::gpu_loops::ausm_faces(d_, deterministic, sched);
        ek::gpu_loops::ausm_boundaries(d_, deterministic, sched);
        break;
    }
    add_terms();
  }

  template <class T>
  // Euler<P>::Impl::fill_ghosts: at most 16 components per exchange call, so any number of
  // components goes in calls of 16.
  void fill_ghosts(T* v, std::size_t components) {
    std::array<T*, 16> c{};
    for (std::size_t first = 0; first < components; first += c.size()) {
      const std::size_t count = std::min(c.size(), components - first);
      for (std::size_t k = 0; k < count; ++k) c[k] = v + (first + k) * n_;
      ghosts_.forward(std::span<T* const>(c.data(), count), MemorySpace::cuda, stream_);
    }
  }

  // RMS of each residual over the owned cells of every rank, from exact sums.
  std::vector<double> norms() {
    runtime::gpu::exact_sums<6>(static_cast<std::int64_t>(owned_),
                                SquareTerm<R>{res_.get(), static_cast<std::int64_t>(n_)},
                                sums_.get(), stream_);
    auto sums = runtime::gpu::to_host<6>(sums_.get(), stream_);
    ranks_.sum(sums);
    std::vector<double> out(5);
    for (std::size_t k = 0; k < 5; ++k) out[k] = std::sqrt(sums[k].value() / sums[5].value());
    return out;
  }

  std::optional<index_t> invalid() {
    invalid_states<P><<<blocks(owned_), block, 0, stream_>>>(index(owned_), index(n_), u_.get(),
                                                             ref_, gamma_, p_ref_, flag_.get());
    check(cudaGetLastError(), "invalid_states");
    if (const auto c = least_invalid()) return c;
    invalid_residuals<<<blocks(owned_), block, 0, stream_>>>(index(owned_), index(n_), res_.get(),
                                                             flag_.get());
    check(cudaGetLastError(), "invalid_residuals");
    return least_invalid();
  }

  // As the host's first_invalid_cell: the flagged owned cell with the least cell_global_id.
  std::optional<index_t> least_invalid() {
    if (gpu::first_index(index(owned_), flag_.get(), stream_) == index(owned_)) return std::nullopt;
    const unsigned long long none = ~0ULL;
    best_.upload(std::span<const unsigned long long>(&none, 1), stream_);
    const std::int64_t* ids = ids_.size() == n_ ? ids_.get() : nullptr;
    least_flagged<<<blocks(owned_), block, 0, stream_>>>(index(owned_), flag_.get(), ids,
                                                         best_.get());
    cell_of<<<blocks(owned_), block, 0, stream_>>>(index(owned_), flag_.get(), ids, best_.get(),
                                                   cell_.get());
    check(cudaGetLastError(), "least_invalid");
    return cell_.download(stream_)[0];
  }

  cudaStream_t stream_;
  runtime::Ranks& ranks_;
  runtime::GhostExchange& ghosts_;
  Scheme scheme_;
  bool second_order_;
  bool planar_;
  std::size_t n_, owned_, nf_;
  double gamma_, p_ref_ = 0.0;
  Five ref_{};
  gpu::DeviceMesh mesh_;
  gpu::DeviceArray<double> u_, stage_, volume_, dt_;
  gpu::DeviceArray<element<decltype(D::u)>> state_;
  gpu::DeviceArray<element<decltype(D::prim)>> prim_;
  gpu::DeviceArray<element<decltype(D::prim_b)>> prim_b_;
  gpu::DeviceArray<element<decltype(D::grad)>> grad_;
  gpu::DeviceArray<element<decltype(D::wmin)>> wmin_, wmax_, phi_;
  gpu::DeviceArray<element<decltype(D::lam)>> lam_;
  gpu::DeviceArray<element<decltype(D::res)>> res_;
  gpu::DeviceArray<element<decltype(D::source)>> source_;
  gpu::DeviceArray<element<decltype(D::bc_kind)>> bc_kind_;
  gpu::DeviceArray<element<decltype(D::bc)>> bc_;
  gpu::DeviceArray<std::uint8_t> flag_;
  gpu::DeviceArray<std::int64_t> ids_;  // cell_global_id; empty for the identity
  gpu::DeviceArray<unsigned long long> best_;
  gpu::DeviceArray<index_t> cell_;
  gpu::DeviceArray<runtime::gpu::ExactWords> sums_;
  // The quadratic gradient's (GradientFit::quadratic only).
  gpu::DeviceArray<element<decltype(D::band)>> band_;
  gpu::DeviceArray<kernels::offset_t> band_offsets_;
  gpu::DeviceArray<index_t> band_entries_;
  gpu::DeviceArray<element<decltype(D::band_weight)>> band_weight_;
  gpu::DeviceArray<element<decltype(D::grad_band)>> grad_band_;
  kernels::Topology band_topo_{};
  D d_{};
};

}  // namespace autan::physics::device
