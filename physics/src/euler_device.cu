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

namespace autan::physics {

namespace {

namespace ek = autan::kernels::euler;
namespace gpu = autan::kernels::gpu;
using kernels::index_t;

constexpr int block = 256;

int blocks(std::size_t n) { return static_cast<int>((n + block - 1) / block); }

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

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class P>
class CudaStepper final : public DeviceStepper<P> {
  using D = ek::Data<P>;
  using S = typename P::state;
  using Gr = typename P::grad;
  using R = typename P::residual;
  using Result = typename DeviceStepper<P>::Result;

 public:
  CudaStepper(const Mesh& mesh, const Geometry& geometry, const Euler<P>& host, double gamma,
              bool second_order, runtime::Ranks& ranks, void* stream)
      : stream_(static_cast<cudaStream_t>(stream)),
        ranks_(ranks),
        scheme_(host.scheme()),
        second_order_(second_order),
        n_(static_cast<std::size_t>(mesh.n_cells())),
        owned_(static_cast<std::size_t>(mesh.n_owned_cells())),
        nf_(static_cast<std::size_t>(mesh.n_faces())),
        gamma_(gamma),
        mesh_(mesh, kernels::pack_geometry(mesh, geometry), &host.stencil(),
              {.block = block,
               .face_scratch_width = 5,
               .incidence_scratch_width = 0,
               .stream = stream_}),
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
        sums_(6, stream_) {
    const D& h = host.data();
    source_.upload(std::span(h.source, 5 * n_), stream_);
    bc_kind_.upload(std::span(h.bc_kind, nf_), stream_);
    bc_.upload(std::span(h.bc, 5 * nf_), stream_);
    if (second_order_ && !scheme_.limiter) {
      phi_.upload(std::vector<S>(5 * n_, S(1)), stream_);
    }
    mesh_.set_stencil_weights(host.gradient_weights());
    mesh_.bind(d_);
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
    const auto& r = scheme_.reference;
    p_ref_ = (gamma - 1.0) * (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]);
  }

  Result step(double cfl, std::optional<double> dt, bool local) override {
    Result out;
    residual(u_.get());
    out.residuals = norms();
    double h = dt.value_or(0.0);
    const double* local_dt = nullptr;
    if (!dt) {
      ek::gpu_loops::spectral_radius(d_, mesh_.schedule());
      local_time_step<<<blocks(n_), block, 0, stream_>>>(index(n_), cfl, volume_.get(), lam_.get(),
                                                         dt_.get());
      check(cudaGetLastError(), "local_time_step");
      h = gpu::min(index(n_), dt_.get(), stream_);
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

  void upload(std::span<const double> u) override { u_.upload(u, stream_); }
  void download(std::span<double> u) override { u_.download(u, stream_); }

 private:
  static index_t index(std::size_t n) { return static_cast<index_t>(n); }

  // Euler<P>::residual of the stored state of `u`.
  void residual(const double* u) {
    store_state<<<blocks(n_), block, 0, stream_>>>(index(n_), u, ref_, state_.get());
    check(cudaGetLastError(), "store_state");
    const auto& sched = mesh_.schedule();
    ek::gpu_loops::primitives(d_, sched);
    ek::gpu_loops::boundary_states(d_, sched);
    if (second_order_) {
      ek::gpu_loops::gradients(d_, sched);
      if (scheme_.limiter) ek::gpu_loops::limiter(d_, sched);
    }
    ek::gpu_loops::sources(d_, sched);
    constexpr auto deterministic = gpu::Strategy::store_gather;
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
    auto c = gpu::first_index(index(owned_), flag_.get(), stream_);
    if (c < index(owned_)) return c;
    invalid_residuals<<<blocks(owned_), block, 0, stream_>>>(index(owned_), index(n_), res_.get(),
                                                             flag_.get());
    check(cudaGetLastError(), "invalid_residuals");
    c = gpu::first_index(index(owned_), flag_.get(), stream_);
    if (c < index(owned_)) return c;
    return std::nullopt;
  }

  cudaStream_t stream_;
  runtime::Ranks& ranks_;
  Scheme scheme_;
  bool second_order_;
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
  gpu::DeviceArray<runtime::gpu::ExactWords> sums_;
  D d_{};
};

}  // namespace

template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const Euler<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      void* stream) {
  return std::make_unique<CudaStepper<P>>(mesh, geometry, host, gamma, second_order, ranks, stream);
}

template std::unique_ptr<DeviceStepper<kernels::fp64>> make_device_stepper(
    const Mesh&, const Geometry&, const Euler<kernels::fp64>&, double, bool, runtime::Ranks&,
    void*);
template std::unique_ptr<DeviceStepper<kernels::fp32_acc64>> make_device_stepper(
    const Mesh&, const Geometry&, const Euler<kernels::fp32_acc64>&, double, bool, runtime::Ranks&,
    void*);

}  // namespace autan::physics
