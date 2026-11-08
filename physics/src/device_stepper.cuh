#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "autan/kernels/device.cuh"
#include "autan/physics/euler_device.hpp"
#include "autan/runtime/exact_sum.cuh"
#include "autan/solvers/gcr_gpu.hpp"
#include "autan/solvers/line_smoother.hpp"
#include "autan/solvers/line_sweep_gpu.hpp"
#include "euler_gpu.cuh"
#include "euler_jacobian_device.cuh"

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

// EulerPhysics::add_boundary_gradient_term: dv = State(v).
template <class S>
__global__ void to_state(std::int64_t n, const double* v, S* s) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) s[i] = static_cast<S>(v[i]);
}

// EulerPhysics::steady_step: shift_c = lambda_c / CFL.
template <class S>
__global__ void shift_kernel(index_t n, const S* lambda, double cfl, double* shift) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c < n) shift[c] = div(static_cast<double>(lambda[c]), cfl);
}

// rhs = -R.
template <class R>
__global__ void negate_kernel(std::int64_t m, const R* res, double* rhs) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < m) rhs[i] = -static_cast<double>(res[i]);
}

// y += dres, as add_boundary_gradient_term; nothing while *active is 0.
template <class R>
__global__ void add_kernel(std::int64_t m, const R* dres, double* y, const int* active) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m || *active == 0) return;
  y[i] = add(y[i], static_cast<double>(dres[i]));
}

// EulerPhysics::pressure: (gamma - 1) (v_4 - 0.5 (v_1^2 + v_2^2 + v_3^2) / v_0), v = u + a du.
__device__ inline double pressure_at(const double* u, const double* du, std::size_t n,
                                     std::size_t c, double a, double gamma_minus_one) {
  double v[5];
  for (std::size_t k = 0; k < 5; ++k) v[k] = add(u[k * n + c], mul(a, du[k * n + c]));
  const double q = add(add(mul(v[1], v[1]), mul(v[2], v[2])), mul(v[3], v[3]));
  return mul(gamma_minus_one, sub(v[4], div(mul(0.5, q), v[0])));
}

namespace {  // non-template kernels: one copy per translation unit including this header

// EulerPhysics::admissible_step, per owned cell: the step halved from 1 until rho and p keep
// at least (1 - max_decrease) of their values.
__global__ void admissible_kernel(index_t owned, index_t n, const double* u, const double* du,
                                  double gamma_minus_one, double max_decrease, double min_step,
                                  double* alpha) {
  const index_t c = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= owned) return;
  const auto cs = static_cast<std::size_t>(c), ns = static_cast<std::size_t>(n);
  double a = 1.0;
  const double rho = u[cs], drho = du[cs];
  if (drho < 0.0) a = fmin(a, div(mul(max_decrease, rho), -drho));
  const double p = pressure_at(u, du, ns, cs, 0.0, gamma_minus_one);
  const double floor = mul(sub(1.0, max_decrease), p);
  while (a >= min_step && !(pressure_at(u, du, ns, cs, a, gamma_minus_one) >= floor)) {
    a = mul(a, 0.5);
  }
  alpha[cs] = a;
}

// The least of v over n entries, from 1: one block, a fixed tree (a minimum is exact).
__global__ void __launch_bounds__(1024) min_kernel(std::int64_t m, const double* v, double* out) {
  __shared__ double least[1024];
  double l = 1.0;
  for (std::int64_t i = threadIdx.x; i < m; i += blockDim.x) l = fmin(l, v[i]);
  least[threadIdx.x] = l;
  __syncthreads();
  for (unsigned int h = blockDim.x / 2; h > 0; h /= 2) {
    if (threadIdx.x < h) least[threadIdx.x] = fmin(least[threadIdx.x], least[threadIdx.x + h]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = least[0];
}

// u += alpha du.
__global__ void apply_kernel(std::int64_t m, double alpha, const double* du, double* u) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < m) u[i] = add(u[i], mul(alpha, du[i]));
}

}  // namespace

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
        moments_(moments || !planar_),
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
    kept_boundaries(mesh, host);
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

  void assemble(std::span<const double> shift) override {
    if (shift.size() != n_) throw std::invalid_argument("assemble: a shift per cell");
    ensure_jacobian();
    shift_.upload(shift, stream_);
    residual(u_.get(), true);
    assemble_kernel();
  }

  // The Jacobian's arrays, allocated on first use, as on the host: explicit steps carry none.
  void ensure_jacobian() {
    if (!moments_) {
      throw std::logic_error(
          "the device Jacobian reads face moments, which a planar stepper omits");
    }
    if (mesh_.topology().incidence_stride != 1) {
      throw std::logic_error("the device Jacobian assembles the CSR incidence layout only");
    }
    if (dwdu_.size() == 0) {
      face_jac_ = gpu::DeviceArray<J>(50 * nf_, stream_);
      bnd_jac_ = gpu::DeviceArray<J>(25 * nf_, stream_);
      dwdu_ = gpu::DeviceArray<J>(25 * n_, stream_);
      shift_ = gpu::DeviceArray<double>(n_, stream_);
      jac_.diag = gpu::DeviceArray<double>(25 * n_, stream_);
      jac_.off = gpu::DeviceArray<J>(25 * static_cast<std::size_t>(mesh_.topology().n_incidences),
                                     stream_);
      d_.face_jac = face_jac_.get();
      d_.bnd_jac = bnd_jac_.get();
      d_.dwdu = dwdu_.get();
    }
  }

  // The blocks from the face blocks and dW/dU of the last evaluation and the shift.
  void assemble_kernel() {
    device::assemble_blocks<J><<<blocks(n_), block, 0, stream_>>>(
        mesh_.topology(), face_jac_.get(), bnd_jac_.get(), dwdu_.get(), shift_.get(),
        jac_.diag.get(), jac_.off.get());
    check(cudaGetLastError(), "assemble_blocks");
  }

  void boundary_gradient_tangent(std::span<const double> du, std::span<double> dres) override {
    if (!jacobian_current_) {
      throw std::logic_error("the device gradient tangent needs the state of the last assemble");
    }
    if (du.size() != 5 * n_ || dres.size() != 5 * n_) {
      throw std::invalid_argument("boundary_gradient_tangent: 5 values per cell");
    }
    if (tangent_du_.size() == 0) {
      tangent_du_ = gpu::DeviceArray<double>(5 * n_, stream_);
      tangent_dres_ = gpu::DeviceArray<R>(5 * n_, stream_);
    }
    tangent_du_.upload(du, stream_);
    boundary_gradient_tangent(tangent_du_.get(), tangent_dres_.get());
    const auto r = tangent_dres_.download(stream_);
    for (std::size_t i = 0; i < r.size(); ++i) dres[i] = static_cast<double>(r[i]);
  }

  std::vector<double> evaluate_jacobian() override {
    ensure_jacobian();
    residual(u_.get(), true);
    auto out = norms();
    ek::gpu_loops::spectral_radius(d_, mesh_.schedule());
    add_spectral_radius();
    return out;
  }

  void assemble_blocks(double cfl) override {
    if (!jacobian_current_) throw std::logic_error("assemble_blocks follows evaluate_jacobian");
    shift_kernel<<<blocks(n_), block, 0, stream_>>>(index(n_), lam_.get(), cfl, shift_.get());
    check(cudaGetLastError(), "shift");
    assemble_kernel();
  }

  void set_lines(const solvers::LineSmoother& lines) override {
    if (owned_ != n_) throw std::invalid_argument("the device line smoother runs on one rank");
    // The stepper's incidence layout, read once: the slots of the off-diagonal blocks.
    const auto& t = mesh_.topology();
    std::vector<kernels::offset_t> offsets(n_ + 1);
    std::vector<index_t> faces(static_cast<std::size_t>(t.n_incidences));
    check(cudaMemcpyAsync(offsets.data(), t.cell_face_offsets, offsets.size() * sizeof(offsets[0]),
                          cudaMemcpyDeviceToHost, stream_),
          "incidences");
    check(cudaMemcpyAsync(faces.data(), t.cell_faces, faces.size() * sizeof(faces[0]),
                          cudaMemcpyDeviceToHost, stream_),
          "incidences");
    check(cudaStreamSynchronize(stream_), "incidences");
    lines_ = std::make_unique<solvers::gpu::LineSweep>(lines, offsets, faces);
    gcr_ = std::make_unique<solvers::gpu::Gcr>(static_cast<std::int64_t>(5 * n_), 10, stream_);
    du_ = gpu::DeviceArray<double>(5 * n_, stream_);
    krylov_ = gpu::DeviceArray<double>(5 * n_, stream_);
    tangent_dres_ = gpu::DeviceArray<R>(5 * n_, stream_);
    alpha_ = gpu::DeviceArray<double>(n_, stream_);
    least_ = gpu::DeviceArray<double>(1, stream_);
  }

  typename DeviceStepper<P>::LinearSolve solve(int sweeps, int max_cycles, double tolerance,
                                               bool gradient_term) override {
    if (!lines_) throw std::logic_error("solve needs set_lines");
    if (!jacobian_current_) throw std::logic_error("solve follows evaluate_jacobian");
    typename DeviceStepper<P>::LinearSolve out;
    const auto m = static_cast<std::int64_t>(5 * n_);
    negate_kernel<<<blocks(5 * n_), block, 0, stream_>>>(m, res_.get(), krylov_.get());
    check(cudaMemsetAsync(du_.get(), 0, 5 * n_ * sizeof(double), stream_), "du");
    try {
      lines_->set_operator(jac_.diag.get(), jac_.off.get(), stream_);
    } catch (const std::domain_error&) {
      out.singular = true;
      out.residual_drop = std::numeric_limits<double>::quiet_NaN();
      return out;
    }
    const auto report = gcr_->solve(
        [&](const double* x, double* y, const int* active, void* stream) {
          lines_->multiply(x, y, stream, active);
          if (gradient_term) {
            boundary_gradient_tangent(x, tangent_dres_.get());
            add_kernel<<<blocks(5 * n_), block, 0, static_cast<cudaStream_t>(stream)>>>(
                m, tangent_dres_.get(), y, active);
            check(cudaGetLastError(), "add_kernel");
          }
        },
        [&](const double* r, double* z, const int* active, void* stream) {
          lines_->sweep(r, z, sweeps, stream, active);
        },
        krylov_.get(), du_.get(), max_cycles, tolerance);
    out.cycles = report.cycles;
    out.residual_drop = report.residual_drop;
    return out;
  }

  double admissible_step(double gamma_minus_one, double max_decrease, double min_step) override {
    admissible_kernel<<<blocks(owned_), block, 0, stream_>>>(index(owned_), index(n_), u_.get(),
                                                             du_.get(), gamma_minus_one,
                                                             max_decrease, min_step, alpha_.get());
    min_kernel<<<1, 1024, 0, stream_>>>(static_cast<std::int64_t>(owned_), alpha_.get(),
                                        least_.get());
    check(cudaGetLastError(), "admissible_step");
    return least_.download(stream_)[0];
  }

  void apply_step(double alpha) override {
    apply_kernel<<<blocks(5 * n_), block, 0, stream_>>>(static_cast<std::int64_t>(5 * n_), alpha,
                                                        du_.get(), u_.get());
    check(cudaGetLastError(), "apply_step");
  }

  void download_step(std::span<double> du) override { du_.download(du, stream_); }

  std::optional<index_t> first_invalid() override { return invalid(); }

  DeviceStepperBlocks jacobian_blocks() override {
    return download_blocks(mesh_.topology(), jac_, stream_);
  }

  void upload(std::span<const double> u) override { u_.upload(u, stream_); }
  void download(std::span<double> u) override { u_.download(u, stream_); }

 protected:
  virtual void add_terms() {}
  virtual void add_spectral_radius() {}

  static index_t index(std::size_t n) { return static_cast<index_t>(n); }

  // Euler<P>::residual of the stored state of `u`; with `jacobian`,
  // Euler<P>::residual_and_jacobian: dW/dU and the face blocks too, from the loops that are not
  // planar.
  void residual(const double* u, bool jacobian = false) {
    jacobian_current_ = jacobian;
    store_state<<<blocks(n_), block, 0, stream_>>>(index(n_), u, ref_, state_.get());
    check(cudaGetLastError(), "store_state");
    const auto& sched = mesh_.schedule();
    // As Euler<P>::Impl::cell_loops: each loop's output into the ghosts before a later loop
    // reads it there.
    if (jacobian) {
      ek::gpu_loops::primitives_jacobian(d_, sched);
      fill_ghosts(dwdu_.get(), 25);
    } else {
      ek::gpu_loops::primitives(d_, sched);
    }
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
    if (jacobian) {
      switch (scheme_.flux) {
        case Flux::roe:
          ek::gpu_loops::roe_faces_jacobian(d_, deterministic, sched);
          ek::gpu_loops::roe_boundaries_jacobian(d_, deterministic, sched);
          break;
        case Flux::hllc:
          ek::gpu_loops::hllc_faces_jacobian(d_, deterministic, sched);
          ek::gpu_loops::hllc_boundaries_jacobian(d_, deterministic, sched);
          break;
        case Flux::ausm_up:
          ek::gpu_loops::ausm_faces_jacobian(d_, deterministic, sched);
          ek::gpu_loops::ausm_boundaries_jacobian(d_, deterministic, sched);
          break;
      }
      return;
    }
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

  // Euler<P>::gradient_tangent(s, State(du), dres, Faces::boundary) at the state of the last
  // assemble, dres zeroed first, as EulerPhysics::add_boundary_gradient_term calls it.
  void boundary_gradient_tangent(const double* du, R* dres) {
    check(cudaMemsetAsync(dres, 0, 5 * n_ * sizeof(R), stream_), "dres");
    if (!second_order_) return;
    using Gr = element<decltype(D::grad_tan)>;
    if (prim_tan_.size() == 0) {
      dv_ = gpu::DeviceArray<S>(5 * n_, stream_);
      prim_tan_ = gpu::DeviceArray<S>(5 * n_, stream_);
      zero_state_ = gpu::DeviceArray<S>(5 * n_, stream_);
      wmin_tan_ = gpu::DeviceArray<S>(5 * n_, stream_);
      wmax_tan_ = gpu::DeviceArray<S>(5 * n_, stream_);
      grad_tan_ = gpu::DeviceArray<Gr>(15 * n_, stream_);
      if (scheme_.gradient_fit == GradientFit::quadratic) {
        grad_band_tan_ = gpu::DeviceArray<element<decltype(D::grad_band_tan)>>(15 * n_, stream_);
        band_tan_ = gpu::DeviceArray<element<decltype(D::band_tan)>>(n_, stream_);
      }
      if (n_kept_ > 0) {
        bnd_state_jac_ = gpu::DeviceArray<J>(25 * nf_, stream_);
        prim_b_tan_ = gpu::DeviceArray<double>(5 * (nf_ - ni_), stream_);
        discarded_ = gpu::DeviceArray<R>(5 * n_, stream_);
      }
    }
    to_state<<<blocks(5 * n_), block, 0, stream_>>>(static_cast<std::int64_t>(5 * n_), du,
                                                    dv_.get());
    check(cudaGetLastError(), "to_state");
    const auto& sched = mesh_.schedule();
    auto d = d_;
    d.u_tan = dv_.get();
    d.prim_tan = prim_tan_.get();
    d.source_tan = zero_state_.get();
    ek::gpu_loops::primitives_tangent(d, sched);
    fill_ghosts(prim_tan_.get(), 5);
    d.grad_tan = grad_tan_.get();
    d.wmin_tan = wmin_tan_.get();
    d.wmax_tan = wmax_tan_.get();
    if (scheme_.gradient_fit == GradientFit::quadratic) {
      d.grad_band_tan = grad_band_tan_.get();
      d.band_tan = band_tan_.get();
      auto b = d;
      b.topo = band_topo_;
      ek::gpu_loops::band_gradients_tangent(b, sched);
      ek::gpu_loops::quadratic_gradients_tangent(d, sched);
    } else {
      ek::gpu_loops::gradients_tangent(d, sched);
    }
    if (n_kept_ > 0) {
      // Euler<P>::Impl::kept_boundaries_tangent: every boundary face's state Jacobian, its
      // increment discarded, then the kept cells' boundary values' tangent.
      auto k = d;
      k.bnd_state_jac = bnd_state_jac_.get();
      check(cudaMemsetAsync(discarded_.get(), 0, 5 * n_ * sizeof(R), stream_), "discarded");
      k.res = discarded_.get();
      k.topo.n_owned_faces = k.topo.n_faces;
      ek::gpu_loops::boundary_states_jacobian(k, gpu::Strategy::store_gather, sched);
      boundary_state_tangent<J, S><<<blocks(nf_ - ni_), block, 0, stream_>>>(
          index(ni_), index(nf_), index(n_), bnd_owner_.get(), bnd_state_jac_.get(),
          prim_tan_.get(), prim_b_tan_.get());
      kept_gradient_tangent<Gr, W><<<blocks(n_kept_), block, 0, stream_>>>(
          static_cast<std::int64_t>(n_kept_), kept_cells_.get(), kept_offsets_.get(),
          kept_boundary_.get(), kept_weight_.get(), index(n_), index(nf_ - ni_), prim_b_tan_.get(),
          grad_tan_.get());
      check(cudaGetLastError(), "kept boundaries tangent");
    }
    fill_ghosts(grad_tan_.get(), 15);
    // The fluxes with only the gradients moving: dW and the limiter held.
    d.prim_tan = zero_state_.get();
    d.phi_tan = zero_state_.get();
    d.res_tan = dres;
    constexpr auto deterministic = gpu::Strategy::store_gather;
    switch (scheme_.flux) {
      case Flux::roe:
        ek::gpu_loops::roe_boundaries_tangent(d, deterministic, sched);
        break;
      case Flux::hllc:
        ek::gpu_loops::hllc_boundaries_tangent(d, deterministic, sched);
        break;
      case Flux::ausm_up:
        ek::gpu_loops::ausm_boundaries_tangent(d, deterministic, sched);
        break;
    }
  }

  // The kept cells of the host's gradient fit and, per cell, its boundary entries in stencil
  // order with their weights, as Euler<P>::Impl::kept_boundaries_tangent reads them.
  void kept_boundaries(const Mesh& mesh, const Euler<P>& host) {
    ni_ = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto& stencil = host.stencil();
    const auto nst = stencil.entries.size();
    const auto* w = host.data().stencil_lsq_weight;
    std::vector<double> fitted;
    if (scheme_.gradient_fit == GradientFit::quadratic) fitted = host.quadratic_gradient().fitted;
    std::vector<index_t> cells, boundary;
    std::vector<kernels::offset_t> offsets{0};
    std::vector<W> weights;
    for (const std::size_t c : host.kept_cells()) {
      if (!fitted.empty() && fitted[c] > 0.5) continue;
      for (auto k = static_cast<std::size_t>(stencil.offsets[c]);
           k < static_cast<std::size_t>(stencil.offsets[c + 1]); ++k) {
        const index_t e = stencil.entries[k];
        if (e >= 0) continue;
        boundary.push_back(static_cast<index_t>(static_cast<std::size_t>(-e - 1) - ni_));
        for (std::size_t u = 0; u < 3; ++u) weights.push_back(w[u * nst + k]);
      }
      cells.push_back(static_cast<index_t>(c));
      offsets.push_back(static_cast<kernels::offset_t>(boundary.size()));
    }
    n_kept_ = cells.size();
    if (n_kept_ == 0) return;
    kept_cells_ = gpu::DeviceArray<index_t>(std::span<const index_t>(cells), stream_);
    kept_offsets_ =
        gpu::DeviceArray<kernels::offset_t>(std::span<const kernels::offset_t>(offsets), stream_);
    kept_boundary_ = gpu::DeviceArray<index_t>(std::span<const index_t>(boundary), stream_);
    kept_weight_ = gpu::DeviceArray<W>(std::span<const W>(weights), stream_);
    bnd_owner_ = gpu::DeviceArray<index_t>(mesh.owner(), stream_);
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
  bool moments_;
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
  // The implicit step's (assemble and the boundary gradient tangent only).
  using J = element<decltype(D::face_jac)>;
  using W = element<decltype(D::stencil_lsq_weight)>;
  std::size_t ni_ = 0, n_kept_ = 0;
  bool jacobian_current_ = false;  // the evaluated arrays are those of the last assemble
  // The implicit steady step's (set_lines).
  std::unique_ptr<solvers::gpu::LineSweep> lines_;
  std::unique_ptr<solvers::gpu::Gcr> gcr_;
  gpu::DeviceArray<double> du_, krylov_, alpha_, least_;
  gpu::DeviceArray<double> tangent_du_, prim_b_tan_;
  gpu::DeviceArray<R> tangent_dres_, discarded_;
  gpu::DeviceArray<S> dv_, prim_tan_, zero_state_, wmin_tan_, wmax_tan_;
  gpu::DeviceArray<element<decltype(D::grad_tan)>> grad_tan_;
  gpu::DeviceArray<element<decltype(D::grad_band_tan)>> grad_band_tan_;
  gpu::DeviceArray<element<decltype(D::band_tan)>> band_tan_;
  gpu::DeviceArray<J> bnd_state_jac_;
  gpu::DeviceArray<index_t> kept_cells_, kept_boundary_, bnd_owner_;
  gpu::DeviceArray<kernels::offset_t> kept_offsets_;
  gpu::DeviceArray<W> kept_weight_;
  gpu::DeviceArray<J> face_jac_, bnd_jac_, dwdu_;
  gpu::DeviceArray<double> shift_;
  JacobianBlocks<J> jac_;
  D d_{};
};

}  // namespace autan::physics::device
