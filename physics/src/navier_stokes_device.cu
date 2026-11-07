#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "autan/physics/navier_stokes_device.hpp"
#include "autan/physics/spalart_allmaras_device.hpp"
#include "device_stepper.cuh"
#include "navier_stokes_gpu.cuh"
#include "spalart_allmaras_gpu.cuh"

namespace autan::physics {

namespace {

namespace nk = autan::kernels::navier_stokes;
namespace sk = autan::kernels::spalart_allmaras;
namespace gpu = autan::kernels::gpu;
using device::blocks;
using device::check;
using device::element;

// NavierStokes<P>::spectral_radius: lambda += 4 lambda_v in the state's type (Blazek 2015,
// eq. 6.20), correctly rounded as on the host.
template <class S>
__global__ void add_viscous_radius(kernels::index_t n, const S* lam_v, S* lam) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  lam[c] = device::add(lam[c], device::mul(S(4), lam_v[c]));
}

template <class Ptr>
gpu::DeviceArray<element<Ptr>> copy_of(Ptr host, std::size_t n, cudaStream_t stream) {
  return gpu::DeviceArray<element<Ptr>>(std::span<const element<Ptr>>(host, n), stream);
}

template <class P>
class NavierStokesStepper final : public device::CudaStepper<P> {
  using Base = device::CudaStepper<P>;
  using N = nk::Data<P>;

 public:
  NavierStokesStepper(const Mesh& mesh, const Geometry& geometry, const NavierStokes<P>& host,
                      double gamma, bool second_order, runtime::Ranks& ranks,
                      runtime::GhostExchange& ghosts, void* stream, const kernels::GlobalKeys* keys)
      : Base(mesh, geometry, host.euler(), gamma, second_order, ranks, ghosts, stream, keys,
             /*moments=*/true, /*release=*/false),
        quadratic_(host.euler().scheme().gradient_fit == GradientFit::quadratic) {
    const N& h = host.data();
    const auto n = this->n_;
    const auto nf = this->nf_;
    const auto s = this->stream_;
    const auto nst = static_cast<std::size_t>(h.topo.n_stencil);
    fit_offsets_ = copy_of(h.topo.stencil_offsets, n + 1, s);
    fit_entries_ = copy_of(h.topo.stencil, nst, s);
    for (std::size_t k = 0; k < 7; ++k) {
      const auto* q = std::array{h.q0, h.q1, h.q2, h.q3, h.q4, h.q5, h.q6}[k];
      q_[k] = copy_of(q, 3 * nst, s);
      fits_[k] = gpu::DeviceArray<element<decltype(N::fit0)>>(15 * n, s);
    }
    quad_ = copy_of(h.quad, n, s);
    mu_t_ = copy_of(h.mu_t, n, s);
    mu_t_b_ = copy_of(h.mu_t_b, nf, s);
    ut_ = gpu::DeviceArray<element<decltype(N::ut)>>(5 * n, s);
    ut_b_ = gpu::DeviceArray<element<decltype(N::ut_b)>>(5 * nf, s);
    lam_v_ = gpu::DeviceArray<element<decltype(N::lam_v)>>(n, s);
    if (quadratic_) {
      grad_v_ = gpu::DeviceArray<element<decltype(N::grad)>>(15 * n, s);
      lo_ = gpu::DeviceArray<element<decltype(Base::D::wmin)>>(5 * n, s);
      hi_ = gpu::DeviceArray<element<decltype(Base::D::wmax)>>(5 * n, s);
    }
    this->mesh_.bind(v_);
    this->mesh_.release_host();
    // NavierStokes<P>::Impl::share: the flow's arrays and topology, the boundary fit's stencil.
    v_.topo.n_stencil = static_cast<kernels::offset_t>(nst);
    v_.topo.stencil_offsets = fit_offsets_.get();
    v_.topo.stencil = fit_entries_.get();
    v_.topo.stencil_twin = nullptr;
    v_.topo.stencil_stride = 1;
    v_.prim = this->d_.prim;
    v_.grad = quadratic_ ? grad_v_.get() : this->d_.grad;
    v_.prim_b = this->d_.prim_b;
    v_.bc_kind = this->d_.bc_kind;
    v_.bc = this->d_.bc;
    v_.res = this->d_.res;
    v_.ut = ut_.get();
    v_.ut_b = ut_b_.get();
    v_.q0 = q_[0].get();
    v_.q1 = q_[1].get();
    v_.q2 = q_[2].get();
    v_.q3 = q_[3].get();
    v_.q4 = q_[4].get();
    v_.q5 = q_[5].get();
    v_.q6 = q_[6].get();
    v_.fit0 = fits_[0].get();
    v_.fit1 = fits_[1].get();
    v_.fit2 = fits_[2].get();
    v_.fit3 = fits_[3].get();
    v_.fit4 = fits_[4].get();
    v_.fit5 = fits_[5].get();
    v_.fit6 = fits_[6].get();
    v_.quad = quad_.get();
    v_.mu_t = mu_t_.get();
    v_.mu_t_b = mu_t_b_.get();
    v_.lam_v = lam_v_.get();
    v_.gas_constant[0] = h.gas_constant[0];
    v_.p_ref[0] = h.p_ref[0];
    v_.gamma[0] = h.gamma[0];
    v_.mu_ref[0] = h.mu_ref[0];
    v_.t_ref[0] = h.t_ref[0];
    v_.sutherland_s[0] = h.sutherland_s[0];
    v_.sutherland[0] = h.sutherland[0];
    v_.prandtl[0] = h.prandtl[0];
    v_.prandtl_t[0] = h.prandtl_t[0];
  }

  // For the turbulence model on the same device: the flow's arrays of its last evaluation and
  // the eddy viscosity it reads, as NavierStokes<P>::arrays() gives them on the host.
  [[nodiscard]] const typename Base::D& flow() const noexcept { return this->d_; }
  [[nodiscard]] const N& viscous() const noexcept { return v_; }
  [[nodiscard]] gpu::DeviceMesh& mesh() noexcept { return this->mesh_; }
  [[nodiscard]] cudaStream_t stream() const noexcept { return this->stream_; }
  [[nodiscard]] std::size_t n_cells() const noexcept { return this->n_; }
  [[nodiscard]] std::size_t n_faces() const noexcept { return this->nf_; }

 private:
  // NavierStokes<P>::residual after the Euler residual: under the quadratic fit the linear
  // gradient (Euler<P>::linear_gradient, its limiter bounds into scratch), then the boundary fit
  // and the viscous fluxes.
  void add_terms() override {
    const auto& sched = this->mesh_.schedule();
    if (quadratic_) {
      auto d = this->d_;
      d.grad = grad_v_.get();
      d.wmin = lo_.get();
      d.wmax = hi_.get();
      kernels::euler::gpu_loops::gradients(d, sched);
    }
    nk::gpu_loops::temperatures_velocities(v_, sched);
    nk::gpu_loops::boundary_temperatures_velocities(v_, sched);
    nk::gpu_loops::cubic_fits(v_, sched);
    constexpr auto deterministic = gpu::Strategy::store_gather;
    nk::gpu_loops::viscous_faces(v_, deterministic, sched);
    nk::gpu_loops::viscous_boundaries(v_, deterministic, sched);
  }

  void add_spectral_radius() override {
    nk::gpu_loops::viscous_spectral_radius(v_, this->mesh_.schedule());
    add_viscous_radius<<<blocks(this->n_), device::block, 0, this->stream_>>>(
        Base::index(this->n_), lam_v_.get(), this->lam_.get());
    check(cudaGetLastError(), "add_viscous_radius");
  }

  bool quadratic_;
  gpu::DeviceArray<kernels::offset_t> fit_offsets_;
  gpu::DeviceArray<kernels::index_t> fit_entries_;
  std::array<gpu::DeviceArray<element<decltype(N::q0)>>, 7> q_;
  std::array<gpu::DeviceArray<element<decltype(N::fit0)>>, 7> fits_;
  gpu::DeviceArray<element<decltype(N::quad)>> quad_;
  gpu::DeviceArray<element<decltype(N::mu_t)>> mu_t_, mu_t_b_, lam_v_;
  gpu::DeviceArray<element<decltype(N::ut)>> ut_, ut_b_;
  gpu::DeviceArray<element<decltype(N::grad)>> grad_v_;
  gpu::DeviceArray<element<decltype(Base::D::wmin)>> lo_, hi_;
  N v_{};
};

// rho nt stored in the state's type, as the host's callers store it: State(v).
template <class S>
__global__ void store_scalar(kernels::index_t n, const double* v, S* s) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  s[c] = static_cast<S>(v[c]);
}

// SpalartAllmaras<P> on the device of a NavierStokesStepper<P>: the loops of
// kernels/spalart_allmaras.py on the flow's arrays, in the host's order.
template <class P>
class SpalartAllmarasDevice final : public DeviceSpalartAllmaras<P> {
  using A = sk::Data<P>;

 public:
  SpalartAllmarasDevice(const SpalartAllmaras<P>& host, NavierStokesStepper<P>& flow)
      : flow_(flow), n_(flow.n_cells()), nf_(flow.n_faces()) {
    const A& h = host.data();
    const auto s = flow.stream();
    input_ = gpu::DeviceArray<double>(n_, s);
    rnu_ = gpu::DeviceArray<element<decltype(A::rnu)>>(n_, s);
    nt_ = gpu::DeviceArray<element<decltype(A::nt)>>(n_, s);
    nt_b_ = gpu::DeviceArray<element<decltype(A::nt_b)>>(nf_, s);
    nt_bc_ = copy_of(h.nt_bc, nf_, s);
    source_ = copy_of(h.nt_source, n_, s);
    wall_ = copy_of(h.wall_distance, n_, s);
    nt_grad_ = gpu::DeviceArray<element<decltype(A::nt_grad)>>(3 * n_, s);
    res_ = gpu::DeviceArray<element<decltype(A::nt_res)>>(n_, s);
    flow.mesh().bind(a_);
    // SpalartAllmaras<P>::Impl::share: the flow's arrays, gradient stencil and weights.
    const auto& d = flow.flow();
    const auto& v = flow.viscous();
    a_.topo = d.topo;
    a_.stencil_lsq_weight = d.stencil_lsq_weight;
    a_.prim = d.prim;
    a_.grad = v.grad;
    a_.prim_b = d.prim_b;
    a_.bc_kind = d.bc_kind;
    a_.mu_t = v.mu_t;
    a_.mu_t_b = v.mu_t_b;
    a_.rnu = rnu_.get();
    a_.nt = nt_.get();
    a_.nt_b = nt_b_.get();
    a_.nt_bc = nt_bc_.get();
    a_.nt_source = source_.get();
    a_.wall_distance = wall_.get();
    a_.nt_grad = nt_grad_.get();
    a_.nt_res = res_.get();
    a_.gas_constant[0] = h.gas_constant[0];
    a_.p_ref[0] = h.p_ref[0];
    a_.mu_ref[0] = h.mu_ref[0];
    a_.t_ref[0] = h.t_ref[0];
    a_.sutherland_s[0] = h.sutherland_s[0];
    a_.sutherland[0] = h.sutherland[0];
    a_.ft2_on[0] = h.ft2_on[0];
  }

  void residual(std::span<const double> rnu, std::span<double> res) override {
    const auto& sched = flow_.mesh().schedule();
    store(rnu);
    sk::gpu_loops::nt_values(a_, sched);
    sk::gpu_loops::boundary_values(a_, sched);
    sk::gpu_loops::nt_gradients(a_, sched);
    sk::gpu_loops::nt_sources(a_, sched);
    constexpr auto deterministic = gpu::Strategy::store_gather;
    sk::gpu_loops::nt_faces(a_, deterministic, sched);
    sk::gpu_loops::nt_boundaries(a_, deterministic, sched);
    const auto r = res_.download(flow_.stream());
    for (std::size_t i = 0; i < r.size(); ++i) res[i] = static_cast<double>(r[i]);
  }

  void eddy_viscosity(std::span<const double> rnu) override {
    const auto& sched = flow_.mesh().schedule();
    store(rnu);
    sk::gpu_loops::nt_values(a_, sched);
    sk::gpu_loops::boundary_values(a_, sched);
    sk::gpu_loops::eddy_viscosities(a_, sched);
    sk::gpu_loops::boundary_eddy_viscosities(a_, sched);
  }

 private:
  void store(std::span<const double> rnu) {
    if (rnu.size() != n_) throw std::invalid_argument("rho nt does not match the cells");
    input_.upload(rnu, flow_.stream());
    store_scalar<<<blocks(n_), device::block, 0, flow_.stream()>>>(
        static_cast<kernels::index_t>(n_), input_.get(), rnu_.get());
    check(cudaGetLastError(), "store_scalar");
  }

  NavierStokesStepper<P>& flow_;
  std::size_t n_, nf_;
  gpu::DeviceArray<double> input_;
  gpu::DeviceArray<element<decltype(A::rnu)>> rnu_;
  gpu::DeviceArray<element<decltype(A::nt)>> nt_;
  gpu::DeviceArray<element<decltype(A::nt_b)>> nt_b_;
  gpu::DeviceArray<element<decltype(A::nt_bc)>> nt_bc_;
  gpu::DeviceArray<element<decltype(A::nt_source)>> source_;
  gpu::DeviceArray<element<decltype(A::wall_distance)>> wall_;
  gpu::DeviceArray<element<decltype(A::nt_grad)>> nt_grad_;
  gpu::DeviceArray<element<decltype(A::nt_res)>> res_;
  A a_{};
};

}  // namespace

template <class P>
std::unique_ptr<DeviceStepper<P>> make_device_stepper(const Mesh& mesh, const Geometry& geometry,
                                                      const NavierStokes<P>& host, double gamma,
                                                      bool second_order, runtime::Ranks& ranks,
                                                      runtime::GhostExchange& ghosts, void* stream,
                                                      const kernels::GlobalKeys* keys) {
  return std::make_unique<NavierStokesStepper<P>>(mesh, geometry, host, gamma, second_order, ranks,
                                                  ghosts, stream, keys);
}

template <class P>
std::unique_ptr<DeviceSpalartAllmaras<P>> make_device_spalart_allmaras(
    const SpalartAllmaras<P>& host, DeviceStepper<P>& flow) {
  auto* ns = dynamic_cast<NavierStokesStepper<P>*>(&flow);
  if (ns == nullptr) {
    throw std::invalid_argument("Spalart-Allmaras on a device needs a Navier-Stokes stepper");
  }
  return std::make_unique<SpalartAllmarasDevice<P>>(host, *ns);
}

template std::unique_ptr<DeviceSpalartAllmaras<kernels::fp64>> make_device_spalart_allmaras(
    const SpalartAllmaras<kernels::fp64>&, DeviceStepper<kernels::fp64>&);
template std::unique_ptr<DeviceSpalartAllmaras<kernels::fp32_acc64>> make_device_spalart_allmaras(
    const SpalartAllmaras<kernels::fp32_acc64>&, DeviceStepper<kernels::fp32_acc64>&);

template std::unique_ptr<DeviceStepper<kernels::fp64>> make_device_stepper(
    const Mesh&, const Geometry&, const NavierStokes<kernels::fp64>&, double, bool, runtime::Ranks&,
    runtime::GhostExchange&, void*, const kernels::GlobalKeys*);
template std::unique_ptr<DeviceStepper<kernels::fp32_acc64>> make_device_stepper(
    const Mesh&, const Geometry&, const NavierStokes<kernels::fp32_acc64>&, double, bool,
    runtime::Ranks&, runtime::GhostExchange&, void*, const kernels::GlobalKeys*);

}  // namespace autan::physics
