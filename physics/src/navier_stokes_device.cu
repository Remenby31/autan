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
        quadratic_(host.euler().scheme().gradient_fit == GradientFit::quadratic),
        origin_(host.data().mu_t) {
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
  // Whether `mu_t` is the eddy viscosity array of the host flow this stepper copies.
  [[nodiscard]] bool copies(const void* mu_t) const noexcept { return mu_t == origin_; }

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
  // The host flow's eddy viscosity array: the turbulence model of that flow shares it.
  const void* origin_;
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

// The j-th incidence of cell c, in the CSR or the sliced layout (kernels::Topology).
__device__ inline kernels::offset_t incidence(const kernels::Topology& t, kernels::index_t c,
                                              kernels::offset_t j) {
  return t.cell_face_offsets[c] + j * t.incidence_stride;
}
__device__ inline kernels::offset_t incidences(const kernels::Topology& t, kernels::index_t c) {
  return t.incidence_length != nullptr ? t.incidence_length[c]
                                       : t.cell_face_offsets[c + 1] - t.cell_face_offsets[c];
}

// SpalartAllmaras<P>::assemble's diagonal under global keys: shift + max(0, cell block), then the
// cell's faces in Topology::cell_faces order, in the host's arithmetic.
template <class J>
__global__ void sa_assemble_diagonal(kernels::Topology t, const double* shift, const J* cell_jac,
                                     const J* face_jac, const J* bnd_jac, double* diag) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= t.n_cells) return;
  const double x = static_cast<double>(cell_jac[c]);
  double d = shift[c] + (0.0 < x ? x : 0.0);
  const auto m = incidences(t, c);
  for (kernels::offset_t j = 0; j < m; ++j) {
    const auto f = t.cell_faces[incidence(t, c, j)];
    if (f >= t.n_interior_faces) {
      d += static_cast<double>(bnd_jac[f]);
    } else if (t.owner[f] == c) {
      d += static_cast<double>(face_jac[f]);
    } else {
      d -= static_cast<double>(face_jac[t.n_faces + f]);
    }
  }
  diag[c] = d;
}

// A(owner, neighbour) = dF/d(rho nt) of the neighbour, A(neighbour, owner) = -dF/d(rho nt) of the
// owner (no reversed faces on one rank).
template <class J>
__global__ void sa_assemble_off_diagonal(kernels::Topology t, const J* face_jac, double* upper,
                                         double* lower) {
  const auto f = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (f >= t.n_interior_faces) return;
  upper[f] = static_cast<double>(face_jac[t.n_faces + f]);
  lower[f] = -static_cast<double>(face_jac[f]);
}

// One colour of SpalartAllmaras<P>::solve: each thread relaxes one cell of the colour, its row
// summing its interior faces in Topology::cell_faces order. No two cells of a colour are
// neighbours, so the cells of a launch read none of each other's values.
__global__ void sa_relax_colour(kernels::Topology t, const kernels::index_t* cells,
                                kernels::index_t count, const double* diag, const double* upper,
                                const double* lower, const double* b, double* dx) {
  const auto i = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const auto c = cells[i];
  double r = b[c];
  const auto m = incidences(t, c);
  for (kernels::offset_t j = 0; j < m; ++j) {
    const auto f = t.cell_faces[incidence(t, c, j)];
    if (f >= t.n_interior_faces) continue;
    const auto o = t.owner[f];
    const auto n = t.neighbour[f];
    r -= o == c ? upper[f] * dx[n] : lower[f] * dx[o];
  }
  dx[c] = r / diag[c];
}

// b = -R in FP64, as the host's step forms it.
template <class R>
__global__ void sa_negate(kernels::index_t n, const R* res, double* b) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  b[c] = -static_cast<double>(res[c]);
}

// update_rho_nt, in the host's arithmetic: std::max(a, b) is (a < b ? b : a) and std::clamp(v,
// lo, hi) is (v < lo ? lo : hi < v ? hi : v).
__global__ void sa_update(kernels::index_t n, double* q, const double* dx, const double* rho,
                          double nt_inf, double alpha) {
  const auto c = static_cast<kernels::index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) return;
  const double a = fabs(q[c]);
  const double reference = rho[c] * nt_inf;
  const double scale = a < reference ? reference : a;
  const double lo = -sa_max_decrease * scale;
  const double hi = sa_max_increase * scale;
  const double v = dx[c];
  q[c] += alpha * (v < lo ? lo : hi < v ? hi : v);
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
    cell_jac_ = gpu::DeviceArray<element<decltype(A::nt_cell_jac)>>(n_, s);
    face_jac_ = gpu::DeviceArray<element<decltype(A::nt_face_jac)>>(2 * nf_, s);
    bnd_jac_ = gpu::DeviceArray<element<decltype(A::nt_bnd_jac)>>(nf_, s);
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
    a_.nt_cell_jac = cell_jac_.get();
    a_.nt_face_jac = face_jac_.get();
    a_.nt_bnd_jac = bnd_jac_.get();
    a_.gas_constant[0] = h.gas_constant[0];
    a_.p_ref[0] = h.p_ref[0];
    a_.mu_ref[0] = h.mu_ref[0];
    a_.t_ref[0] = h.t_ref[0];
    a_.sutherland_s[0] = h.sutherland_s[0];
    a_.sutherland[0] = h.sutherland[0];
    a_.ft2_on[0] = h.ft2_on[0];
    // The implicit step's scope, from the host's topology: global keys (gathers in the global
    // order), no ghost cell, no face reversed against its global orientation.
    const auto& ht = h.topo;
    bool reversed = false;
    if (ht.face_reversed != nullptr) {
      for (std::size_t f = 0; f < nf_; ++f) reversed = reversed || ht.face_reversed[f] != 0;
    }
    one_part_ = ht.rank_independent && ht.n_owned_cells == ht.n_cells && !reversed;
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

  void residual_and_jacobian(std::span<const double> rnu, std::span<double> res) override {
    implicit_scope();
    evaluate_jacobian(rnu);
    const auto r = res_.download(flow_.stream());
    if (res.size() != r.size()) throw std::invalid_argument("res does not match the cells");
    for (std::size_t i = 0; i < r.size(); ++i) res[i] = static_cast<double>(r[i]);
  }

  void jacobians(std::span<double> cell, std::span<double> face,
                 std::span<double> boundary) override {
    implicit_scope();
    widen(cell_jac_.download(flow_.stream()), cell);
    widen(face_jac_.download(flow_.stream()), face);
    widen(bnd_jac_.download(flow_.stream()), boundary);
  }

  void assemble(std::span<const double> shift, std::span<double> diag, std::span<double> upper,
                std::span<double> lower) override {
    implicit_scope();
    assemble_on_device(shift);
    diag_.download(diag, flow_.stream());
    upper_.download(upper, flow_.stream());
    lower_.download(lower, flow_.stream());
  }

  void solve(std::span<const double> diag, std::span<const double> upper,
             std::span<const double> lower, std::span<const double> b, std::span<double> dx,
             int sweeps, const std::vector<std::vector<std::size_t>>& colours) override {
    implicit_scope();
    system();
    diag_.upload(diag, flow_.stream());
    upper_.upload(upper, flow_.stream());
    lower_.upload(lower, flow_.stream());
    b_.upload(b, flow_.stream());
    dx_.upload(std::span<const double>(dx), flow_.stream());
    sweep(sweeps, colours);
    dx_.download(dx, flow_.stream());
  }

  void step(std::span<double> rnu, std::span<const double> shift, std::span<const double> rho,
            double nt_inf, double alpha, int sweeps,
            const std::vector<std::vector<std::size_t>>& colours) override {
    implicit_scope();
    if (rho.size() != n_) throw std::invalid_argument("rho does not match the cells");
    evaluate_jacobian(rnu);
    assemble_on_device(shift);
    const auto st = flow_.stream();
    sa_negate<<<blocks(n_), device::block, 0, st>>>(index(n_), res_.get(), b_.get());
    check(cudaGetLastError(), "sa_negate");
    check(cudaMemsetAsync(dx_.get(), 0, n_ * sizeof(double), st), "sa dx zero");
    sweep(sweeps, colours);
    rho_.upload(rho, st);
    q_.upload(std::span<const double>(rnu), st);
    sa_update<<<blocks(n_), device::block, 0, st>>>(index(n_), q_.get(), dx_.get(), rho_.get(),
                                                    nt_inf, alpha);
    check(cudaGetLastError(), "sa_update");
    q_.download(rnu, st);
  }

 private:
  void store(std::span<const double> rnu) {
    if (rnu.size() != n_) throw std::invalid_argument("rho nt does not match the cells");
    input_.upload(rnu, flow_.stream());
    store_scalar<<<blocks(n_), device::block, 0, flow_.stream()>>>(
        static_cast<kernels::index_t>(n_), input_.get(), rnu_.get());
    check(cudaGetLastError(), "store_scalar");
  }

  static kernels::index_t index(std::size_t n) { return static_cast<kernels::index_t>(n); }

  template <class T>
  static void widen(const std::vector<T>& v, std::span<double> out) {
    if (out.size() != v.size()) throw std::invalid_argument("a Jacobian block size does not match");
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
  }

  // The implicit step's scope (ADR 0005, B; #163 step 7b): one rank, no ghost cells, the global
  // order equal to the local one, so the gathers follow SpalartAllmaras<P>::assemble's keyed path.
  void implicit_scope() const {
    if (!one_part_) {
      throw std::invalid_argument(
          "the device SA step runs on one rank without ghost cells, under identity keys");
    }
  }

  void evaluate_jacobian(std::span<const double> rnu) {
    const auto& sched = flow_.mesh().schedule();
    store(rnu);
    sk::gpu_loops::nt_values(a_, sched);
    sk::gpu_loops::boundary_values(a_, sched);
    sk::gpu_loops::nt_gradients(a_, sched);
    sk::gpu_loops::nt_sources_jacobian(a_, sched);
    constexpr auto deterministic = gpu::Strategy::store_gather;
    sk::gpu_loops::nt_faces_jacobian(a_, deterministic, sched);
    sk::gpu_loops::nt_boundaries_jacobian(a_, deterministic, sched);
  }

  void system() {
    if (diag_.size() == n_) return;
    const auto st = flow_.stream();
    shift_ = gpu::DeviceArray<double>(n_, st);
    diag_ = gpu::DeviceArray<double>(n_, st);
    upper_ = gpu::DeviceArray<double>(static_cast<std::size_t>(a_.topo.n_interior_faces), st);
    lower_ = gpu::DeviceArray<double>(static_cast<std::size_t>(a_.topo.n_interior_faces), st);
    b_ = gpu::DeviceArray<double>(n_, st);
    dx_ = gpu::DeviceArray<double>(n_, st);
    rho_ = gpu::DeviceArray<double>(n_, st);
    q_ = gpu::DeviceArray<double>(n_, st);
  }

  void assemble_on_device(std::span<const double> shift) {
    system();
    const auto st = flow_.stream();
    shift_.upload(shift, st);
    sa_assemble_diagonal<<<blocks(n_), device::block, 0, st>>>(
        a_.topo, shift_.get(), cell_jac_.get(), face_jac_.get(), bnd_jac_.get(), diag_.get());
    check(cudaGetLastError(), "sa_assemble_diagonal");
    const auto ni = static_cast<std::size_t>(a_.topo.n_interior_faces);
    if (ni > 0) {
      sa_assemble_off_diagonal<<<blocks(ni), device::block, 0, st>>>(a_.topo, face_jac_.get(),
                                                                     upper_.get(), lower_.get());
      check(cudaGetLastError(), "sa_assemble_off_diagonal");
    }
  }

  // Colours 0 ... n-1 then n-1 ... 0, `sweeps` times, one launch per colour in stream order.
  void sweep(int sweeps, const std::vector<std::vector<std::size_t>>& colours) {
    const auto st = flow_.stream();
    colour_cells_.clear();
    for (const auto& cells : colours) {
      std::vector<kernels::index_t> v(cells.size());
      for (std::size_t i = 0; i < cells.size(); ++i) v[i] = index(cells[i]);
      colour_cells_.emplace_back(std::span<const kernels::index_t>(v), st);
    }
    auto colour = [&](std::size_t k) {
      const auto count = colours[k].size();
      if (count == 0) return;
      sa_relax_colour<<<blocks(count), device::block, 0, st>>>(
          a_.topo, colour_cells_[k].get(), index(count), diag_.get(), upper_.get(), lower_.get(),
          b_.get(), dx_.get());
      check(cudaGetLastError(), "sa_relax_colour");
    };
    for (int s = 0; s < sweeps; ++s) {
      for (std::size_t k = 0; k < colours.size(); ++k) colour(k);
      for (std::size_t k = colours.size(); k-- > 0;) colour(k);
    }
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
  gpu::DeviceArray<element<decltype(A::nt_cell_jac)>> cell_jac_;
  gpu::DeviceArray<element<decltype(A::nt_face_jac)>> face_jac_;
  gpu::DeviceArray<element<decltype(A::nt_bnd_jac)>> bnd_jac_;
  gpu::DeviceArray<double> shift_, diag_, upper_, lower_, b_, dx_, rho_, q_;
  std::vector<gpu::DeviceArray<kernels::index_t>> colour_cells_;
  bool one_part_ = false;
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
  if (ns == nullptr || !ns->copies(host.data().mu_t)) {
    throw std::invalid_argument(
        "Spalart-Allmaras on a device needs the stepper of the flow it is coupled to");
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
