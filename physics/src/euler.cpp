#include "autan/physics/euler.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "euler_cpu.hpp"

namespace autan::physics {

namespace ek = autan::kernels::euler;

template <class P>
struct Euler<P>::Impl {
  using G = typename P::geom;
  using Gr = typename P::grad;

  Scheme scheme;
  kernels::Topology topo{};
  std::vector<G> face_normal, face_area_magnitude, face_centroid, face_moment, cell_centroid,
      cell_volume, stencil_lsq_weight;
  kernels::CellStencil stencil;
  std::vector<double> bc_kind;
  std::vector<State> bc, prim_b, prim, wmin, wmax, phi, source, prim_tan, zero_state;
  std::vector<Gr> grad, zero_grad;
  ek::Data<P> d{};

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme s) : scheme(s) {
    const auto packed = kernels::pack_geometry(mesh, geometry);
    topo = kernels::topology(mesh.view());
    stencil =
        kernels::cell_stencil(mesh, packed,
                              s.gradient == GradientStencil::vertex ? kernels::StencilKind::vertex
                                                                    : kernels::StencilKind::face2);
    topo.n_stencil = static_cast<kernels::offset_t>(stencil.entries.size());
    topo.stencil_offsets = stencil.offsets.data();
    topo.stencil = stencil.entries.data();
    topo.stencil_twin = stencil.twin.data();
    stencil_lsq_weight = kernels::converted<G>(stencil.lsq_weight);
    face_normal = kernels::converted<G>(packed.face_normal);
    face_area_magnitude = kernels::converted<G>(packed.face_area_magnitude);
    face_centroid = kernels::converted<G>(packed.face_centroid);
    face_moment = kernels::converted<G>(packed.face_moment);
    cell_centroid = kernels::converted<G>(packed.cell_centroid);
    cell_volume = kernels::converted<G>(packed.cell_volume);
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    bc_kind.assign(nf, 0.0);
    bc.assign(5 * nf, State(0));
    prim_b.assign(5 * nf, State(0));
    prim.assign(5 * nc, State(0));
    wmin.assign(5 * nc, State(0));
    wmax.assign(5 * nc, State(0));
    phi.assign(5 * nc, State(scheme.second_order && !scheme.limiter ? 1 : 0));
    source.assign(5 * nc, State(0));
    prim_tan.assign(5 * nc, State(0));
    zero_state.assign(5 * nc, State(0));
    grad.assign(15 * nc, Gr(0));
    zero_grad.assign(15 * nc, Gr(0));

    d.topo = topo;
    d.prim = prim.data();
    d.grad = grad.data();
    d.wmin = wmin.data();
    d.wmax = wmax.data();
    d.phi = phi.data();
    d.source = source.data();
    d.prim_b = prim_b.data();
    d.bc_kind = bc_kind.data();
    d.bc = bc.data();
    d.face_normal = face_normal.data();
    d.face_area_magnitude = face_area_magnitude.data();
    d.face_centroid = face_centroid.data();
    d.face_moment = face_moment.data();
    d.cell_centroid = cell_centroid.data();
    d.cell_volume = cell_volume.data();
    d.stencil_lsq_weight = stencil_lsq_weight.data();
    d.gamma[0] = typename P::param(gas.gamma);
    d.gas_constant[0] = typename P::param(gas.gas_constant);
    d.mach_ref[0] = typename P::param(scheme.mach_ref);
    const auto& r = scheme.reference;
    if (!(r[0] > 0.0)) throw std::invalid_argument("reference density must be positive");
    for (std::size_t c = 0; c < 5; ++c) d.u_ref[c] = typename P::param(r[c]);
    d.p_ref[0] = typename P::param((gas.gamma - 1.0) *
                                   (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]));
    // Venkatakrishnan (1995), eq. 5, threshold eps^2 = (K h)^3 made dimensional:
    // eps_k^2 = (K / L)^3 V q_k^2.
    const double k3 = std::pow(scheme.venkat_k / scheme.length, 3);
    for (std::size_t c = 0; c < 5; ++c) {
      d.limiter_scale[c] = typename P::param(k3 * scheme.scale[c] * scheme.scale[c]);
    }
  }

  void faces() {
    switch (scheme.flux) {
      case Flux::roe:
        ek::cpu::roe_faces(d);
        ek::cpu::roe_boundaries(d);
        break;
      case Flux::hllc:
        ek::cpu::hllc_faces(d);
        ek::cpu::hllc_boundaries(d);
        break;
      case Flux::ausm_up:
        ek::cpu::ausm_faces(d);
        ek::cpu::ausm_boundaries(d);
        break;
    }
  }

  void faces_tangent() {
    switch (scheme.flux) {
      case Flux::roe:
        ek::cpu::roe_faces_tangent(d);
        ek::cpu::roe_boundaries_tangent(d);
        break;
      case Flux::hllc:
        ek::cpu::hllc_faces_tangent(d);
        ek::cpu::hllc_boundaries_tangent(d);
        break;
      case Flux::ausm_up:
        ek::cpu::ausm_faces_tangent(d);
        ek::cpu::ausm_boundaries_tangent(d);
        break;
    }
  }
};

template <class P>
Euler<P>::Euler(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme)
    : impl_(std::make_unique<Impl>(mesh, geometry, gas, scheme)) {}

template <class P>
Euler<P>::~Euler() = default;

template <class P>
index_t Euler<P>::n_cells() const noexcept {
  return impl_->topo.n_cells;
}

template <class P>
index_t Euler<P>::n_boundary_faces() const noexcept {
  return impl_->topo.n_faces - impl_->topo.n_interior_faces;
}

template <class P>
void Euler<P>::set_boundary(std::span<const BoundaryKind> kind, std::span<const double> data) {
  auto& m = *impl_;
  const auto nb = static_cast<std::size_t>(n_boundary_faces());
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  if (kind.size() != nb || data.size() != 5 * nb) {
    throw std::invalid_argument("boundary arrays do not match the boundary faces");
  }
  for (std::size_t b = 0; b < nb; ++b) {
    m.bc_kind[ni + b] = static_cast<double>(kind[b]);
    for (std::size_t c = 0; c < 5; ++c) {
      m.bc[c * nf + ni + b] = State(data[c * nb + b]);
    }
  }
}

template <class P>
void Euler<P>::set_source(std::span<const double> source) {
  auto& m = *impl_;
  if (source.size() != m.source.size()) {
    throw std::invalid_argument("source array does not match the cells");
  }
  for (std::size_t i = 0; i < source.size(); ++i) {
    m.source[i] = State(source[i]);
  }
}

template <class P>
void Euler<P>::state(std::span<const double> u, std::span<State> s) const {
  const auto n = static_cast<std::size_t>(n_cells());
  const auto& r = impl_->scheme.reference;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < n; ++c) s[k * n + c] = State(u[k * n + c] - r[k]);
  }
}

template <class P>
void Euler<P>::residual(std::span<const State> u, std::span<Res> res) {
  auto& m = *impl_;
  m.d.u = const_cast<State*>(u.data());
  m.d.res = res.data();
  ek::cpu::primitives(m.d);
  ek::cpu::boundary_states(m.d);
  if (m.scheme.second_order) {
    ek::cpu::gradients(m.d);
    if (m.scheme.limiter) {
      ek::cpu::limiter(m.d);
    }
  }
  ek::cpu::sources(m.d);
  m.faces();
}

template <class P>
void Euler<P>::first_order_tangent(std::span<const State> u, std::span<const State> du,
                                   std::span<Res> dres) {
  auto& m = *impl_;
  auto d = m.d;
  d.u = const_cast<State*>(u.data());
  d.grad = m.zero_grad.data();
  d.phi = m.zero_state.data();
  ek::cpu::primitives(d);
  d.u_tan = const_cast<State*>(du.data());
  d.prim_tan = m.prim_tan.data();
  d.grad_tan = m.zero_grad.data();
  d.phi_tan = m.zero_state.data();
  d.source_tan = m.zero_state.data();
  d.res_tan = dres.data();
  ek::cpu::primitives_tangent(d);
  ek::cpu::sources_tangent(d);
  // The first-order fluxes use the primitives only: grad and phi are zero above.
  auto saved = m.d;
  m.d = d;
  m.faces_tangent();
  m.d = saved;
}

template <class P>
void Euler<P>::spectral_radius(std::span<State> lambda) {
  auto& m = *impl_;
  m.d.lam = lambda.data();
  ek::cpu::spectral_radius(m.d);
}

template <class P>
void Euler<P>::boundary_pressure(std::span<double> p) const {
  const auto& m = *impl_;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(m.topo.owner[f]);
    double dp = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
      // grad is 5 x 3 per cell, component (4, j) at (4 * 3 + j) * n_cells + c.
      dp += static_cast<double>(m.grad[(12 + j) * nc + c]) *
            (static_cast<double>(m.face_centroid[j * nf + f]) -
             static_cast<double>(m.cell_centroid[j * nc + c]));
    }
    p[f - ni] = static_cast<double>(m.d.p_ref[0]) + static_cast<double>(m.prim[4 * nc + c]) +
                static_cast<double>(m.phi[4 * nc + c]) * dp;
  }
}

template class Euler<kernels::fp64>;
template class Euler<kernels::fp32_acc64>;

}  // namespace autan::physics
