#include "autan/physics/spalart_allmaras.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "spalart_allmaras_cpu.hpp"

namespace autan::physics {

namespace sk = autan::kernels::spalart_allmaras;

template <class P>
struct SpalartAllmaras<P>::Impl {
  using J = typename P::jac;

  NavierStokes<P>& flow;
  std::size_t nc, nf, ni;
  template <class Ptr>
  using stored = std::vector<std::remove_cvref_t<std::remove_pointer_t<Ptr>>>;
  using D = sk::Data<P>;
  stored<decltype(D::face_normal)> face_normal;
  stored<decltype(D::face_area_magnitude)> face_area_magnitude;
  stored<decltype(D::face_centroid)> face_centroid;
  stored<decltype(D::cell_centroid)> cell_centroid;
  stored<decltype(D::cell_volume)> cell_volume;
  stored<decltype(D::wall_distance)> wall_distance;
  std::vector<State> nt, nt_b, nt_bc, source;
  std::vector<typename P::grad> nt_grad;
  std::vector<J> cell_jac, face_jac, bnd_jac;
  std::vector<index_t> owner, neighbour;
  std::vector<offset_t> cell_offsets;  // cell -> interior faces CSR
  std::vector<index_t> cell_faces;
  sk::Data<P> d{};

  Impl(const Mesh& mesh, const Geometry& geometry, NavierStokes<P>& f, const Transport& t, bool ft2)
      : flow(f),
        nc(static_cast<std::size_t>(mesh.n_cells())),
        nf(static_cast<std::size_t>(mesh.n_faces())),
        ni(static_cast<std::size_t>(mesh.n_interior_faces())) {
    const auto packed = kernels::pack_geometry(mesh, geometry);
    face_normal =
        kernels::converted<typename decltype(face_normal)::value_type>(packed.face_normal);
    face_area_magnitude = kernels::converted<typename decltype(face_area_magnitude)::value_type>(
        packed.face_area_magnitude);
    face_centroid =
        kernels::converted<typename decltype(face_centroid)::value_type>(packed.face_centroid);
    cell_centroid =
        kernels::converted<typename decltype(cell_centroid)::value_type>(packed.cell_centroid);
    cell_volume =
        kernels::converted<typename decltype(cell_volume)::value_type>(packed.cell_volume);
    wall_distance.assign(nc, typename decltype(wall_distance)::value_type(1));
    nt.assign(nc, State(0));
    nt_b.assign(nf, State(0));
    nt_bc.assign(nf, State(0));
    source.assign(nc, State(0));
    nt_grad.assign(3 * nc, typename P::grad(0));
    cell_jac.assign(nc, J(0));
    face_jac.assign(2 * nf, J(0));
    bnd_jac.assign(nf, J(0));
    owner.assign(mesh.owner().begin(), mesh.owner().end());
    neighbour.assign(mesh.neighbour().begin(), mesh.neighbour().end());
    cell_offsets.assign(nc + 1, 0);
    for (std::size_t g = 0; g < ni; ++g) {
      ++cell_offsets[static_cast<std::size_t>(owner[g]) + 1];
      ++cell_offsets[static_cast<std::size_t>(neighbour[g]) + 1];
    }
    for (std::size_t c = 0; c < nc; ++c) cell_offsets[c + 1] += cell_offsets[c];
    cell_faces.assign(static_cast<std::size_t>(cell_offsets[nc]), 0);
    auto fill = cell_offsets;
    for (std::size_t g = 0; g < ni; ++g) {
      cell_faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(owner[g])]++)] =
          static_cast<index_t>(g);
      cell_faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(neighbour[g])]++)] =
          static_cast<index_t>(g);
    }

    d.nt = nt.data();
    d.nt_b = nt_b.data();
    d.nt_bc = nt_bc.data();
    d.nt_source = source.data();
    d.nt_grad = nt_grad.data();
    d.wall_distance = wall_distance.data();
    d.nt_cell_jac = cell_jac.data();
    d.nt_face_jac = face_jac.data();
    d.nt_bnd_jac = bnd_jac.data();
    d.face_normal = face_normal.data();
    d.face_area_magnitude = face_area_magnitude.data();
    d.face_centroid = face_centroid.data();
    d.cell_centroid = cell_centroid.data();
    d.cell_volume = cell_volume.data();
    using Pa = typename P::param;
    d.mu_ref[0] = Pa(t.mu_ref);
    d.t_ref[0] = Pa(t.sutherland ? t.t_ref : 1.0);
    d.sutherland_s[0] = Pa(t.sutherland ? t.s : 0.0);
    d.sutherland[0] = Pa(t.sutherland ? 1.0 : 0.0);
    d.ft2_on[0] = Pa(ft2 ? 1.0 : 0.0);
    share();
  }

  // The flow's arrays of its last evaluation (reallocated by Euler<P> on first use).
  void share() {
    const auto a = flow.arrays();
    d.topo = a.flow.topo;
    d.prim = a.flow.prim;
    d.grad = a.flow.grad;
    d.prim_b = a.flow.prim_b;
    d.bc_kind = a.flow.bc_kind;
    d.stencil_lsq_weight = a.flow.stencil_lsq_weight;
    d.mu_t = a.mu_t;
    d.mu_t_b = a.mu_t_b;
    d.p_ref[0] = typename P::param(a.p_ref);
    d.gas_constant[0] = typename P::param(a.gas_constant);
  }

  void prepare(std::span<const State> s, std::span<Res> res) {
    share();
    d.rnu = const_cast<State*>(s.data());
    d.nt_res = res.data();
    sk::cpu::nt_values(d);
    sk::cpu::boundary_values(d);
    sk::cpu::nt_gradients(d);
  }
};

template <class P>
SpalartAllmaras<P>::SpalartAllmaras(const Mesh& mesh, const Geometry& geometry,
                                    NavierStokes<P>& flow, const Transport& transport, bool ft2)
    : impl_(std::make_unique<Impl>(mesh, geometry, flow, transport, ft2)) {}

template <class P>
SpalartAllmaras<P>::~SpalartAllmaras() = default;

template <class P>
void SpalartAllmaras<P>::set_boundary(std::span<const double> nt) {
  auto& m = *impl_;
  if (nt.size() != m.nf - m.ni) throw std::invalid_argument("nt array does not match the faces");
  for (std::size_t b = 0; b < nt.size(); ++b) m.nt_bc[m.ni + b] = State(nt[b]);
}

template <class P>
void SpalartAllmaras<P>::set_wall_distance(std::span<const double> d) {
  auto& m = *impl_;
  if (d.size() != m.nc) throw std::invalid_argument("wall distance does not match the cells");
  for (std::size_t c = 0; c < m.nc; ++c) {
    if (!(d[c] > 0.0)) throw std::invalid_argument("wall distance must be positive");
    m.wall_distance[c] = static_cast<typename decltype(m.wall_distance)::value_type>(d[c]);
  }
}

template <class P>
void SpalartAllmaras<P>::set_source(std::span<const double> source) {
  auto& m = *impl_;
  if (source.size() != m.nc) throw std::invalid_argument("source does not match the cells");
  for (std::size_t c = 0; c < m.nc; ++c) m.source[c] = State(source[c]);
}

template <class P>
void SpalartAllmaras<P>::residual(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.prepare(s, res);
  sk::cpu::nt_sources(m.d);
  sk::cpu::nt_faces(m.d);
  sk::cpu::nt_boundaries(m.d);
}

template <class P>
void SpalartAllmaras<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.prepare(s, res);
  sk::cpu::nt_sources_jacobian(m.d);
  sk::cpu::nt_faces_jacobian(m.d);
  sk::cpu::nt_boundaries_jacobian(m.d);
}

template <class P>
void SpalartAllmaras<P>::assemble(std::span<const double> shift, std::span<double> diag,
                                  std::span<double> upper, std::span<double> lower) const {
  const auto& m = *impl_;
  for (std::size_t c = 0; c < m.nc; ++c) {
    diag[c] = shift[c] + std::max(0.0, static_cast<double>(m.cell_jac[c]));
  }
  // R_o += F, R_n -= F; F depends on rho nt of the owner (block 0) and the neighbour (block 1).
  for (std::size_t f = 0; f < m.ni; ++f) {
    const auto o = static_cast<std::size_t>(m.owner[f]);
    const auto n = static_cast<std::size_t>(m.neighbour[f]);
    const double jl = static_cast<double>(m.face_jac[f]);
    const double jr = static_cast<double>(m.face_jac[m.nf + f]);
    diag[o] += jl;
    upper[f] = jr;
    lower[f] = -jl;
    diag[n] -= jr;
  }
  for (std::size_t f = m.ni; f < m.nf; ++f) {
    diag[static_cast<std::size_t>(m.owner[f])] += static_cast<double>(m.bnd_jac[f]);
  }
}

template <class P>
void SpalartAllmaras<P>::solve(std::span<const double> diag, std::span<const double> upper,
                               std::span<const double> lower, std::span<const double> b,
                               std::span<double> dx, int sweeps) const {
  const auto& m = *impl_;
  auto relax = [&](std::size_t c) {
    double r = b[c];
    for (auto k = m.cell_offsets[c]; k < m.cell_offsets[c + 1]; ++k) {
      const auto f = static_cast<std::size_t>(m.cell_faces[static_cast<std::size_t>(k)]);
      const auto o = static_cast<std::size_t>(m.owner[f]);
      const auto n = static_cast<std::size_t>(m.neighbour[f]);
      r -= o == c ? upper[f] * dx[n] : lower[f] * dx[o];
    }
    dx[c] = r / diag[c];
  };
  for (int s = 0; s < sweeps; ++s) {
    for (std::size_t c = 0; c < m.nc; ++c) relax(c);
    for (std::size_t c = m.nc; c-- > 0;) relax(c);
  }
}

template <class P>
void SpalartAllmaras<P>::eddy_viscosity(std::span<const State> s) {
  auto& m = *impl_;
  m.share();
  m.d.rnu = const_cast<State*>(s.data());
  sk::cpu::nt_values(m.d);
  sk::cpu::boundary_values(m.d);
  sk::cpu::eddy_viscosities(m.d);
  sk::cpu::boundary_eddy_viscosities(m.d);
}

template class SpalartAllmaras<kernels::fp64>;
template class SpalartAllmaras<kernels::fp32_acc64>;

}  // namespace autan::physics
