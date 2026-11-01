#include "autan/physics/navier_stokes.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "navier_stokes_cpu.hpp"

namespace autan::physics {

namespace nk = autan::kernels::navier_stokes;

template <class P>
struct NavierStokes<P>::Impl {
  using G = typename P::geom;

  Euler<P> euler;
  kernels::Topology topo{};
  std::vector<G> face_normal, face_area_magnitude, face_centroid, face_moment, cell_centroid,
      cell_volume;
  std::vector<State> mu_t, mu_t_b, lam_v, traction;
  std::vector<typename P::jac> visc_jac, visc_bnd_jac;
  nk::Data<P> d{};
  const typename P::jac* dwdu = nullptr;

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme, Transport t)
      : euler(mesh, geometry, gas, scheme) {
    if (!(t.mu_ref > 0.0) || !(t.prandtl > 0.0) || !(t.prandtl_turbulent > 0.0) ||
        (t.sutherland && (!(t.t_ref > 0.0) || !(t.s >= 0.0)))) {
      throw std::invalid_argument("transport properties must be positive");
    }
    const auto packed = kernels::pack_geometry(mesh, geometry);
    topo = kernels::topology(mesh.view());
    face_normal = kernels::converted<G>(packed.face_normal);
    face_area_magnitude = kernels::converted<G>(packed.face_area_magnitude);
    face_centroid = kernels::converted<G>(packed.face_centroid);
    face_moment = kernels::converted<G>(packed.face_moment);
    cell_centroid = kernels::converted<G>(packed.cell_centroid);
    cell_volume = kernels::converted<G>(packed.cell_volume);
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    mu_t.assign(nc, State(0));
    mu_t_b.assign(nf, State(0));
    lam_v.assign(nc, State(0));
    traction.assign(5 * nf, State(0));
    visc_jac.assign(50 * nf, typename P::jac(0));
    visc_bnd_jac.assign(25 * nf, typename P::jac(0));

    d.topo = topo;
    const auto a = euler.arrays();
    d.prim = a.prim;
    d.grad = a.grad;
    d.prim_b = a.prim_b;
    d.bc_kind = a.bc_kind;
    d.bc = a.bc;
    dwdu = a.dwdu;
    d.mu_t = mu_t.data();
    d.mu_t_b = mu_t_b.data();
    d.lam_v = lam_v.data();
    d.traction = traction.data();
    d.visc_jac = visc_jac.data();
    d.visc_bnd_jac = visc_bnd_jac.data();
    d.face_normal = face_normal.data();
    d.face_area_magnitude = face_area_magnitude.data();
    d.face_centroid = face_centroid.data();
    d.face_moment = face_moment.data();
    d.cell_centroid = cell_centroid.data();
    d.cell_volume = cell_volume.data();
    using Pa = typename P::param;
    d.gamma[0] = Pa(gas.gamma);
    d.gas_constant[0] = Pa(gas.gas_constant);
    const auto& r = scheme.reference;
    d.p_ref[0] = Pa((gas.gamma - 1.0) *
                    (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]));
    d.mu_ref[0] = Pa(t.mu_ref);
    d.t_ref[0] = Pa(t.sutherland ? t.t_ref : 1.0);
    d.sutherland_s[0] = Pa(t.sutherland ? t.s : 0.0);
    d.sutherland[0] = Pa(t.sutherland ? 1.0 : 0.0);
    d.prandtl[0] = Pa(t.prandtl);
    d.prandtl_t[0] = Pa(t.prandtl_turbulent);
  }
};

template <class P>
NavierStokes<P>::NavierStokes(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
                              Transport transport)
    : impl_(std::make_unique<Impl>(mesh, geometry, gas, scheme, transport)) {}

template <class P>
NavierStokes<P>::~NavierStokes() = default;

template <class P>
index_t NavierStokes<P>::n_cells() const noexcept {
  return impl_->euler.n_cells();
}

template <class P>
index_t NavierStokes<P>::n_boundary_faces() const noexcept {
  return impl_->euler.n_boundary_faces();
}

template <class P>
void NavierStokes<P>::set_boundary(std::span<const BoundaryKind> kind,
                                   std::span<const double> data) {
  impl_->euler.set_boundary(kind, data);
}

template <class P>
void NavierStokes<P>::set_second_order(bool on) {
  impl_->euler.set_second_order(on);
}

template <class P>
void NavierStokes<P>::set_source(std::span<const double> source) {
  impl_->euler.set_source(source);
}

template <class P>
void NavierStokes<P>::set_eddy_viscosity(std::span<const double> cells,
                                         std::span<const double> boundary_faces) {
  auto& m = *impl_;
  const auto nb = static_cast<std::size_t>(n_boundary_faces());
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  if (cells.size() != m.mu_t.size() || boundary_faces.size() != nb) {
    throw std::invalid_argument("eddy viscosity arrays do not match the mesh");
  }
  for (std::size_t c = 0; c < cells.size(); ++c) m.mu_t[c] = State(cells[c]);
  for (std::size_t b = 0; b < nb; ++b) m.mu_t_b[ni + b] = State(boundary_faces[b]);
}

template <class P>
void NavierStokes<P>::state(std::span<const double> u, std::span<State> s) const {
  impl_->euler.state(u, s);
}

template <class P>
void NavierStokes<P>::residual(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.euler.residual(s, res);
  m.d.res = res.data();
  nk::cpu::viscous_faces(m.d);
  nk::cpu::viscous_boundaries(m.d);
}

template <class P>
void NavierStokes<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.euler.residual_and_jacobian(s, res);
  m.d.res = res.data();
  nk::cpu::viscous_faces_jacobian(m.d);
  nk::cpu::viscous_boundaries_jacobian(m.d);
}

template <class P>
void NavierStokes<P>::assemble(std::span<const double> shift, BlockMatrix& a) const {
  const auto& m = *impl_;
  m.euler.assemble(shift, a);
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  const auto* dwdu = m.dwdu;
  // Entry (i, j) of block k of entity e of an n-entity Jac dataset: (k * 25 + i * 5 + j) * n + e.
  auto add = [](double* out, const typename P::jac* j, std::size_t nj, std::size_t ej,
                std::size_t kj, const typename P::jac* dm, std::size_t nd, std::size_t ed,
                double sign) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t c = 0; c < 5; ++c) {
        double t = 0.0;
        for (std::size_t k = 0; k < 5; ++k) {
          t += static_cast<double>(j[(25 * kj + 5 * r + k) * nj + ej]) *
               static_cast<double>(dm[(5 * k + c) * nd + ed]);
        }
        out[5 * r + c] += sign * t;
      }
    }
  };
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(m.topo.owner[f]);
    const auto n = static_cast<std::size_t>(m.topo.neighbour[f]);
    add(a.diagonal(o), m.visc_jac.data(), nf, f, 0, dwdu, nc, o, 1.0);
    add(a.upper(f), m.visc_jac.data(), nf, f, 1, dwdu, nc, n, 1.0);
    add(a.lower(f), m.visc_jac.data(), nf, f, 0, dwdu, nc, o, -1.0);
    add(a.diagonal(n), m.visc_jac.data(), nf, f, 1, dwdu, nc, n, -1.0);
  }
  for (std::size_t f = ni; f < nf; ++f) {
    const auto o = static_cast<std::size_t>(m.topo.owner[f]);
    add(a.diagonal(o), m.visc_bnd_jac.data(), nf, f, 0, dwdu, nc, o, 1.0);
  }
}

template <class P>
void NavierStokes<P>::spectral_radius(std::span<State> lambda) {
  auto& m = *impl_;
  m.euler.spectral_radius(lambda);
  nk::cpu::viscous_spectral_radius(m.d);
  // Blazek (2015), eq. 6.20, C = 4 for cell-centred schemes.
  for (std::size_t c = 0; c < lambda.size(); ++c) lambda[c] += State(4) * m.lam_v[c];
}

template <class P>
void NavierStokes<P>::boundary_pressure(std::span<double> p, std::span<double> force) const {
  impl_->euler.boundary_pressure(p, force);
}

template <class P>
void NavierStokes<P>::viscous_traction(std::span<double> t) const {
  auto& m = *impl_;
  nk::cpu::viscous_tractions(m.d);
  const auto nb = static_cast<std::size_t>(n_boundary_faces());
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t b = 0; b < nb; ++b) {
      t[k * nb + b] = static_cast<double>(m.traction[(1 + k) * nf + ni + b]);
    }
  }
}

template class NavierStokes<kernels::fp64>;
template class NavierStokes<kernels::fp32_acc64>;

}  // namespace autan::physics
