#include "autan/physics/navier_stokes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "navier_stokes_cpu.hpp"

namespace autan::physics {

namespace nk = autan::kernels::navier_stokes;

template <class P>
struct NavierStokes<P>::Impl {
  Euler<P> euler;
  kernels::Topology topo{};
  // Each geometry array has the element type its Data<P> member stores (P::geom_value for
  // geometry the bodies read only as values).
  template <class Ptr>
  using stored = std::vector<std::remove_cvref_t<std::remove_pointer_t<Ptr>>>;
  using D = nk::Data<P>;
  stored<decltype(D::face_normal)> face_normal;
  stored<decltype(D::face_area_magnitude)> face_area_magnitude;
  stored<decltype(D::face_centroid)> face_centroid;
  stored<decltype(D::face_moment)> face_moment;
  stored<decltype(D::cell_centroid)> cell_centroid;
  stored<decltype(D::cell_volume)> cell_volume;
  std::vector<State> mu_t, mu_t_b, lam_v, traction, ut, ut_b;
  // The boundary fit's weights: 19 per stencil entry in seven 3-vectors (kernels/navier_stokes.py).
  std::array<stored<decltype(D::q0)>, 7> q;
  stored<decltype(D::quad)> quad;
  // The boundary fit's stencil: the two rings of cells around each boundary cell (cells sharing a
  // node with it or with one of those) and their boundary faces; empty for the other cells.
  std::vector<offset_t> fit_offsets;
  std::vector<index_t> fit_entries;
  kernels::CellStencil vertex;
  std::array<std::vector<typename P::grad>, 7> fits;
  const Geometry& geo;  // FP64 geometry of the boundary fit
  std::vector<typename P::jac> visc_jac, visc_bnd_jac;
  nk::Data<P> d{};
  const typename P::jac* dwdu = nullptr;

  // The Euler arrays this residual reads; Euler<P> allocates its Jacobian blocks on first use.
  void share() {
    const auto a = euler.arrays();
    d.prim = a.prim;
    d.grad = a.grad;
    d.prim_b = a.prim_b;
    d.bc_kind = a.bc_kind;
    d.bc = a.bc;
    dwdu = a.dwdu;
    d.topo = a.topo;
    d.topo.n_stencil = static_cast<offset_t>(fit_entries.size());
    d.topo.stencil_offsets = fit_offsets.data();
    d.topo.stencil = fit_entries.data();
    d.topo.stencil_twin = nullptr;  // the fit has no adjoint
  }

  // The fit's stencil: the two rings of cells around each cell owning a face whose condition gives
  // T and the velocity (cells sharing a node with it or with one of those) and their boundary
  // faces; empty for the other cells, which the fit's loop then skips.
  void fit_stencil(std::span<const BoundaryKind> kind) {
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    std::vector<char> fitted(nc, 0);
    for (std::size_t f = ni; f < nf; ++f) {
      if (kind[f - ni] == BoundaryKind::state || kind[f - ni] == BoundaryKind::wall) {
        fitted[static_cast<std::size_t>(topo.owner[f])] = 1;
      }
    }
    // Their face neighbours too: their face2 least-squares stencil reaches those boundary faces
    // on one side only, which leaves the faces between them first order.
    for (std::size_t f = 0; f < ni; ++f) {
      const auto own = static_cast<std::size_t>(topo.owner[f]);
      const auto nbr = static_cast<std::size_t>(topo.neighbour[f]);
      if (fitted[own] == 1 && fitted[nbr] == 0) fitted[nbr] = 2;
      if (fitted[nbr] == 1 && fitted[own] == 0) fitted[own] = 2;
    }
    auto ring = [&](std::size_t c) {
      return std::span<const index_t>(
          vertex.entries.data() + vertex.offsets[c],
          static_cast<std::size_t>(vertex.offsets[c + 1] - vertex.offsets[c]));
    };
    fit_offsets.assign(nc + 1, 0);
    fit_entries.clear();
    std::vector<index_t> entries;
    for (std::size_t c = 0; c < nc; ++c) {
      if (fitted[c] != 0) {
        entries.assign(ring(c).begin(), ring(c).end());
        for (const auto e : ring(c)) {
          if (e >= 0) {
            const auto second = ring(static_cast<std::size_t>(e));
            entries.insert(entries.end(), second.begin(), second.end());
          }
        }
        std::sort(entries.begin(), entries.end());
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
        for (const auto e : entries) {
          if (e != static_cast<index_t>(c)) fit_entries.push_back(e);
        }
      }
      fit_offsets[c + 1] = static_cast<offset_t>(fit_entries.size());
    }
    d.topo.n_stencil = static_cast<offset_t>(fit_entries.size());
    d.topo.stencil_offsets = fit_offsets.data();
    d.topo.stencil = fit_entries.data();
  }

  // Cubic least-squares fit over the two-ring stencil of every cell with a boundary face (docs/
  // design/ns-sa.md, Boundary conditions): theta (19 values: gradient, Hessian, third derivatives)
  // minimises sum_k w_k (phi_k - phi_c - b_k . theta)^2, w_k = 1/|d_k|^2, b_k the monomials of
  // kernels/navier_stokes.py at d_k, over the stencil's cells and its boundary faces whose
  // condition gives T and the velocity there (manufactured, walls). Monomials in a direction where
  // no entry extends (the span of a one-cell-thick 2D mesh) are left out, their derivatives zero. A
  // cell whose entries do not determine the rest is refused. Stores M^-1 w_k b_k per entry.
  void fit_weights(std::span<const BoundaryKind> kind) {
    constexpr std::size_t nt = 19;
    fit_stencil(kind);
    const auto& t = d.topo;
    const auto nc = static_cast<std::size_t>(t.n_cells);
    const auto ni = static_cast<std::size_t>(t.n_interior_faces);
    const auto nst = static_cast<std::size_t>(t.n_stencil);
    for (auto& v : q) v.assign(3 * nst, {});
    std::fill(quad.begin(), quad.end(), typename decltype(quad)::value_type(0));
    const auto& g = geo;
    std::vector<std::array<double, nt>> rows;
    std::vector<double> w;
    for (std::size_t c = 0; c < nc; ++c) {
      const auto begin = static_cast<std::size_t>(t.stencil_offsets[c]);
      const auto end = static_cast<std::size_t>(t.stencil_offsets[c + 1]);
      if (begin == end) continue;
      rows.assign(end - begin, {});
      w.assign(end - begin, 0.0);
      double scale = 0.0;
      for (std::size_t k = begin; k < end; ++k) {
        const index_t e = t.stencil[k];
        std::array<double, 3> x{};
        if (e >= 0) {
          const auto i = static_cast<std::size_t>(e);
          x = {g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]};
        } else {
          const auto f = static_cast<std::size_t>(-e - 1);
          if (kind[f - ni] != BoundaryKind::state && kind[f - ni] != BoundaryKind::wall) continue;
          x = {g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]};
        }
        const double dx = x[0] - g.cell_centroid_x[c];
        const double dy = x[1] - g.cell_centroid_y[c];
        const double dz = x[2] - g.cell_centroid_z[c];
        const double d2 = dx * dx + dy * dy + dz * dz;
        scale = std::max(scale, std::sqrt(d2));
        w[k - begin] = 1.0 / d2;
        rows[k - begin] = {dx,
                           dy,
                           dz,
                           0.5 * dx * dx,
                           0.5 * dy * dy,
                           0.5 * dz * dz,
                           dx * dy,
                           dx * dz,
                           dy * dz,
                           dx * dx * dx / 6.0,
                           dy * dy * dy / 6.0,
                           dz * dz * dz / 6.0,
                           0.5 * dx * dx * dy,
                           0.5 * dx * dx * dz,
                           0.5 * dx * dy * dy,
                           0.5 * dy * dy * dz,
                           0.5 * dx * dz * dz,
                           0.5 * dy * dz * dz,
                           dx * dy * dz};
      }
      // Unknowns scaled by the stencil's size keep the normal matrix's entries of order one.
      const std::array<int, nt> degree{1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
      std::array<double, nt> unit{};
      for (std::size_t i = 0; i < nt; ++i) unit[i] = std::pow(scale, degree[i]);
      // Columns of the weighted, scaled basis that vanish (no extent in a direction) are dropped.
      std::array<double, nt> norm{};
      for (std::size_t k = 0; k < rows.size(); ++k) {
        for (std::size_t i = 0; i < nt; ++i) norm[i] += w[k] * std::pow(rows[k][i] / unit[i], 2);
      }
      const double largest = *std::max_element(norm.begin(), norm.end());
      std::vector<std::size_t> used;
      for (std::size_t i = 0; i < nt; ++i) {
        if (norm[i] > 1e-20 * largest) used.push_back(i);
      }
      const auto m = used.size();
      std::vector<double> a(m * m, 0.0), inv(m * m, 0.0);
      for (std::size_t k = 0; k < rows.size(); ++k) {
        if (w[k] == 0.0) continue;
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < m; ++j) {
            a[m * i + j] +=
                w[k] * (rows[k][used[i]] / unit[used[i]]) * (rows[k][used[j]] / unit[used[j]]);
          }
        }
      }
      // Gauss-Jordan with partial pivoting, the rank judged against the matrix's own scale.
      for (std::size_t i = 0; i < m; ++i) inv[(m + 1) * i] = 1.0;
      double diag = 0.0;
      for (std::size_t i = 0; i < m; ++i) diag = std::max(diag, std::abs(a[(m + 1) * i]));
      for (std::size_t col = 0; col < m; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < m; ++r) {
          if (std::abs(a[m * r + col]) > std::abs(a[m * piv + col])) piv = r;
        }
        if (!(std::abs(a[m * piv + col]) > 1e-12 * diag)) {
          throw std::invalid_argument(
              "the boundary fit of the viscous fluxes is undetermined at cell " +
              std::to_string(c) + " (" + std::to_string(end - begin) + " stencil entries)");
        }
        for (std::size_t j = 0; j < m; ++j) {
          std::swap(a[m * col + j], a[m * piv + j]);
          std::swap(inv[m * col + j], inv[m * piv + j]);
        }
        const double pv = a[m * col + col];
        for (std::size_t j = 0; j < m; ++j) {
          a[m * col + j] /= pv;
          inv[m * col + j] /= pv;
        }
        for (std::size_t r = 0; r < m; ++r) {
          if (r == col || a[m * r + col] == 0.0) continue;
          const double fct = a[m * r + col];
          for (std::size_t j = 0; j < m; ++j) {
            a[m * r + j] -= fct * a[m * col + j];
            inv[m * r + j] -= fct * inv[m * col + j];
          }
        }
      }
      quad[c] = typename decltype(quad)::value_type(1);
      for (std::size_t k = begin; k < end; ++k) {
        const double wk = w[k - begin];
        if (wk == 0.0) continue;
        std::array<double, 21> coef{};
        for (std::size_t i = 0; i < m; ++i) {
          double v = 0.0;
          for (std::size_t j = 0; j < m; ++j) {
            v += inv[m * i + j] * wk * rows[k - begin][used[j]] / unit[used[j]];
          }
          coef[used[i]] = v / unit[used[i]];
        }
        for (std::size_t b = 0; b < 7; ++b) {
          for (std::size_t j = 0; j < 3; ++j) {
            q[b][j * nst + k] =
                static_cast<typename decltype(q)::value_type::value_type>(coef[3 * b + j]);
          }
        }
      }
    }
    d.q0 = q[0].data();
    d.q1 = q[1].data();
    d.q2 = q[2].data();
    d.q3 = q[3].data();
    d.q4 = q[4].data();
    d.q5 = q[5].data();
    d.q6 = q[6].data();
  }

  void fit() {
    nk::cpu::temperatures_velocities(d);
    nk::cpu::boundary_temperatures_velocities(d);
    nk::cpu::cubic_fits(d);
  }

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme, Transport t)
      : euler(mesh, geometry, gas, scheme), geo(geometry) {
    if (!(t.mu_ref > 0.0) || !(t.prandtl > 0.0) || !(t.prandtl_turbulent > 0.0) ||
        (t.sutherland && (!(t.t_ref > 0.0) || !(t.s >= 0.0)))) {
      throw std::invalid_argument("transport properties must be positive");
    }
    const auto packed = kernels::pack_geometry(mesh, geometry);
    topo = kernels::topology(mesh.view());
    vertex = kernels::cell_stencil(mesh, packed, kernels::StencilKind::vertex);
    face_normal =
        kernels::converted<typename decltype(face_normal)::value_type>(packed.face_normal);
    face_area_magnitude = kernels::converted<typename decltype(face_area_magnitude)::value_type>(
        packed.face_area_magnitude);
    face_centroid =
        kernels::converted<typename decltype(face_centroid)::value_type>(packed.face_centroid);
    face_moment =
        kernels::converted<typename decltype(face_moment)::value_type>(packed.face_moment);
    cell_centroid =
        kernels::converted<typename decltype(cell_centroid)::value_type>(packed.cell_centroid);
    cell_volume =
        kernels::converted<typename decltype(cell_volume)::value_type>(packed.cell_volume);
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    mu_t.assign(nc, State(0));
    mu_t_b.assign(nf, State(0));
    lam_v.assign(nc, State(0));
    traction.assign(5 * nf, State(0));
    ut.assign(5 * nc, State(0));
    ut_b.assign(5 * nf, State(0));
    quad.assign(nc, {});
    for (auto& v : fits) v.assign(15 * nc, typename P::grad(0));
    visc_jac.assign(50 * nf, typename P::jac(0));
    visc_bnd_jac.assign(25 * nf, typename P::jac(0));

    d.topo = topo;
    share();
    fit_weights(std::vector<BoundaryKind>(nf - static_cast<std::size_t>(topo.n_interior_faces),
                                          BoundaryKind::state));
    d.mu_t = mu_t.data();
    d.mu_t_b = mu_t_b.data();
    d.lam_v = lam_v.data();
    d.traction = traction.data();
    d.ut = ut.data();
    d.ut_b = ut_b.data();
    d.quad = quad.data();
    d.fit0 = fits[0].data();
    d.fit1 = fits[1].data();
    d.fit2 = fits[2].data();
    d.fit3 = fits[3].data();
    d.fit4 = fits[4].data();
    d.fit5 = fits[5].data();
    d.fit6 = fits[6].data();
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
    d.p_ref[0] =
        Pa((gas.gamma - 1.0) * (r[4] - 0.5 * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / r[0]));
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
  impl_->share();
  impl_->fit_weights(kind);
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
typename NavierStokes<P>::Arrays NavierStokes<P>::arrays() noexcept {
  auto& m = *impl_;
  return {m.euler.arrays(), m.mu_t.data(), m.mu_t_b.data(), static_cast<double>(m.d.p_ref[0]),
          static_cast<double>(m.d.gas_constant[0])};
}

template <class P>
void NavierStokes<P>::state(std::span<const double> u, std::span<State> s) const {
  impl_->euler.state(u, s);
}

template <class P>
void NavierStokes<P>::residual(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.euler.residual(s, res);
  m.share();
  m.fit();
  m.d.res = res.data();
  nk::cpu::viscous_faces(m.d);
  nk::cpu::viscous_boundaries(m.d);
}

template <class P>
void NavierStokes<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  m.euler.residual_and_jacobian(s, res);
  m.share();
  m.fit();
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
                std::size_t kj, const typename P::jac* dm, std::size_t n_dm, std::size_t ed,
                double sign) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t c = 0; c < 5; ++c) {
        double t = 0.0;
        for (std::size_t k = 0; k < 5; ++k) {
          t += static_cast<double>(j[(25 * kj + 5 * r + k) * nj + ej]) *
               static_cast<double>(dm[(5 * k + c) * n_dm + ed]);
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
