#include "autan/physics/navier_stokes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
  std::vector<State> mu_t, mu_t_b, lam_v, traction, ut, ut_b;
  std::vector<G> q_grad, q_hdiag, q_hoff, quad;
  // The quadratic fit's stencil: every cell sharing a node with a boundary cell, and their
  // boundary faces (kernels::StencilKind::vertex); empty for the other cells.
  std::vector<offset_t> fit_offsets;
  std::vector<index_t> fit_entries;
  std::vector<typename P::grad> fit_g, fit_hd, fit_ho;
  const Geometry& geo;  // FP64 geometry of the quadratic fit
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

  // Quadratic least-squares fit over the stencil of every cell with a boundary face (docs/design/
  // ns-sa.md, Boundary conditions): theta = (grad, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz) minimises
  // sum_k w_k (phi_k - phi_c - b_k . theta)^2, b_k = (d, d_x^2/2, d_y^2/2, d_z^2/2, d_x d_y, d_x
  // d_z, d_y d_z), w_k = 1/|d_k|^2, over the stencil's cells and its boundary faces whose condition
  // gives T and the velocity there (manufactured, walls). Stores M^-1 w_k b_k per entry.
  void fit_weights(std::span<const BoundaryKind> kind) {
    const auto& t = d.topo;
    const auto nc = static_cast<std::size_t>(t.n_cells);
    const auto ni = static_cast<std::size_t>(t.n_interior_faces);
    const auto nst = static_cast<std::size_t>(t.n_stencil);
    q_grad.assign(3 * nst, G(0));
    q_hdiag.assign(3 * nst, G(0));
    q_hoff.assign(3 * nst, G(0));
    std::fill(quad.begin(), quad.end(), G(0));
    std::vector<char> boundary_cell(nc, 0);
    for (std::size_t f = ni; f < static_cast<std::size_t>(t.n_faces); ++f) {
      boundary_cell[static_cast<std::size_t>(t.owner[f])] = 1;
    }
    const auto& g = geo;
    for (std::size_t c = 0; c < nc; ++c) {
      if (boundary_cell[c] == 0) continue;
      const auto begin = static_cast<std::size_t>(t.stencil_offsets[c]);
      const auto end = static_cast<std::size_t>(t.stencil_offsets[c + 1]);
      std::vector<std::array<double, 9>> rows(end - begin);
      std::vector<double> w(end - begin, 0.0);
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
        const std::array<double, 3> dx{x[0] - g.cell_centroid_x[c], x[1] - g.cell_centroid_y[c],
                                       x[2] - g.cell_centroid_z[c]};
        const double d2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        scale = std::max(scale, std::sqrt(d2));
        w[k - begin] = 1.0 / d2;
        rows[k - begin] = {dx[0],
                           dx[1],
                           dx[2],
                           0.5 * dx[0] * dx[0],
                           0.5 * dx[1] * dx[1],
                           0.5 * dx[2] * dx[2],
                           dx[0] * dx[1],
                           dx[0] * dx[2],
                           dx[1] * dx[2]};
      }
      // Scaled unknowns (grad s, H s^2) keep the normal matrix's entries of order one.
      const std::array<double, 9> unit{scale,         scale,         scale,
                                       scale * scale, scale * scale, scale * scale,
                                       scale * scale, scale * scale, scale * scale};
      std::array<double, 81> a{};
      for (std::size_t k = 0; k < rows.size(); ++k) {
        if (w[k] == 0.0) continue;
        for (std::size_t i = 0; i < 9; ++i) {
          for (std::size_t j = 0; j < 9; ++j) {
            a[9 * i + j] += w[k] * (rows[k][i] / unit[i]) * (rows[k][j] / unit[j]);
          }
        }
      }
      // Inverse by Gauss-Jordan with partial pivoting; a cell whose entries do not determine a
      // quadratic keeps the linear gradient (quad = 0).
      std::array<double, 81> inv{};
      for (std::size_t i = 0; i < 9; ++i) inv[10 * i] = 1.0;
      double largest = 0.0;
      for (std::size_t i = 0; i < 9; ++i) largest = std::max(largest, std::abs(a[10 * i]));
      bool regular = largest > 0.0;
      for (std::size_t col = 0; col < 9 && regular; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < 9; ++r) {
          if (std::abs(a[9 * r + col]) > std::abs(a[9 * piv + col])) piv = r;
        }
        if (std::abs(a[9 * piv + col]) < 1e-10 * largest) {
          regular = false;
          break;
        }
        for (std::size_t j = 0; j < 9; ++j) {
          std::swap(a[9 * col + j], a[9 * piv + j]);
          std::swap(inv[9 * col + j], inv[9 * piv + j]);
        }
        const double p = a[9 * col + col];
        for (std::size_t j = 0; j < 9; ++j) {
          a[9 * col + j] /= p;
          inv[9 * col + j] /= p;
        }
        for (std::size_t r = 0; r < 9; ++r) {
          if (r == col) continue;
          const double f = a[9 * r + col];
          if (f == 0.0) continue;
          for (std::size_t j = 0; j < 9; ++j) {
            a[9 * r + j] -= f * a[9 * col + j];
            inv[9 * r + j] -= f * inv[9 * col + j];
          }
        }
      }
      if (!regular) continue;
      quad[c] = G(1);
      for (std::size_t k = begin; k < end; ++k) {
        const double wk = w[k - begin];
        if (wk == 0.0) continue;
        std::array<double, 9> coef{};
        for (std::size_t i = 0; i < 9; ++i) {
          for (std::size_t j = 0; j < 9; ++j) {
            coef[i] += inv[9 * i + j] * wk * rows[k - begin][j] / unit[j];
          }
          coef[i] /= unit[i];
        }
        for (std::size_t j = 0; j < 3; ++j) {
          q_grad[j * nst + k] = G(coef[j]);
          q_hdiag[j * nst + k] = G(coef[3 + j]);
          q_hoff[j * nst + k] = G(coef[6 + j]);
        }
      }
    }
    d.q_grad = q_grad.data();
    d.q_hdiag = q_hdiag.data();
    d.q_hoff = q_hoff.data();
  }

  void fit() {
    nk::cpu::temperatures_velocities(d);
    nk::cpu::boundary_temperatures_velocities(d);
    nk::cpu::quadratic_fits(d);
  }

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme, Transport t)
      : euler(mesh, geometry, gas, scheme), geo(geometry) {
    if (!(t.mu_ref > 0.0) || !(t.prandtl > 0.0) || !(t.prandtl_turbulent > 0.0) ||
        (t.sutherland && (!(t.t_ref > 0.0) || !(t.s >= 0.0)))) {
      throw std::invalid_argument("transport properties must be positive");
    }
    const auto packed = kernels::pack_geometry(mesh, geometry);
    topo = kernels::topology(mesh.view());
    {
      const auto vertex = kernels::cell_stencil(mesh, packed, kernels::StencilKind::vertex);
      const auto ncell = static_cast<std::size_t>(mesh.n_cells());
      std::vector<char> boundary_cell(ncell, 0);
      for (auto f = mesh.n_interior_faces(); f < mesh.n_faces(); ++f) {
        boundary_cell[static_cast<std::size_t>(mesh.owner()[static_cast<std::size_t>(f)])] = 1;
      }
      fit_offsets.assign(ncell + 1, 0);
      for (std::size_t c = 0; c < ncell; ++c) {
        const auto b = static_cast<std::size_t>(vertex.offsets[c]);
        const auto e = static_cast<std::size_t>(vertex.offsets[c + 1]);
        if (boundary_cell[c] != 0) {
          fit_entries.insert(fit_entries.end(),
                             vertex.entries.begin() + static_cast<std::ptrdiff_t>(b),
                             vertex.entries.begin() + static_cast<std::ptrdiff_t>(e));
        }
        fit_offsets[c + 1] = static_cast<offset_t>(fit_entries.size());
      }
    }
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
    ut.assign(5 * nc, State(0));
    ut_b.assign(5 * nf, State(0));
    quad.assign(nc, G(0));
    fit_g.assign(15 * nc, typename P::grad(0));
    fit_hd.assign(15 * nc, typename P::grad(0));
    fit_ho.assign(15 * nc, typename P::grad(0));
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
    d.fit_g = fit_g.data();
    d.fit_hd = fit_hd.data();
    d.fit_ho = fit_ho.data();
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
