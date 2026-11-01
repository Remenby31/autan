#include "autan/physics/navier_stokes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
  bool bounded = false;  // set_boundary called: the fit is built
  kernels::CellStencil vertex;
  std::array<std::vector<typename P::grad>, 7> fits;
  const Geometry& geo;  // FP64 geometry of the boundary fit
  std::vector<typename P::jac> visc_jac, visc_bnd_jac;
  nk::Data<P> d{};
  const typename P::jac* dwdu = nullptr;
  mutable std::pair<double, double> fit_norms{};
  // Per cell, its face neighbours and the faces between them (the block pattern's off-diagonals).
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> adjacent;

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

  // Cubic least-squares fit of every cell owning a face whose condition gives T and the velocity
  // (manufactured, walls) and of its face neighbours (docs/design/ns-sa.md, Boundary conditions):
  // theta (19 values: gradient, Hessian, third derivatives) minimises
  // sum_k w_k (phi_k - phi_c - b_k . theta)^2, b_k the monomials of kernels/navier_stokes.py at
  // d_k, over the cells within two rings of the cell (sharing a node with it or with one of those)
  // and their boundary faces with such a condition. Monomials in a direction where no entry extends
  // (the span of a one-cell-thick 2D mesh) are left out, their derivatives zero. A cell whose two
  // rings leave the rest undetermined (a corner where a wall meets an outflow) takes three; one
  // whose three rings do not either is refused. Stores M^-1 w_k b_k per entry; the cells without
  // a fit get an empty stencil, which the fit's loop skips.
  void fit_weights(std::span<const BoundaryKind> kind) {
    constexpr std::size_t nt = 19;
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    auto gives = [&](std::size_t f) {
      return kind[f - ni] == BoundaryKind::state || kind[f - ni] == BoundaryKind::wall;
    };
    std::vector<char> fitted(nc, 0);
    for (std::size_t f = ni; f < nf; ++f) {
      if (gives(f)) fitted[static_cast<std::size_t>(topo.owner[f])] = 1;
    }
    // Their face neighbours too: their face2 least-squares stencil reaches those boundary faces
    // on one side only, which leaves the faces between them first order.
    for (std::size_t f = 0; f < ni; ++f) {
      const auto own = static_cast<std::size_t>(topo.owner[f]);
      const auto nbr = static_cast<std::size_t>(topo.neighbour[f]);
      if (fitted[own] == 1 && fitted[nbr] == 0) fitted[nbr] = 2;
      if (fitted[nbr] == 1 && fitted[own] == 0) fitted[own] = 2;
    }
    auto ring = [&](index_t c) {
      const auto i = static_cast<std::size_t>(c);
      return std::span<const index_t>(
          vertex.entries.data() + vertex.offsets[i],
          static_cast<std::size_t>(vertex.offsets[i + 1] - vertex.offsets[i]));
    };
    // The cells within `rings` rings of c and their boundary faces with a condition giving T
    // and the velocity, c excluded.
    auto gather = [&](std::size_t c, int rings) {
      std::vector<index_t> cells{static_cast<index_t>(c)}, out;
      for (int r = 0; r < rings; ++r) {
        const auto reached = cells.size();
        for (std::size_t k = 0; k < reached; ++k) {
          for (const auto e : ring(cells[k])) {
            if (e >= 0)
              cells.push_back(e);
            else if (gives(static_cast<std::size_t>(-e - 1)))
              out.push_back(e);
          }
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
      }
      for (const auto e : cells) {
        if (e != static_cast<index_t>(c)) out.push_back(e);
      }
      std::sort(out.begin(), out.end());
      out.erase(std::unique(out.begin(), out.end()), out.end());
      return out;
    };
    constexpr std::array<std::array<int, 3>, nt> power{{{1, 0, 0},
                                                        {0, 1, 0},
                                                        {0, 0, 1},
                                                        {2, 0, 0},
                                                        {0, 2, 0},
                                                        {0, 0, 2},
                                                        {1, 1, 0},
                                                        {1, 0, 1},
                                                        {0, 1, 1},
                                                        {3, 0, 0},
                                                        {0, 3, 0},
                                                        {0, 0, 3},
                                                        {2, 1, 0},
                                                        {2, 0, 1},
                                                        {1, 2, 0},
                                                        {0, 2, 1},
                                                        {1, 0, 2},
                                                        {0, 1, 2},
                                                        {1, 1, 1}}};
    const auto& g = geo;
    // theta's weights per entry of c's stencil, or false if the entries leave it undetermined.
    auto solve = [&](std::size_t c, std::span<const index_t> entries,
                     std::vector<std::array<double, nt>>& coef) {
      std::vector<std::array<double, nt>> rows(entries.size());
      std::vector<double> w(entries.size());
      std::array<double, 3> extent{};
      for (std::size_t k = 0; k < entries.size(); ++k) {
        const index_t e = entries[k];
        std::array<double, 3> x{};
        if (e >= 0) {
          const auto i = static_cast<std::size_t>(e);
          x = {g.cell_centroid_x[i], g.cell_centroid_y[i], g.cell_centroid_z[i]};
        } else {
          const auto f = static_cast<std::size_t>(-e - 1);
          x = {g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]};
        }
        const double dx = x[0] - g.cell_centroid_x[c];
        const double dy = x[1] - g.cell_centroid_y[c];
        const double dz = x[2] - g.cell_centroid_z[c];
        extent = {std::max(extent[0], std::abs(dx)), std::max(extent[1], std::abs(dy)),
                  std::max(extent[2], std::abs(dz))};
        for (std::size_t i = 0; i < nt; ++i) {
          rows[k][i] = std::pow(dx, power[i][0]) * std::pow(dy, power[i][1]) *
                       std::pow(dz, power[i][2]) /
                       (std::tgamma(power[i][0] + 1.0) * std::tgamma(power[i][1] + 1.0) *
                        std::tgamma(power[i][2] + 1.0));
        }
      }
      // Each unknown scaled by the stencil's extent along its directions, and the weights 1/|d|^2
      // taken in the scaled coordinates, keep the normal matrix of order one and well conditioned
      // on stretched cells: in physical coordinates, the entries across a wall-resolved layer
      // (10^3:1 and more) would outweigh the others by the aspect ratio squared.
      const double widest = std::max({extent[0], extent[1], extent[2]});
      std::array<double, nt> unit{};
      std::vector<std::size_t> used;
      for (std::size_t i = 0; i < nt; ++i) {
        unit[i] = 1.0;
        bool spans = true;
        for (std::size_t a = 0; a < 3; ++a) {
          if (power[i][a] == 0) continue;
          spans = spans && extent[a] > 1e-12 * widest;
          unit[i] *= std::pow(extent[a], power[i][a]);
        }
        if (spans) used.push_back(i);
      }
      for (std::size_t k = 0; k < rows.size(); ++k) {
        double s2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a) {
          if (extent[a] > 1e-12 * widest) s2 += std::pow(rows[k][a] / extent[a], 2);
        }
        w[k] = 1.0 / s2;
      }
      const auto m = used.size();
      std::vector<double> a(m * m, 0.0), inv(m * m, 0.0);
      for (std::size_t k = 0; k < rows.size(); ++k) {
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
        if (!(std::abs(a[m * piv + col]) > 1e-12 * diag)) return false;
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
      coef.assign(entries.size(), {});
      for (std::size_t k = 0; k < entries.size(); ++k) {
        for (std::size_t i = 0; i < m; ++i) {
          double v = 0.0;
          for (std::size_t j = 0; j < m; ++j) {
            v += inv[m * i + j] * w[k] * rows[k][used[j]] / unit[used[j]];
          }
          coef[k][used[i]] = v / unit[used[i]];
        }
      }
      return true;
    };
    fit_offsets.assign(nc + 1, 0);
    fit_entries.clear();
    std::vector<std::array<double, nt>> coef, all;
    std::fill(quad.begin(), quad.end(), typename decltype(quad)::value_type(0));
    for (std::size_t c = 0; c < nc; ++c) {
      if (fitted[c] != 0) {
        auto entries = gather(c, 2);
        if (!solve(c, entries, coef)) {
          entries = gather(c, 3);
          if (!solve(c, entries, coef)) {
            throw std::invalid_argument(
                "the boundary fit of the viscous fluxes is undetermined at cell " +
                std::to_string(c) + " (" + std::to_string(entries.size()) +
                " entries within three rings)");
          }
        }
        fit_entries.insert(fit_entries.end(), entries.begin(), entries.end());
        all.insert(all.end(), coef.begin(), coef.end());
        quad[c] = typename decltype(quad)::value_type(1);
      }
      fit_offsets[c + 1] = static_cast<offset_t>(fit_entries.size());
    }
    d.topo.n_stencil = static_cast<offset_t>(fit_entries.size());
    d.topo.stencil_offsets = fit_offsets.data();
    d.topo.stencil = fit_entries.data();
    const auto nst = fit_entries.size();
    for (auto& v : q) v.assign(3 * nst, {});
    for (std::size_t k = 0; k < nst; ++k) {
      for (std::size_t b = 0; b < 7; ++b) {
        for (std::size_t j = 0; j < 3; ++j) {
          if (3 * b + j < nt) {
            q[b][j * nst + k] =
                static_cast<typename decltype(q)::value_type::value_type>(all[k][3 * b + j]);
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

  // Coefficient i (0-18) of the boundary fit at stencil entry k (kernels/navier_stokes.py order).
  [[nodiscard]] double coef(std::size_t k, std::size_t i) const {
    const auto nst = static_cast<std::size_t>(d.topo.n_stencil);
    return static_cast<double>(q[i / 3][(i % 3) * nst + k]);
  }

  // Gradient at r of the cubic whose coefficients are those of entry k, and its rise from 0 to s.
  void basis(std::size_t k, const std::array<double, 3>& r, const std::array<double, 3>& sg,
             std::array<double, 3>& grad, double& rise) const {
    std::array<double, 19> c{};
    for (std::size_t i = 0; i < 19; ++i) c[i] = coef(k, i);
    auto gradient = [&](const std::array<double, 3>& v) {
      const double x = v[0], y = v[1], z = v[2];
      return std::array<double, 3>{
          c[0] + c[3] * x + c[6] * y + c[7] * z + 0.5 * x * x * c[9] + x * y * c[12] +
              x * z * c[13] + 0.5 * y * y * c[14] + 0.5 * z * z * c[16] + y * z * c[18],
          c[1] + c[6] * x + c[4] * y + c[8] * z + 0.5 * y * y * c[10] + 0.5 * x * x * c[12] +
              x * y * c[14] + y * z * c[15] + 0.5 * z * z * c[17] + x * z * c[18],
          c[2] + c[7] * x + c[8] * y + c[5] * z + 0.5 * z * z * c[11] + 0.5 * x * x * c[13] +
              0.5 * y * y * c[15] + x * z * c[16] + y * z * c[17] + x * y * c[18]};
    };
    grad = gradient(r);
    const double x = sg[0], y = sg[1], z = sg[2];
    rise = c[0] * x + c[1] * y + c[2] * z + 0.5 * (c[3] * x * x + c[4] * y * y + c[5] * z * z) +
           c[6] * x * y + c[7] * x * z + c[8] * y * z +
           (c[9] * x * x * x + c[10] * y * y * y + c[11] * z * z * z) / 6.0 +
           0.5 * (c[12] * x * x * y + c[13] * x * x * z + c[14] * x * y * y) +
           0.5 * (c[15] * y * y * z + c[16] * x * z * z + c[17] * y * z * z) + c[18] * x * y * z;
  }

  // The viscous fluxes' dependence through the boundary fit, which the face loops' Jacobians hold
  // fixed: dF/dU_k = dF/dG dG/dphi_k dphi/dW_k dW/dU_k (phi = T, u, v, w; G their face gradients)
  // for every stencil entry k, added where the residual's cell and k are the same or face
  // neighbours (the block pattern); returns the Frobenius norms of the blocks added and of those
  // left out.
  std::pair<double, double> fit_jacobian(BlockMatrix& a) const {
    const auto& t = d.topo;
    const auto nc = static_cast<std::size_t>(t.n_cells);
    const auto nf = static_cast<std::size_t>(t.n_faces);
    const auto ni = static_cast<std::size_t>(t.n_interior_faces);
    const auto& g = geo;
    const double rgas = static_cast<double>(d.gas_constant[0]);
    const double pref = static_cast<double>(d.p_ref[0]);
    const double gm = static_cast<double>(d.gamma[0]);
    const double cp = gm * rgas / (gm - 1.0);
    const double mref = static_cast<double>(d.mu_ref[0]), tref = static_cast<double>(d.t_ref[0]);
    const double sth = static_cast<double>(d.sutherland_s[0]);
    const bool law = static_cast<double>(d.sutherland[0]) > 0.5;
    const double prl = static_cast<double>(d.prandtl[0]), prt = static_cast<double>(d.prandtl_t[0]);
    auto prim = [&](std::size_t c, std::size_t k) {
      return static_cast<double>(d.prim[k * nc + c]);
    };
    auto temp = [&](std::size_t c) { return (prim(c, 4) + pref) / (prim(c, 0) * rgas); };
    auto mu = [&](double tt) {
      return law ? mref * std::pow(tt / tref, 1.5) * (tref + sth) / (tt + sth) : mref;
    };
    auto centroid = [&](std::size_t c) {
      return std::array<double, 3>{g.cell_centroid_x[c], g.cell_centroid_y[c],
                                   g.cell_centroid_z[c]};
    };
    double placed = 0.0, outside = 0.0;
    // Block (row, k) of the pattern, null outside it.
    auto block = [&](std::size_t row, std::size_t k) -> double* {
      if (k == row) return a.diagonal(row);
      for (const auto& [nb, face] : adjacent[row]) {
        if (nb == k) {
          return static_cast<std::size_t>(t.owner[face]) == row ? a.upper(face) : a.lower(face);
        }
      }
      return nullptr;
    };
    // dF/dphi (5 x 4) -> block dF/dU_k (5 x 5) into `out` with `sign`.
    auto chain = [&](const std::array<std::array<double, 4>, 5>& dfdphi, std::size_t k, double* out,
                     double sign) {
      const double rho = prim(k, 0), tt = temp(k);
      std::array<std::array<double, 5>, 4> dphidw{};
      dphidw[0][0] = -tt / rho;
      dphidw[0][4] = 1.0 / (rho * rgas);
      for (std::size_t i = 1; i < 4; ++i) dphidw[i][i] = 1.0;
      double norm = 0.0;
      for (std::size_t r = 0; r < 5; ++r) {
        std::array<double, 5> dfdw{};
        for (std::size_t c = 0; c < 5; ++c) {
          for (std::size_t i = 0; i < 4; ++i) dfdw[c] += dfdphi[r][i] * dphidw[i][c];
        }
        for (std::size_t c = 0; c < 5; ++c) {
          double v = 0.0;
          for (std::size_t j = 0; j < 5; ++j) {
            v += dfdw[j] * static_cast<double>(dwdu[(5 * j + c) * nc + k]);
          }
          if (out != nullptr) out[5 * r + c] += sign * v;
          norm += v * v;
        }
      }
      return norm;
    };
    // One fitted side s of face f: F's derivatives through s's fit, weight w, rise along sg.
    auto side = [&](std::size_t f, std::size_t s, double w, const std::array<double, 3>& r,
                    const std::array<double, 3>& sg, double dist, double k_cond, double mue,
                    const std::array<double, 3>& uf, auto&& place) {
      const double area =
          std::sqrt(g.face_area_x[f] * g.face_area_x[f] + g.face_area_y[f] * g.face_area_y[f] +
                    g.face_area_z[f] * g.face_area_z[f]);
      const std::array<double, 3> n{g.face_area_x[f] / area, g.face_area_y[f] / area,
                                    g.face_area_z[f] / area};
      const std::array<double, 3> e{sg[0] / dist, sg[1] / dist, sg[2] / dist};
      // dF/dG: F = -A (0, tau n, u . tau n + k grad T . n), tau_ia n_a with
      // d(tau n)_i / dG_aj = mu (delta_ia n_j + delta_ij n_a - 2/3 delta_aj n_i).
      std::array<std::array<std::array<double, 3>, 4>, 5> dfdg{};
      for (std::size_t a2 = 0; a2 < 3; ++a2) {
        for (std::size_t j = 0; j < 3; ++j) {
          for (std::size_t i = 0; i < 3; ++i) {
            const double dt = mue * ((i == a2 ? n[j] : 0.0) + (i == j ? n[a2] : 0.0) -
                                     (a2 == j ? 2.0 / 3.0 * n[i] : 0.0));
            dfdg[1 + i][1 + a2][j] = -area * dt;
            dfdg[4][1 + a2][j] += -area * uf[i] * dt;
          }
        }
      }
      for (std::size_t j = 0; j < 3; ++j) dfdg[4][0][j] = -area * k_cond * n[j];
      const auto begin = static_cast<std::size_t>(t.stencil_offsets[s]);
      const auto end = static_cast<std::size_t>(t.stencil_offsets[s + 1]);
      std::array<double, 3> self{};
      for (std::size_t k = begin; k < end; ++k) {
        std::array<double, 3> bg{};
        double rise = 0.0;
        basis(k, r, sg, bg, rise);
        std::array<double, 3> v{};
        for (std::size_t j = 0; j < 3; ++j) v[j] = w * (bg[j] - rise / dist * e[j]);
        for (std::size_t j = 0; j < 3; ++j) self[j] -= v[j];
        const index_t entry = t.stencil[k];
        if (entry < 0) continue;  // boundary values are data
        std::array<std::array<double, 4>, 5> dfdphi{};
        for (std::size_t row = 0; row < 5; ++row) {
          for (std::size_t ph = 0; ph < 4; ++ph) {
            for (std::size_t j = 0; j < 3; ++j) dfdphi[row][ph] += dfdg[row][ph][j] * v[j];
          }
        }
        place(static_cast<std::size_t>(entry), dfdphi);
      }
      std::array<std::array<double, 4>, 5> dfdphi{};
      for (std::size_t row = 0; row < 5; ++row) {
        for (std::size_t ph = 0; ph < 4; ++ph) {
          for (std::size_t j = 0; j < 3; ++j) dfdphi[row][ph] += dfdg[row][ph][j] * self[j];
        }
      }
      place(s, dfdphi);
    };
    for (std::size_t f = 0; f < ni; ++f) {
      const auto l = static_cast<std::size_t>(t.owner[f]);
      const auto r = static_cast<std::size_t>(t.neighbour[f]);
      const bool fl = static_cast<double>(quad[l]) > 0.5, fr = static_cast<double>(quad[r]) > 0.5;
      if (!fl && !fr) continue;
      const auto xl = centroid(l), xr = centroid(r);
      const std::array<double, 3> xf{g.face_centroid_x[f], g.face_centroid_y[f],
                                     g.face_centroid_z[f]};
      const std::array<double, 3> dlr{xr[0] - xl[0], xr[1] - xl[1], xr[2] - xl[2]};
      const double dist = std::sqrt(dlr[0] * dlr[0] + dlr[1] * dlr[1] + dlr[2] * dlr[2]);
      const double tf = 0.5 * (temp(l) + temp(r));
      const double mt = 0.5 * (static_cast<double>(mu_t[l]) + static_cast<double>(mu_t[r]));
      const double mf = mu(tf);
      const double k_cond = cp * (mf / prl + mt / prt);
      const std::array<double, 3> uf{0.5 * (prim(l, 1) + prim(r, 1)),
                                     0.5 * (prim(l, 2) + prim(r, 2)),
                                     0.5 * (prim(l, 3) + prim(r, 3))};
      const double wl = fl && fr ? 0.5 : (fl ? 1.0 : 0.0);
      const double wr = fl && fr ? 0.5 : (fr ? 1.0 : 0.0);
      // R_l += F, R_r -= F: a dependence on cell l goes to (l, l) and (r, l), on r to (l, r) and
      // (r, r).
      auto place = [&](std::size_t k, const std::array<std::array<double, 4>, 5>& dfdphi) {
        for (const auto& [row, sign] : {std::pair{l, 1.0}, std::pair{r, -1.0}}) {
          double* out = block(row, k);
          (out != nullptr ? placed : outside) += chain(dfdphi, k, out, sign);
        }
      };
      if (fl) {
        const std::array<double, 3> rf{xf[0] - xl[0], xf[1] - xl[1], xf[2] - xl[2]};
        side(f, l, wl, rf, dlr, dist, k_cond, mf + mt, uf, place);
      }
      if (fr) {
        // The rise of r's fit runs from r to l, against e: its correction enters with + sign,
        // which `side` gives through the reversed segment.
        const std::array<double, 3> rf{xf[0] - xr[0], xf[1] - xr[1], xf[2] - xr[2]};
        const std::array<double, 3> drl{-dlr[0], -dlr[1], -dlr[2]};
        side(f, r, wr, rf, drl, dist, k_cond, mf + mt, uf, place);
      }
    }
    const auto* kind = d.bc_kind;
    for (std::size_t f = ni; f < nf; ++f) {
      const auto c = static_cast<std::size_t>(t.owner[f]);
      if (static_cast<double>(quad[c]) <= 0.5) continue;
      const double kf = static_cast<double>(kind[f]);
      if (kf > 1.5 && kf < 2.5) continue;  // slip: no viscous flux
      const bool adiabatic = kf > 4.5 && static_cast<double>(d.bc[f]) <= 0.0;
      const auto xc = centroid(c);
      const std::array<double, 3> rf{g.face_centroid_x[f] - xc[0], g.face_centroid_y[f] - xc[1],
                                     g.face_centroid_z[f] - xc[2]};
      const double dist = std::sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]);
      const double mf = mu(temp(c));
      const double mt = static_cast<double>(mu_t_b[f]);
      const double k_cond = adiabatic ? 0.0 : cp * (mf / prl + mt / prt);
      const std::array<double, 3> ub{static_cast<double>(d.prim_b[1 * nf + f]),
                                     static_cast<double>(d.prim_b[2 * nf + f]),
                                     static_cast<double>(d.prim_b[3 * nf + f])};
      auto place = [&](std::size_t k, const std::array<std::array<double, 4>, 5>& dfdphi) {
        double* out = block(c, k);
        (out != nullptr ? placed : outside) += chain(dfdphi, k, out, 1.0);
      };
      side(f, c, 1.0, rf, rf, dist, k_cond, mf + mt, ub, place);
    }
    return {std::sqrt(placed), std::sqrt(outside)};
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
    adjacent.assign(static_cast<std::size_t>(mesh.n_cells()), {});
    for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
      const auto o = static_cast<std::size_t>(mesh.owner()[f]);
      const auto n = static_cast<std::size_t>(mesh.neighbour()[f]);
      adjacent[o].emplace_back(n, f);
      adjacent[n].emplace_back(o, f);
    }
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
    // No fit until set_boundary gives the conditions it depends on.
    fit_weights(std::vector<BoundaryKind>(nf - static_cast<std::size_t>(topo.n_interior_faces),
                                          BoundaryKind::slip));
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
  impl_->bounded = true;
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
  if (!m.bounded) throw std::logic_error("NavierStokes: set_boundary before the residual");
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
  if (!m.bounded) throw std::logic_error("NavierStokes: set_boundary before the residual");
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
  m.fit_norms = m.fit_jacobian(a);
}

template <class P>
std::pair<double, double> NavierStokes<P>::fit_jacobian_norms() const {
  return impl_->fit_norms;
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
