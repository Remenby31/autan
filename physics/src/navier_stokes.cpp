#include "autan/physics/navier_stokes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "navier_stokes_cpu.hpp"

namespace autan::physics {

namespace nk = autan::kernels::navier_stokes;

template <class P>
struct NavierStokes<P>::Impl {
  Euler<P> euler;
  // The viscous terms' gradient: the reconstruction's under GradientFit::linear; the linear one in
  // grad_v under quadratic, which serves the convective reconstruction only.
  bool quadratic;
  std::vector<typename P::grad> grad_v;
  // fit_jacobian's stand-in for a block of the pattern when it computes a product.
  mutable double pattern_mark = 0.0;
  // gradient_tangent's: the viscous gradients', the primitives', T and u's and the fits'
  // tangents, and zero tangents of the other inputs.
  std::vector<typename P::grad> grad_v_tan;
  std::array<std::vector<typename P::grad>, 7> fits_tan;
  std::vector<State> zero_state, prim_tan, ut_tan;
  std::vector<typename P::geom> zero_geom;
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
  // Under global keys the face loops gather (Euler<P>'s topology, which share() takes).
  bool keyed;
  std::vector<std::int64_t> cell_key, face_key;  // global ids, under keys only
  std::size_t owned;                             // the owned cells, numbered first
  GhostFill* ghost_fill = nullptr;
  // Below vertex4 ghosts (#651): each rank fits its owned cells only; a fitted ghost face neighbour
  // of an owned cell has the fitted coefficients its owner computes (`fits`, exchanged each
  // residual), and its owner's whole fit row from setup, in the owner's entry order: fit_jacobian
  // evaluates it entry by entry as the owner evaluates its own, so the self term sums the same
  // terms in the same order. An entry is a local cell, or -1 for a boundary face or a cell this
  // rank does not hold, which enter the self term only.
  bool exchange = false;
  std::vector<std::int64_t> global_id;
  using FitRow = std::vector<std::pair<index_t, std::array<double, 19>>>;
  std::vector<FitRow> ghost_rows;  // per ghost cell, numbered from `owned`
  // Under keys: the interior faces, then the boundary faces, each in ascending global id.
  std::vector<std::size_t> face_order;
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
    d.grad = quadratic ? grad_v.data() : a.grad;
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

  // Fills the ghost values of `n` components of an array laid out c * n_cells + i, at most 16 a
  // call (GhostFill).
  template <class T>
  void fill_ghosts(std::vector<T>& v, std::size_t n) const {
    if (ghost_fill == nullptr) return;
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    for (std::size_t first = 0; first < n; first += 16) {
      const std::size_t count = std::min<std::size_t>(16, n - first);
      std::array<T*, 16> components{};
      for (std::size_t k = 0; k < count; ++k) components[k] = v.data() + (first + k) * nc;
      ghost_fill->forward(std::span<T* const>(components.data(), count));
    }
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
    // On a partition, only the owned cells and their face neighbours get a fit: the viscous faces
    // of the owned cells read no other. With vertex4 ghosts the rank computes them all, their
    // three-ring stencils whole, so each is the fit of the whole mesh; below vertex4 it computes
    // the owned cells' only and receives the ghosts' (exchange_fit_rows, `fits`).
    if (owned < nc) {
      std::vector<char> near(nc, 0);
      std::fill(near.begin(), near.begin() + static_cast<std::ptrdiff_t>(owned), 1);
      for (std::size_t f = 0; f < ni; ++f) {
        const auto own = static_cast<std::size_t>(topo.owner[f]);
        const auto nbr = static_cast<std::size_t>(topo.neighbour[f]);
        if (own < owned) near[nbr] = 1;
        if (nbr < owned) near[own] = 1;
      }
      for (std::size_t c = 0; c < nc; ++c) {
        if (near[c] == 0) fitted[c] = 0;
      }
    }
    if (exchange) {
      bool ghost_fits = false;
      for (std::size_t c = owned; c < nc; ++c) ghost_fits = ghost_fits || fitted[c] != 0;
      if (ghost_fits && ghost_fill == nullptr) {
        throw std::logic_error("NavierStokes: set_ghosts before set_boundary on a partition");
      }
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
      if (!cell_key.empty()) {
        // Under global keys the boundary faces, then the cells, in ascending global id: the
        // normal matrix sums its entries, and the nearest boundary face is chosen, in that
        // order, the same in any cell order (runtime.md, Ranks).
        std::sort(out.begin(), out.end(), [&](index_t a, index_t b) {
          if ((a < 0) != (b < 0)) return a < 0;
          return a < 0 ? face_key[static_cast<std::size_t>(-a - 1)] <
                             face_key[static_cast<std::size_t>(-b - 1)]
                       : cell_key[static_cast<std::size_t>(a)] <
                             cell_key[static_cast<std::size_t>(b)];
        });
      }
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
    // The index in `power` of a derivative's multi-index.
    auto unknown = [&](const std::array<int, 3>& p) {
      for (std::size_t i = 0; i < nt; ++i) {
        if (power[i] == p) return i;
      }
      return nt;
    };
    // theta's weights per entry of c's stencil, or false if the entries leave it undetermined.
    auto solve = [&](std::size_t c, std::span<const index_t> entries,
                     std::vector<std::array<double, nt>>& coef) {
      const auto& g = geo;
      std::vector<std::array<double, 3>> off(entries.size());
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
        off[k] = {x[0] - g.cell_centroid_x[c], x[1] - g.cell_centroid_y[c],
                  x[2] - g.cell_centroid_z[c]};
        for (std::size_t a = 0; a < 3; ++a) {
          extent[a] = std::max(extent[a], std::abs(off[k][a]));
        }
      }
      // The weights 1/|d|^2, |d| measured with each axis scaled by the stencil's extent along it:
      // in physical coordinates, the entries across a wall-resolved layer (10^3:1 and more) would
      // outweigh the others by the aspect ratio squared.
      const double widest = std::max({extent[0], extent[1], extent[2]});
      std::vector<double> w(entries.size());
      for (std::size_t k = 0; k < entries.size(); ++k) {
        double s2 = 0.0;
        for (std::size_t a = 0; a < 3; ++a) {
          if (extent[a] > 1e-12 * widest) s2 += std::pow(off[k][a] / extent[a], 2);
        }
        w[k] = 1.0 / s2;
      }
      // The normal matrix is formed and solved in the frame of the wall: the unit normal of the
      // stencil's boundary face nearest to the cell, completed by the global axis least aligned
      // with it, each unknown scaled by the stencil's extent along its local directions. On a
      // sloped or curved wall the global axes mix the wall-normal spacing with the slope, and the
      // matrix scaled along them is singular to round-off (#448); on a wall along an axis the
      // frame is the global one. The cubic monomials span the same space in any rotated frame, so
      // with the same weights the fit is the same polynomial.
      std::array<std::array<double, 3>, 3> axes{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
      double nearest = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < entries.size(); ++k) {
        if (entries[k] >= 0) continue;
        const auto f = static_cast<std::size_t>(-entries[k] - 1);
        const double r2 = off[k][0] * off[k][0] + off[k][1] * off[k][1] + off[k][2] * off[k][2];
        const std::array<double, 3> s{g.face_area_x[f], g.face_area_y[f], g.face_area_z[f]};
        const double area = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        if (!(r2 < nearest) || !(area > 0.0)) continue;
        nearest = r2;
        const std::array<double, 3> n{s[0] / area, s[1] / area, s[2] / area};
        std::size_t least = 0;
        for (std::size_t ax = 1; ax < 3; ++ax) {
          if (std::abs(n[ax]) < std::abs(n[least])) least = ax;
        }
        std::array<double, 3> t{};
        t[least] = 1.0;
        for (std::size_t ax = 0; ax < 3; ++ax) t[ax] -= n[least] * n[ax];
        const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        for (auto& v : t) v /= tn;
        axes = {{t,
                 {n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]},
                 n}};
      }
      std::vector<std::array<double, nt>> rows(entries.size());
      std::array<double, 3> local{};
      for (std::size_t k = 0; k < entries.size(); ++k) {
        std::array<double, 3> r{};
        for (std::size_t a = 0; a < 3; ++a) {
          r[a] = axes[a][0] * off[k][0] + axes[a][1] * off[k][1] + axes[a][2] * off[k][2];
          local[a] = std::max(local[a], std::abs(r[a]));
        }
        for (std::size_t i = 0; i < nt; ++i) {
          rows[k][i] = std::pow(r[0], power[i][0]) * std::pow(r[1], power[i][1]) *
                       std::pow(r[2], power[i][2]) /
                       (std::tgamma(power[i][0] + 1.0) * std::tgamma(power[i][1] + 1.0) *
                        std::tgamma(power[i][2] + 1.0));
        }
      }
      // Monomials in a local direction where no entry extends (the span of a one-layer mesh) are
      // left out, their derivatives zero.
      const double wide = std::max({local[0], local[1], local[2]});
      std::array<double, nt> unit{};
      std::vector<std::size_t> used;
      for (std::size_t i = 0; i < nt; ++i) {
        unit[i] = 1.0;
        bool spans = true;
        for (std::size_t a = 0; a < 3; ++a) {
          if (power[i][a] == 0) continue;
          spans = spans && local[a] > 1e-12 * wide;
          unit[i] *= std::pow(local[a], power[i][a]);
        }
        if (spans) used.push_back(i);
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
      // Global derivatives from local ones: with y = axes x, d^n phi / dx_i1..dx_in =
      // sum over a1..an of axes[a1][i1]..axes[an][in] d^n psi / dy_a1..dy_an.
      std::array<std::array<double, nt>, nt> rotate{};
      for (std::size_t i = 0; i < nt; ++i) {
        std::array<std::size_t, 3> seq{};
        std::size_t n = 0;
        for (std::size_t ax = 0; ax < 3; ++ax) {
          for (int r = 0; r < power[i][ax]; ++r) seq[n++] = ax;
        }
        std::size_t count = 1;
        for (std::size_t r = 0; r < n; ++r) count *= 3;
        for (std::size_t code = 0; code < count; ++code) {
          std::array<int, 3> p{};
          double f = 1.0;
          std::size_t rest = code;
          for (std::size_t r = 0; r < n; ++r) {
            const std::size_t la = rest % 3;
            rest /= 3;
            ++p[la];
            f *= axes[la][seq[r]];
          }
          rotate[i][unknown(p)] += f;
        }
      }
      coef.assign(entries.size(), {});
      for (std::size_t k = 0; k < entries.size(); ++k) {
        std::array<double, nt> theta{};
        for (std::size_t i = 0; i < m; ++i) {
          double v = 0.0;
          for (std::size_t j = 0; j < m; ++j) {
            v += inv[m * i + j] * w[k] * rows[k][used[j]] / unit[used[j]];
          }
          theta[used[i]] = v / unit[used[i]];
        }
        for (std::size_t i = 0; i < nt; ++i) {
          double v = 0.0;
          for (std::size_t j = 0; j < nt; ++j) v += rotate[i][j] * theta[j];
          coef[k][i] = v;
        }
      }
      return true;
    };
    fit_offsets.assign(nc + 1, 0);
    fit_entries.clear();
    std::vector<std::array<double, nt>> coef, all;
    std::fill(quad.begin(), quad.end(), typename decltype(quad)::value_type(0));
    for (std::size_t c = 0; c < nc; ++c) {
      if (exchange && c >= owned) {
        if (fitted[c] != 0) quad[c] = typename decltype(quad)::value_type(1);
      } else if (fitted[c] != 0) {
        auto entries = gather(c, 2);
        if (!solve(c, entries, coef)) {
          entries = gather(c, 3);
          if (!solve(c, entries, coef)) {
            throw std::invalid_argument(
                "the boundary fit of the viscous fluxes is undetermined at cell " +
                std::to_string(cell_key.empty() ? static_cast<std::int64_t>(c) : cell_key[c]) +
                " (" + std::to_string(entries.size()) + " entries within three rings)");
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
    if (exchange && ghost_fill != nullptr) exchange_fit_rows();
  }

  // The fit rows of the owned cells to the ranks holding them as ghosts, and theirs from them:
  // every entry in the owner's order, a cell keyed by its global id, a boundary face by -1.
  void exchange_fit_rows() {
    constexpr std::size_t nt = 19;
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto& t = d.topo;
    CellRows rows;
    rows.stride = static_cast<int>(nt);
    rows.offsets.reserve(owned + 1);
    for (std::size_t c = 0; c < owned; ++c) {
      const auto begin = static_cast<std::size_t>(t.stencil_offsets[c]);
      const auto end = static_cast<std::size_t>(t.stencil_offsets[c + 1]);
      for (std::size_t k = begin; k < end; ++k) {
        const index_t e = t.stencil[k];
        rows.keys.push_back(e < 0 ? -1 : global_id[static_cast<std::size_t>(e)]);
        for (std::size_t i = 0; i < nt; ++i) rows.values.push_back(coef(k, i));
      }
      rows.offsets.push_back(static_cast<offset_t>(rows.keys.size()));
    }
    // Only the rows this rank reads travel: fitted ghosts face-adjacent to an owned cell.
    std::vector<std::uint8_t> wanted(nc - owned, 0);
    for (std::size_t g = owned; g < nc; ++g) {
      bool near = false;
      for (const auto& nb : adjacent[g]) near = near || nb.first < owned;
      wanted[g - owned] = near && static_cast<double>(quad[g]) > 0.5 ? 1 : 0;
    }
    const CellRows got = ghost_fill->forward_rows(rows, wanted);
    std::unordered_map<std::int64_t, std::size_t> local;
    local.reserve(nc);
    for (std::size_t c = 0; c < nc; ++c) local.emplace(global_id[c], c);
    ghost_rows.assign(nc - owned, {});
    for (std::size_t g = owned; g < nc; ++g) {
      const std::size_t r = g - owned;
      const auto begin = static_cast<std::size_t>(got.offsets[r]);
      const auto end = static_cast<std::size_t>(got.offsets[r + 1]);
      bool near = false;
      for (const auto& nb : adjacent[g]) near = near || nb.first < owned;
      if (!near) continue;
      const bool fitted = static_cast<double>(quad[g]) > 0.5;
      if ((begin < end) != fitted) {
        throw Error(ErrorCode::halo_inconsistent,
                    "a ghost's boundary fit differs from its owner's at cell " +
                        std::to_string(global_id[g]));
      }
      if (!fitted) continue;
      auto& row = ghost_rows[r];
      row.reserve(end - begin);
      for (std::size_t e = begin; e < end; ++e) {
        std::array<double, nt> c{};
        for (std::size_t i = 0; i < nt; ++i) c[i] = got.values[nt * e + i];
        const auto at = got.keys[e] < 0 ? local.end() : local.find(got.keys[e]);
        row.emplace_back(at == local.end() ? index_t{-1} : static_cast<index_t>(at->second), c);
      }
    }
  }

  // Coefficient i (0-18) of the boundary fit at stencil entry k (kernels/navier_stokes.py order).
  [[nodiscard]] double coef(std::size_t k, std::size_t i) const {
    const auto nst = static_cast<std::size_t>(d.topo.n_stencil);
    return static_cast<double>(q[i / 3][(i % 3) * nst + k]);
  }

  [[nodiscard]] std::array<double, 19> coefs(std::size_t k) const {
    std::array<double, 19> c{};
    for (std::size_t i = 0; i < 19; ++i) c[i] = coef(k, i);
    return c;
  }

  // Gradient at r of the cubic whose coefficients are c, and its rise from 0 to s.
  static void basis(const std::array<double, 19>& c, const std::array<double, 3>& r,
                    const std::array<double, 3>& sg, std::array<double, 3>& grad, double& rise) {
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

  // The fits' tangent through their adiabatic-wall entries, which cubic_fits_tangent takes as
  // fixed data (boundary_temperatures_velocities is frozen): such a wall's temperature is the
  // cell's extrapolated to the face, T_b = T + grad T . (x_f - x_c), with T = p / (rho R) and
  // grad T = (grad p - R T grad rho) / (rho R), so
  // dT_b = dT + (d grad p - R dT grad rho - R T d grad rho) / (rho R) . r - grad T . r drho / rho.
  // Adds outer((dT_b, 0, 0, 0, 0), q_b) of each such entry to fit b's tangent.
  void adiabatic_fit_tangent() {
    const auto& t = d.topo;
    const auto nc = static_cast<std::size_t>(t.n_cells);
    const auto nst = static_cast<std::size_t>(t.n_stencil);
    const double rgas = static_cast<double>(d.gas_constant[0]);
    const double pref = static_cast<double>(d.p_ref[0]);
    const auto& g = geo;
    for (std::size_t c = 0; c < nc; ++c) {
      for (auto k = static_cast<std::size_t>(t.stencil_offsets[c]);
           k < static_cast<std::size_t>(t.stencil_offsets[c + 1]); ++k) {
        const index_t entry = t.stencil[k];
        if (entry >= 0) continue;
        const auto f = static_cast<std::size_t>(-entry - 1);
        const auto l = static_cast<std::size_t>(t.owner[f]);
        const double kf = static_cast<double>(d.bc_kind[f]);
        if (!(kf > 4.5 && static_cast<double>(d.bc[f]) <= 0.0)) continue;
        auto w = [&](std::size_t i) { return static_cast<double>(d.prim[i * nc + l]); };
        auto dw = [&](std::size_t i) { return static_cast<double>(prim_tan[i * nc + l]); };
        auto gw = [&](std::size_t i, std::size_t j) {
          return static_cast<double>(d.grad[(3 * i + j) * nc + l]);
        };
        auto dgw = [&](std::size_t i, std::size_t j) {
          return static_cast<double>(grad_v_tan[(3 * i + j) * nc + l]);
        };
        const double rho = w(0), temp = (w(4) + pref) / (rho * rgas);
        const double dtemp = dw(4) / (rho * rgas) - temp * dw(0) / rho;
        const std::array<double, 3> r{g.face_centroid_x[f] - g.cell_centroid_x[l],
                                      g.face_centroid_y[f] - g.cell_centroid_y[l],
                                      g.face_centroid_z[f] - g.cell_centroid_z[l]};
        double dtb = dtemp;
        for (std::size_t j = 0; j < 3; ++j) {
          const double gt = (gw(4, j) - rgas * temp * gw(0, j)) / (rho * rgas);
          const double dgt =
              (dgw(4, j) - rgas * dtemp * gw(0, j) - rgas * temp * dgw(0, j)) / (rho * rgas) -
              gt * dw(0) / rho;
          dtb += dgt * r[j];
        }
        for (std::size_t b = 0; b < 7; ++b) {
          for (std::size_t j = 0; j < 3; ++j) {
            fits_tan[b][j * nc + c] +=
                typename P::grad(dtb * static_cast<double>(q[b][j * nst + k]));
          }
        }
      }
    }
  }

  // The viscous fluxes' dependence through the boundary fit, which the face loops' Jacobians hold
  // fixed: dF/dU_k = dF/dG dG/dphi_k dphi/dW_k dW/dU_k (phi = T, u, v, w; G their face gradients)
  // for every stencil entry k, added where the residual's cell and k are the same or face
  // neighbours (the block pattern); returns the Frobenius norms of the blocks added and of those
  // left out. With `product`, it subtracts instead the blocks it would add times du from dres, on
  // the chosen faces.
  struct FitProduct {
    const State* du;
    Res* dres;
    bool interior, boundary;
  };
  std::pair<double, double> fit_jacobian(BlockMatrix* a, const FitProduct* product) const {
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
      if (k == row) return a != nullptr ? a->diagonal(row) : &pattern_mark;
      for (const auto& [nb, face] : adjacent[row]) {
        if (nb == k) {
          if (a == nullptr) return &pattern_mark;
          return static_cast<std::size_t>(t.owner[face]) == row ? a->upper(face) : a->lower(face);
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
          if (out != nullptr && out != &pattern_mark) out[5 * r + c] += sign * v;
          norm += v * v;
        }
      }
      return norm;
    };
    // With `product`: row's residual += sign dF/dU_k du_k.
    auto apply = [&](std::size_t row, std::size_t k,
                     const std::array<std::array<double, 4>, 5>& dfdphi, double sign) {
      std::array<double, 25> m{};
      (void)chain(dfdphi, k, m.data(), sign);
      for (std::size_t r = 0; r < 5; ++r) {
        double v = 0.0;
        for (std::size_t c = 0; c < 5; ++c) {
          v += m[5 * r + c] * static_cast<double>(product->du[c * nc + k]);
        }
        product->dres[r * nc + row] += Res(v);
      }
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
      std::array<double, 3> self{};
      // Entry `entry` (a cell, or negative for a boundary value or a cell not held here) with
      // coefficients c: owned and received rows go through it alike, in their owner's order.
      auto term = [&](index_t entry, const std::array<double, 19>& c) {
        std::array<double, 3> bg{};
        double rise = 0.0;
        basis(c, r, sg, bg, rise);
        std::array<double, 3> v{};
        for (std::size_t j = 0; j < 3; ++j) v[j] = w * (bg[j] - rise / dist * e[j]);
        for (std::size_t j = 0; j < 3; ++j) self[j] -= v[j];
        if (entry < 0) return;  // boundary values are data
        std::array<std::array<double, 4>, 5> dfdphi{};
        for (std::size_t row = 0; row < 5; ++row) {
          for (std::size_t ph = 0; ph < 4; ++ph) {
            for (std::size_t j = 0; j < 3; ++j) dfdphi[row][ph] += dfdg[row][ph][j] * v[j];
          }
        }
        place(static_cast<std::size_t>(entry), dfdphi);
      };
      if (exchange && s >= owned) {
        for (const auto& [entry, c] : ghost_rows[s - owned]) term(entry, c);
      } else {
        const auto begin = static_cast<std::size_t>(t.stencil_offsets[s]);
        const auto end = static_cast<std::size_t>(t.stencil_offsets[s + 1]);
        for (std::size_t k = begin; k < end; ++k) term(t.stencil[k], coefs(k));
      }
      std::array<std::array<double, 4>, 5> dfdphi{};
      for (std::size_t row = 0; row < 5; ++row) {
        for (std::size_t ph = 0; ph < 4; ++ph) {
          for (std::size_t j = 0; j < 3; ++j) dfdphi[row][ph] += dfdg[row][ph][j] * self[j];
        }
      }
      place(s, dfdphi);
    };
    // Under global keys the faces in ascending global id (face_order), so that a block or a
    // residual row sums its faces' contributions in the same order in any cell order.
    auto at = [&](std::size_t slot) { return face_order.empty() ? slot : face_order[slot]; };
    for (std::size_t slot = 0; slot < (product == nullptr || product->interior ? ni : 0); ++slot) {
      const auto f = at(slot);
      const auto l = static_cast<std::size_t>(t.owner[f]);
      const auto r = static_cast<std::size_t>(t.neighbour[f]);
      const bool fl = static_cast<double>(quad[l]) > 0.5, fr = static_cast<double>(quad[r]) > 0.5;
      if (!fl && !fr) continue;
      // Below vertex4 a ghost's fit is its owner's to report: halo-only faces are left out.
      if (exchange && l >= owned && r >= owned) continue;
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
          if (product != nullptr) {
            if (out != nullptr) apply(row, k, dfdphi, -sign);
            continue;
          }
          // The norms count owned rows only: on a partition each block's row belongs to one rank.
          const double n2 = chain(dfdphi, k, out, sign);
          if (row < owned) (out != nullptr ? placed : outside) += n2;
        }
      };
      auto side_l = [&] {
        const std::array<double, 3> rf{xf[0] - xl[0], xf[1] - xl[1], xf[2] - xl[2]};
        side(f, l, wl, rf, dlr, dist, k_cond, mf + mt, uf, place);
      };
      auto side_r = [&] {
        // The rise of r's fit runs from r to l, against e: its correction enters with + sign,
        // which `side` gives through the reversed segment.
        const std::array<double, 3> rf{xf[0] - xr[0], xf[1] - xr[1], xf[2] - xr[2]};
        const std::array<double, 3> drl{-dlr[0], -dlr[1], -dlr[2]};
        side(f, r, wr, rf, drl, dist, k_cond, mf + mt, uf, place);
      };
      // Both sides add to the same blocks: the global left's first, in any orientation.
      if (t.face_reversed != nullptr && t.face_reversed[f] != 0) {
        if (fr) side_r();
        if (fl) side_l();
      } else {
        if (fl) side_l();
        if (fr) side_r();
      }
    }
    const auto* kind = d.bc_kind;
    for (std::size_t slot = ni; slot < (product == nullptr || product->boundary ? nf : ni);
         ++slot) {
      const auto f = at(slot);
      const auto c = static_cast<std::size_t>(t.owner[f]);
      if (static_cast<double>(quad[c]) <= 0.5) continue;
      if (exchange && c >= owned) continue;
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
        if (product != nullptr) {
          if (out != nullptr) apply(c, k, dfdphi, -1.0);
          return;
        }
        const double n2 = chain(dfdphi, k, out, 1.0);
        if (c < owned) (out != nullptr ? placed : outside) += n2;
      };
      side(f, c, 1.0, rf, rf, dist, k_cond, mf + mt, ub, place);
    }
    return {std::sqrt(placed), std::sqrt(outside)};
  }

  // A face or boundary loop: its gather under global keys, else its scatter.
  template <class Loop>
  void run(Loop loop) const {
    if (keyed) {
      loop(kernels::Strategy::store_gather);
    } else {
      loop();
    }
  }

  // `data` over every local cell and face: the generated cell and boundary loops cover owned ones
  // only. Pointwise values and the fits of the cells whose stencils the ghosts hold whole (owned
  // cells and their face neighbours) then have their owner's bits; the face loops that sum into the
  // owned cells' residuals keep the owned range.
  [[nodiscard]] static nk::Data<P> halo(nk::Data<P> data) {
    data.topo.n_owned_cells = data.topo.n_cells;
    data.topo.n_owned_interior_faces = data.topo.n_interior_faces;
    data.topo.n_owned_faces = data.topo.n_faces;
    return data;
  }

  void fit() {
    const auto h = halo(d);
    nk::cpu::temperatures_velocities(h);
    nk::cpu::boundary_temperatures_velocities(h);
    nk::cpu::cubic_fits(h);
    if (exchange) {
      for (auto& f : fits) fill_ghosts(f, 15);
    }
  }

  Impl(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme, Transport t,
       const kernels::GlobalKeys* keys, std::optional<GhostStencil> ghosts)
      : euler(mesh, geometry, gas, scheme, keys, ghosts),
        quadratic(scheme.gradient_fit == GradientFit::quadratic),
        keyed(keys != nullptr),
        owned(static_cast<std::size_t>(mesh.n_owned_cells())),
        geo(geometry) {
    // The boundary fits of the owned cells reach three vertex rings of them: on a partition, its
    // ghosts must hold three; with four, the ranks also fit their ghost face neighbours.
    const bool partitioned = ghosts || mesh.n_cells() != mesh.n_owned_cells();
    if (partitioned && ghosts.value_or(GhostStencil::face2) < GhostStencil::vertex3) {
      throw Error(ErrorCode::partition_stencil_not_local,
                  "the viscous boundary fits' vertex rings reach beyond the partition's ghosts");
    }
    // Every rank of a partition takes part in the collective exchanges, a rank without ghosts too.
    exchange = ghosts.has_value() && *ghosts < GhostStencil::vertex4;
    if (exchange) {
      const auto ids = mesh.cell_global_id();
      global_id.assign(ids.begin(), ids.end());
    }
    if (keys != nullptr) {
      cell_key.assign(keys->cell.begin(), keys->cell.end());
      face_key.assign(keys->face.begin(), keys->face.end());
    }
    if (quadratic)
      grad_v.assign(15 * static_cast<std::size_t>(mesh.n_cells()), typename P::grad(0));
    if (!(t.mu_ref > 0.0) || !(t.prandtl > 0.0) || !(t.prandtl_turbulent > 0.0) ||
        (t.sutherland && (!(t.t_ref > 0.0) || !(t.s >= 0.0)))) {
      throw std::invalid_argument("transport properties must be positive");
    }
    const auto packed = kernels::pack_geometry(mesh, geometry, keys);
    topo = kernels::topology(mesh.view());
    if (!face_key.empty()) {
      const auto nf = static_cast<std::size_t>(topo.n_faces);
      const auto ni = static_cast<std::ptrdiff_t>(topo.n_interior_faces);
      face_order.resize(nf);
      std::iota(face_order.begin(), face_order.end(), std::size_t{0});
      auto by_key = [&](std::size_t a, std::size_t b) { return face_key[a] < face_key[b]; };
      std::sort(face_order.begin(), face_order.begin() + ni, by_key);
      std::sort(face_order.begin() + ni, face_order.end(), by_key);
    }
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

GhostStencil navier_stokes_ghost_stencil(bool ranks_reproducible) noexcept {
  return ranks_reproducible ? GhostStencil::vertex4 : GhostStencil::vertex3;
}

template <class P>
NavierStokes<P>::NavierStokes(const Mesh& mesh, const Geometry& geometry, Gas gas, Scheme scheme,
                              Transport transport, const kernels::GlobalKeys* keys,
                              std::optional<GhostStencil> ghosts)
    : impl_(std::make_unique<Impl>(mesh, geometry, gas, scheme, transport, keys, ghosts)) {}

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
void NavierStokes<P>::set_boundary_data(std::span<const double> data) {
  impl_->euler.set_boundary_data(data);
  impl_->share();
}

template <class P>
void NavierStokes<P>::set_ghosts(GhostFill* ghosts) noexcept {
  impl_->euler.set_ghosts(ghosts);
  impl_->ghost_fill = ghosts;
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
  auto flow = m.euler.arrays();
  if (m.quadratic) flow.grad = m.grad_v.data();
  return {flow, m.mu_t.data(), m.mu_t_b.data(), static_cast<double>(m.d.p_ref[0]),
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
  if (m.quadratic) {
    m.euler.linear_gradient(m.grad_v);
    m.fill_ghosts(m.grad_v, 15);
  }
  m.share();
  m.fit();
  m.d.res = res.data();
  m.run([&](auto... strategy) { nk::cpu::viscous_faces(m.d, strategy...); });
  m.run([&](auto... strategy) { nk::cpu::viscous_boundaries(m.d, strategy...); });
}

template <class P>
void NavierStokes<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  if (!m.bounded) throw std::logic_error("NavierStokes: set_boundary before the residual");
  m.euler.residual_and_jacobian(s, res);
  if (m.quadratic) {
    m.euler.linear_gradient(m.grad_v);
    m.fill_ghosts(m.grad_v, 15);
  }
  m.share();
  m.fit();
  m.d.res = res.data();
  m.run([&](auto... strategy) { nk::cpu::viscous_faces_jacobian(m.d, strategy...); });
  m.run([&](auto... strategy) { nk::cpu::viscous_boundaries_jacobian(m.d, strategy...); });
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
  if (m.keyed) {
    // As Euler<P>::assemble under global keys: the face loops evaluate each face in its global
    // orientation (visc_jac: dF/dW of the global left, block 0, and right, block 1), each diagonal
    // block gathers its cell's faces in Topology::cell_faces order (the global order), and the
    // off-diagonal blocks are placed in BlockMatrix's local orientation.
    const auto& t = m.d.topo;
    auto left = [&](std::size_t f) {
      const bool rev = t.face_reversed != nullptr && t.face_reversed[f] != 0;
      return static_cast<std::size_t>(rev ? t.neighbour[f] : t.owner[f]);
    };
    for (std::size_t c = 0; c < nc; ++c) {
      for (auto k = t.cell_face_offsets[c]; k < t.cell_face_offsets[c + 1]; ++k) {
        const auto f = static_cast<std::size_t>(t.cell_faces[k]);
        if (f >= ni) {
          add(a.diagonal(c), m.visc_bnd_jac.data(), nf, f, 0, dwdu, nc, c, 1.0);
        } else if (left(f) == c) {
          add(a.diagonal(c), m.visc_jac.data(), nf, f, 0, dwdu, nc, c, 1.0);
        } else {
          add(a.diagonal(c), m.visc_jac.data(), nf, f, 1, dwdu, nc, c, -1.0);
        }
      }
    }
    for (std::size_t f = 0; f < ni; ++f) {
      const auto o = static_cast<std::size_t>(t.owner[f]);
      const auto n = static_cast<std::size_t>(t.neighbour[f]);
      const bool rev = left(f) != o;
      add(rev ? a.lower(f) : a.upper(f), m.visc_jac.data(), nf, f, 1, dwdu, nc, rev ? o : n, 1.0);
      add(rev ? a.upper(f) : a.lower(f), m.visc_jac.data(), nf, f, 0, dwdu, nc, rev ? n : o, -1.0);
    }
  } else {
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
  m.fit_norms = m.fit_jacobian(&a, nullptr);
}

template <class P>
void NavierStokes<P>::gradient_tangent(std::span<const State> s, std::span<const State> du,
                                       std::span<Res> dres, typename Euler<P>::Faces faces) {
  auto& m = *impl_;
  m.euler.gradient_tangent(s, du, dres, faces);
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  if (m.grad_v_tan.size() != 15 * nc) {
    m.grad_v_tan.assign(15 * nc, typename P::grad(0));
    m.zero_state.assign(5 * nc, State(0));
    m.zero_geom.assign(nc, typename P::geom(0));
  }
  // The viscous fluxes with the cell gradients and the boundary fits moving, the face states, the
  // eddy viscosity and the fits' boundary values held: the fits' tangent covers every stencil
  // entry, so the blocks `assemble` already adds for them are subtracted (fit_jacobian).
  m.euler.linear_gradient_tangent(s, du, m.grad_v_tan);
  m.fill_ghosts(m.grad_v_tan, 15);
  // Under the linear fit the viscous terms read the reconstruction's gradient, zero at first order.
  if (!m.quadratic && !m.euler.scheme().second_order) {
    std::fill(m.grad_v_tan.begin(), m.grad_v_tan.end(), typename P::grad(0));
  }
  if (m.prim_tan.size() != 5 * nc) {
    m.prim_tan.assign(5 * nc, State(0));
    m.ut_tan.assign(5 * nc, State(0));
    for (auto& f : m.fits_tan) f.assign(15 * nc, typename P::grad(0));
  }
  m.euler.primitive_tangent(s, du, m.prim_tan);
  m.fill_ghosts(m.prim_tan, 5);
  auto d = m.d;
  d.prim_tan = m.prim_tan.data();
  d.ut_tan = m.ut_tan.data();
  nk::cpu::temperatures_velocities_tangent(Impl::halo(d));
  if (m.exchange) m.fill_ghosts(m.ut_tan, 5);
  d.fit0_tan = m.fits_tan[0].data();
  d.fit1_tan = m.fits_tan[1].data();
  d.fit2_tan = m.fits_tan[2].data();
  d.fit3_tan = m.fits_tan[3].data();
  d.fit4_tan = m.fits_tan[4].data();
  d.fit5_tan = m.fits_tan[5].data();
  d.fit6_tan = m.fits_tan[6].data();
  nk::cpu::cubic_fits_tangent(Impl::halo(d));
  m.adiabatic_fit_tangent();
  if (m.exchange) {
    for (auto& f : m.fits_tan) m.fill_ghosts(f, 15);
  }
  d.prim_tan = m.zero_state.data();
  d.ut_tan = m.zero_state.data();
  d.mu_t_tan = m.zero_state.data();
  d.lam_v_tan = m.zero_state.data();
  d.quad_tan = m.zero_geom.data();
  d.grad_tan = m.grad_v_tan.data();
  d.res_tan = dres.data();
  const bool interior = faces == Euler<P>::Faces::interior;
  if (interior) {
    m.run([&](auto... strategy) { nk::cpu::viscous_faces_tangent(d, strategy...); });
  } else {
    m.run([&](auto... strategy) { nk::cpu::viscous_boundaries_tangent(d, strategy...); });
  }
  const typename Impl::FitProduct product{du.data(), dres.data(), interior, !interior};
  (void)m.fit_jacobian(nullptr, &product);
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

template <class P>
typename Euler<P>::QuadraticGradient NavierStokes<P>::quadratic_gradient() const {
  return impl_->euler.quadratic_gradient();
}

template <class P>
typename Euler<P>::LinearCells NavierStokes<P>::linear_cells() const noexcept {
  return impl_->euler.linear_cells();
}

template <class P>
const Euler<P>& NavierStokes<P>::euler() const noexcept {
  return impl_->euler;
}

template <class P>
const kernels::navier_stokes::Data<P>& NavierStokes<P>::data() const {
  if (!impl_->bounded) throw std::logic_error("NavierStokes: set_boundary before data()");
  return impl_->d;
}

template class NavierStokes<kernels::fp64>;
template class NavierStokes<kernels::fp32_acc64>;

}  // namespace autan::physics
