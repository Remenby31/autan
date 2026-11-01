#include "autan/physics/euler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "euler_cpu.hpp"

namespace autan::physics {

namespace ek = autan::kernels::euler;

template <class P>
struct Euler<P>::Impl {
  using Gr = typename P::grad;

  Scheme scheme;
  kernels::Topology topo{};
  // Each geometry array has the element type its Data<P> member stores (the generator stores
  // geometry that bodies read only as values in P::geom_value).
  template <class Ptr>
  using stored = std::vector<std::remove_cvref_t<std::remove_pointer_t<Ptr>>>;
  using D = ek::Data<P>;
  stored<decltype(D::face_normal)> face_normal;
  stored<decltype(D::face_area_magnitude)> face_area_magnitude;
  stored<decltype(D::face_centroid)> face_centroid;
  stored<decltype(D::face_moment)> face_moment;
  stored<decltype(D::cell_centroid)> cell_centroid;
  stored<decltype(D::cell_volume)> cell_volume;
  stored<decltype(D::stencil_lsq_weight)> stencil_lsq_weight;
  kernels::CellStencil stencil;
  // Boundary band: its quadratic fit's stencil (two vertex rings) and weights, as a topology of
  // its own for band_gradients (empty outside the band).
  kernels::CellStencil vertex;
  stored<decltype(D::band)> band;
  stored<decltype(D::band_weight)> band_weight;
  std::vector<kernels::offset_t> band_offsets;
  std::vector<index_t> band_entries;
  kernels::Topology band_topo{};
  std::vector<double> moments;   // per cell, xx yy zz xy xz yz about the centroid per unit volume
  std::size_t linear_cells = 0;  // band cells that keep the linear fit
  index_t first_linear = -1;
  std::vector<double> bc_kind;
  std::vector<State> bc, prim_b, prim, wmin, wmax, phi, source, prim_tan, zero_state;
  std::vector<typename P::jac> face_jac, bnd_jac, dwdu;
  std::vector<Gr> grad, grad_band, grad_v, zero_grad;
  bool viscous = false;  // enable_viscous_gradient: band_gradients and grad_v are computed
  std::vector<BoundaryKind> kinds;  // the last set_boundary's
  ek::Data<P> d{};

  template <class T, class Source>
  static void fill(std::vector<T>& to, const Source& from) {
    to = kernels::converted<T>(from);
  }

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
    fill(stencil_lsq_weight, stencil.lsq_weight);
    vertex = kernels::cell_stencil(mesh, packed, kernels::StencilKind::vertex);
    compute_moments(mesh, geometry);
    fill(face_normal, packed.face_normal);
    fill(face_area_magnitude, packed.face_area_magnitude);
    fill(face_centroid, packed.face_centroid);
    fill(face_moment, packed.face_moment);
    fill(cell_centroid, packed.cell_centroid);
    fill(cell_volume, packed.cell_volume);
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
    grad_band.assign(15 * nc, Gr(0));
    grad_v.assign(15 * nc, Gr(0));
    zero_grad.assign(15 * nc, Gr(0));
    band.assign(nc, typename decltype(band)::value_type(0));
    band_offsets.assign(nc + 1, 0);
    band_topo = topo;
    band_topo.n_stencil = 0;
    band_topo.stencil_offsets = band_offsets.data();
    band_topo.stencil = band_entries.data();
    band_topo.stencil_twin = nullptr;

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
    d.band = band.data();
    d.grad_band = grad_band.data();
    d.grad_v = grad_v.data();
    d.band_weight = band_weight.data();
    d.gamma[0] = typename P::param(gas.gamma);
    d.gas_constant[0] = typename P::param(gas.gas_constant);
    if (!(scheme.mach_ref > 0.0)) {
      throw std::invalid_argument("AUSM+-up needs a positive cut-off Mach number");
    }
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

  // Least-squares weights over the stencil, 1/|d|^2 weighting as kernels::cell_stencil, but
  // with only the boundary faces whose condition gives the state there: elsewhere the boundary
  // value is the adjacent cell's, which sits at the cell centroid and not at the face, and would
  // make the gradient inconsistent next to slip walls and extrapolating boundaries. A cell whose
  // remaining entries do not determine a gradient keeps every boundary face.
  void compute_gradient_weights(std::span<const BoundaryKind> kind) {
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
    const auto nst = stencil.entries.size();
    std::vector<double> w(3 * nst, 0.0);
    constexpr double coordinate_eps =
        std::numeric_limits<typename decltype(cell_centroid)::value_type>::epsilon();
    for (std::size_t c = 0; c + 1 < stencil.offsets.size(); ++c) {
      const auto begin = static_cast<std::size_t>(stencil.offsets[c]);
      const auto end = static_cast<std::size_t>(stencil.offsets[c + 1]);
      auto weights = [&](bool all_boundaries) {
        std::array<double, 9> a{};
        std::vector<std::array<double, 4>> offs;  // offset and weight per entry, 0 if excluded
        for (std::size_t k = begin; k < end; ++k) {
          const index_t e = stencil.entries[k];
          std::array<double, 3> x{};
          bool used = true;
          if (e >= 0) {
            for (std::size_t j = 0; j < 3; ++j) {
              x[j] = static_cast<double>(cell_centroid[j * nc + static_cast<std::size_t>(e)]);
            }
          } else {
            const auto f = static_cast<std::size_t>(-e - 1);
            used = all_boundaries || kind[f - ni] == BoundaryKind::state;
            for (std::size_t j = 0; j < 3; ++j) {
              x[j] = static_cast<double>(face_centroid[j * nf + f]);
            }
          }
          // Offsets within 64 ulps of the coordinates are round-off, not geometry: across a
          // one-cell-thick slab the centroids differ only by them, and an offset kept at 1e-17
          // would let the fit invert round-off into a gradient across the layer.
          std::array<double, 4> dk{};
          for (std::size_t j = 0; j < 3; ++j) {
            const double xc = static_cast<double>(cell_centroid[j * nc + c]);
            const double round_off = 64 * coordinate_eps * std::max(std::abs(x[j]), std::abs(xc));
            dk[j] = x[j] - xc;
            if (std::abs(dk[j]) <= round_off) dk[j] = 0.0;
          }
          dk[3] = used ? 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]) : 0.0;
          for (std::size_t u = 0; u < 3; ++u) {
            for (std::size_t v = 0; v < 3; ++v) a[3 * u + v] += dk[3] * dk[u] * dk[v];
          }
          offs.push_back(dk);
        }
        const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                           a[2] * (a[3] * a[7] - a[4] * a[6]);
        const double scale = a[0] * a[4] * a[8];
        if (!(std::abs(det) > 1e-10 * std::abs(scale))) return false;
        const std::array<double, 9> inv = {
            (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
            (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
            (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
            (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
            (a[0] * a[4] - a[1] * a[3]) / det};
        for (std::size_t k = begin; k < end; ++k) {
          const auto& dk = offs[k - begin];
          for (std::size_t u = 0; u < 3; ++u) {
            w[u * nst + k] =
                dk[3] * (inv[3 * u] * dk[0] + inv[3 * u + 1] * dk[1] + inv[3 * u + 2] * dk[2]);
          }
        }
        return true;
      };
      if (!weights(false)) weights(true);
    }
    fill(stencil_lsq_weight, w);
    d.stencil_lsq_weight = stencil_lsq_weight.data();
    kinds.assign(kind.begin(), kind.end());
    if (viscous) band_weights(kind);
  }

  // The boundary band's weights (kernels/euler.py, band_gradients): for every cell within three
  // face rings of the boundary, the gradient at its centroid of the quadratic
  // phi_c + g . d + d^T H d / 2 fitted, with weights 1/|d|^2, to the cell averages of the cells
  // within two vertex rings (the quadratic's average over cell k adds H : M_k / 2, M_k its second
  // moment about its centroid, and phi_c is cell c's average) and to the states of their boundary
  // faces that give one (point values). Each unknown is scaled by the stencil's extent along its
  // directions and the weights taken in those scaled coordinates; directions where no entry
  // extends (the span of a one-cell-thick slab) are left out. Solved by Householder QR. A cell
  // with fewer than 1.5 entries per unknown, or whose entries leave the fit undetermined, keeps
  // the linear least-squares gradient: band_gradient() reports how many. Boundary faces are
  // weighted entries, not equality constraints: as constraints (Ollivier-Gooch and Van Altena
  // 2002) they made the implicit solution diverge on the hybrid manufactured-solution meshes.
  void band_weights(std::span<const BoundaryKind> kind) {
    const auto nc = static_cast<std::size_t>(topo.n_cells);
    const auto ni = static_cast<std::size_t>(topo.n_interior_faces);
    const auto nf = static_cast<std::size_t>(topo.n_faces);
    std::vector<int> ring(nc, 0);
    for (std::size_t f = ni; f < nf; ++f) ring[static_cast<std::size_t>(topo.owner[f])] = 1;
    for (int depth = 2; depth <= 3; ++depth) {
      auto next = ring;
      for (std::size_t f = 0; f < ni; ++f) {
        const auto o = static_cast<std::size_t>(topo.owner[f]);
        const auto n = static_cast<std::size_t>(topo.neighbour[f]);
        if (ring[o] == depth - 1 && ring[n] == 0) next[n] = depth;
        if (ring[n] == depth - 1 && ring[o] == 0) next[o] = depth;
      }
      ring = std::move(next);
    }
    auto gives = [&](index_t e) {
      return kind[static_cast<std::size_t>(-e - 1) - ni] == BoundaryKind::state;
    };
    auto around = [&](index_t c) {
      const auto i = static_cast<std::size_t>(c);
      return std::span<const index_t>(
          vertex.entries.data() + vertex.offsets[i],
          static_cast<std::size_t>(vertex.offsets[i + 1] - vertex.offsets[i]));
    };
    auto centre = [&](index_t e) {
      std::array<double, 3> x{};
      for (std::size_t j = 0; j < 3; ++j) {
        x[j] = e >= 0
                   ? static_cast<double>(cell_centroid[j * nc + static_cast<std::size_t>(e)])
                   : static_cast<double>(face_centroid[j * nf + static_cast<std::size_t>(-e - 1)]);
      }
      return x;
    };
    // Quadratic monomials as (i, j) pairs of directions after the three linear ones.
    constexpr std::size_t nt = 9;
    constexpr std::array<std::array<int, 2>, nt> pair{
        {{0, -1}, {1, -1}, {2, -1}, {0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};
    std::fill(band.begin(), band.end(), typename decltype(band)::value_type(0));
    band_offsets.assign(nc + 1, 0);
    band_entries.clear();
    linear_cells = 0;
    first_linear = -1;
    std::vector<std::array<double, 3>> weights;
    std::vector<index_t> cells, points;
    for (std::size_t c = 0; c < nc; ++c) {
      band_offsets[c + 1] = static_cast<kernels::offset_t>(band_entries.size());
      if (ring[c] == 0) continue;
      cells.assign(1, static_cast<index_t>(c));
      points.clear();
      for (int r = 0; r < 2; ++r) {
        const auto reached = cells.size();
        for (std::size_t k = 0; k < reached; ++k) {
          for (const auto e : around(cells[k])) {
            if (e >= 0)
              cells.push_back(e);
            else if (gives(e))
              points.push_back(e);
          }
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
      }
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());
      for (const auto e : cells) {
        if (e != static_cast<index_t>(c)) points.push_back(e);
      }
      const auto xc = centre(static_cast<index_t>(c));
      auto offset = [&](index_t e) {
        const auto x = centre(e);
        return std::array<double, 3>{x[0] - xc[0], x[1] - xc[1], x[2] - xc[2]};
      };
      std::array<double, 3> extent{};
      for (const auto e : points) {
        const auto dk = offset(e);
        for (std::size_t j = 0; j < 3; ++j) extent[j] = std::max(extent[j], std::abs(dk[j]));
      }
      const double widest = std::max({extent[0], extent[1], extent[2]});
      auto spans = [&](int j) {
        return j < 0 || extent[static_cast<std::size_t>(j)] > 1e-12 * widest;
      };
      std::vector<std::size_t> used;
      for (std::size_t i = 0; i < nt; ++i) {
        if (spans(i < 3 ? static_cast<int>(i) : pair[i][0]) && spans(pair[i][1])) used.push_back(i);
      }
      const auto m = used.size();
      // Scaled row of entry e: d / extent, and (d_i d_j + M_k,ij - M_c,ij) / (extent_i extent_j)
      // for a cell k, (d_i d_j - M_c,ij) / (...) for a face (its state is a point value).
      auto row = [&](index_t e) {
        const auto dk = offset(e);
        std::vector<double> r(m);
        for (std::size_t u = 0; u < m; ++u) {
          const auto i = used[u];
          if (i < 3) {
            r[u] = dk[i] / extent[i];
          } else {
            const auto a = static_cast<std::size_t>(pair[i][0]);
            const auto b = static_cast<std::size_t>(pair[i][1]);
            double v = dk[a] * dk[b] - moment(c, a, b);
            if (e >= 0) v += moment(static_cast<std::size_t>(e), a, b);
            r[u] = v / (extent[a] * extent[b]);
          }
        }
        return r;
      };
      auto sqrt_weight = [&](index_t e) {
        const auto dk = offset(e);
        double s2 = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
          if (spans(static_cast<int>(j))) s2 += std::pow(dk[j] / extent[j], 2);
        }
        return 1.0 / std::sqrt(s2);
      };
      const auto n = points.size();
      auto keep_linear = [&] {
        ++linear_cells;
        if (first_linear < 0) first_linear = static_cast<index_t>(c);
      };
      if (2 * n < 3 * m) {
        keep_linear();
        continue;
      }
      // theta = R^-1 Q^T W^1/2 y from W^1/2 A = Q R: the weights G = R^-1 Q^T W^1/2 (m x n).
      std::vector<double> b(n * m), sw(n);
      for (std::size_t k = 0; k < n; ++k) {
        const auto r = row(points[k]);
        sw[k] = sqrt_weight(points[k]);
        for (std::size_t i = 0; i < m; ++i) b[i * n + k] = sw[k] * r[i];
      }
      std::vector<double> qb;
      if (!householder(n, m, b, m, qb)) {
        keep_linear();
        continue;
      }
      std::vector<double> ga(m * n);  // column-major m x n
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t ii = m; ii-- > 0;) {
          double v = qb[ii * n + k];
          for (std::size_t jj = ii + 1; jj < m; ++jj) v -= b[jj * n + ii] * ga[k * m + jj];
          ga[k * m + ii] = v / b[ii * n + ii];
        }
        for (std::size_t i = 0; i < m; ++i) ga[k * m + i] *= sw[k];
      }
      auto gradient = [&](const double* col) {
        std::array<double, 3> g{};
        for (std::size_t u = 0; u < m; ++u) {
          if (used[u] < 3) g[used[u]] = col[u] / extent[used[u]];
        }
        return g;
      };
      for (std::size_t k = 0; k < n; ++k) {
        band_entries.push_back(points[k]);
        weights.push_back(gradient(&ga[k * m]));
      }
      band[c] = typename decltype(band)::value_type(1);
      band_offsets[c + 1] = static_cast<kernels::offset_t>(band_entries.size());
    }
    const auto nst = band_entries.size();
    std::vector<double> w(3 * nst);
    for (std::size_t k = 0; k < nst; ++k) {
      for (std::size_t j = 0; j < 3; ++j) w[j * nst + k] = weights[k][j];
    }
    fill(band_weight, w);
    d.band_weight = band_weight.data();
    band_topo.n_stencil = static_cast<kernels::offset_t>(nst);
    band_topo.stencil_offsets = band_offsets.data();
    band_topo.stencil = band_entries.data();
  }

  // Second moment of cell c about its centroid per unit volume, component (a, b).
  [[nodiscard]] double moment(std::size_t c, std::size_t a, std::size_t b) const {
    static constexpr std::array<std::array<std::size_t, 3>, 3> slot{
        {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}}};
    return moments[slot[a][b] * static_cast<std::size_t>(topo.n_cells) + c];
  }

  // Second moments (xx, yy, zz, xy, xz, yz) of every cell about its centroid per unit volume,
  // from the tetrahedra (centroid, face node average, two consecutive face nodes):
  // int x x^T dV = V / 20 (sum_i x_i x_i^T + (sum_i x_i)(sum_i x_i)^T) over each one's vertices.
  void compute_moments(const Mesh& mesh, const Geometry& g) {
    const auto nc = static_cast<std::size_t>(mesh.n_cells());
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    moments.assign(6 * nc, 0.0);
    std::vector<double> volume(nc, 0.0);
    const auto xs = mesh.x(), ys = mesh.y(), zs = mesh.z();
    const auto offsets = mesh.face_node_offsets();
    const auto nodes = mesh.face_nodes();
    for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
      const auto begin = static_cast<std::size_t>(offsets[f]);
      const auto end = static_cast<std::size_t>(offsets[f + 1]);
      std::array<double, 3> mid{};
      for (auto k = begin; k < end; ++k) {
        const auto v = static_cast<std::size_t>(nodes[k]);
        mid = {mid[0] + xs[v], mid[1] + ys[v], mid[2] + zs[v]};
      }
      for (auto& x : mid) x /= static_cast<double>(end - begin);
      for (int side = 0; side < (f < ni ? 2 : 1); ++side) {
        const auto c = static_cast<std::size_t>(side == 0 ? mesh.owner()[f] : mesh.neighbour()[f]);
        const std::array<double, 3> o{g.cell_centroid_x[c], g.cell_centroid_y[c],
                                      g.cell_centroid_z[c]};
        for (auto k = begin; k < end; ++k) {
          const auto v0 = static_cast<std::size_t>(nodes[k]);
          const auto v1 = static_cast<std::size_t>(nodes[k + 1 < end ? k + 1 : begin]);
          const std::array<std::array<double, 3>, 3> t{
              {{mid[0] - o[0], mid[1] - o[1], mid[2] - o[2]},
               {xs[v0] - o[0], ys[v0] - o[1], zs[v0] - o[2]},
               {xs[v1] - o[0], ys[v1] - o[1], zs[v1] - o[2]}}};
          // Signed volume, positive for the owner (normals point out of it).
          double vol = (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) -
                        t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) +
                        t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) /
                       6.0;
          if (side == 1) vol = -vol;
          volume[c] += vol;
          const std::array<double, 3> sum{t[0][0] + t[1][0] + t[2][0], t[0][1] + t[1][1] + t[2][1],
                                          t[0][2] + t[1][2] + t[2][2]};
          static constexpr std::array<std::array<std::size_t, 2>, 6> ab{
              {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};
          for (std::size_t s = 0; s < 6; ++s) {
            const auto [a, b] = ab[s];
            double v = sum[a] * sum[b];
            for (const auto& x : t) v += x[a] * x[b];
            moments[s * nc + c] += vol / 20.0 * v;
          }
        }
      }
    }
    for (std::size_t s = 0; s < 6; ++s) {
      for (std::size_t c = 0; c < nc; ++c) moments[s * nc + c] /= volume[c];
    }
  }

  // Householder QR of the column-major rows x cols matrix a: on return its upper triangle is R,
  // and q the first qcols columns of Q (rows x qcols, column-major). False if a diagonal entry of
  // R is below 1e-10 of the largest column norm (rank deficient).
  static bool householder(std::size_t rows, std::size_t cols, std::vector<double>& a,
                          std::size_t qcols, std::vector<double>& q) {
    double largest = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < rows; ++i) s += a[j * rows + i] * a[j * rows + i];
      largest = std::max(largest, std::sqrt(s));
    }
    std::vector<double> v(rows * cols, 0.0), vv(cols, 0.0);
    auto reflect = [&](std::size_t j, double* col) {
      double s = 0.0;
      for (std::size_t i = j; i < rows; ++i) s += v[j * rows + i] * col[i];
      s *= 2.0 / vv[j];
      for (std::size_t i = j; i < rows; ++i) col[i] -= s * v[j * rows + i];
    };
    for (std::size_t j = 0; j < cols; ++j) {
      double norm = 0.0;
      for (std::size_t i = j; i < rows; ++i) norm += a[j * rows + i] * a[j * rows + i];
      norm = std::sqrt(norm);
      if (!(norm > 1e-10 * largest)) return false;
      const double alpha = a[j * rows + j] > 0.0 ? -norm : norm;
      for (std::size_t i = j; i < rows; ++i) v[j * rows + i] = a[j * rows + i];
      v[j * rows + j] -= alpha;
      for (std::size_t i = j; i < rows; ++i) vv[j] += v[j * rows + i] * v[j * rows + i];
      for (std::size_t k = j; k < cols; ++k) reflect(j, &a[k * rows]);
    }
    // Q E = H_0 ... H_{cols-1} E, E the first qcols columns of the identity.
    q.assign(rows * qcols, 0.0);
    for (std::size_t k = 0; k < qcols; ++k) {
      q[k * rows + k] = 1.0;
      for (std::size_t j = cols; j-- > 0;) reflect(j, &q[k * rows]);
    }
    return true;
  }

  void viscous_gradients() {
    auto b = d;
    b.topo = band_topo;
    ek::cpu::band_gradients(b);
    ek::cpu::viscous_gradients(d);
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

  void faces_jacobian() {
    switch (scheme.flux) {
      case Flux::roe:
        ek::cpu::roe_faces_jacobian(d);
        ek::cpu::roe_boundaries_jacobian(d);
        break;
      case Flux::hllc:
        ek::cpu::hllc_faces_jacobian(d);
        ek::cpu::hllc_boundaries_jacobian(d);
        break;
      case Flux::ausm_up:
        ek::cpu::ausm_faces_jacobian(d);
        ek::cpu::ausm_boundaries_jacobian(d);
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
  m.compute_gradient_weights(kind);
}

template <class P>
std::vector<double> Euler<P>::gradient_weights() const {
  const auto& w = impl_->stencil_lsq_weight;
  return {w.begin(), w.end()};
}

template <class P>
void Euler<P>::enable_viscous_gradient() {
  auto& m = *impl_;
  if (m.viscous) return;
  m.viscous = true;
  if (!m.kinds.empty()) m.band_weights(m.kinds);
}

template <class P>
std::span<const typename P::grad> Euler<P>::viscous_gradient() const noexcept {
  return impl_->grad_v;
}

template <class P>
typename Euler<P>::BandGradient Euler<P>::band_gradient() const {
  const auto& m = *impl_;
  return {{m.band.begin(), m.band.end()},
          {m.band_offsets.begin(), m.band_offsets.end()},
          m.band_entries,
          {m.band_weight.begin(), m.band_weight.end()},
          m.linear_cells,
          m.first_linear};
}

template <class P>
void Euler<P>::set_second_order(bool on) {
  auto& m = *impl_;
  m.scheme.second_order = on;
  if (!on) {
    std::fill(m.grad.begin(), m.grad.end(), typename P::grad(0));
    std::fill(m.grad_v.begin(), m.grad_v.end(), typename P::grad(0));
    std::fill(m.phi.begin(), m.phi.end(), State(0));
  } else if (!m.scheme.limiter) {
    std::fill(m.phi.begin(), m.phi.end(), State(1));
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
    if (m.viscous) m.viscous_gradients();
  }
  ek::cpu::sources(m.d);
  m.faces();
}

template <class P>
void Euler<P>::residual_and_jacobian(std::span<const State> s, std::span<Res> res) {
  auto& m = *impl_;
  if (m.dwdu.empty()) {
    // Allocated on first use: residual-only users (explicit, unsteady) carry no Jacobian.
    const auto nc = static_cast<std::size_t>(m.topo.n_cells);
    const auto nf = static_cast<std::size_t>(m.topo.n_faces);
    m.face_jac.assign(50 * nf, typename P::jac(0));
    m.bnd_jac.assign(25 * nf, typename P::jac(0));
    m.dwdu.assign(25 * nc, typename P::jac(0));
    m.d.face_jac = m.face_jac.data();
    m.d.bnd_jac = m.bnd_jac.data();
    m.d.dwdu = m.dwdu.data();
  }
  m.d.u = const_cast<State*>(s.data());
  m.d.res = res.data();
  ek::cpu::primitives_jacobian(m.d);
  ek::cpu::boundary_states(m.d);
  if (m.scheme.second_order) {
    ek::cpu::gradients(m.d);
    if (m.scheme.limiter) {
      ek::cpu::limiter(m.d);
    }
    if (m.viscous) m.viscous_gradients();
  }
  ek::cpu::sources(m.d);
  m.faces_jacobian();
}

template <class P>
void Euler<P>::assemble(std::span<const double> shift, BlockMatrix& a) const {
  const auto& m = *impl_;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  // Block k of a structure-of-arrays Jac dataset over n entities: entry (i, j) of entity e at
  // (k * 25 + i * 5 + j) * n + e.
  auto block = [](const std::vector<typename P::jac>& v, std::size_t n, std::size_t e,
                  std::size_t k) {
    std::array<double, 25> b{};
    for (std::size_t q = 0; q < 25; ++q) b[q] = static_cast<double>(v[(25 * k + q) * n + e]);
    return b;
  };
  // out += sign * J D.
  auto add_product = [](double* out, const std::array<double, 25>& j,
                        const std::array<double, 25>& dm, double sign) {
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t c = 0; c < 5; ++c) {
        double t = 0.0;
        for (std::size_t k = 0; k < 5; ++k) t += j[5 * r + k] * dm[5 * k + c];
        out[5 * r + c] += sign * t;
      }
    }
  };
  a.zero();
  for (std::size_t c = 0; c < nc; ++c) {
    double* dg = a.diagonal(c);
    for (std::size_t i = 0; i < 5; ++i) dg[6 * i] += shift[c];
  }
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(m.topo.owner[f]);
    const auto n = static_cast<std::size_t>(m.topo.neighbour[f]);
    const auto jl = block(m.face_jac, nf, f, 0);
    const auto jr = block(m.face_jac, nf, f, 1);
    const auto dl = block(m.dwdu, nc, o, 0);
    const auto dr = block(m.dwdu, nc, n, 0);
    // R_o += F, R_n -= F.
    add_product(a.diagonal(o), jl, dl, 1.0);
    add_product(a.upper(f), jr, dr, 1.0);
    add_product(a.lower(f), jl, dl, -1.0);
    add_product(a.diagonal(n), jr, dr, -1.0);
  }
  for (std::size_t f = ni; f < nf; ++f) {
    const auto o = static_cast<std::size_t>(m.topo.owner[f]);
    add_product(a.diagonal(o), block(m.bnd_jac, nf, f, 0), block(m.dwdu, nc, o, 0), 1.0);
  }
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
void Euler<P>::boundary_pressure(std::span<double> p, std::span<double> force) const {
  // The face state of the boundary flux (reconstruct in kernels/euler.py, with its fallback to
  // the cell value) and, for the force, the pressure correction of warped faces (wall_warp).
  const auto& m = *impl_;
  const auto nc = static_cast<std::size_t>(m.topo.n_cells);
  const auto nf = static_cast<std::size_t>(m.topo.n_faces);
  const auto ni = static_cast<std::size_t>(m.topo.n_interior_faces);
  const auto nb = nf - ni;
  // grad is 5 x 3 per cell, row-major: component (i, j) at (3 i + j) * n_cells + c.
  auto g = [&](std::size_t i, std::size_t j, std::size_t c) {
    return static_cast<double>(m.grad[(3 * i + j) * nc + c]);
  };
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(m.topo.owner[f]);
    double drho = 0.0, dp = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
      const double r = static_cast<double>(m.face_centroid[j * nf + f]) -
                       static_cast<double>(m.cell_centroid[j * nc + c]);
      drho += g(0, j, c) * r;
      dp += g(4, j, c) * r;
    }
    const double rho_c = static_cast<double>(m.prim[c]);
    // prim[4] is the gauge pressure p - p_ref.
    const double p_c = static_cast<double>(m.d.p_ref[0]) + static_cast<double>(m.prim[4 * nc + c]);
    const double rho_f = rho_c + static_cast<double>(m.phi[c]) * drho;
    double p_f = p_c + static_cast<double>(m.phi[4 * nc + c]) * dp;
    if (!(rho_f > 0.0 && p_f > 0.0)) p_f = p_c;
    p[f - ni] = p_f;
    if (force.empty()) continue;
    const double phi_p = static_cast<double>(m.phi[4 * nc + c]);
    for (std::size_t k = 0; k < 3; ++k) {
      // p S + phi_p M^T grad p, M row-major 3 x 3 per face: entry (j, k) at (3 j + k) * nf + f.
      double warp = 0.0;
      for (std::size_t j = 0; j < 3; ++j) {
        warp += static_cast<double>(m.face_moment[(3 * j + k) * nf + f]) * g(4, j, c);
      }
      const double area_k = static_cast<double>(m.face_normal[k * nf + f]) *
                            static_cast<double>(m.face_area_magnitude[f]);
      force[k * nb + (f - ni)] = p_f * area_k + phi_p * warp;
    }
  }
}

template class Euler<kernels::fp64>;
template class Euler<kernels::fp32_acc64>;

}  // namespace autan::physics
