#include "autan/kernels/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace autan::kernels {

namespace {

using Vec = std::array<double, 3>;

Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

std::array<double, 9> inverse3(const std::array<double, 9>& m) {
  const double c00 = m[4] * m[8] - m[5] * m[7];
  const double c01 = m[5] * m[6] - m[3] * m[8];
  const double c02 = m[3] * m[7] - m[4] * m[6];
  const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  const double s = 1.0 / det;
  return {c00 * s, (m[2] * m[7] - m[1] * m[8]) * s, (m[1] * m[5] - m[2] * m[4]) * s,
          c01 * s, (m[0] * m[8] - m[2] * m[6]) * s, (m[2] * m[3] - m[0] * m[5]) * s,
          c02 * s, (m[1] * m[6] - m[0] * m[7]) * s, (m[0] * m[4] - m[1] * m[3]) * s};
}

/// Weighted least-squares weights w_k = M^-1 d_k / |d_k|^2, M = sum_k d_k d_k^T / |d_k|^2, of one
/// cell's entries, written to out[u * stride + first + k].
void store_lsq_weights(const std::vector<Vec>& d, std::vector<double>& out, std::size_t stride,
                       std::size_t first) {
  std::array<double, 9> mat{};
  for (const Vec& dk : d) {
    const double w = 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
    for (std::size_t u = 0; u < 3; ++u) {
      for (std::size_t v = 0; v < 3; ++v) {
        mat[3 * u + v] += w * dk[u] * dk[v];
      }
    }
  }
  const auto inv = inverse3(mat);
  for (std::size_t j = 0; j < d.size(); ++j) {
    const Vec& dk = d[j];
    const double w = 1.0 / (dk[0] * dk[0] + dk[1] * dk[1] + dk[2] * dk[2]);
    for (std::size_t u = 0; u < 3; ++u) {
      out[u * stride + first + j] =
          w * (inv[3 * u] * dk[0] + inv[3 * u + 1] * dk[1] + inv[3 * u + 2] * dk[2]);
    }
  }
}

}  // namespace

Topology topology(const MeshView& mesh) noexcept {
  return {mesh.n_cells,
          mesh.n_owned_cells,
          mesh.n_faces,
          mesh.n_interior_faces,
          mesh.n_owned_interior_faces,
          mesh.n_owned_faces,
          mesh.cell_face_offsets[mesh.n_cells],
          mesh.owner,
          mesh.neighbour,
          mesh.cell_face_offsets,
          mesh.cell_faces,
          nullptr};
}

std::vector<offset_t> incidence_twins(const Mesh& mesh) {
  const auto off = mesh.cell_face_offsets();
  const auto cf = mesh.cell_faces();
  const auto own = mesh.owner();
  const auto nei = mesh.neighbour();
  std::vector<offset_t> twin(cf.size(), -1);
  for (std::size_t c = 0; c + 1 < off.size(); ++c) {
    for (auto k = off[c]; k < off[c + 1]; ++k) {
      const index_t f = cf[static_cast<std::size_t>(k)];
      const auto fi = static_cast<std::size_t>(f);
      if (fi >= nei.size()) {
        continue;
      }
      const auto other =
          static_cast<std::size_t>(static_cast<std::size_t>(own[fi]) == c ? nei[fi] : own[fi]);
      // Faces of a cell are listed in increasing order (mesh contract).
      const auto first = cf.begin() + off[other];
      const auto last = cf.begin() + off[other + 1];
      twin[static_cast<std::size_t>(k)] = std::lower_bound(first, last, f) - cf.begin();
    }
  }
  return twin;
}

PackedGeometry pack_geometry(const Mesh& mesh, const Geometry& g) {
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  PackedGeometry p;
  p.face_area.resize(3 * nf);
  p.face_centroid.resize(3 * nf);
  p.face_moment.assign(9 * nf, 0.0);
  p.cell_centroid.resize(3 * nc);
  p.cell_volume = g.cell_volume;
  for (std::size_t f = 0; f < nf; ++f) {
    p.face_area[f] = g.face_area_x[f];
    p.face_area[nf + f] = g.face_area_y[f];
    p.face_area[2 * nf + f] = g.face_area_z[f];
    p.face_centroid[f] = g.face_centroid_x[f];
    p.face_centroid[nf + f] = g.face_centroid_y[f];
    p.face_centroid[2 * nf + f] = g.face_centroid_z[f];
  }
  for (std::size_t c = 0; c < nc; ++c) {
    p.cell_centroid[c] = g.cell_centroid_x[c];
    p.cell_centroid[nc + c] = g.cell_centroid_y[c];
    p.cell_centroid[2 * nc + c] = g.cell_centroid_z[c];
  }
  p.face_area_magnitude = g.face_area;
  p.face_normal.resize(3 * nf);
  p.face_r_left.resize(3 * nf);
  p.face_r_right.assign(3 * nf, 0.0);
  {
    const auto own = mesh.owner();
    const auto nei = mesh.neighbour();
    for (std::size_t f = 0; f < nf; ++f) {
      const auto l = static_cast<std::size_t>(own[f]);
      for (std::size_t u = 0; u < 3; ++u) {
        p.face_normal[u * nf + f] = p.face_area[u * nf + f] / g.face_area[f];
        p.face_r_left[u * nf + f] = p.face_centroid[u * nf + f] - p.cell_centroid[u * nc + l];
        if (f < nei.size()) {
          const auto r = static_cast<std::size_t>(nei[f]);
          p.face_r_right[u * nf + f] = p.face_centroid[u * nf + f] - p.cell_centroid[u * nc + r];
        }
      }
    }
  }

  // Face moments over the same fan as the core's geometry (triangles (m, p_i, p_i+1)).
  const auto x = mesh.x(), y = mesh.y(), z = mesh.z();
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    const auto nodes = mesh.face_nodes(f);
    Vec m{0.0, 0.0, 0.0};
    for (const index_t node : nodes) {
      const auto n = static_cast<std::size_t>(node);
      m = {m[0] + x[n], m[1] + y[n], m[2] + z[n]};
    }
    const auto k = static_cast<double>(nodes.size());
    m = {m[0] / k, m[1] / k, m[2] / k};
    const auto fi = static_cast<std::size_t>(f);
    const Vec xf{g.face_centroid_x[fi], g.face_centroid_y[fi], g.face_centroid_z[fi]};
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const auto i0 = static_cast<std::size_t>(nodes[i]);
      const auto i1 = static_cast<std::size_t>(nodes[(i + 1) % nodes.size()]);
      const Vec a{x[i0], y[i0], z[i0]};
      const Vec b{x[i1], y[i1], z[i1]};
      const Vec s = cross(sub(a, m), sub(b, m));
      const Vec c{(m[0] + a[0] + b[0]) / 3.0, (m[1] + a[1] + b[1]) / 3.0,
                  (m[2] + a[2] + b[2]) / 3.0};
      const Vec r = sub(c, xf);
      for (std::size_t u = 0; u < 3; ++u) {
        for (std::size_t v = 0; v < 3; ++v) {
          p.face_moment[(3 * u + v) * nf + fi] += 0.5 * r[u] * s[v];
        }
      }
    }
  }

  const auto offsets = mesh.cell_face_offsets();
  const auto cell_faces = mesh.cell_faces();
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  const auto ni = static_cast<std::size_t>(offsets.back());
  p.lsq_weight.assign(3 * ni, 0.0);
  std::vector<Vec> d;
  for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
    const auto ci = static_cast<std::size_t>(c);
    const Vec xc{p.cell_centroid[ci], p.cell_centroid[nc + ci], p.cell_centroid[2 * nc + ci]};
    d.clear();
    for (auto k = offsets[ci]; k < offsets[ci + 1]; ++k) {
      const auto f = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(k)]);
      Vec xo;
      if (f < neighbour.size()) {
        const auto o = static_cast<std::size_t>(owner[f] == c ? neighbour[f] : owner[f]);
        xo = {p.cell_centroid[o], p.cell_centroid[nc + o], p.cell_centroid[2 * nc + o]};
      } else {
        xo = {p.face_centroid[f], p.face_centroid[nf + f], p.face_centroid[2 * nf + f]};
      }
      d.push_back(sub(xo, xc));
    }
    store_lsq_weights(d, p.lsq_weight, ni, static_cast<std::size_t>(offsets[ci]));
  }
  return p;
}

CellStencil cell_stencil(const Mesh& mesh, const PackedGeometry& g, StencilKind kind) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto n_owned = static_cast<std::size_t>(mesh.n_owned_cells());
  const auto off = mesh.cell_face_offsets();
  const auto cf = mesh.cell_faces();
  const auto own = mesh.owner();
  const auto nei = mesh.neighbour();
  const std::size_t nint = nei.size();
  auto faces_of = [&](std::size_t c) {
    return cf.subspan(static_cast<std::size_t>(off[c]),
                      static_cast<std::size_t>(off[c + 1] - off[c]));
  };
  auto across = [&](std::size_t c, std::size_t f) {
    return static_cast<std::size_t>(static_cast<std::size_t>(own[f]) == c ? nei[f] : own[f]);
  };

  // Cells and boundary faces touching each node (vertex stencil).
  std::vector<std::vector<index_t>> node_cells, node_bfaces;
  if (kind == StencilKind::vertex) {
    node_cells.resize(static_cast<std::size_t>(mesh.n_points()));
    node_bfaces.resize(node_cells.size());
    for (std::size_t f = 0; f < nf; ++f) {
      for (const index_t n : mesh.face_nodes(static_cast<index_t>(f))) {
        auto& cells = node_cells[static_cast<std::size_t>(n)];
        cells.push_back(own[f]);
        if (f < nint) {
          cells.push_back(nei[f]);
        } else {
          node_bfaces[static_cast<std::size_t>(n)].push_back(static_cast<index_t>(f));
        }
      }
    }
  }

  CellStencil st;
  st.offsets.assign(nc + 1, 0);
  std::vector<index_t> cells, bfaces;
  for (std::size_t c = 0; c < nc; ++c) {
    cells.clear();
    bfaces.clear();
    if (kind == StencilKind::vertex) {
      for (const index_t f : faces_of(c)) {
        for (const index_t n : mesh.face_nodes(f)) {
          const auto& around = node_cells[static_cast<std::size_t>(n)];
          cells.insert(cells.end(), around.begin(), around.end());
          const auto& bound = node_bfaces[static_cast<std::size_t>(n)];
          bfaces.insert(bfaces.end(), bound.begin(), bound.end());
        }
      }
    } else {
      for (const index_t f : faces_of(c)) {
        const auto fi = static_cast<std::size_t>(f);
        if (fi >= nint) {
          bfaces.push_back(f);
          continue;
        }
        const std::size_t n = across(c, fi);
        cells.push_back(static_cast<index_t>(n));
        for (const index_t f2 : faces_of(n)) {
          const auto f2i = static_cast<std::size_t>(f2);
          if (f2i >= nint) {
            bfaces.push_back(f2);
          } else {
            cells.push_back(static_cast<index_t>(across(n, f2i)));
          }
        }
      }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    cells.erase(std::remove(cells.begin(), cells.end(), static_cast<index_t>(c)), cells.end());
    std::sort(bfaces.begin(), bfaces.end());
    bfaces.erase(std::unique(bfaces.begin(), bfaces.end()), bfaces.end());
    st.entries.insert(st.entries.end(), cells.begin(), cells.end());
    for (const index_t f : bfaces) {
      st.entries.push_back(-f - 1);
    }
    st.offsets[c + 1] = static_cast<offset_t>(st.entries.size());
  }

  const std::size_t ns = st.entries.size();
  st.twin.assign(ns, -1);
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = st.offsets[c]; k < st.offsets[c + 1]; ++k) {
      const index_t e = st.entries[static_cast<std::size_t>(k)];
      if (e < 0) {
        continue;
      }
      const auto n = static_cast<std::size_t>(e);
      const auto first = st.entries.begin() + st.offsets[n];
      const auto last = st.entries.begin() + st.offsets[n + 1];
      const auto cells_end = std::find_if(first, last, [](index_t x) { return x < 0; });
      const auto it = std::lower_bound(first, cells_end, static_cast<index_t>(c));
      if (it != cells_end && *it == static_cast<index_t>(c)) {
        st.twin[static_cast<std::size_t>(k)] = it - st.entries.begin();
      }
    }
  }

  st.lsq_weight.assign(3 * ns, 0.0);
  std::vector<Vec> d;
  for (std::size_t c = 0; c < n_owned; ++c) {
    const Vec xc{g.cell_centroid[c], g.cell_centroid[nc + c], g.cell_centroid[2 * nc + c]};
    d.clear();
    for (auto k = st.offsets[c]; k < st.offsets[c + 1]; ++k) {
      const index_t e = st.entries[static_cast<std::size_t>(k)];
      Vec xo;
      if (e >= 0) {
        const auto o = static_cast<std::size_t>(e);
        xo = {g.cell_centroid[o], g.cell_centroid[nc + o], g.cell_centroid[2 * nc + o]};
      } else {
        const auto f = static_cast<std::size_t>(-e - 1);
        xo = {g.face_centroid[f], g.face_centroid[nf + f], g.face_centroid[2 * nf + f]};
      }
      d.push_back(sub(xo, xc));
    }
    store_lsq_weights(d, st.lsq_weight, ns, static_cast<std::size_t>(st.offsets[c]));
  }
  return st;
}

}  // namespace autan::kernels
