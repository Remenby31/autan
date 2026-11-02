#include "autan/kernels/topology.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace autan::kernels {

SlicedRows slice_rows(std::span<const offset_t> offsets, int slice) {
  if (slice < 1) {
    throw std::invalid_argument("E1101: slice must be at least 1");
  }
  const std::size_t nc = offsets.size() - 1;
  const auto w = static_cast<std::size_t>(slice);
  SlicedRows out;
  out.length.resize(nc);
  out.position.resize(static_cast<std::size_t>(offsets[nc]));
  out.starts.resize(nc + 1);
  offset_t start = 0;
  for (std::size_t s = 0; s * w < nc; ++s) {
    offset_t longest = 0;
    for (std::size_t l = 0; l < w && s * w + l < nc; ++l) {
      const std::size_t c = s * w + l;
      longest = std::max(longest, offsets[c + 1] - offsets[c]);
    }
    for (std::size_t l = 0; l < w && s * w + l < nc; ++l) {
      const std::size_t c = s * w + l;
      out.starts[c] = start + static_cast<offset_t>(l);
      out.length[c] = static_cast<index_t>(offsets[c + 1] - offsets[c]);
      for (offset_t j = 0; j < out.length[c]; ++j) {
        out.position[static_cast<std::size_t>(offsets[c] + j)] =
            out.starts[c] + j * static_cast<offset_t>(slice);
      }
    }
    start += longest * static_cast<offset_t>(slice);
  }
  out.starts[nc] = start;
  return out;
}

SlicedIncidences sliced_incidences(const Topology& csr, int slice) {
  const auto nc = static_cast<std::size_t>(csr.n_cells);
  auto rows = slice_rows(std::span(csr.cell_face_offsets, nc + 1), slice);
  SlicedIncidences out;
  out.stride = slice;
  const auto n = static_cast<std::size_t>(rows.starts[nc]);
  out.cell_faces.assign(n, 0);
  out.cell.assign(n, -1);
  out.twin.assign(n, -1);
  for (std::size_t c = 0; c < nc; ++c) {
    for (auto k = csr.cell_face_offsets[c]; k < csr.cell_face_offsets[c + 1]; ++k) {
      const auto p = static_cast<std::size_t>(rows.position[static_cast<std::size_t>(k)]);
      const index_t f = csr.cell_faces[k];
      out.cell_faces[p] = f;
      if (f < csr.n_interior_faces) {
        out.cell[p] = csr.owner[f] == static_cast<index_t>(c) ? csr.neighbour[f] : csr.owner[f];
      }
      const offset_t t = csr.incidence_twin == nullptr ? -1 : csr.incidence_twin[k];
      out.twin[p] = t < 0 ? -1 : rows.position[static_cast<std::size_t>(t)];
    }
  }
  out.cell_face_offsets = std::move(rows.starts);
  out.length = std::move(rows.length);
  out.position = std::move(rows.position);
  return out;
}

std::vector<double> sliced_values(const SlicedIncidences& s, const std::vector<double>& csr,
                                  int comps) {
  const std::size_t ne = s.position.size();
  const std::size_t n = s.cell_faces.size();
  const auto nk = static_cast<std::size_t>(comps);
  if (csr.size() != nk * ne) {
    throw std::invalid_argument("E1101: per-incidence values of the wrong size");
  }
  std::vector<double> out(nk * n, 0.0);
  for (std::size_t k = 0; k < ne; ++k) {
    for (std::size_t i = 0; i < nk; ++i) {
      out[i * n + static_cast<std::size_t>(s.position[k])] = csr[i * ne + k];
    }
  }
  return out;
}

Topology GraphTopology::topology() const {
  Topology t{};
  const auto ne = static_cast<index_t>(owner.size());
  t.n_cells = t.n_owned_cells = n_nodes;
  t.n_faces = t.n_interior_faces = t.n_owned_interior_faces = t.n_owned_faces = ne;
  t.n_incidences = static_cast<offset_t>(cell_faces.size());
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = cell_face_offsets.data();
  t.cell_faces = cell_faces.data();
  if (stride != 1) {
    t.incidence_stride = stride;
    t.incidence_length = length.data();
    t.incidence_cell = cell.data();
  }
  return t;
}

GraphTopology graph_topology(index_t n_nodes, std::span<const index_t> lo,
                             std::span<const index_t> hi, int slice) {
  if (lo.size() != hi.size()) {
    throw std::invalid_argument("E1101: edge lists of different sizes");
  }
  GraphTopology g;
  g.n_nodes = n_nodes;
  g.owner.assign(lo.begin(), lo.end());
  g.neighbour.assign(hi.begin(), hi.end());
  const auto nn = static_cast<std::size_t>(n_nodes);
  g.cell_face_offsets.assign(nn + 1, 0);
  for (std::size_t e = 0; e < lo.size(); ++e) {
    if (lo[e] < 0 || lo[e] >= hi[e] || hi[e] >= n_nodes) {
      throw std::invalid_argument("E1101: edge " + std::to_string(e) +
                                  " is not a pair lo < hi < n_nodes");
    }
    ++g.cell_face_offsets[static_cast<std::size_t>(lo[e]) + 1];
    ++g.cell_face_offsets[static_cast<std::size_t>(hi[e]) + 1];
  }
  for (std::size_t i = 0; i < nn; ++i) g.cell_face_offsets[i + 1] += g.cell_face_offsets[i];
  g.cell_faces.resize(static_cast<std::size_t>(g.cell_face_offsets[nn]));
  std::vector<offset_t> fill(g.cell_face_offsets.begin(), g.cell_face_offsets.end() - 1);
  for (std::size_t e = 0; e < lo.size(); ++e) {
    for (const index_t node : {lo[e], hi[e]}) {
      g.cell_faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(node)]++)] =
          static_cast<index_t>(e);
    }
  }
  if (slice > 0) {
    auto s = sliced_incidences(g.topology(), slice);
    g.cell_face_offsets = std::move(s.cell_face_offsets);
    g.cell_faces = std::move(s.cell_faces);
    g.length = std::move(s.length);
    g.cell = std::move(s.cell);
    g.stride = s.stride;
  }
  return g;
}

ColourOrder colour_order(std::span<const std::int32_t> colours, std::int32_t n_colours,
                         index_t n_owned) {
  ColourOrder out;
  out.offsets.assign(static_cast<std::size_t>(n_colours) + 1, 0);
  const auto no = static_cast<std::size_t>(n_owned);
  for (std::size_t c = 0; c < no; ++c) {
    if (colours[c] < 0 || colours[c] >= n_colours) {
      throw std::invalid_argument("E1101: colour of row " + std::to_string(c) +
                                  " outside [0, n_colours)");
    }
    ++out.offsets[static_cast<std::size_t>(colours[c]) + 1];
  }
  for (std::size_t k = 0; k < static_cast<std::size_t>(n_colours); ++k) {
    out.offsets[k + 1] += out.offsets[k];
  }
  out.new_to_old.resize(colours.size());
  std::vector<index_t> fill(out.offsets.begin(), out.offsets.end() - 1);
  for (std::size_t c = 0; c < no; ++c) {
    out.new_to_old[static_cast<std::size_t>(fill[static_cast<std::size_t>(colours[c])]++)] =
        static_cast<index_t>(c);
  }
  for (std::size_t c = no; c < colours.size(); ++c) {
    out.new_to_old[c] = static_cast<index_t>(c);
  }
  return out;
}

}  // namespace autan::kernels
