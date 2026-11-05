#include "autan/core/lines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace autan {

Lines smoother_lines(const Mesh& mesh, const Geometry& geometry, double ratio) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw std::invalid_argument("lines of a mesh with ghost cells");
  }
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  const auto zone = mesh.cell_zone();
  const auto& gm = geometry;
  std::vector<double> weight(ni);
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(owner[f]);
    const auto nb = static_cast<std::size_t>(neighbour[f]);
    const double dx = gm.cell_centroid_x[nb] - gm.cell_centroid_x[o];
    const double dy = gm.cell_centroid_y[nb] - gm.cell_centroid_y[o];
    const double dz = gm.cell_centroid_z[nb] - gm.cell_centroid_z[o];
    weight[f] = gm.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  // Each cell's interior faces in increasing face index, CSR.
  std::vector<offset_t> first(n + 1, 0);
  for (std::size_t f = 0; f < ni; ++f) {
    ++first[static_cast<std::size_t>(owner[f]) + 1];
    ++first[static_cast<std::size_t>(neighbour[f]) + 1];
  }
  for (std::size_t c = 0; c < n; ++c) first[c + 1] += first[c];
  std::vector<std::size_t> faces(static_cast<std::size_t>(first.back()));
  {
    std::vector<offset_t> fill(first.begin(), first.end() - 1);
    for (std::size_t f = 0; f < ni; ++f) {
      faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(owner[f])]++)] = f;
      faces[static_cast<std::size_t>(fill[static_cast<std::size_t>(neighbour[f])]++)] = f;
    }
  }
  // Each neighbour of a cell with the sum of the couplings of the faces they share, in the order
  // the neighbours first appear.
  const auto couplings = [&](std::size_t c) {
    std::vector<std::pair<std::size_t, double>> out;
    for (auto k = first[c]; k < first[c + 1]; ++k) {
      const std::size_t f = faces[static_cast<std::size_t>(k)];
      const auto o = static_cast<std::size_t>(owner[f]);
      const std::size_t m = o == c ? static_cast<std::size_t>(neighbour[f]) : o;
      const auto it = std::ranges::find_if(out, [&](const auto& p) { return p.first == m; });
      if (it == out.end()) {
        out.emplace_back(m, weight[f]);
      } else {
        it->second += weight[f];
      }
    }
    return out;
  };
  // Weakest and strongest coupling of each cell. Boundary faces join in only for a cell with at
  // most two neighbours, whose anisotropy they alone show (a column of cells), as A_f over twice
  // the distance to the face (the distance to the mirror cell); the side planes of a
  // one-cell-thick mesh, far weaker than any neighbour, would otherwise make every cell
  // anisotropic.
  std::vector<double> weakest(n, std::numeric_limits<double>::infinity());
  std::vector<double> strongest(n, 0.0);
  std::vector<std::size_t> neighbours(n, 0);
  std::vector<char> at_boundary(n, 0);
  for (std::size_t c = 0; c < n; ++c) {
    for (const auto& [m, w] : couplings(c)) {
      weakest[c] = std::min(weakest[c], w);
      strongest[c] = std::max(strongest[c], w);
      ++neighbours[c];
    }
  }
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(owner[f]);
    at_boundary[c] = 1;
    if (neighbours[c] > 2) continue;
    const double dx = gm.face_centroid_x[f] - gm.cell_centroid_x[c];
    const double dy = gm.face_centroid_y[f] - gm.cell_centroid_y[c];
    const double dz = gm.face_centroid_z[f] - gm.cell_centroid_z[c];
    const double w = gm.face_area[f] / (2.0 * std::sqrt(dx * dx + dy * dy + dz * dz));
    weakest[c] = std::min(weakest[c], w);
    strongest[c] = std::max(strongest[c], w);
  }
  // Seeds: the anisotropic cells, those on the boundary first, then by decreasing anisotropy.
  std::vector<std::size_t> seeds;
  for (std::size_t c = 0; c < n; ++c) {
    if (strongest[c] >= ratio * weakest[c]) seeds.push_back(c);
  }
  std::ranges::stable_sort(seeds, [&](std::size_t a, std::size_t b) {
    if (at_boundary[a] != at_boundary[b]) return at_boundary[a] != 0;
    return strongest[a] / weakest[a] > strongest[b] / weakest[b];
  });
  std::vector<char> in_line(n, 0);
  const auto grow = [&](std::size_t start, std::vector<index_t>& out) {
    std::size_t cur = start;
    for (;;) {
      std::size_t best = n;
      double best_w = 0.0;
      for (const auto& [m, w] : couplings(cur)) {
        if (in_line[m] != 0 || zone[m] != zone[cur]) continue;
        if (w < ratio * weakest[cur] || w < ratio * weakest[m]) continue;
        if (w > best_w || (w == best_w && m < best)) {
          best_w = w;
          best = m;
        }
      }
      if (best == n) return;
      in_line[best] = 1;
      out.push_back(static_cast<index_t>(best));
      cur = best;
    }
  };
  Lines lines;
  for (const std::size_t s : seeds) {
    if (in_line[s] != 0) continue;
    in_line[s] = 1;
    std::vector<index_t> forward{static_cast<index_t>(s)}, backward;
    grow(s, forward);
    grow(s, backward);
    if (forward.size() + backward.size() < 2) {
      in_line[s] = 0;
      continue;
    }
    std::vector<index_t> line(backward.rbegin(), backward.rend());
    line.insert(line.end(), forward.begin(), forward.end());
    // A line touching the boundary at one end starts there, so that neighbouring wall lines run
    // the same way.
    if (at_boundary[static_cast<std::size_t>(line.front())] == 0 &&
        at_boundary[static_cast<std::size_t>(line.back())] != 0) {
      std::ranges::reverse(line);
    }
    lines.cells.insert(lines.cells.end(), line.begin(), line.end());
    lines.offsets.push_back(static_cast<offset_t>(lines.cells.size()));
  }
  return lines;
}

std::size_t cells_in_lines(const Lines& lines, std::size_t min_length) {
  std::size_t count = 0;
  for (std::size_t l = 0; l + 1 < lines.offsets.size(); ++l) {
    const auto length = static_cast<std::size_t>(lines.offsets[l + 1] - lines.offsets[l]);
    if (length >= min_length) count += length;
  }
  return count;
}

}  // namespace autan
