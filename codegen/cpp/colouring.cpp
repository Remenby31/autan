#include "autan/kernels/colouring.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace autan::kernels {

HostColouring colour_faces(const Topology& topo, index_t first, index_t last) {
  // Greedy, faces in order: a face takes the least colour none of the faces already coloured
  // at its cells has. stamp[k] == f marks colour k taken for face f: memory in the faces and
  // colours, however many faces a cell has.
  std::vector<index_t> colour(static_cast<std::size_t>(last - first));
  std::vector<index_t> stamp;
  index_t n_colours = 0;
  auto mark = [&](index_t cell, index_t f) {
    const auto c = static_cast<std::size_t>(cell);
    const offset_t begin = topo.cell_face_offsets[c];
    // CSR, or the sliced layout (Topology::incidence_stride).
    const bool sliced = topo.incidence_length != nullptr;
    const offset_t count = sliced ? topo.incidence_length[c]
                                  : topo.cell_face_offsets[c + 1] - topo.cell_face_offsets[c];
    const offset_t step = sliced ? topo.incidence_stride : 1;
    for (offset_t j = 0; j < count; ++j) {
      const index_t g = topo.cell_faces[begin + j * step];
      if (g >= first && g < f) {
        stamp[static_cast<std::size_t>(colour[static_cast<std::size_t>(g - first)])] = f;
      }
    }
  };
  for (index_t f = first; f < last; ++f) {
    const index_t l = topo.owner[f];
    const index_t r = f < topo.n_interior_faces ? topo.neighbour[f] : -1;
    stamp.resize(static_cast<std::size_t>(n_colours) + 1, -1);
    mark(l, f);
    if (r >= 0) {
      mark(r, f);
    }
    index_t k = 0;
    while (stamp[static_cast<std::size_t>(k)] == f) {
      ++k;
    }
    colour[static_cast<std::size_t>(f - first)] = k;
    n_colours = std::max(n_colours, k + 1);
  }
  HostColouring out;
  out.offsets.assign(static_cast<std::size_t>(n_colours) + 1, 0);
  for (const index_t k : colour) {
    ++out.offsets[static_cast<std::size_t>(k) + 1];
  }
  for (std::size_t k = 0; k < static_cast<std::size_t>(n_colours); ++k) {
    out.offsets[k + 1] += out.offsets[k];
  }
  out.faces.resize(colour.size());
  std::vector<index_t> next(out.offsets.begin(), out.offsets.end() - 1);
  for (index_t f = first; f < last; ++f) {
    const auto k = static_cast<std::size_t>(colour[static_cast<std::size_t>(f - first)]);
    out.faces[static_cast<std::size_t>(next[k]++)] = f;
  }
  return out;
}

}  // namespace autan::kernels
