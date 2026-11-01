#include "autan/kernels/colouring.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace autan::kernels {

HostColouring colour_faces(const Topology& topo, index_t first, index_t last) {
  // used[c] is a bit set of the colours already incident to cell c (up to 64 per word).
  std::vector<std::vector<std::uint64_t>> used(static_cast<std::size_t>(topo.n_cells));
  std::vector<index_t> colour(static_cast<std::size_t>(last - first));
  index_t n_colours = 0;
  auto taken = [&](index_t cell, index_t k) {
    const auto& u = used[static_cast<std::size_t>(cell)];
    const auto w = static_cast<std::size_t>(k / 64);
    return w < u.size() && ((u[w] >> (k % 64)) & 1U) != 0;
  };
  auto take = [&](index_t cell, index_t k) {
    auto& u = used[static_cast<std::size_t>(cell)];
    const auto w = static_cast<std::size_t>(k / 64);
    if (u.size() <= w) {
      u.resize(w + 1, 0);
    }
    u[w] |= std::uint64_t{1} << (k % 64);
  };
  for (index_t f = first; f < last; ++f) {
    const index_t l = topo.owner[f];
    const index_t r = f < topo.n_interior_faces ? topo.neighbour[f] : -1;
    index_t k = 0;
    while (taken(l, k) || (r >= 0 && taken(r, k))) {
      ++k;
    }
    take(l, k);
    if (r >= 0) {
      take(r, k);
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
