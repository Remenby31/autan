#include "autan/solvers/device_order.hpp"

#include <algorithm>
#include <cstddef>

namespace autan::solvers::gpu {

std::vector<std::vector<index_t>> device_orders(std::span<const Level> levels) {
  std::vector<std::vector<index_t>> orders;
  if (levels.empty()) return orders;
  std::vector<index_t> order(levels[0].n_nodes), rank(levels[0].n_nodes);
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = rank[i] = static_cast<index_t>(i);
  orders.push_back(order);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    const Level& f = levels[l];
    const std::size_t n = f.n_nodes, nc = levels[l + 1].n_nodes;
    std::vector<index_t> least(nc, static_cast<index_t>(n));
    for (std::size_t h = 0; h < n; ++h) {
      index_t& m = least[static_cast<std::size_t>(f.coarse_node[h])];
      m = std::min(m, rank[h]);
    }
    std::vector<index_t> coarse(nc), coarse_rank(nc);
    for (std::size_t H = 0; H < nc; ++H) coarse[H] = static_cast<index_t>(H);
    std::sort(coarse.begin(), coarse.end(), [&](index_t a, index_t b) {
      return least[static_cast<std::size_t>(a)] < least[static_cast<std::size_t>(b)];
    });
    for (std::size_t J = 0; J < nc; ++J) {
      coarse_rank[static_cast<std::size_t>(coarse[J])] = static_cast<index_t>(J);
    }
    orders.push_back(coarse);
    rank = std::move(coarse_rank);
  }
  return orders;
}

}  // namespace autan::solvers::gpu
