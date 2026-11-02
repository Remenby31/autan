#include "autan/solvers/galerkin_lists.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace autan::solvers::gpu {

namespace {

// Inverse of a device order: the device index of each host node.
std::vector<index_t> ranks(std::span<const index_t> order, std::size_t n, const char* what) {
  if (order.size() != n) {
    throw std::invalid_argument(std::string("galerkin_lists: ") + what + " order has " +
                                std::to_string(order.size()) + " nodes, the level " +
                                std::to_string(n));
  }
  std::vector<index_t> rank(n, -1);
  for (std::size_t j = 0; j < n; ++j) {
    const auto h = static_cast<std::size_t>(order[j]);
    if (h >= n || rank[h] != -1) {
      throw std::invalid_argument(std::string("galerkin_lists: ") + what +
                                  " order is not a permutation");
    }
    rank[h] = static_cast<index_t>(j);
  }
  return rank;
}

void check_slots(const EdgeSlots& s, std::size_t n_edges, offset_t n_slots, const char* what) {
  if (s.owner.size() != n_edges || s.neighbour.size() != n_edges) {
    throw std::invalid_argument(std::string("galerkin_lists: ") + what +
                                " slots do not cover the level's edges");
  }
  for (std::size_t e = 0; e < n_edges; ++e) {
    for (const offset_t p : {s.owner[e], s.neighbour[e]}) {
      if (p < 0 || (n_slots >= 0 && p >= n_slots)) {
        throw std::invalid_argument(std::string("galerkin_lists: ") + what + " slot " +
                                    std::to_string(p) + " out of range");
      }
    }
  }
}

}  // namespace

GalerkinLists galerkin_lists(const Level& fine, const Level& coarse,
                             std::span<const index_t> fine_order,
                             std::span<const index_t> coarse_order, const EdgeSlots& fine_slots,
                             const EdgeSlots& coarse_slots, offset_t n_coarse_slots) {
  const std::size_t n = fine.n_nodes, nc = coarse.n_nodes;
  const std::size_t n_edges = fine.owner.size(), nc_edges = coarse.owner.size();
  if (fine.coarse_node.size() != n || fine.coarse_edge.size() != n_edges ||
      fine.coarse_flipped.size() != n_edges) {
    throw std::invalid_argument("galerkin_lists: the fine level has no coarser level");
  }
  const std::vector<index_t> fine_rank = ranks(fine_order, n, "fine");
  const std::vector<index_t> coarse_rank = ranks(coarse_order, nc, "coarse");
  check_slots(fine_slots, n_edges, -1, "fine");
  check_slots(coarse_slots, nc_edges, n_coarse_slots, "coarse");

  GalerkinLists g;
  // Diagonal: children's diagonals first (increasing host fine index), then internal edges.
  std::vector<offset_t> count(nc + 1, 0);
  for (std::size_t h = 0; h < n; ++h) {
    const auto H = static_cast<std::size_t>(fine.coarse_node[h]);
    if (H >= nc) throw std::invalid_argument("galerkin_lists: coarse node out of range");
    ++count[static_cast<std::size_t>(coarse_rank[H]) + 1];
  }
  for (std::size_t e = 0; e < n_edges; ++e) {
    if (fine.coarse_edge[e] >= 0) continue;
    const auto H =
        static_cast<std::size_t>(fine.coarse_node[static_cast<std::size_t>(fine.owner[e])]);
    count[static_cast<std::size_t>(coarse_rank[H]) + 1] += 2;
  }
  for (std::size_t J = 0; J < nc; ++J) count[J + 1] += count[J];
  g.diag_offsets = count;
  g.diag.resize(static_cast<std::size_t>(count.back()));
  std::vector<offset_t> fill(count.begin(), count.end() - 1);
  auto push_diag = [&](std::size_t H, std::int64_t term) {
    g.diag[static_cast<std::size_t>(fill[static_cast<std::size_t>(coarse_rank[H])]++)] = term;
  };
  for (std::size_t h = 0; h < n; ++h) {
    push_diag(static_cast<std::size_t>(fine.coarse_node[h]), fine_rank[h]);
  }
  for (std::size_t e = 0; e < n_edges; ++e) {
    if (fine.coarse_edge[e] >= 0) continue;
    const auto H =
        static_cast<std::size_t>(fine.coarse_node[static_cast<std::size_t>(fine.owner[e])]);
    push_diag(H, -(fine_slots.owner[e] + 1));
    push_diag(H, -(fine_slots.neighbour[e] + 1));
  }

  // Off-diagonal: per coarse slot, the fine blocks of its coarse edge in increasing fine edge.
  // A flipped fine edge's owner lies in the coarse neighbour, so its blocks swap roles.
  const auto slots = static_cast<std::size_t>(n_coarse_slots);
  std::vector<offset_t> off_count(slots + 1, 0);
  for (std::size_t e = 0; e < n_edges; ++e) {
    const index_t E = fine.coarse_edge[e];
    if (E < 0) continue;
    if (static_cast<std::size_t>(E) >= nc_edges) {
      throw std::invalid_argument("galerkin_lists: coarse edge out of range");
    }
    ++off_count[static_cast<std::size_t>(coarse_slots.owner[static_cast<std::size_t>(E)]) + 1];
    ++off_count[static_cast<std::size_t>(coarse_slots.neighbour[static_cast<std::size_t>(E)]) + 1];
  }
  for (std::size_t s = 0; s < slots; ++s) off_count[s + 1] += off_count[s];
  g.off_offsets = off_count;
  g.off.resize(static_cast<std::size_t>(off_count.back()));
  std::vector<offset_t> off_fill(off_count.begin(), off_count.end() - 1);
  for (std::size_t e = 0; e < n_edges; ++e) {
    const index_t E = fine.coarse_edge[e];
    if (E < 0) continue;
    const bool flip = fine.coarse_flipped[e] != 0;
    const auto Es = static_cast<std::size_t>(E);
    g.off[static_cast<std::size_t>(off_fill[static_cast<std::size_t>(coarse_slots.owner[Es])]++)] =
        flip ? fine_slots.neighbour[e] : fine_slots.owner[e];
    g.off[static_cast<std::size_t>(
        off_fill[static_cast<std::size_t>(coarse_slots.neighbour[Es])]++)] =
        flip ? fine_slots.owner[e] : fine_slots.neighbour[e];
  }
  return g;
}

}  // namespace autan::solvers::gpu
