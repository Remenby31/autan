#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/partition/partition.hpp"

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

[[noreturn]] void invalid(const std::string& why) {
  throw Error(ErrorCode::partition_invalid_request, why);
}

// The nodes and edges part q holds of a level: owned nodes, ghost nodes,
// and the edges with an end owned or a remote child.
struct Share {
  std::vector<std::size_t> owned, ghosts, edges;
};

class Level {
 public:
  Level(const GraphLevel& level, std::span<const std::int32_t> parts, std::int32_t n_parts,
        std::span<const index_t> coarse_node, std::span<const std::int32_t> coarse_parts)
      : level_(level), parts_(parts), coarse_node_(coarse_node), coarse_parts_(coarse_parts) {
    const std::size_t n = level.n_nodes;
    if (level.global_id.size() != n || parts.size() != n ||
        level.owner.size() != level.neighbour.size())
      invalid("the graph level's arrays do not match its nodes and edges");
    if (!coarse_node.empty() && coarse_node.size() != n) invalid("coarse_node has the wrong size");
    for (const std::int32_t p : parts) {
      if (p < 0 || p >= n_parts) invalid("part " + std::to_string(p) + " out of range");
    }
    adjacent_.resize(n);
    for (std::size_t e = 0; e < level.owner.size(); ++e) {
      const index_t a = level.owner[e], b = level.neighbour[e];
      if (a < 0 || b < 0 || at(a) >= n || at(b) >= n) invalid("an edge's node is out of range");
      adjacent_[at(a)].push_back(e);
      adjacent_[at(b)].push_back(e);
    }
    for (const index_t c : coarse_node) {
      if (c < 0 || at(c) >= coarse_parts.size()) invalid("a coarse node is out of range");
    }
    for (const std::int32_t p : coarse_parts) {
      if (p < 0 || p >= n_parts) invalid("coarse part " + std::to_string(p) + " out of range");
    }
  }

  [[nodiscard]] std::int32_t edge_part(std::size_t e) const { return parts_[at(level_.owner[e])]; }

  [[nodiscard]] Share share(std::int32_t q) const {
    Share s;
    std::set<std::size_t> core;  // owned and remote children
    for (std::size_t v = 0; v < level_.n_nodes; ++v) {
      const bool owned = parts_[v] == q;
      const bool child = !owned && !coarse_node_.empty() && coarse_parts_[at(coarse_node_[v])] == q;
      if (owned) s.owned.push_back(v);
      if (owned || child) core.insert(v);
    }
    std::set<std::size_t> ghosts, edges;
    for (const std::size_t v : core) {
      if (parts_[v] != q) ghosts.insert(v);
      for (const std::size_t e : adjacent_[v]) {
        edges.insert(e);
        for (const index_t w : {level_.owner[e], level_.neighbour[e]}) {
          if (parts_[at(w)] != q) ghosts.insert(at(w));
        }
      }
    }
    const auto by_id = [&](std::size_t a, std::size_t b) {
      return level_.global_id[a] < level_.global_id[b];
    };
    std::ranges::sort(s.owned, by_id);
    s.ghosts.assign(ghosts.begin(), ghosts.end());
    std::ranges::sort(s.ghosts, [&](std::size_t a, std::size_t b) {
      return std::tuple(parts_[a], level_.global_id[a]) <
             std::tuple(parts_[b], level_.global_id[b]);
    });
    s.edges.assign(edges.begin(), edges.end());
    std::ranges::stable_sort(s.edges, [&](std::size_t a, std::size_t b) {
      const bool own_a = edge_part(a) == q, own_b = edge_part(b) == q;
      if (own_a != own_b) return own_a;
      return std::tuple(own_a ? 0 : edge_part(a), a) < std::tuple(own_b ? 0 : edge_part(b), b);
    });
    return s;
  }

 private:
  const GraphLevel& level_;
  std::span<const std::int32_t> parts_;
  std::span<const index_t> coarse_node_;
  std::span<const std::int32_t> coarse_parts_;
  std::vector<std::vector<std::size_t>> adjacent_;
};

// A HaloPattern from what this rank receives (its ghost items, in order,
// with their owning parts) and what it sends (per part, its owned items
// that part holds as ghosts, in that part's order, as local indices).
HaloPattern pattern(std::int32_t rank, std::int32_t n_parts,
                    const std::vector<std::int32_t>& ghost_parts,
                    const std::vector<std::vector<index_t>>& sends) {
  HaloPattern h;
  h.rank = rank;
  h.n_ranks = n_parts;
  std::set<std::int32_t> neighbours(ghost_parts.begin(), ghost_parts.end());
  for (std::int32_t q = 0; q < n_parts; ++q) {
    if (!sends[at(q)].empty()) neighbours.insert(q);
  }
  h.neighbours.assign(neighbours.begin(), neighbours.end());
  std::size_t g = 0;
  for (const std::int32_t q : h.neighbours) {
    while (g < ghost_parts.size() && ghost_parts[g] == q) ++g;
    h.recv_offsets.push_back(static_cast<index_t>(g));
    h.send_cells.insert(h.send_cells.end(), sends[at(q)].begin(), sends[at(q)].end());
    h.send_offsets.push_back(static_cast<index_t>(h.send_cells.size()));
  }
  std::vector<std::pair<index_t, index_t>> slots(h.send_cells.size());
  for (std::size_t s = 0; s < slots.size(); ++s)
    slots[s] = {h.send_cells[s], static_cast<index_t>(s)};
  std::ranges::sort(slots);
  for (std::size_t s = 0; s < slots.size(); ++s) {
    if (s == 0 || slots[s].first != slots[s - 1].first) {
      if (s > 0) h.reverse_offsets.push_back(static_cast<index_t>(s));
      h.reverse_cells.push_back(slots[s].first);
    }
    h.reverse_slots.push_back(slots[s].second);
  }
  if (!slots.empty()) h.reverse_offsets.push_back(static_cast<index_t>(slots.size()));
  return h;
}

}  // namespace

std::vector<std::int32_t> coarse_parts(std::span<const std::int32_t> parts,
                                       std::span<const index_t> coarse_node,
                                       std::span<const std::int64_t> global_id,
                                       std::size_t n_coarse) {
  if (parts.size() != coarse_node.size() || global_id.size() != coarse_node.size())
    invalid("parts, coarse_node and global_id differ in size");
  std::vector<std::int32_t> out(n_coarse, -1);
  std::vector<std::int64_t> least(n_coarse, 0);
  for (std::size_t v = 0; v < coarse_node.size(); ++v) {
    const index_t c = coarse_node[v];
    if (c < 0 || at(c) >= n_coarse) invalid("a coarse node is out of range");
    if (out[at(c)] == -1 || global_id[v] < least[at(c)]) {
      least[at(c)] = global_id[v];
      out[at(c)] = parts[v];
    }
  }
  if (std::ranges::find(out, -1) != out.end()) invalid("a coarse node has no child");
  return out;
}

std::vector<index_t> remote_children(std::int32_t rank, std::span<const std::int32_t> parts,
                                     std::span<const index_t> coarse_node,
                                     std::span<const std::int32_t> coarse_parts) {
  if (parts.size() != coarse_node.size()) invalid("parts and coarse_node differ in size");
  std::vector<index_t> out;
  for (std::size_t v = 0; v < parts.size(); ++v) {
    const index_t c = coarse_node[v];
    if (c < 0 || at(c) >= coarse_parts.size()) invalid("a coarse node is out of range");
    if (parts[v] != rank && coarse_parts[at(c)] == rank) out.push_back(static_cast<index_t>(v));
  }
  return out;
}

LocalGraph extract_graph(const GraphLevel& level, std::span<const std::int32_t> parts,
                         std::int32_t n_parts, std::int32_t rank,
                         std::span<const index_t> coarse_node,
                         std::span<const std::int32_t> coarse_parts) {
  if (rank < 0 || rank >= n_parts) invalid("rank " + std::to_string(rank) + " out of range");
  const Level graph(level, parts, n_parts, coarse_node, coarse_parts);
  const Share mine = graph.share(rank);
  LocalGraph out;
  out.n_owned = mine.owned.size();
  std::vector<index_t> local(level.n_nodes, -1);
  for (const auto& list : {mine.owned, mine.ghosts}) {
    for (const std::size_t v : list) {
      local[v] = static_cast<index_t>(out.node_source.size());
      out.node_source.push_back(static_cast<index_t>(v));
      out.node_global_id.push_back(level.global_id[v]);
    }
  }
  std::vector<index_t> local_edge(level.owner.size(), -1);
  for (const std::size_t e : mine.edges) {
    const index_t a = local[at(level.owner[e])], b = local[at(level.neighbour[e])];
    local_edge[e] = static_cast<index_t>(out.owner.size());
    out.owner.push_back(std::min(a, b));
    out.neighbour.push_back(std::max(a, b));
    out.edge_global_id.push_back(static_cast<std::int64_t>(e));
    out.edge_reversed.push_back(a > b ? 1 : 0);
    if (graph.edge_part(e) == rank) ++out.n_owned_edges;
  }
  // What each other part holds of ours: its ghost nodes and edges we own, in its order.
  std::vector<std::vector<index_t>> node_sends(at(n_parts)), edge_sends(at(n_parts));
  for (std::int32_t q = 0; q < n_parts; ++q) {
    if (q == rank) continue;
    const Share theirs = graph.share(q);
    for (const std::size_t v : theirs.ghosts) {
      if (parts[v] == rank) node_sends[at(q)].push_back(local[v]);
    }
    for (const std::size_t e : theirs.edges) {
      if (graph.edge_part(e) == rank) edge_sends[at(q)].push_back(local_edge[e]);
    }
  }
  std::vector<std::int32_t> ghost_parts, ghost_edge_parts;
  for (const std::size_t v : mine.ghosts) ghost_parts.push_back(parts[v]);
  for (std::size_t k = out.n_owned_edges; k < mine.edges.size(); ++k)
    ghost_edge_parts.push_back(graph.edge_part(mine.edges[k]));
  out.node_halo = pattern(rank, n_parts, ghost_parts, node_sends);
  out.edge_halo = pattern(rank, n_parts, ghost_edge_parts, edge_sends);
  return out;
}

}  // namespace autan
