#include "autan/solvers/distributed_operator.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

#include "blocks.hpp"

namespace autan::solvers {

namespace {

using detail::add5;
using detail::b2;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

std::vector<std::int64_t> identity_ids(std::size_t n) {
  std::vector<std::int64_t> ids(n);
  std::iota(ids.begin(), ids.end(), std::int64_t{0});
  return ids;
}

std::vector<std::size_t> by_global_id(const LocalGraph& g) {
  std::vector<std::size_t> order(g.node_global_id.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    return g.node_global_id[a] < g.node_global_id[b];
  });
  return order;
}

std::vector<std::size_t> edges_by_global_id(const LocalGraph& g) {
  std::vector<std::size_t> order(g.edge_global_id.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    return g.edge_global_id[a] < g.edge_global_id[b];
  });
  return order;
}

std::size_t global_owner(const LocalGraph& g, std::size_t k) {
  return at(g.edge_reversed[k] != 0 ? g.neighbour[k] : g.owner[k]);
}

}  // namespace

DistributedOperator::DistributedOperator(const std::vector<Level>& levels,
                                         std::span<const std::int32_t> cell_parts,
                                         std::int32_t n_parts, std::int32_t rank,
                                         LevelExchange& exchange)
    : levels_(levels), exchange_(exchange) {
  if (levels.empty()) throw std::invalid_argument("a hierarchy has at least one level");
  if (cell_parts.size() != levels.front().n_nodes)
    throw std::invalid_argument("one part per cell of the finest level");
  const std::size_t n_levels = levels.size();
  parts_.resize(n_levels);
  parts_[0].assign(cell_parts.begin(), cell_parts.end());
  std::vector<std::vector<std::int64_t>> ids(n_levels);
  for (std::size_t l = 0; l < n_levels; ++l) ids[l] = identity_ids(levels[l].n_nodes);
  for (std::size_t l = 0; l + 1 < n_levels; ++l) {
    parts_[l + 1] = level_parts(parts_[l], levels[l].coarse_node, ids[l], levels[l + 1].n_nodes,
                                levels[l + 1].line_offsets, levels[l + 1].line_nodes);
  }
  for (std::size_t l = 0; l < n_levels; ++l) {
    const Level& v = levels[l];
    const GraphLevel level{
        .n_nodes = v.n_nodes, .owner = v.owner, .neighbour = v.neighbour, .global_id = ids[l]};
    const bool coarser = l + 1 < n_levels;
    graphs_.push_back(extract_graph(
        level, parts_[l], n_parts, rank,
        coarser ? std::span<const index_t>(v.coarse_node) : std::span<const index_t>{},
        coarser ? std::span<const std::int32_t>(parts_[l + 1]) : std::span<const std::int32_t>{}));
    if (coarser) {
      parents_.push_back(
          parent_import(parts_[l], v.coarse_node, parts_[l + 1], ids[l + 1], n_parts, rank));
    }
  }
  node_offsets_.resize(n_levels);
  node_edges_.resize(n_levels);
  for (std::size_t l = 0; l < n_levels; ++l) {
    const LocalGraph& g = graphs_[l];
    std::vector<std::vector<index_t>> incident(g.n_owned);
    for (const std::size_t k : edges_by_global_id(g)) {
      for (const index_t end : {g.owner[k], g.neighbour[k]}) {
        if (at(end) < g.n_owned) incident[at(end)].push_back(static_cast<index_t>(k));
      }
    }
    auto& offsets = node_offsets_[l];
    offsets.assign(1, 0);
    for (const auto& list : incident) {
      node_edges_[l].insert(node_edges_[l].end(), list.begin(), list.end());
      offsets.push_back(static_cast<offset_t>(node_edges_[l].size()));
    }
  }
  diag_.resize(n_levels);
  upper_.resize(n_levels);
  lower_.resize(n_levels);
}

void DistributedOperator::forward_vector(const HaloPattern& pattern, std::size_t n_owned,
                                         std::size_t n_items, std::span<double> values) {
  std::array<double*, 5> components{};
  for (std::size_t c = 0; c < 5; ++c) components[c] = values.data() + c * n_items;
  exchange_.forward(pattern, n_owned, components);
}

void DistributedOperator::forward_blocks(const HaloPattern& pattern, std::size_t n_owned,
                                         std::size_t n_items, std::vector<double>& blocks) {
  std::vector<double> soa(b2 * n_items);
  for (std::size_t i = 0; i < n_owned; ++i) {
    for (std::size_t q = 0; q < b2; ++q) soa[q * n_items + i] = blocks[b2 * i + q];
  }
  std::array<double*, b2> components{};
  for (std::size_t q = 0; q < b2; ++q) components[q] = soa.data() + q * n_items;
  exchange_.forward(pattern, n_owned, std::span<double* const>(components.data(), 16));
  exchange_.forward(pattern, n_owned, std::span<double* const>(components.data() + 16, b2 - 16));
  for (std::size_t i = n_owned; i < n_items; ++i) {
    for (std::size_t q = 0; q < b2; ++q) blocks[b2 * i + q] = soa[q * n_items + i];
  }
}

void DistributedOperator::set_operator(std::span<const double> diagonal,
                                       std::span<const double> upper,
                                       std::span<const double> lower) {
  const LocalGraph& g = graphs_[0];
  const std::size_t n = g.node_global_id.size(), m = g.edge_global_id.size();
  if (diagonal.size() != b2 * g.n_owned || upper.size() != b2 * g.n_owned_edges ||
      lower.size() != b2 * g.n_owned_edges)
    throw std::invalid_argument("one block per owned node and per owned edge of level 0");
  diag_[0].assign(b2 * n, 0.0);
  upper_[0].assign(b2 * m, 0.0);
  lower_[0].assign(b2 * m, 0.0);
  std::ranges::copy(diagonal, diag_[0].begin());
  std::ranges::copy(upper, upper_[0].begin());
  std::ranges::copy(lower, lower_[0].begin());
  forward_blocks(g.node_halo, g.n_owned, n, diag_[0]);
  forward_blocks(g.edge_halo, g.n_owned_edges, m, upper_[0]);
  forward_blocks(g.edge_halo, g.n_owned_edges, m, lower_[0]);
  for (std::size_t l = 0; l + 1 < graphs_.size(); ++l) galerkin(l);
}

// P^T A P with P constant on each agglomerate, in Multigrid::set_operator's order: a coarse
// diagonal block sums its children's diagonal blocks by increasing fine index, then the blocks of
// the fine edges inside the agglomerate by increasing edge index, upper before lower; a coarse
// off-diagonal block sums the fine edges it holds by increasing edge index, swapped where the
// fine edge runs against it.
void DistributedOperator::galerkin(std::size_t l) {
  const Level& f = levels_[l];
  const LocalGraph& g = graphs_[l];
  const LocalGraph& gc = graphs_[l + 1];
  const std::size_t nc = gc.node_global_id.size(), mc = gc.edge_global_id.size();
  std::vector<index_t> coarse_local(levels_[l + 1].n_nodes, -1);
  for (std::size_t k = 0; k < gc.n_owned; ++k)
    coarse_local[at(gc.node_source[k])] = static_cast<index_t>(k);
  std::vector<index_t> coarse_edge_local(levels_[l + 1].owner.size(), -1);
  for (std::size_t k = 0; k < gc.n_owned_edges; ++k)
    coarse_edge_local[at(gc.edge_global_id[k])] = static_cast<index_t>(k);
  auto& d = diag_[l + 1];
  auto& u = upper_[l + 1];
  auto& w = lower_[l + 1];
  d.assign(b2 * nc, 0.0);
  u.assign(b2 * mc, 0.0);
  w.assign(b2 * mc, 0.0);
  for (const std::size_t i : by_global_id(g)) {
    const index_t k = coarse_local[at(f.coarse_node[at(g.node_source[i])])];
    if (k >= 0) add5(&d[b2 * at(k)], &diag_[l][b2 * i]);
  }
  for (const std::size_t k : edges_by_global_id(g)) {
    const auto e = at(g.edge_global_id[k]);
    const index_t ce = f.coarse_edge[e];
    if (ce < 0) {
      const index_t c = coarse_local[at(f.coarse_node[at(f.owner[e])])];
      if (c < 0) continue;
      add5(&d[b2 * at(c)], &upper_[l][b2 * k]);
      add5(&d[b2 * at(c)], &lower_[l][b2 * k]);
      continue;
    }
    const index_t E = coarse_edge_local[at(ce)];
    if (E < 0) continue;
    const bool flip = f.coarse_flipped[e] != 0;
    add5(&u[b2 * at(E)], flip ? &lower_[l][b2 * k] : &upper_[l][b2 * k]);
    add5(&w[b2 * at(E)], flip ? &upper_[l][b2 * k] : &lower_[l][b2 * k]);
  }
  forward_blocks(gc.node_halo, gc.n_owned, nc, d);
  forward_blocks(gc.edge_halo, gc.n_owned_edges, mc, u);
  forward_blocks(gc.edge_halo, gc.n_owned_edges, mc, w);
}

std::pair<const double*, std::size_t> DistributedOperator::off(std::size_t level, std::size_t k,
                                                               std::size_t i) const {
  const LocalGraph& g = graphs_.at(level);
  const bool owner = global_owner(g, k) == i;
  const std::size_t j = at(g.owner[k]) == i ? at(g.neighbour[k]) : at(g.owner[k]);
  return {owner ? &upper_[level][b2 * k] : &lower_[level][b2 * k], j};
}

void DistributedOperator::multiply(std::size_t level, std::span<double> x, std::span<double> y) {
  const LocalGraph& g = graphs_.at(level);
  const std::size_t n = g.node_global_id.size();
  forward_vector(g.node_halo, g.n_owned, n, x);
  const auto& offsets = node_offsets_[level];
  const auto& edges = node_edges_[level];
  for (std::size_t i = 0; i < g.n_owned; ++i) {
    std::array<double, 5> s{};
    const double* dg = &diag_[level][b2 * i];
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) s[p] += dg[5 * p + q] * x[q * n + i];
    }
    for (auto t = offsets[i]; t < offsets[i + 1]; ++t) {
      const auto k = at(edges[at(t)]);
      const bool owner = global_owner(g, k) == i;
      const double* a = owner ? &upper_[level][b2 * k] : &lower_[level][b2 * k];
      const std::size_t j = at(g.owner[k]) == i ? at(g.neighbour[k]) : at(g.owner[k]);
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) s[p] += a[5 * p + q] * x[q * n + j];
      }
    }
    for (std::size_t p = 0; p < 5; ++p) y[p * n + i] = s[p];
  }
}

void DistributedOperator::restrict_to(std::size_t level, std::span<double> r,
                                      std::span<double> coarse) {
  const LocalGraph& g = graphs_.at(level);
  const LocalGraph& gc = graphs_.at(level + 1);
  const std::size_t n = g.node_global_id.size(), nc = gc.node_global_id.size();
  forward_vector(g.node_halo, g.n_owned, n, r);
  std::vector<index_t> coarse_local(levels_[level + 1].n_nodes, -1);
  for (std::size_t k = 0; k < gc.n_owned; ++k)
    coarse_local[at(gc.node_source[k])] = static_cast<index_t>(k);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t k = 0; k < gc.n_owned; ++k) coarse[c * nc + k] = 0.0;
  }
  const Level& f = levels_[level];
  for (const std::size_t i : by_global_id(g)) {
    const index_t k = coarse_local[at(f.coarse_node[at(g.node_source[i])])];
    if (k < 0) continue;
    for (std::size_t c = 0; c < 5; ++c) coarse[c * nc + at(k)] += r[c * n + i];
  }
}

void DistributedOperator::prolong_from(std::size_t level, std::span<const double> coarse,
                                       std::span<double> e) {
  const LocalGraph& g = graphs_.at(level);
  const LocalGraph& gc = graphs_.at(level + 1);
  const ParentImport& pi = parents_.at(level);
  const std::size_t n = g.node_global_id.size(), nc = gc.node_global_id.size();
  const std::size_t n_owned = pi.owned.size(), items = n_owned + pi.parents.size();
  auto& buffer = parent_values_;
  buffer.assign(5, std::vector<double>(items, 0.0));
  std::array<double*, 5> components{};
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t k = 0; k < n_owned; ++k) buffer[c][k] = coarse[c * nc + k];
    components[c] = buffer[c].data();
  }
  exchange_.forward(pi.halo, n_owned, components);
  std::vector<index_t> slot(levels_[level + 1].n_nodes, -1);
  for (std::size_t k = 0; k < n_owned; ++k) slot[at(pi.owned[k])] = static_cast<index_t>(k);
  for (std::size_t k = 0; k < pi.parents.size(); ++k)
    slot[at(pi.parents[k])] = static_cast<index_t>(n_owned + k);
  const Level& f = levels_[level];
  for (std::size_t i = 0; i < g.n_owned; ++i) {
    const index_t s = slot[at(f.coarse_node[at(g.node_source[i])])];
    if (s < 0) throw std::logic_error("a fine node's parent is neither owned nor imported");
    for (std::size_t c = 0; c < 5; ++c) e[c * n + i] = buffer[c][at(s)];
  }
}

}  // namespace autan::solvers
