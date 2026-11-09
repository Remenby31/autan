#include "partition_multigrid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "autan/runtime/exact_sum.hpp"
#include "autan/solvers/multigrid.hpp"

namespace autan::physics {
namespace {

constexpr std::size_t b2 = 25;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

}  // namespace

RanksCollective::RanksCollective(runtime::Ranks& ranks, runtime::HaloExchanges exchanges)
    : ranks_(ranks), make_(std::move(exchanges)) {}

void RanksCollective::forward(const HaloPattern& pattern, std::size_t n_owned,
                              std::span<double* const> components) {
  auto& slot = exchanges_[{&pattern, n_owned}];
  if (!slot) slot = make_(pattern, static_cast<index_t>(n_owned));
  slot->forward(components, MemorySpace::host, nullptr);
}

double RanksCollective::sum(std::span<const double> terms) {
  std::array<runtime::ExactSum, 1> s;
  for (const double t : terms) s[0].add(t);
  ranks_.sum(s);
  return s[0].value();
}

double RanksCollective::max(double value) {
  return ranks_.max(std::isnan(value) ? std::numeric_limits<double>::infinity() : value);
}

std::vector<double> RanksCollective::all_gather(std::span<const double> values) {
  return ranks_.all_gather(values);
}

std::shared_ptr<const Hierarchy> partition_hierarchy(const Mesh& mesh, const Geometry& geometry) {
  return std::make_shared<const Hierarchy>(
      solvers::agglomerate(mesh, geometry, solvers::MultigridOptions{}.agglomeration));
}

PartitionMultigrid::PartitionMultigrid(const runtime::PhysicsContext& ctx)
    : levels_(*static_cast<const Hierarchy*>(ctx.setup)),
      collective_(ctx.ranks, ctx.halo_exchange),
      mg_(levels_, ctx.cell_parts, ctx.ranks.size(), ctx.ranks.rank(), collective_),
      n_cells_(at(ctx.mesh.n_cells())) {
  const LocalGraph& g = mg_.op().graph(0);
  n_local_ = g.node_global_id.size();
  const auto ids = ctx.mesh.cell_global_id();
  const auto owned = at(ctx.mesh.n_owned_cells());
  if (owned != g.n_owned) {
    throw std::logic_error("the partition's owned cells are not the hierarchy's owned nodes");
  }
  // Owned nodes come in increasing global id; the physics' owned cells in its own order.
  std::vector<std::pair<std::int64_t, std::size_t>> by_id(owned);
  for (std::size_t c = 0; c < owned; ++c) by_id[c] = {ids[c], c};
  std::ranges::sort(by_id);
  cell_of_node_.resize(owned);
  for (std::size_t i = 0; i < owned; ++i) {
    if (by_id[i].first != g.node_global_id[i]) {
      throw std::logic_error("the partition's owned cells are not the hierarchy's owned nodes");
    }
    cell_of_node_[i] = by_id[i].second;
  }
  // Level 0's edges are the interior faces of the mesh as read, in its order (agglomerate).
  std::unordered_map<std::int64_t, std::size_t> face_of_id;
  const auto interior = at(ctx.mesh.n_interior_faces());
  face_of_id.reserve(interior);
  for (std::size_t f = 0; f < interior; ++f) face_of_id.emplace(ctx.face_global_id[f], f);
  face_of_edge_.resize(g.n_owned_edges);
  edge_against_.resize(g.n_owned_edges);
  for (std::size_t k = 0; k < g.n_owned_edges; ++k) {
    const auto it = face_of_id.find(g.edge_global_id[k]);
    if (it == face_of_id.end()) {
      throw std::logic_error("an owned edge of the hierarchy has no face in the partition");
    }
    face_of_edge_[k] = it->second;
    edge_against_[k] = ctx.face_reversed[it->second];
  }
  diag_.resize(b2 * owned);
  upper_.resize(b2 * g.n_owned_edges);
  lower_.resize(b2 * g.n_owned_edges);
  b_.resize(5 * n_local_);
  x_.resize(5 * n_local_);
}

void PartitionMultigrid::set_operator(std::span<const double> diagonal,
                                      std::span<const double> upper,
                                      std::span<const double> lower) {
  for (std::size_t i = 0; i < cell_of_node_.size(); ++i) {
    std::copy_n(diagonal.begin() + static_cast<std::ptrdiff_t>(b2 * cell_of_node_[i]), b2,
                diag_.begin() + static_cast<std::ptrdiff_t>(b2 * i));
  }
  // A face that runs against its edge holds A(edge neighbour, edge owner) as its upper block.
  for (std::size_t k = 0; k < face_of_edge_.size(); ++k) {
    const auto from = static_cast<std::ptrdiff_t>(b2 * face_of_edge_[k]);
    const auto to = static_cast<std::ptrdiff_t>(b2 * k);
    const auto& u = edge_against_[k] != 0 ? lower : upper;
    const auto& l = edge_against_[k] != 0 ? upper : lower;
    std::copy_n(u.begin() + from, b2, upper_.begin() + to);
    std::copy_n(l.begin() + from, b2, lower_.begin() + to);
  }
  mg_.set_operator(diag_, upper_, lower_);
}

void PartitionMultigrid::cycle(std::span<const double> r, std::span<double> z) {
  std::ranges::fill(b_, 0.0);
  std::ranges::fill(x_, 0.0);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < cell_of_node_.size(); ++i)
      b_[k * n_local_ + i] = r[k * n_cells_ + cell_of_node_[i]];
  }
  mg_.cycle(b_, x_);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < cell_of_node_.size(); ++i)
      z[k * n_cells_ + cell_of_node_[i]] = x_[k * n_local_ + i];
  }
}

}  // namespace autan::physics
