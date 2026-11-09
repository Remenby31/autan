#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "autan/runtime/physics.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/distributed_multigrid.hpp"

namespace autan::physics {

/// solvers::LevelCollective over the run's ranks (runtime.md, Ranks): each level pattern's
/// forward exchange, made on its first use; sums through ExactSum and Ranks::sum; the maximum
/// with NaN as +infinity; Ranks::all_gather.
class RanksCollective final : public solvers::LevelCollective {
 public:
  RanksCollective(runtime::Ranks& ranks, runtime::HaloExchanges exchanges);
  void forward(const HaloPattern& pattern, std::size_t n_owned,
               std::span<double* const> components) override;
  double sum(std::span<const double> terms) override;
  double max(double value) override;
  std::vector<double> all_gather(std::span<const double> values) override;

 private:
  runtime::Ranks& ranks_;
  runtime::HaloExchanges make_;
  std::map<std::pair<const HaloPattern*, std::size_t>, std::unique_ptr<runtime::GhostExchange>>
      exchanges_;
};

/// The hierarchy partition_hierarchy builds, on the run's mesh renumbered in RCM order (#259 D2):
/// the levels the one-rank path builds on its RCM compute mesh, in RCM labels, with the
/// renumbering. new_to_old[i] is the read cell at RCM position i, old_to_new its inverse;
/// face_source[e] is the read face of RCM face e and face_reversed[e] 1 where it runs against it.
struct Hierarchy {
  std::vector<solvers::Level> levels;
  std::vector<index_t> new_to_old, old_to_new;
  std::vector<index_t> face_source;
  std::vector<std::uint8_t> face_reversed;
};

/// The agglomeration multigrid on the run's partition (#259 D2): solvers::DistributedMultigrid on
/// the hierarchy of the run's RCM mesh (PhysicsContext::setup, from partition_hierarchy) and its
/// nodes' parts, with the physics' cells and interior faces mapped to level 0 through
/// cell_global_id, face_global_id, face_reversed and the RCM renumbering. Its bits do not depend
/// on the number of ranks nor on the physics' cell order.
class PartitionMultigrid {
 public:
  explicit PartitionMultigrid(const runtime::PhysicsContext& ctx);

  [[nodiscard]] const std::vector<solvers::Level>& levels() const noexcept { return h_.levels; }
  [[nodiscard]] bool direct_coarsest() const noexcept { return mg_.direct_coarsest(); }
  /// The operator from the physics' blocks (BlockMatrix: per cell, per interior face in the
  /// physics' orientation). A singular block on any rank throws std::domain_error on every rank.
  /// Collective.
  void set_operator(std::span<const double> diagonal, std::span<const double> upper,
                    std::span<const double> lower);
  /// One cycle on A z = r from z = 0, over the physics' owned cells (structure of arrays over its
  /// n_cells); z's ghosts are left as they were. Collective.
  void cycle(std::span<const double> r, std::span<double> z);

 private:
  const Hierarchy& h_;
  std::vector<std::int32_t> parts_;  // the part of each level-0 node (RCM order)
  RanksCollective collective_;
  solvers::DistributedMultigrid mg_;
  std::size_t n_cells_ = 0, n_local_ = 0;
  std::vector<std::size_t> cell_of_node_;   // owned level-0 node -> physics cell
  std::vector<std::size_t> face_of_edge_;   // owned level-0 edge -> physics interior face
  std::vector<std::uint8_t> edge_against_;  // 1 where that face runs against the edge
  std::vector<double> diag_, upper_, lower_, b_, x_;
};

/// Builds the hierarchy of `mesh` (as the run read it and scaled it, no ghost cells, with its
/// geometry) on rcm_order(mesh), the geometry carried by renumber_geometry as the run's RCM
/// compute mesh carries it, for PartitionMultigrid.
std::shared_ptr<const Hierarchy> partition_hierarchy(const Mesh& mesh, const Geometry& geometry);

}  // namespace autan::physics
