#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "autan/partition/partition.hpp"
#include "autan/solvers/agglomeration.hpp"

namespace autan::solvers {

/// The forward exchange a distributed multigrid needs from the run's ranks (#259 D2). Collective:
/// every rank calls it with the patterns of the same level, in the same order. Each component
/// holds `n_owned` owned values, then the pattern's ghosts; at most 16 components per call.
class LevelExchange {
 public:
  virtual ~LevelExchange() = default;
  virtual void forward(const HaloPattern& pattern, std::size_t n_owned,
                       std::span<double* const> components) = 0;
};

/// One rank's share of an agglomeration hierarchy on N ranks and of its Galerkin operators
/// (docs/design/multigrid.md, #259 D2). Every rank holds the same global hierarchy (`levels`, from
/// the mesh in read order; global ids are node indices) and the cells' parts; coarse nodes go to
/// parts by level_parts, lines whole. On each level a rank keeps its LocalGraph: owned nodes by
/// increasing id, then ghosts (edge neighbours and remote children), and the edges with an owned
/// end or a remote child. Blocks are 5 x 5 row-major in the level's global orientation (`upper` =
/// A(owner, neighbour)); vectors are structure of arrays over a level's local nodes, component c
/// of local node i at c * n_local + i. Every sum runs in the host Multigrid's order over global
/// indices, so the owned values are the same bits on any number of parts.
class DistributedOperator {
 public:
  DistributedOperator(const std::vector<Level>& levels, std::span<const std::int32_t> cell_parts,
                      std::int32_t n_parts, std::int32_t rank, LevelExchange& exchange);

  [[nodiscard]] std::size_t n_levels() const noexcept { return graphs_.size(); }
  [[nodiscard]] const LocalGraph& graph(std::size_t level) const { return graphs_.at(level); }
  [[nodiscard]] std::size_t n_local(std::size_t level) const {
    return graphs_.at(level).node_global_id.size();
  }

  /// Sets the fine operator from the blocks of level 0's owned nodes and owned edges (the first
  /// n_owned and n_owned_edges of its LocalGraph), then fills the ghosts and builds every coarse
  /// operator P^T A P. Collective.
  void set_operator(std::span<const double> diagonal, std::span<const double> upper,
                    std::span<const double> lower);

  /// Level `level`'s blocks: per local node, per local edge; ghosts filled.
  [[nodiscard]] std::span<const double> diagonal(std::size_t level) const {
    return diag_.at(level);
  }
  [[nodiscard]] std::span<const double> upper(std::size_t level) const { return upper_.at(level); }
  [[nodiscard]] std::span<const double> lower(std::size_t level) const { return lower_.at(level); }

  /// y = A x on the owned nodes of `level`, after a forward exchange of x's ghosts. Collective.
  void multiply(std::size_t level, std::span<double> x, std::span<double> y);
  /// The coarse right-hand side on level + 1's owned nodes: the sum of the children's values of
  /// r, after a forward exchange of r's ghosts on `level`. Collective.
  void restrict_to(std::size_t level, std::span<double> r, std::span<double> coarse);
  /// e on `level`'s owned nodes: the value of each node's parent in `coarse` (owned values of
  /// level + 1), after a forward exchange of the parents other parts own. Collective.
  void prolong_from(std::size_t level, std::span<const double> coarse, std::span<double> e);

 private:
  void forward_vector(const HaloPattern& pattern, std::size_t n_owned, std::size_t n_items,
                      std::span<double> values);
  void forward_blocks(const HaloPattern& pattern, std::size_t n_owned, std::size_t n_items,
                      std::vector<double>& blocks);
  void galerkin(std::size_t level);

  const std::vector<Level>& levels_;
  LevelExchange& exchange_;
  std::vector<LocalGraph> graphs_;
  std::vector<ParentImport> parents_;             // level l to l + 1
  std::vector<std::vector<std::int32_t>> parts_;  // per level, every node's part
  // Per level, local incidence: for each owned node, its local edges in increasing global id.
  std::vector<std::vector<offset_t>> node_offsets_;
  std::vector<std::vector<index_t>> node_edges_;
  std::vector<std::vector<double>> diag_, upper_, lower_;
  std::vector<std::vector<double>> parent_values_;
};

}  // namespace autan::solvers
