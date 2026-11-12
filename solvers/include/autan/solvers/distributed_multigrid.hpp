#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "autan/solvers/distributed_operator.hpp"
#include "autan/solvers/gcr.hpp"
#include "autan/solvers/multigrid.hpp"

namespace autan::solvers {

/// What a distributed multigrid needs from the run's ranks beyond forward exchanges (#259 D2).
/// Collective: every rank calls the same members in the same order.
class LevelCollective : public LevelExchange {
 public:
  /// The sum of every rank's `terms`: the same bits on any number of ranks, in any order.
  virtual double sum(std::span<const double> terms) = 0;
  /// The largest `value` over the ranks (NaN counted as +infinity).
  virtual double max(double value) = 0;
  /// Every rank's `values`, concatenated in rank order.
  virtual std::vector<double> all_gather(std::span<const double> values) = 0;
};

/// The agglomeration multigrid of docs/design/multigrid.md on one rank's share of a partition
/// (#259 D2): Multigrid's cycle, smoother and coarsest solve over DistributedOperator's levels.
/// - Each line is swept on one rank, in the global line colours, with a forward exchange before
///   the first colour and after each one; every coupling sums its edges by global id, so a sweep
///   is Multigrid's, bit for bit, on any number of parts. level_parts keeps the coarse levels'
///   lines whole; the finest level's are whole only if `cell_parts` keeps them so, which the
///   caller ensures (the runtime partitions with the multigrid's finest lines as groups).
/// - The coarsest level, when Multigrid factors it densely, is gathered on every rank in global
///   node order and solved by the same dense LU.
/// - The K-cycle's and the Galerkin scaling's inner products are LevelCollective sums, so they
///   do not depend on the partition. With Cycle::v and Scaling::none, which take no inner
///   product, a cycle gives Multigrid::cycle's bits.
/// Vectors are structure of arrays over a level's local nodes (DistributedOperator).
class DistributedMultigrid {
 public:
  /// Keeps a reference to `levels`, which must outlive the multigrid.
  DistributedMultigrid(const std::vector<Level>& levels, std::span<const std::int32_t> cell_parts,
                       std::int32_t n_parts, std::int32_t rank, LevelCollective& collective,
                       const MultigridOptions& options = {});
  DistributedMultigrid(const std::vector<Level>&&, std::span<const std::int32_t>, std::int32_t,
                       std::int32_t, LevelCollective&, const MultigridOptions& = {}) = delete;
  ~DistributedMultigrid();
  DistributedMultigrid(const DistributedMultigrid&) = delete;
  DistributedMultigrid& operator=(const DistributedMultigrid&) = delete;

  [[nodiscard]] const DistributedOperator& op() const noexcept { return op_; }
  [[nodiscard]] bool direct_coarsest() const noexcept { return direct_; }

  /// Sets the fine operator (owned blocks of level 0, DistributedOperator::set_operator), builds
  /// the coarse operators and factors every level's smoother. A singular block on any rank
  /// throws std::domain_error on every rank. Collective.
  void set_operator(std::span<const double> diagonal, std::span<const double> upper,
                    std::span<const double> lower);
  /// One cycle on A x = b from x, level 0. Collective.
  void cycle(std::span<const double> b, std::span<double> x);
  /// GCR preconditioned by one cycle per iteration, as Multigrid::solve, its inner products and
  /// largest entries reduced over the ranks. Collective.
  KrylovReport solve(std::span<const double> b, std::span<double> x, int max_cycles,
                     double tolerance);

 private:
  struct Smoother;
  void sweep(std::size_t l, std::span<const double> b, std::span<double> x, int sweeps);
  void residual(std::size_t l, std::span<const double> b, std::span<double> x, std::span<double> r);
  void cycle_level(std::size_t l, std::span<const double> b, std::span<double> x);
  void coarse_solve(std::size_t m, std::span<const double> b, std::span<double> x);
  void solve_coarsest(std::span<const double> b, std::span<double> x);
  double dot(std::size_t l, std::span<const double> u, std::span<const double> v);

  const std::vector<Level>& levels_;
  MultigridOptions options_;
  LevelCollective& collective_;
  DistributedOperator op_;
  std::vector<Smoother> smoothers_;
  bool direct_ = false;
  std::vector<double> dense_;  // the coarsest level's LU factors, the same on every rank
  std::vector<std::size_t> pivot_;
  std::vector<std::vector<double>> r_, bc_, xc_, e_, q_;
  std::vector<std::vector<std::vector<double>>> krylov_;
  Gcr gcr_;
  std::vector<double> residual_;
};

}  // namespace autan::solvers
