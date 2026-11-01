#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "autan/partition/partition.hpp"
#include "autan/runtime/physics.hpp"
#include "autan/runtime/ranks.hpp"

namespace autan::runtime {

/// The ranks of a run as the runtime uses them: what a physics sees, plus
/// what moving the outputs to rank 0 needs.
class Comm : public Ranks {
 public:
  /// This rank's index among the ranks of its host.
  [[nodiscard]] virtual int local_rank() const noexcept = 0;
  /// Rank 0's `text`, on every rank.
  [[nodiscard]] virtual std::string broadcast(std::string text) = 0;
  /// On rank 0, the n values whose global indices `ids` each rank gives
  /// (every index in [0, n) exactly once over all ranks); empty elsewhere.
  [[nodiscard]] virtual std::vector<double> gather(std::span<const std::int64_t> ids,
                                                   std::span<const double> values,
                                                   std::int64_t n) = 0;
  /// Ends every process of the run with exit status `status`.
  [[noreturn]] virtual void abort(int status) = 0;
  /// The ghost exchange of a part of this run, for PhysicsContext::ghosts.
  [[nodiscard]] virtual std::unique_ptr<GhostExchange> ghost_exchange(
      const LocalPartition& part) = 0;
  /// Fills the ghost volumes and centroids of `part` (exchange_ghost_geometry).
  virtual void fill_ghosts(LocalPartition& part) = 0;
};

/// The ranks this process runs with: those of MPI_COMM_WORLD when this
/// build has MPI and a launcher started the process, otherwise one rank.
/// Throws E0146 when a launcher started several ranks of a build without MPI.
std::unique_ptr<Comm> make_comm();

/// Whether a launcher (mpiexec, mpirun, srun) started this process, as its
/// environment shows; MPI is initialized only then.
bool launched_by_mpi();

}  // namespace autan::runtime
