#include "comm.hpp"

#include <cstdlib>
#include <initializer_list>
#include <string_view>

#include "autan/core/error.hpp"
#include "autan/partition/device_self.hpp"

#ifdef AUTAN_HAVE_MPI
#include <mpi.h>

#include "autan/partition/halo_exchange.hpp"
#endif

namespace autan::runtime {
namespace {

// What a launcher announces in the environment: Open MPI, the PMI of MPICH,
// Intel MPI and Slurm, and PMIx (which gives the rank, not always the size).
// -1 when absent.
int environment(std::initializer_list<const char*> names) {
  for (const char* name : names) {
    if (const char* v = std::getenv(name)) return std::atoi(v);
  }
  return -1;
}
int launched_size() { return environment({"OMPI_COMM_WORLD_SIZE", "PMI_SIZE"}); }
int launched_rank() { return environment({"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"}); }

// Fills the periodic images of owned cells, the one ghost a part of a
// single-rank run can hold (partitioning.md); without a part, nothing.
class SelfExchange final : public GhostExchange {
 public:
  explicit SelfExchange(const LocalPartition& part) : part_(part) {}
  void forward(std::span<float* const> components, MemorySpace space, void* stream) override {
    run(components, space, stream);
  }
  void forward(std::span<double* const> components, MemorySpace space, void* stream) override {
    run(components, space, stream);
  }

 private:
  // Device memory: the image pairs move to the device on first use (E0145
  // in a build without CUDA).
  template <class T>
  void run(std::span<T* const> components, MemorySpace space, void* stream) {
    if (space == MemorySpace::host) {
      forward_self(part_.halo, part_.mesh.n_owned_cells(), components);
      return;
    }
    if (!device_)
      device_ = std::make_unique<DeviceSelfForward>(part_.halo, part_.mesh.n_owned_cells());
    device_->forward(components, stream);
  }

  const LocalPartition& part_;
  std::unique_ptr<DeviceSelfForward> device_;
};

class Single final : public Comm {
 public:
  [[nodiscard]] int rank() const noexcept override { return 0; }
  [[nodiscard]] int size() const noexcept override { return 1; }
  [[nodiscard]] int local_rank() const noexcept override { return 0; }
  void sum(std::span<ExactSum> /*sums*/) override {}
  [[nodiscard]] double min(double value) override { return value; }
  [[nodiscard]] std::int64_t max(std::int64_t value) override { return value; }
  [[nodiscard]] double max(double value) override { return value; }
  [[nodiscard]] std::string broadcast(std::string text) override { return text; }
  [[nodiscard]] std::vector<double> gather(std::span<const std::int64_t> ids,
                                           std::span<const double> values,
                                           std::int64_t n) override {
    std::vector<double> out(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < ids.size(); ++i) out[static_cast<std::size_t>(ids[i])] = values[i];
    return out;
  }
  [[noreturn]] void abort(int status) override { std::exit(status); }
  [[nodiscard]] std::unique_ptr<GhostExchange> ghost_exchange(const LocalPartition& part) override {
    return std::make_unique<SelfExchange>(part);
  }
  void fill_ghosts(LocalPartition& part) override { fill_ghost_geometry(part); }
};

#ifdef AUTAN_HAVE_MPI

void check(int status, const char* what) {
  if (status != MPI_SUCCESS) {
    throw Error(ErrorCode::halo_inconsistent, std::string("MPI failed in ") + what);
  }
}

// Ghosts filled by the halo exchange; one exchange per memory space, made on
// first use (each is collective, and every rank asks for the same spaces in
// the same order).
class MpiGhosts final : public GhostExchange {
 public:
  MpiGhosts(const LocalPartition& part, MPI_Comm comm) : part_(part), comm_(comm) {}
  void forward(std::span<float* const> components, MemorySpace space, void* stream) override {
    exchange(space).forward(components, stream);
  }
  void forward(std::span<double* const> components, MemorySpace space, void* stream) override {
    exchange(space).forward(components, stream);
  }

 private:
  HaloExchange& exchange(MemorySpace space) {
    auto& slot = space == MemorySpace::host ? host_ : device_;
    if (!slot) {
      slot = std::make_unique<HaloExchange>(part_.halo, part_.mesh.n_owned_cells(), comm_, space);
    }
    return *slot;
  }

  const LocalPartition& part_;
  MPI_Comm comm_;
  std::unique_ptr<HaloExchange> host_, device_;
};

class World final : public Comm {
 public:
  World() {
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
    check(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI_Comm_size");
    MPI_Comm host = MPI_COMM_NULL;
    check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &host),
          "MPI_Comm_split_type");
    check(MPI_Comm_rank(host, &local_rank_), "MPI_Comm_rank");
    MPI_Comm_free(&host);
  }
  [[nodiscard]] int local_rank() const noexcept override { return local_rank_; }
  [[nodiscard]] int rank() const noexcept override { return rank_; }
  [[nodiscard]] int size() const noexcept override { return size_; }

  void sum(std::span<ExactSum> sums) override {
    std::vector<std::int64_t> words;
    words.reserve(sums.size() * ExactSum::words);
    for (auto& s : sums) {
      s.normalize();
      const auto d = s.data();
      words.insert(words.end(), d.begin(), d.end());
    }
    check(MPI_Allreduce(MPI_IN_PLACE, words.data(), static_cast<int>(words.size()), MPI_INT64_T,
                        MPI_SUM, MPI_COMM_WORLD),
          "MPI_Allreduce");
    for (std::size_t k = 0; k < sums.size(); ++k) {
      const auto d = sums[k].data();
      std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(k * ExactSum::words), d.size(),
                  d.begin());
    }
  }
  [[nodiscard]] double min(double value) override {
    check(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
          "MPI_Allreduce");
    return value;
  }
  [[nodiscard]] std::int64_t max(std::int64_t value) override {
    check(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD),
          "MPI_Allreduce");
    return value;
  }
  [[nodiscard]] double max(double value) override {
    check(MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
          "MPI_Allreduce");
    return value;
  }
  [[nodiscard]] std::string broadcast(std::string text) override {
    auto n = static_cast<std::int64_t>(text.size());
    check(MPI_Bcast(&n, 1, MPI_INT64_T, 0, MPI_COMM_WORLD), "MPI_Bcast");
    text.resize(static_cast<std::size_t>(n));
    check(MPI_Bcast(text.data(), static_cast<int>(n), MPI_CHAR, 0, MPI_COMM_WORLD), "MPI_Bcast");
    return text;
  }
  [[nodiscard]] std::vector<double> gather(std::span<const std::int64_t> ids,
                                           std::span<const double> values,
                                           std::int64_t n) override {
    const int count = static_cast<int>(ids.size());
    std::vector<int> counts(rank_ == 0 ? static_cast<std::size_t>(size_) : 0);
    check(MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD),
          "MPI_Gather");
    std::vector<int> displs(counts.size());
    std::int64_t total = 0;
    for (std::size_t r = 0; r < counts.size(); ++r) {
      displs[r] = static_cast<int>(total);
      total += counts[r];
    }
    std::vector<std::int64_t> all_ids(static_cast<std::size_t>(total));
    std::vector<double> all_values(static_cast<std::size_t>(total));
    check(MPI_Gatherv(ids.data(), count, MPI_INT64_T, all_ids.data(), counts.data(), displs.data(),
                      MPI_INT64_T, 0, MPI_COMM_WORLD),
          "MPI_Gatherv");
    check(MPI_Gatherv(values.data(), count, MPI_DOUBLE, all_values.data(), counts.data(),
                      displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
          "MPI_Gatherv");
    if (rank_ != 0) return {};
    std::vector<double> out(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < all_ids.size(); ++i)
      out[static_cast<std::size_t>(all_ids[i])] = all_values[i];
    return out;
  }
  [[noreturn]] void abort(int status) override {
    MPI_Abort(MPI_COMM_WORLD, status);
    std::exit(status);
  }
  [[nodiscard]] std::unique_ptr<GhostExchange> ghost_exchange(const LocalPartition& part) override {
    return std::make_unique<MpiGhosts>(part, MPI_COMM_WORLD);
  }
  void fill_ghosts(LocalPartition& part) override {
    HaloExchange exchange(part.halo, part.mesh.n_owned_cells(), MPI_COMM_WORLD, MemorySpace::host);
    exchange_ghost_geometry(exchange, part.mesh, part.geometry);
  }

 private:
  int rank_ = 0, size_ = 1, local_rank_ = 0;
};

#endif

}  // namespace

bool launched_by_mpi() { return launched_size() >= 0 || launched_rank() >= 0; }

std::unique_ptr<Comm> make_comm() {
#ifdef AUTAN_HAVE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) return std::make_unique<World>();
#endif
  // Under a launcher, only a count of 1 shows a single rank.
  if (launched_by_mpi() && launched_size() != 1) {
    throw Error(ErrorCode::mpi_unavailable,
                "a launcher started several ranks, but this build of autan-run has no MPI");
  }
  return std::make_unique<Single>();
}

}  // namespace autan::runtime
