#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <mpi.h>

#include "autan/core/geometry.hpp"
#include "autan/partition/partition.hpp"

namespace autan {

/// Halo exchange of cell fields between the ranks of a partition, one rank
/// per part. A field is a list of components, each an array of n_cells values
/// (owned then ghost) in `space`; one call moves at most max_components of
/// them, in float or double.
///
/// forward copies owned values to their ghost copies on other ranks. reverse
/// is its transpose: each owned cell adds the ghost values the other ranks
/// hold for it, in ascending rank order, and the ghosts are set to zero. Both
/// are deterministic: the same inputs give the same bits on every run.
///
/// begin_* posts the receives, packs on `stream` (a cudaStream_t, or null for
/// the default stream; ignored on the host), waits for the packing and posts
/// the sends; end_* waits for the messages and unpacks on `stream`, then
/// waits for the unpacking. Between the two, a kernel may update every cell
/// except, for forward, the faces in HaloPattern::interface_faces, and, for
/// reverse, the ghosts. One exchange is in flight at a time.
class HaloExchange {
 public:
  static constexpr int max_components = 16;

  /// Collective over `comm`, whose size must be pattern.n_ranks and rank
  /// pattern.rank. Throws E0144 if the ranks disagree on the halo and E0145
  /// for MemorySpace::cuda without CUDA in this build or in MPI.
  HaloExchange(const HaloPattern& pattern, index_t n_owned_cells, MPI_Comm comm, MemorySpace space);
  ~HaloExchange();
  HaloExchange(const HaloExchange&) = delete;
  HaloExchange& operator=(const HaloExchange&) = delete;

  template <class T>
  void begin_forward(std::span<T* const> components, void* stream = nullptr);
  void end_forward();
  template <class T>
  void begin_reverse(std::span<T* const> components, void* stream = nullptr);
  void end_reverse();

  template <class T>
  void forward(std::span<T* const> components, void* stream = nullptr) {
    begin_forward(components, stream);
    end_forward();
  }
  template <class T>
  void reverse(std::span<T* const> components, void* stream = nullptr) {
    begin_reverse(components, stream);
    end_reverse();
  }

  [[nodiscard]] MemorySpace space() const noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

/// Fills the ghost cell volumes and centroids of a host Geometry from their
/// owners, with a forward exchange on the host, and moves the centroids of
/// periodic images by their translation (Mesh::ghost_translation).
void exchange_ghost_geometry(HaloExchange& exchange, const Mesh& mesh, Geometry& geometry);

}  // namespace autan
