#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Graph partitioner behind partition_cells. METIS is the default; Scotch is
/// the option for builds that carry it. Both are seeded, so the same mesh and
/// options give the same parts on every run.
enum class PartitionMethod : std::uint8_t { metis, scotch };

struct PartitionOptions {
  PartitionMethod method = PartitionMethod::metis;
  /// Cost of each cell, ≥ 1; empty for uniform cost. cell_cost_weights gives
  /// the face-count cost of polyhedral meshes.
  std::vector<std::int32_t> cell_weights;
  /// Allowed load imbalance: the heaviest part may weigh (1 + tolerance)
  /// times the mean.
  double imbalance_tolerance = 0.03;
  std::int32_t seed = 0;
};

/// Number of faces of each cell: the cost of a face loop, which dominates the
/// solver, for cells with from 4 to hundreds of faces.
std::vector<std::int32_t> cell_cost_weights(const Mesh& mesh);

/// Part in [0, n_parts) of every cell of an unpartitioned mesh, from its dual
/// graph (cells joined by interior faces). Throws E0140 for n_parts < 1, more
/// parts than cells or weights of the wrong size or below 1, E0141 if the mesh
/// already has ghost cells, E0142 if the method is not built or fails, and
/// E0143 if a part comes out empty.
std::vector<std::int32_t> partition_cells(const Mesh& mesh, std::int32_t n_parts,
                                          const PartitionOptions& options = {});

/// Quality of a partition, as each part would hold it with one ghost layer.
struct PartitionQuality {
  std::int64_t cut_faces = 0;              ///< interior faces whose cells lie in two parts
  double imbalance = 0.0;                  ///< heaviest part weight / mean part weight
  double ghost_ratio = 0.0;                ///< total ghost cells / total owned cells
  double max_ghost_ratio = 0.0;            ///< largest ghost/owned ratio of one part
  std::int32_t max_neighbours = 0;         ///< largest number of parts one part exchanges with
  std::vector<std::int64_t> owned_cells;   ///< per part
  std::vector<std::int64_t> ghost_cells;   ///< per part
  std::vector<std::int64_t> part_weights;  ///< per part, with the weights given
};

/// Measures `parts` on `mesh` (unpartitioned). `cell_weights` empty means
/// uniform. Throws E0140 if a part index or a size is out of range.
PartitionQuality partition_quality(const Mesh& mesh, std::span<const std::int32_t> parts,
                                   std::int32_t n_parts,
                                   std::span<const std::int32_t> cell_weights = {});

/// What one rank exchanges with the others, its own rank included when it
/// holds periodic images of its cells. Ghost cells are ordered by owning
/// rank, then by index in the global mesh, then by periodic shift, so the ghosts received from
/// neighbours[i] are the contiguous local cells n_owned_cells + [recv_offsets[i],
/// recv_offsets[i+1]). send_cells[send_offsets[i] .. send_offsets[i+1]) are the owned cells whose
/// copies are those ghosts on neighbours[i], in the same order.
struct HaloPattern {
  std::int32_t rank = 0;
  std::int32_t n_ranks = 1;
  std::vector<std::int32_t> neighbours;  ///< ascending
  std::vector<index_t> send_offsets{0};
  std::vector<index_t> send_cells;
  std::vector<index_t> recv_offsets{0};
  /// Reverse accumulation in a fixed order: owned cell reverse_cells[u]
  /// receives the send slots reverse_slots[reverse_offsets[u] ..
  /// reverse_offsets[u+1]), ascending, so by ascending neighbour rank.
  std::vector<index_t> reverse_cells;
  std::vector<index_t> reverse_offsets{0};
  std::vector<index_t> reverse_slots;
  /// Interior faces whose neighbour is a ghost, ascending: the faces a kernel
  /// must defer until the forward exchange completes.
  std::vector<index_t> interface_faces;
};

/// One rank's share of a partitioned mesh.
struct LocalPartition {
  Mesh mesh;
  /// Owned cells and every face carry the global geometry bit for bit (area
  /// vectors negated on faces whose orientation flipped); ghost cells hold
  /// NaN until a forward halo exchange fills them.
  Geometry geometry;
  std::vector<std::int64_t> face_global_id;  ///< face of the global mesh of each local face
  HaloPattern halo;
};

/// Builds the part `rank` of `parts` with `layers` (1 or 2) layers of ghost
/// cells: owned cells in the order of `global`, then ghosts ordered as
/// HaloPattern states; every cell keeps the cell_global_id it has in
/// `global`. With two layers the first ghost layer carries all its faces,
/// so the face-of-face stencil of every owned cell is local. A periodic face
/// becomes an interior face to an image ghost: the cell across it, shifted
/// by the pair's translation (Mesh::ghost_translation), even on one part.
/// `global_geometry` must be compute_geometry(global). Throws E0141 if
/// `global` has ghosts and E0140 if `parts`, `n_parts`, `rank` or `layers`
/// are out of range, `rank` owns no cell, or the periodic translations of
/// `global` span more than three directions or are not linearly independent.
LocalPartition extract_partition(const Mesh& global, const Geometry& global_geometry,
                                 std::span<const std::int32_t> parts, std::int32_t n_parts,
                                 std::int32_t rank, int layers = 2);

}  // namespace autan
