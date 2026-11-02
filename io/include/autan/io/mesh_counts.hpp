#pragma once

#include <cstdint>
#include <optional>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/mesh_info.hpp"

namespace autan::io {

/// The cell order a run computes in (results.md, cell_order).
enum class CellOrder : std::uint8_t { rcm, input };

/// The groups a run on several ranks keeps whole: none, or the smoother's lines
/// (smoother_lines on the mesh as read), which a steady Euler case with
/// linear_solver "lines" forms.
enum class RankGroups : std::uint8_t { none, lines };

struct RanksRequest {
  std::int32_t ranks = 1;
  RankGroups groups = RankGroups::none;
};

/// The MeshInfoRun of `mesh`, an unpartitioned mesh as read, as autan-run would compute on it
/// with `scale` metres per mesh unit (solver-input.md, mesh.scale): on scaled(mesh, scale) and its
/// compute_geometry, whose round-off decides couplings at the lines' threshold:
/// - lines: smoother_lines on the mesh in `order` (for rcm, renumber_cells by
///   rcm_order with renumber_geometry, as the run renumbers), cells in lines of
///   at least 2 and of at least long_line cells;
/// - stencil_entries, periodic_ghosts: the face2 cell stencil's entries and the
///   ghost cells of a single-rank run's mesh (its one-part partition when the
///   mesh has periodic pairs, the mesh otherwise);
/// - with `ranks`, the partition of a run on that many ranks: the run's METIS
///   partition (cell_cost_weights, its groups kept whole), and each rank's owned
///   cells, ghost cells, faces, interior faces and face2 stencil entries.
/// Throws as partition_cells for an invalid rank count (E0140 to E0143).
MeshInfoRun run_counts(const Mesh& mesh, double scale, CellOrder order,
                       std::optional<RanksRequest> ranks = std::nullopt);

}  // namespace autan::io
