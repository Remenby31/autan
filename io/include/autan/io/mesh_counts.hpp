#pragma once

#include <cstdint>
#include <optional>

#include "autan/core/geometry.hpp"
#include "autan/core/ghost_stencil.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/mesh_info.hpp"

namespace autan::io {

/// The cell order a run computes in (results.md, cell_order).
enum class CellOrder : std::uint8_t { rcm, input };

/// The groups a run on several ranks keeps whole (mesh-info.md, partition.groups): none; lines,
/// the line smoother's lines (smoother_lines on the mesh as read), which a steady Euler case with
/// linear_solver "lines" forms; multigrid, the lines of the multigrid's finest level
/// (solvers::agglomerate on the mesh as read, level 0), which a steady Euler case with
/// linear_solver "multigrid" forms.
enum class RankGroups : std::uint8_t { none, lines, multigrid };

struct RanksRequest {
  std::int32_t ranks = 1;
  RankGroups groups = RankGroups::none;
};

/// The MeshInfoRun of `read`, an unpartitioned mesh as read, as autan-run would compute on it
/// with `scale` metres per mesh unit (solver-input.md, mesh.scale): on scaled(read, scale) and its
/// compute_geometry, whose round-off decides couplings at the lines' threshold:
/// - lines: smoother_lines on the mesh in `order` (for rcm, renumber_cells by
///   rcm_order with renumber_geometry, as the run renumbers), cells in lines of
///   at least 2 and of at least long_line cells;
/// - multigrid: the multigrid's levels on the mesh in `multigrid_order` (default `order`),
///   solvers::agglomerate with its default options as a run's multigrid builds them: nodes and
///   edges per level, the mesh first. A run on one rank builds them in its cell order, an
///   inviscid run on a partition (several ranks, or reproducibility "ranks") in RCM order;
/// - stencil_entries, periodic_ghosts: the face2 cell stencil's entries and the
///   ghost cells of a single-rank run's mesh (its one-part partition when the
///   mesh has periodic pairs, the mesh otherwise);
/// - vertex_stencil_entries, band_entries, boundary_fit_entries on that mesh: the vertex cell
///   stencil's entries, and the vertex-ring entries the quadratic gradient's band (two rings)
///   and the viscous boundary fit (three rings, in the cells owning a boundary face and their
///   face neighbours) gather when every boundary face gives a value (mesh-info.md);
/// - with `ranks`, the partition of a run on that many ranks: the run's METIS
///   partition (cell_cost_weights, its groups kept whole), and each rank's owned
///   cells, ghost cells, faces, interior faces, face2 stencil entries and the three ring counts.
/// Every partition holds the ghosts of `ghosts`, face2 or vertex, the stencil the run's physics
/// declares. Throws as partition_cells for an invalid rank count (E0140 to E0143), and E0148 as
/// extract_partition.
MeshInfoRun run_counts(const Mesh& read, double scale, CellOrder order,
                       std::optional<RanksRequest> ranks = std::nullopt,
                       GhostStencil ghosts = GhostStencil::face2,
                       std::optional<CellOrder> multigrid_order = std::nullopt);

}  // namespace autan::io
