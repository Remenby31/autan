#include "autan/io/mesh_counts.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "autan/core/lines.hpp"
#include "autan/core/renumber.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/partition/partition.hpp"

namespace autan::io {
namespace {

std::int64_t stencil_entries(const Mesh& mesh, const Geometry& geometry) {
  const auto packed = kernels::pack_geometry(mesh, geometry);
  return static_cast<std::int64_t>(
      kernels::cell_stencil(mesh, packed, kernels::StencilKind::face2).entries.size());
}

void count_lines(const Mesh& mesh, const Geometry& geometry, MeshInfoRun& out) {
  const Lines lines = smoother_lines(mesh, geometry);
  out.line_cells = static_cast<std::int64_t>(cells_in_lines(lines));
  out.long_line_cells = static_cast<std::int64_t>(cells_in_lines(lines, long_line));
}

}  // namespace

MeshInfoRun run_counts(const Mesh& mesh, const Geometry& geometry, CellOrder order,
                       std::optional<RanksRequest> ranks) {
  MeshInfoRun out;
  if (order == CellOrder::rcm) {
    out.line_order = "rcm";
    AssemblyMap map;
    const std::vector<index_t> new_to_old = rcm_order(mesh);
    const Mesh ordered = renumber_cells(mesh, new_to_old, &map);
    count_lines(ordered, renumber_geometry(geometry, new_to_old, map), out);
  } else {
    out.line_order = "input";
    count_lines(mesh, geometry, out);
  }

  // The stencil and ghosts are sets, the same in any cell order.
  if (mesh.periodic_pairs().empty()) {
    out.stencil_entries = stencil_entries(mesh, geometry);
  } else {
    const std::vector<std::int32_t> one(static_cast<std::size_t>(mesh.n_cells()), 0);
    LocalPartition part = extract_partition(mesh, geometry, one, 1, 0);
    fill_ghost_geometry(part);
    out.periodic_ghosts = part.mesh.n_cells() - part.mesh.n_owned_cells();
    out.stencil_entries = stencil_entries(part.mesh, part.geometry);
  }

  if (!ranks) return out;
  // As the physics forms them (euler_groups): a mesh without lines has no groups.
  Lines groups{.offsets = {}, .cells = {}};
  if (ranks->groups == RankGroups::lines) {
    groups = smoother_lines(mesh, geometry);
    if (groups.cells.empty()) groups.offsets.clear();
  }
  const std::vector<std::int32_t> parts = partition_cells(mesh, ranks->ranks,
                                                          {.cell_weights = cell_cost_weights(mesh),
                                                           .group_offsets = groups.offsets,
                                                           .group_cells = groups.cells});
  MeshInfoRun::Partition p{.ranks = ranks->ranks,
                           .groups = ranks->groups == RankGroups::lines ? "lines" : "none",
                           .cells = {},
                           .ghosts = {},
                           .faces = {},
                           .interior_faces = {},
                           .stencil_entries = {}};
  for (std::int32_t r = 0; r < ranks->ranks; ++r) {
    // Ghost cells hold no geometry here: the counts depend on the connectivity only.
    const LocalPartition part = extract_partition(mesh, geometry, parts, ranks->ranks, r);
    p.cells.push_back(part.mesh.n_owned_cells());
    p.ghosts.push_back(part.mesh.n_cells() - part.mesh.n_owned_cells());
    p.faces.push_back(part.mesh.n_faces());
    p.interior_faces.push_back(part.mesh.n_interior_faces());
    p.stencil_entries.push_back(stencil_entries(part.mesh, part.geometry));
  }
  out.partition = std::move(p);
  return out;
}

}  // namespace autan::io
