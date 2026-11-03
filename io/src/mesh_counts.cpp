#include "autan/io/mesh_counts.hpp"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "autan/core/lines.hpp"
#include "autan/core/renumber.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/partition/partition.hpp"
#include "autan/solvers/agglomeration.hpp"

namespace autan::io {
namespace {

std::int64_t stencil_entries(const Mesh& mesh, const Geometry& geometry) {
  const auto packed = kernels::pack_geometry(mesh, geometry);
  return static_cast<std::int64_t>(
      kernels::cell_stencil(mesh, packed, kernels::StencilKind::face2).entries.size());
}

// What depends on the cell order: the smoother's lines, and the multigrid's hierarchy, which
// the run builds with its default options on one rank.
void count_ordered(const Mesh& mesh, const Geometry& geometry, MeshInfoRun& out) {
  const Lines lines = smoother_lines(mesh, geometry);
  out.line_cells = static_cast<std::int64_t>(cells_in_lines(lines));
  out.long_line_cells = static_cast<std::int64_t>(cells_in_lines(lines, long_line));
  for (const auto& level : solvers::agglomerate(mesh, geometry)) {
    out.multigrid_cells.push_back(static_cast<std::int64_t>(level.n_nodes));
    out.multigrid_edges.push_back(static_cast<std::int64_t>(level.owner.size()));
  }
}

}  // namespace

MeshInfoRun run_counts(const Mesh& read, double scale, CellOrder order,
                       std::optional<RanksRequest> ranks) {
  // As prepare() does: the coordinates in metres before any geometry.
  std::optional<Mesh> metres;
  const Mesh& mesh = scale == 1.0 ? read : metres.emplace(scaled(read, scale));
  const Geometry geometry = compute_geometry(mesh);
  MeshInfoRun out;
  out.line_scale = scale;
  if (order == CellOrder::rcm) {
    out.line_order = "rcm";
    AssemblyMap map;
    const std::vector<index_t> new_to_old = rcm_order(mesh);
    const Mesh ordered = renumber_cells(mesh, new_to_old, &map);
    count_ordered(ordered, renumber_geometry(geometry, new_to_old, map), out);
  } else {
    out.line_order = "input";
    count_ordered(mesh, geometry, out);
  }

  // The stencil and ghosts are sets, the same in any cell order. On one rank, the run's partition
  // is its one part (Run::make_part): the mesh itself without periodic pairs, else the part
  // whose image ghosts make each periodic face interior.
  MeshInfoRun::Partition one_part{.ranks = 1,
                                  .groups = "none",
                                  .cells = {},
                                  .ghosts = {},
                                  .faces = {},
                                  .interior_faces = {},
                                  .stencil_entries = {},
                                  .nodes = {},
                                  .face_nodes = {},
                                  .sent_cells = {}};
  const auto count_part = [](MeshInfoRun::Partition& p, const LocalPartition& part,
                             std::int64_t stencil) {
    p.cells.push_back(part.mesh.n_owned_cells());
    p.ghosts.push_back(part.mesh.n_cells() - part.mesh.n_owned_cells());
    p.faces.push_back(part.mesh.n_faces());
    p.interior_faces.push_back(part.mesh.n_interior_faces());
    p.stencil_entries.push_back(stencil);
    p.nodes.push_back(part.mesh.n_points());
    p.face_nodes.push_back(static_cast<std::int64_t>(part.mesh.face_nodes().size()));
    p.sent_cells.push_back(static_cast<std::int64_t>(part.halo.send_cells.size()));
  };
  if (mesh.periodic_pairs().empty()) {
    out.stencil_entries = stencil_entries(mesh, geometry);
    one_part.cells = {mesh.n_cells()};
    one_part.ghosts = {0};
    one_part.faces = {mesh.n_faces()};
    one_part.interior_faces = {mesh.n_interior_faces()};
    one_part.stencil_entries = {out.stencil_entries};
    one_part.nodes = {mesh.n_points()};
    one_part.face_nodes = {static_cast<std::int64_t>(mesh.face_nodes().size())};
    one_part.sent_cells = {0};
  } else {
    const std::vector<std::int32_t> one(static_cast<std::size_t>(mesh.n_cells()), 0);
    LocalPartition part = extract_partition(mesh, geometry, one, 1, 0);
    fill_ghost_geometry(part);
    out.periodic_ghosts = part.mesh.n_cells() - part.mesh.n_owned_cells();
    out.stencil_entries = stencil_entries(part.mesh, part.geometry);
    count_part(one_part, part, out.stencil_entries);
  }

  if (!ranks || ranks->ranks == 1) {
    out.partition = std::move(one_part);
    return out;
  }
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
                           .stencil_entries = {},
                           .nodes = {},
                           .face_nodes = {},
                           .sent_cells = {}};
  for (std::int32_t r = 0; r < ranks->ranks; ++r) {
    // Ghost cells hold no geometry here: the counts depend on the connectivity only.
    const LocalPartition part = extract_partition(mesh, geometry, parts, ranks->ranks, r);
    count_part(p, part, stencil_entries(part.mesh, part.geometry));
  }
  out.partition = std::move(p);
  return out;
}

}  // namespace autan::io
