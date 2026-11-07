#include "autan/io/mesh_counts.hpp"

#include <algorithm>
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

struct Stencils {
  std::int64_t face2 = 0, vertex = 0, band = 0, boundary_fit = 0;
};

// The cells within `rings` vertex rings of c, c excluded, and the boundary faces in the vertex
// stencils of those within rings - 1: what the quadratic gradient's band (two rings) and the
// viscous boundary fit (two, or three where two leave it undetermined) gather when every
// boundary face gives a value (physics/src/euler.cpp band_weights, navier_stokes.cpp
// fit_weights). `mark` and `bmark` hold the last cell that listed each cell and boundary face.
std::int64_t gathered(const kernels::CellStencil& vertex, index_t c, int rings,
                      std::vector<index_t>& mark, std::vector<index_t>& bmark, index_t nint,
                      std::vector<index_t>& layer, std::vector<index_t>& next) {
  std::int64_t count = 0;
  layer.assign(1, c);
  mark[static_cast<std::size_t>(c)] = c;
  for (int r = 0; r < rings; ++r) {
    next.clear();
    for (const index_t e : layer) {
      const auto i = static_cast<std::size_t>(e);
      for (auto k = vertex.offsets[i]; k < vertex.offsets[i + 1]; ++k) {
        const index_t x = vertex.entries[static_cast<std::size_t>(k)];
        if (x >= 0) {
          if (mark[static_cast<std::size_t>(x)] != c) {
            mark[static_cast<std::size_t>(x)] = c;
            next.push_back(x);
            ++count;
          }
        } else if (auto& m = bmark[static_cast<std::size_t>(-1 - x - nint)]; m != c) {
          m = c;
          ++count;
        }
      }
    }
    layer.swap(next);
  }
  return count;
}

Stencils stencils(const Mesh& mesh, const Geometry& geometry) {
  const auto packed = kernels::pack_geometry(mesh, geometry);
  Stencils out;
  out.face2 = static_cast<std::int64_t>(
      kernels::cell_stencil(mesh, packed, kernels::StencilKind::face2).entries.size());
  const auto vertex = kernels::cell_stencil(mesh, packed, kernels::StencilKind::vertex);
  out.vertex = static_cast<std::int64_t>(vertex.entries.size());
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const index_t nint = mesh.n_interior_faces();
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  // The cells the boundary fit serves when every boundary face is a wall: their owners and
  // the owners' face neighbours.
  std::vector<char> fitted(nc, 0);
  for (auto f = static_cast<std::size_t>(nint); f < owner.size(); ++f)
    fitted[static_cast<std::size_t>(owner[f])] = 1;
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    const auto a = static_cast<std::size_t>(owner[f]), b = static_cast<std::size_t>(neighbour[f]);
    if (fitted[a] == 1 && fitted[b] == 0) fitted[b] = 2;
    if (fitted[b] == 1 && fitted[a] == 0) fitted[a] = 2;
  }
  std::vector<index_t> mark(nc, -1), bmark(owner.size() - static_cast<std::size_t>(nint), -1);
  std::vector<index_t> layer, next;
  for (std::size_t c = 0; c < nc; ++c) {
    const auto cell = static_cast<index_t>(c);
    out.band += gathered(vertex, cell, 2, mark, bmark, nint, layer, next);
  }
  std::ranges::fill(mark, -1);
  std::ranges::fill(bmark, -1);
  for (std::size_t c = 0; c < nc; ++c) {
    if (fitted[c] != 0)
      out.boundary_fit +=
          gathered(vertex, static_cast<index_t>(c), 3, mark, bmark, nint, layer, next);
  }
  return out;
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
                                  .vertex_stencil_entries = {},
                                  .band_entries = {},
                                  .boundary_fit_entries = {},
                                  .nodes = {},
                                  .face_nodes = {},
                                  .sent_cells = {}};
  const auto count_part = [](MeshInfoRun::Partition& p, const LocalPartition& part,
                             const Stencils& stencil) {
    p.cells.push_back(part.mesh.n_owned_cells());
    p.ghosts.push_back(part.mesh.n_cells() - part.mesh.n_owned_cells());
    p.faces.push_back(part.mesh.n_faces());
    p.interior_faces.push_back(part.mesh.n_interior_faces());
    p.stencil_entries.push_back(stencil.face2);
    p.vertex_stencil_entries.push_back(stencil.vertex);
    p.band_entries.push_back(stencil.band);
    p.boundary_fit_entries.push_back(stencil.boundary_fit);
    p.nodes.push_back(part.mesh.n_points());
    p.face_nodes.push_back(static_cast<std::int64_t>(part.mesh.face_nodes().size()));
    p.sent_cells.push_back(static_cast<std::int64_t>(part.halo.send_cells.size()));
  };
  const auto set_counts = [&out](const Stencils& st) {
    out.stencil_entries = st.face2;
    out.vertex_stencil_entries = st.vertex;
    out.band_entries = st.band;
    out.boundary_fit_entries = st.boundary_fit;
  };
  if (mesh.periodic_pairs().empty()) {
    set_counts(stencils(mesh, geometry));
    one_part.cells = {mesh.n_cells()};
    one_part.ghosts = {0};
    one_part.faces = {mesh.n_faces()};
    one_part.interior_faces = {mesh.n_interior_faces()};
    one_part.stencil_entries = {out.stencil_entries};
    one_part.vertex_stencil_entries = {out.vertex_stencil_entries};
    one_part.band_entries = {out.band_entries};
    one_part.boundary_fit_entries = {out.boundary_fit_entries};
    one_part.nodes = {mesh.n_points()};
    one_part.face_nodes = {static_cast<std::int64_t>(mesh.face_nodes().size())};
    one_part.sent_cells = {0};
  } else {
    const std::vector<std::int32_t> one(static_cast<std::size_t>(mesh.n_cells()), 0);
    LocalPartition part = extract_partition(mesh, geometry, one, 1, 0);
    fill_ghost_geometry(part);
    out.periodic_ghosts = part.mesh.n_cells() - part.mesh.n_owned_cells();
    const Stencils st = stencils(part.mesh, part.geometry);
    set_counts(st);
    count_part(one_part, part, st);
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
                           .vertex_stencil_entries = {},
                           .band_entries = {},
                           .boundary_fit_entries = {},
                           .nodes = {},
                           .face_nodes = {},
                           .sent_cells = {}};
  for (std::int32_t r = 0; r < ranks->ranks; ++r) {
    // Ghost cells hold no geometry here: the counts depend on the connectivity only.
    const LocalPartition part = extract_partition(mesh, geometry, parts, ranks->ranks, r);
    count_part(p, part, stencils(part.mesh, part.geometry));
  }
  out.partition = std::move(p);
  return out;
}

}  // namespace autan::io
