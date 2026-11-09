#include "autan/io/mesh_counts.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// Each cell's nodes and each node's cells and boundary faces (CSR): the vertex cell stencil
// (kernels::cell_stencil, StencilKind::vertex) without materialising it. A cell's vertex
// stencil is the other cells at its nodes and the boundary faces with one of its nodes.
struct Incidence {
  std::vector<std::int64_t> cell_off, node_off, bface_off;
  std::vector<index_t> cell_nodes, node_cells, node_bfaces;
};

Incidence incidence(const Mesh& mesh) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nn = static_cast<std::size_t>(mesh.n_points());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto nint = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  std::vector<std::int64_t> face_off(nc + 1, 0);
  for (std::size_t f = 0; f < nf; ++f) {
    ++face_off[static_cast<std::size_t>(owner[f]) + 1];
    if (f < nint) ++face_off[static_cast<std::size_t>(neighbour[f]) + 1];
  }
  for (std::size_t c = 0; c < nc; ++c) face_off[c + 1] += face_off[c];
  std::vector<index_t> faces(static_cast<std::size_t>(face_off[nc]));
  {
    std::vector<std::int64_t> at(face_off.begin(), face_off.end() - 1);
    for (std::size_t f = 0; f < nf; ++f) {
      faces[static_cast<std::size_t>(at[static_cast<std::size_t>(owner[f])]++)] =
          static_cast<index_t>(f);
      if (f < nint)
        faces[static_cast<std::size_t>(at[static_cast<std::size_t>(neighbour[f])]++)] =
            static_cast<index_t>(f);
    }
  }
  Incidence in;
  constexpr auto unseen = static_cast<std::size_t>(-1);
  std::vector<std::size_t> stamp(nn, unseen);
  // A cell's distinct nodes, counted, then listed.
  const auto each_node = [&](std::size_t c, auto&& visit) {
    for (auto k = face_off[c]; k < face_off[c + 1]; ++k) {
      for (const index_t v : mesh.face_nodes(faces[static_cast<std::size_t>(k)])) {
        if (stamp[static_cast<std::size_t>(v)] != c) {
          stamp[static_cast<std::size_t>(v)] = c;
          visit(v);
        }
      }
    }
  };
  in.cell_off.assign(nc + 1, 0);
  in.node_off.assign(nn + 1, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    each_node(c, [&](index_t v) {
      ++in.cell_off[c + 1];
      ++in.node_off[static_cast<std::size_t>(v) + 1];
    });
  }
  for (std::size_t c = 0; c < nc; ++c) in.cell_off[c + 1] += in.cell_off[c];
  for (std::size_t v = 0; v < nn; ++v) in.node_off[v + 1] += in.node_off[v];
  in.cell_nodes.resize(static_cast<std::size_t>(in.cell_off[nc]));
  in.node_cells.resize(static_cast<std::size_t>(in.node_off[nn]));
  std::ranges::fill(stamp, unseen);
  std::vector<std::int64_t> at(in.node_off.begin(), in.node_off.end() - 1);
  for (std::size_t c = 0; c < nc; ++c) {
    auto k = in.cell_off[c];
    each_node(c, [&](index_t v) {
      in.cell_nodes[static_cast<std::size_t>(k++)] = v;
      in.node_cells[static_cast<std::size_t>(at[static_cast<std::size_t>(v)]++)] =
          static_cast<index_t>(c);
    });
  }
  in.bface_off.assign(nn + 1, 0);
  for (std::size_t f = nint; f < nf; ++f) {
    for (const index_t v : mesh.face_nodes(static_cast<index_t>(f)))
      ++in.bface_off[static_cast<std::size_t>(v) + 1];
  }
  for (std::size_t v = 0; v < nn; ++v) in.bface_off[v + 1] += in.bface_off[v];
  in.node_bfaces.resize(static_cast<std::size_t>(in.bface_off[nn]));
  std::ranges::copy(in.bface_off.begin(), in.bface_off.end() - 1, at.begin());
  for (std::size_t f = nint; f < nf; ++f) {
    for (const index_t v : mesh.face_nodes(static_cast<index_t>(f)))
      in.node_bfaces[static_cast<std::size_t>(at[static_cast<std::size_t>(v)]++)] =
          static_cast<index_t>(f - nint);
  }
  return in;
}

// The cells within `rings` vertex rings of c, c excluded, and the boundary faces in the vertex
// stencils of those within rings - 1: what the quadratic gradient's band (two rings) and the
// viscous boundary fit (two, or three where two leave it undetermined) gather when every
// boundary face gives a value (physics/src/euler.cpp band_weights, navier_stokes.cpp
// fit_weights). `mark`, `nmark` and `bmark` hold the last cell that listed each cell, node and
// boundary face: a node's cells and faces are listed once.
struct Walk {
  std::vector<index_t> mark, nmark, bmark, layer, next;
};

std::int64_t gathered(const Incidence& in, index_t c, int rings, Walk& w) {
  std::int64_t count = 0;
  w.layer.assign(1, c);
  w.mark[static_cast<std::size_t>(c)] = c;
  for (int r = 0; r < rings; ++r) {
    w.next.clear();
    for (const index_t e : w.layer) {
      const auto i = static_cast<std::size_t>(e);
      for (auto k = in.cell_off[i]; k < in.cell_off[i + 1]; ++k) {
        const auto v = static_cast<std::size_t>(in.cell_nodes[static_cast<std::size_t>(k)]);
        if (w.nmark[v] == c) continue;
        w.nmark[v] = c;
        for (auto j = in.node_off[v]; j < in.node_off[v + 1]; ++j) {
          const index_t x = in.node_cells[static_cast<std::size_t>(j)];
          if (w.mark[static_cast<std::size_t>(x)] != c) {
            w.mark[static_cast<std::size_t>(x)] = c;
            w.next.push_back(x);
            ++count;
          }
        }
        for (auto j = in.bface_off[v]; j < in.bface_off[v + 1]; ++j) {
          auto& m = w.bmark[static_cast<std::size_t>(in.node_bfaces[static_cast<std::size_t>(j)])];
          if (m != c) {
            m = c;
            ++count;
          }
        }
      }
    }
    w.layer.swap(w.next);
  }
  return count;
}

Stencils stencils(const Mesh& mesh, const Geometry& geometry) {
  Stencils out;
  out.face2 =
      static_cast<std::int64_t>(kernels::cell_stencil(mesh, kernels::pack_geometry(mesh, geometry),
                                                      kernels::StencilKind::face2)
                                    .entries.size());
  const Incidence in = incidence(mesh);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nint = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  // The cells the boundary fit serves when every boundary face gives the temperature and the
  // velocity: their owners and the owners' face neighbours.
  std::vector<char> fitted(nc, 0);
  for (auto f = nint; f < owner.size(); ++f) fitted[static_cast<std::size_t>(owner[f])] = 1;
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    const auto a = static_cast<std::size_t>(owner[f]), b = static_cast<std::size_t>(neighbour[f]);
    if (fitted[a] == 1 && fitted[b] == 0) fitted[b] = 2;
    if (fitted[b] == 1 && fitted[a] == 0) fitted[a] = 2;
  }
  const auto fresh = [&] {
    return Walk{.mark = std::vector<index_t>(nc, -1),
                .nmark = std::vector<index_t>(static_cast<std::size_t>(mesh.n_points()), -1),
                .bmark = std::vector<index_t>(owner.size() - nint, -1),
                .layer = {},
                .next = {}};
  };
  Walk w = fresh();
  for (std::size_t c = 0; c < nc; ++c) {
    out.vertex += gathered(in, static_cast<index_t>(c), 1, w);
  }
  w = fresh();
  for (std::size_t c = 0; c < nc; ++c) out.band += gathered(in, static_cast<index_t>(c), 2, w);
  w = fresh();
  for (std::size_t c = 0; c < nc; ++c) {
    if (fitted[c] != 0) out.boundary_fit += gathered(in, static_cast<index_t>(c), 3, w);
  }
  return out;
}

// The smoother's lines, which depend on the cell order.
void count_lines(const Mesh& mesh, const Geometry& geometry, MeshInfoRun& out) {
  const Lines lines = smoother_lines(mesh, geometry);
  out.line_cells = static_cast<std::int64_t>(cells_in_lines(lines));
  out.long_line_cells = static_cast<std::int64_t>(cells_in_lines(lines, long_line));
}

// The multigrid's hierarchy, built with the run's default options; it depends on the cell order.
void count_levels(const Mesh& mesh, const Geometry& geometry, MeshInfoRun& out) {
  for (const auto& level : solvers::agglomerate(mesh, geometry)) {
    out.multigrid_cells.push_back(static_cast<std::int64_t>(level.n_nodes));
    out.multigrid_edges.push_back(static_cast<std::int64_t>(level.owner.size()));
  }
}

const char* order_name(CellOrder order) { return order == CellOrder::rcm ? "rcm" : "input"; }

// Calls visit with the mesh and its geometry in `order`: as read for input, renumbered by
// rcm_order (renumber_cells, renumber_geometry) for rcm, as the run renumbers.
template <class Visit>
void in_order(const Mesh& mesh, const Geometry& geometry, CellOrder order, const Visit& visit) {
  if (order == CellOrder::input) {
    visit(mesh, geometry);
    return;
  }
  AssemblyMap map;
  const std::vector<index_t> new_to_old = rcm_order(mesh);
  const Mesh ordered = renumber_cells(mesh, new_to_old, &map);
  visit(ordered, renumber_geometry(geometry, new_to_old, map));
}

}  // namespace

MeshInfoRun run_counts(const Mesh& read, double scale, CellOrder order,
                       std::optional<RanksRequest> ranks,
                       std::optional<CellOrder> multigrid_order) {
  // As prepare() does: the coordinates in metres before any geometry.
  std::optional<Mesh> metres;
  const Mesh& mesh = scale == 1.0 ? read : metres.emplace(scaled(read, scale));
  const Geometry geometry = compute_geometry(mesh);
  const CellOrder levels_order = multigrid_order.value_or(order);
  MeshInfoRun out;
  out.line_scale = scale;
  out.line_order = order_name(order);
  out.multigrid_order = order_name(levels_order);
  in_order(mesh, geometry, order, [&](const Mesh& m, const Geometry& g) {
    count_lines(m, g, out);
    if (levels_order == order) count_levels(m, g, out);
  });
  if (levels_order != order) {
    in_order(mesh, geometry, levels_order,
             [&](const Mesh& m, const Geometry& g) { count_levels(m, g, out); });
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
