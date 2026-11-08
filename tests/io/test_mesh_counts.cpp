#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/mesh_counts.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/partition/partition.hpp"
#include "fixtures.hpp"

namespace {

using autan::io::CellOrder;

// The unit cube cut into n^3 hexahedra, stretched by `stretch` along x and y.
autan::CellwiseMesh box(int n, double stretch) {
  auto cw = autan::test::box(n, autan::test::BoxFill::hexa);
  for (auto& v : cw.x) v *= stretch;
  for (auto& v : cw.y) v *= stretch;
  return cw;
}

autan::io::MeshInfoRun counts(autan::CellwiseMesh cw, CellOrder order,
                              std::optional<autan::io::RanksRequest> ranks = std::nullopt,
                              autan::GhostStencil ghosts = autan::GhostStencil::face2) {
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cw)));
  return autan::io::run_counts(mesh, 1.0, order, ranks, ghosts);
}

// The face2 stencil of the n^3 hexahedra of a box, from its definition (kernels::StencilKind):
// each cell's face neighbours and theirs, itself excluded, are the cells at L1 index distance 1
// or 2; its boundary faces and those of its face neighbours are their sides on the box's faces.
std::int64_t face2_entries(int n) {
  const auto in = [n](int i) { return i >= 0 && i < n; };
  const auto sides = [n](int i, int j, int k) {
    int s = 0;
    for (const int x : {i, j, k}) s += (x == 0 ? 1 : 0) + (x == n - 1 ? 1 : 0);
    return s;
  };
  std::int64_t total = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        for (int a = -2; a <= 2; ++a) {
          for (int b = -2; b <= 2; ++b) {
            for (int c = -2; c <= 2; ++c) {
              const int d = std::abs(a) + std::abs(b) + std::abs(c);
              if (!in(i + a) || !in(j + b) || !in(k + c)) continue;
              if (d == 1 || d == 2) ++total;
              if (d <= 1) total += sides(i + a, j + b, k + c);
            }
          }
        }
      }
    }
  }
  return total;
}

struct Rings {
  std::int64_t vertex = 0, band = 0, boundary_fit = 0;
};

// The ring counts of the n^3 hexahedra of a box, from their definition (mesh-info.md): the cells
// within r vertex rings of a cell are those at Chebyshev index distance 1 to r; the boundary
// faces in the vertex stencils of the cells within r - 1 rings are, on each side of the box
// within r - 1 cells of the cell, the squares within r of it along the side.
Rings box_rings(int n) {
  const auto within = [n](int i, int r) { return std::min(i + r, n - 1) - std::max(i - r, 0) + 1; };
  const auto gathered = [&](int i, int j, int k, int r) {
    const std::array<int, 3> at{i, j, k};
    std::int64_t total = std::int64_t{within(i, r)} * within(j, r) * within(k, r) - 1;
    for (std::size_t a = 0; a < 3; ++a) {
      const std::size_t b = (a + 1) % 3, c = (a + 2) % 3;
      const std::int64_t side = std::int64_t{within(at[b], r)} * within(at[c], r);
      if (at[a] <= r - 1) total += side;
      if (n - 1 - at[a] <= r - 1) total += side;
    }
    return total;
  };
  Rings out;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        out.vertex += gathered(i, j, k, 1);
        out.band += gathered(i, j, k, 2);
        // The fit serves the cells owning a boundary face and their face neighbours.
        const int edge = std::min({i, j, k, n - 1 - i, n - 1 - j, n - 1 - k});
        if (edge <= 1) out.boundary_fit += gathered(i, j, k, 3);
      }
    }
  }
  return out;
}

// The same counts from the mesh's face-to-node connectivity alone, without the cell stencil:
// two cells are in each other's ring when they share a node, a boundary face is in a cell's
// stencil when it has one of the cell's nodes.
Rings counted_rings(const autan::Mesh& mesh) {
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  const auto nint = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  std::vector<std::set<autan::index_t>> nodes(nc);
  std::vector<std::vector<std::size_t>> cells_at(static_cast<std::size_t>(mesh.n_points()));
  std::vector<std::vector<std::size_t>> bfaces_at(cells_at.size());
  for (std::size_t f = 0; f < nf; ++f) {
    for (const autan::index_t v : mesh.face_nodes(static_cast<autan::index_t>(f))) {
      nodes[static_cast<std::size_t>(owner[f])].insert(v);
      if (f < nint) nodes[static_cast<std::size_t>(neighbour[f])].insert(v);
      if (f >= nint) bfaces_at[static_cast<std::size_t>(v)].push_back(f);
    }
  }
  for (std::size_t c = 0; c < nc; ++c) {
    for (const autan::index_t v : nodes[c]) cells_at[static_cast<std::size_t>(v)].push_back(c);
  }
  const auto gathered = [&](std::size_t c, int rings) {
    std::set<std::size_t> cells{c}, faces;
    for (int r = 0; r < rings; ++r) {
      std::set<std::size_t> next = cells;
      for (const std::size_t e : cells) {
        for (const autan::index_t v : nodes[e]) {
          for (const std::size_t x : cells_at[static_cast<std::size_t>(v)]) next.insert(x);
          for (const std::size_t f : bfaces_at[static_cast<std::size_t>(v)]) faces.insert(f);
        }
      }
      cells = std::move(next);
    }
    return static_cast<std::int64_t>(cells.size() - 1 + faces.size());
  };
  std::vector<int> fitted(nc, 0);
  for (std::size_t f = nint; f < nf; ++f) fitted[static_cast<std::size_t>(owner[f])] = 1;
  for (std::size_t f = 0; f < nint; ++f) {
    const auto a = static_cast<std::size_t>(owner[f]), b = static_cast<std::size_t>(neighbour[f]);
    if (fitted[a] == 1 && fitted[b] == 0) fitted[b] = 2;
    if (fitted[b] == 1 && fitted[a] == 0) fitted[a] = 2;
  }
  Rings out;
  for (std::size_t c = 0; c < nc; ++c) {
    out.vertex += gathered(c, 1);
    out.band += gathered(c, 2);
    if (fitted[c] != 0) out.boundary_fit += gathered(c, 3);
  }
  return out;
}

}  // namespace

TEST_CASE("MeshInfo counts the vertex rings of the quadratic gradient and the boundary fit",
          "[mesh-info]") {
  // A box of hexahedra against the formula, and the same box, tetrahedra, a hybrid box and a
  // polyhedral mesh against an independent count over the node connectivity.
  for (const int n : {4, 7}) {
    CAPTURE(n);
    const auto c = counts(box(n, 1.0), CellOrder::rcm);
    const Rings expected = box_rings(n);
    CHECK(c.vertex_stencil_entries == expected.vertex);
    CHECK(c.band_entries == expected.band);
    CHECK(c.boundary_fit_entries == expected.boundary_fit);
  }
  for (const auto fill :
       {autan::test::BoxFill::hexa, autan::test::BoxFill::tetra, autan::test::BoxFill::hybrid}) {
    CAPTURE(autan::test::to_string(fill));
    const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::box(5, fill)));
    const auto c = autan::io::run_counts(mesh, 1.0, CellOrder::input);
    const Rings expected = counted_rings(mesh);
    CHECK(c.vertex_stencil_entries == expected.vertex);
    // The runtime's own vertex stencil, the one the quadratic gradient walks.
    const auto runtime_stencil = autan::kernels::cell_stencil(
        mesh, autan::kernels::pack_geometry(mesh, autan::compute_geometry(mesh)),
        autan::kernels::StencilKind::vertex);
    CHECK(c.vertex_stencil_entries == static_cast<std::int64_t>(runtime_stencil.entries.size()));
    CHECK(c.band_entries == expected.band);
    CHECK(c.boundary_fit_entries == expected.boundary_fit);
    REQUIRE(c.partition);
    CHECK(c.partition->vertex_stencil_entries == std::vector<std::int64_t>{expected.vertex});
    CHECK(c.partition->band_entries == std::vector<std::int64_t>{expected.band});
    CHECK(c.partition->boundary_fit_entries == std::vector<std::int64_t>{expected.boundary_fit});
  }
  const autan::Mesh poly =
      autan::io::read_mesh(std::filesystem::path(AUTAN_TEST_MESH_DIR) / "poly" / "polyMesh");
  const auto c = autan::io::run_counts(poly, 1.0, CellOrder::rcm);
  const Rings expected = counted_rings(poly);
  CHECK(c.vertex_stencil_entries == expected.vertex);
  CHECK(c.band_entries == expected.band);
  CHECK(c.boundary_fit_entries == expected.boundary_fit);
}

TEST_CASE("MeshInfo counts the cells in the smoother's lines", "[mesh-info]") {
  // Hexahedra 10 times wider than tall couple 100 times more strongly across their z faces
  // (A_f / |d|) than across the others: each column of n cells is a line (Mavriplis 1998), long
  // from 4 cells (long_line). Cubes couple equally in every direction: no line.
  const int n = GENERATE(3, 4);
  const CellOrder order = GENERATE(CellOrder::rcm, CellOrder::input);
  CAPTURE(n, order == CellOrder::rcm);
  const auto flat = counts(box(n, 10.0), order);
  CHECK(flat.line_order == (order == CellOrder::rcm ? "rcm" : "input"));
  CHECK(flat.line_cells == n * n * n);
  CHECK(flat.long_line_cells == (n >= 4 ? n * n * n : 0));
  const auto cubes = counts(box(n, 1.0), order);
  CHECK(cubes.line_cells == 0);
  CHECK(cubes.long_line_cells == 0);
  // The multigrid's finest level is the mesh: its cells, and its interior faces 3 n^2 (n - 1).
  for (const auto& c : {flat, cubes}) {
    REQUIRE(!c.multigrid_cells.empty());
    CHECK(c.multigrid_cells.size() == c.multigrid_edges.size());
    CHECK(c.multigrid_cells[0] == n * n * n);
    CHECK(c.multigrid_edges[0] == 3 * n * n * (n - 1));
    for (std::size_t l = 1; l < c.multigrid_cells.size(); ++l)
      CHECK(c.multigrid_cells[l] < c.multigrid_cells[l - 1]);
  }
}

TEST_CASE("MeshInfo counts a single-rank run's stencil entries and periodic ghosts",
          "[mesh-info]") {
  const int n = 4;
  const auto plain = counts(box(n, 1.0), CellOrder::rcm);
  CHECK(plain.stencil_entries == face2_entries(n));
  CHECK(plain.periodic_ghosts == 0);
  // On one rank the partition is the mesh itself: a box of n^3 hexahedra.
  REQUIRE(plain.partition);
  const auto& one = *plain.partition;
  CHECK(one.ranks == 1);
  CHECK(one.groups == "none");
  CHECK(one.cells == std::vector<std::int64_t>{n * n * n});
  CHECK(one.ghosts == std::vector<std::int64_t>{0});
  CHECK(one.faces == std::vector<std::int64_t>{3 * n * n * (n + 1)});
  CHECK(one.interior_faces == std::vector<std::int64_t>{3 * n * n * (n - 1)});
  CHECK(one.nodes == std::vector<std::int64_t>{(n + 1) * (n + 1) * (n + 1)});
  CHECK(one.face_nodes == std::vector<std::int64_t>{4 * 3 * n * n * (n + 1)});
  CHECK(one.sent_cells == std::vector<std::int64_t>{0});
  CHECK(one.stencil_entries == std::vector<std::int64_t>{face2_entries(n)});
  // Periodic along x: two layers of images across each of xmin and xmax (extract_partition).
  auto cw = box(n, 1.0);
  cw.face_zones[0].kind = autan::BcKind::periodic;
  cw.face_zones[1].kind = autan::BcKind::periodic;
  cw.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const auto periodic = counts(std::move(cw), CellOrder::rcm);
  CHECK(periodic.periodic_ghosts == 2 * 2 * n * n);
  // Its one part: the image ghosts, each sent by the rank to itself, and their faces' nodes.
  REQUIRE(periodic.partition);
  const auto& images = *periodic.partition;
  CHECK(images.ranks == 1);
  CHECK(images.ghosts == std::vector<std::int64_t>{2 * 2 * n * n});
  CHECK(images.sent_cells == images.ghosts);
  CHECK(images.face_nodes[0] == 4 * images.faces[0]);
  CHECK(images.nodes[0] > (n + 1) * (n + 1) * (n + 1));
  CHECK(images.stencil_entries == std::vector<std::int64_t>{periodic.stencil_entries});
  // The rings over the image ghosts too: the counts of the one part's local mesh.
  auto pw = box(n, 1.0);
  pw.face_zones[0].kind = autan::BcKind::periodic;
  pw.face_zones[1].kind = autan::BcKind::periodic;
  pw.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(pw)));
  const std::vector<std::int32_t> rank0(static_cast<std::size_t>(mesh.n_cells()), 0);
  const Rings ring = counted_rings(
      autan::extract_partition(mesh, autan::compute_geometry(mesh), rank0, 1, 0).mesh);
  CHECK(periodic.vertex_stencil_entries == ring.vertex);
  CHECK(periodic.band_entries == ring.band);
  CHECK(periodic.boundary_fit_entries == ring.boundary_fit);
  CHECK(images.vertex_stencil_entries == std::vector<std::int64_t>{ring.vertex});
  CHECK(images.band_entries == std::vector<std::int64_t>{ring.band});
  CHECK(images.boundary_fit_entries == std::vector<std::int64_t>{ring.boundary_fit});
  CHECK(periodic.stencil_entries > plain.stencil_entries);
}

#ifdef AUTAN_HAVE_METIS
TEST_CASE("MeshInfo counts each rank of a partitioned run", "[mesh-info]") {
  const int n = 4;
  const auto groups = GENERATE(autan::io::RankGroups::none, autan::io::RankGroups::lines);
  const auto ghost_stencil = GENERATE(autan::GhostStencil::face2, autan::GhostStencil::vertex,
                                      autan::GhostStencil::vertex4);
  CAPTURE(groups == autan::io::RankGroups::lines, autan::name(ghost_stencil));
  const auto c = counts(box(n, 10.0), CellOrder::rcm,
                        autan::io::RanksRequest{.ranks = 3, .groups = groups}, ghost_stencil);
  REQUIRE(c.partition);
  const auto& p = *c.partition;
  CHECK(p.ranks == 3);
  CHECK(p.groups == (groups == autan::io::RankGroups::lines ? "lines" : "none"));
  CHECK(p.ghost_stencil == autan::name(ghost_stencil));
  REQUIRE(p.cells.size() == 3);
  std::int64_t owned = 0, ghosts = 0, sent = 0;
  for (std::size_t r = 0; r < 3; ++r) {
    CAPTURE(r);
    owned += p.cells[r];
    ghosts += p.ghosts[r];
    sent += p.sent_cells[r];
    // Hexahedra: every face has 4 nodes; a rank's nodes are some of the box's (n + 1)^3.
    CHECK(p.face_nodes[r] == 4 * p.faces[r]);
    CHECK(p.nodes[r] > 0);
    CHECK(p.nodes[r] <= (n + 1) * (n + 1) * (n + 1));
    CHECK(p.ghosts[r] > 0);
    CHECK(p.interior_faces[r] < p.faces[r]);
    CHECK(p.stencil_entries[r] > 0);
    // The lines are the box's columns: kept whole, each rank owns whole columns.
    if (groups == autan::io::RankGroups::lines) CHECK(p.cells[r] % n == 0);
  }
  CHECK(owned == n * n * n);
  // Each ghost is one cell its owner sends.
  CHECK(sent == ghosts);
  // Each rank's rings over its local mesh, ghosts included, as the run partitions it.
  if (groups == autan::io::RankGroups::none) {
    const autan::Mesh mesh = autan::assemble(autan::to_face_based(box(n, 10.0)));
    const auto geometry = autan::compute_geometry(mesh);
    const auto parts = autan::partition_cells(
        mesh, 3,
        {.cell_weights = autan::cell_cost_weights(mesh), .group_offsets = {}, .group_cells = {}});
    for (std::int32_t r = 0; r < 3; ++r) {
      CAPTURE(r);
      const auto part = autan::extract_partition(mesh, geometry, parts, 3, r, ghost_stencil);
      const Rings ring = counted_rings(part.mesh);
      const auto i = static_cast<std::size_t>(r);
      CHECK(p.ghosts[i] == part.mesh.n_cells() - part.mesh.n_owned_cells());
      CHECK(p.vertex_stencil_entries[i] == ring.vertex);
      CHECK(p.band_entries[i] == ring.band);
      CHECK(p.boundary_fit_entries[i] == ring.boundary_fit);
    }
  }
  // More ranks than cells: the partitioner's refusal (E0140).
  try {
    (void)counts(box(2, 1.0), CellOrder::rcm, autan::io::RanksRequest{.ranks = 9, .groups = {}});
    FAIL("partitioned 8 cells over 9 ranks");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::partition_invalid_request);
  }
}
#else
TEST_CASE("MeshInfo has no partition counts without METIS", "[mesh-info]") {
  // The partitioner the run uses is not built: the refusal a run on several ranks would give.
  try {
    (void)counts(box(4, 1.0), CellOrder::rcm, autan::io::RanksRequest{.ranks = 2, .groups = {}});
    FAIL("partitioned without METIS");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::partition_backend_failed);
  }
}
#endif
