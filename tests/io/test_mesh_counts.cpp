#include <cstdint>
#include <cstdlib>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/mesh_counts.hpp"
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
                              std::optional<autan::io::RanksRequest> ranks = std::nullopt) {
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cw)));
  return autan::io::run_counts(mesh, autan::compute_geometry(mesh), order, ranks);
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

}  // namespace

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
}

TEST_CASE("MeshInfo counts a single-rank run's stencil entries and periodic ghosts",
          "[mesh-info]") {
  const int n = 4;
  const auto plain = counts(box(n, 1.0), CellOrder::rcm);
  CHECK(plain.stencil_entries == face2_entries(n));
  CHECK(plain.periodic_ghosts == 0);
  CHECK_FALSE(plain.partition);
  // Periodic along x: two layers of images across each of xmin and xmax (extract_partition).
  auto cw = box(n, 1.0);
  cw.face_zones[0].kind = autan::BcKind::periodic;
  cw.face_zones[1].kind = autan::BcKind::periodic;
  cw.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const auto periodic = counts(std::move(cw), CellOrder::rcm);
  CHECK(periodic.periodic_ghosts == 2 * 2 * n * n);
  CHECK(periodic.stencil_entries > plain.stencil_entries);
}

TEST_CASE("MeshInfo counts each rank of a partitioned run", "[mesh-info]") {
  const int n = 4;
  const auto groups = GENERATE(autan::io::RankGroups::none, autan::io::RankGroups::lines);
  CAPTURE(groups == autan::io::RankGroups::lines);
  const auto c =
      counts(box(n, 10.0), CellOrder::rcm, autan::io::RanksRequest{.ranks = 3, .groups = groups});
  REQUIRE(c.partition);
  const auto& p = *c.partition;
  CHECK(p.ranks == 3);
  CHECK(p.groups == (groups == autan::io::RankGroups::lines ? "lines" : "none"));
  REQUIRE(p.cells.size() == 3);
  std::int64_t owned = 0;
  for (std::size_t r = 0; r < 3; ++r) {
    CAPTURE(r);
    owned += p.cells[r];
    CHECK(p.ghosts[r] > 0);
    CHECK(p.interior_faces[r] < p.faces[r]);
    CHECK(p.stencil_entries[r] > 0);
    // The lines are the box's columns: kept whole, each rank owns whole columns.
    if (groups == autan::io::RankGroups::lines) CHECK(p.cells[r] % n == 0);
  }
  CHECK(owned == n * n * n);
  // More ranks than cells: the partitioner's refusal (E0140).
  try {
    (void)counts(box(2, 1.0), CellOrder::rcm, autan::io::RanksRequest{.ranks = 9, .groups = {}});
    FAIL("partitioned 8 cells over 9 ranks");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::partition_invalid_request);
  }
}
