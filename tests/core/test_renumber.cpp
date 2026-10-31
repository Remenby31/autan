#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinRel;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

index_t bandwidth(const autan::Mesh& mesh) {
  index_t b = 0;
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    b = std::max(b, mesh.neighbour()[at(f)] - mesh.owner()[at(f)]);
  }
  return b;
}

std::vector<index_t> shuffled(index_t n, unsigned seed) {
  std::vector<index_t> p(at(n));
  std::iota(p.begin(), p.end(), 0);
  std::mt19937 rng(seed);
  for (std::size_t i = p.size(); i > 1; --i) std::swap(p[i - 1], p[rng() % i]);
  return p;
}

}  // namespace

TEST_CASE("RCM orders a chain of cells with bandwidth 1", "[renumber]") {
  // An n x 1 x 1 row of hexahedra is a path graph: its optimal bandwidth is 1
  // and Cuthill-McKee from an end (a pseudo-peripheral cell) reaches it.
  autan::CellwiseMesh cw;
  for (int i = 0; i <= 12; ++i)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j) {
        cw.x.push_back(i);
        cw.y.push_back(j);
        cw.z.push_back(k);
      }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(4 * i + 2 * k + j); };
  for (int i = 0; i < 12; ++i) {
    cw.add_cell(autan::CellShape::hexa,
                std::array<index_t, 8>{p(i, 0, 0), p(i + 1, 0, 0), p(i + 1, 1, 0), p(i, 1, 0),
                                       p(i, 0, 1), p(i + 1, 0, 1), p(i + 1, 1, 1), p(i, 1, 1)},
                0);
  }
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::Mesh chain = autan::assemble(autan::to_face_based(cw));
  const autan::Mesh mixed = autan::renumber_cells(chain, shuffled(chain.n_cells(), 3));
  REQUIRE(bandwidth(mixed) > 1);
  const autan::Mesh ordered = autan::renumber_cells(mixed, autan::rcm_order(mixed));
  CHECK(bandwidth(ordered) == 1);
}

TEST_CASE("Renumbering permutes cells and keeps the mesh", "[renumber]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(4, autan::test::BoxFill::hybrid, 0.3)));
  const std::vector<index_t> order = autan::rcm_order(mesh);
  std::vector<index_t> sorted(order);
  std::sort(sorted.begin(), sorted.end());
  std::vector<index_t> identity(sorted.size());
  std::iota(identity.begin(), identity.end(), 0);
  REQUIRE(sorted == identity);

  const autan::Mesh shuffled_mesh = autan::renumber_cells(mesh, shuffled(mesh.n_cells(), 11));
  const autan::Mesh renumbered =
      autan::renumber_cells(shuffled_mesh, autan::rcm_order(shuffled_mesh));
  CHECK(bandwidth(renumbered) < bandwidth(shuffled_mesh));
  CHECK(autan::rcm_order(shuffled_mesh) == autan::rcm_order(shuffled_mesh));

  const autan::Mesh again = autan::renumber_cells(mesh, order);
  const autan::Geometry g0 = autan::compute_geometry(mesh);
  const autan::Geometry g1 = autan::compute_geometry(again);
  REQUIRE(again.n_faces() == mesh.n_faces());
  for (std::size_t i = 0; i < order.size(); ++i) {
    CHECK_THAT(g1.cell_volume[i], WithinRel(g0.cell_volume[at(order[i])], 1e-13));
    CHECK(again.cell_zone()[i] == mesh.cell_zone()[at(order[i])]);
    CHECK(again.cell_global_id()[i] == static_cast<std::int64_t>(i));
  }
  for (std::size_t z = 0; z + 1 < mesh.boundary_zone_offsets().size(); ++z) {
    CHECK(again.boundary_zone_offsets()[z + 1] - again.boundary_zone_offsets()[z] ==
          mesh.boundary_zone_offsets()[z + 1] - mesh.boundary_zone_offsets()[z]);
  }
}

TEST_CASE("Renumbering rejects an invalid permutation", "[renumber]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(1, autan::test::BoxFill::tetra)));
  std::vector<index_t> bad(at(mesh.n_cells()), 0);
  try {
    (void)autan::renumber_cells(mesh, bad);
    FAIL("expected E0102");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_index_out_of_range);
  }
}
