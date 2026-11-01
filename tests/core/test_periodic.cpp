#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/renumber.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// The unit box with its sides xmin..zmax (zones 0..5) periodic in the given
// axes: pair k joins side 2k to side 2k+1 by the unit translation along k.
autan::MeshInput periodic_box(int n, autan::test::BoxFill fill, std::vector<int> axes,
                              const std::function<void(autan::CellwiseMesh&)>& edit = {}) {
  autan::CellwiseMesh mesh = autan::test::box(n, fill, 0.2);
  for (const int k : axes) {
    for (const int side : {2 * k, 2 * k + 1}) {
      mesh.face_zones[static_cast<std::size_t>(side)].kind = autan::BcKind::periodic;
    }
    std::array<double, 3> t{};
    t[static_cast<std::size_t>(k)] = 1.0;
    mesh.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
  }
  if (edit) edit(mesh);
  return autan::to_face_based(std::move(mesh));
}

void require_code(autan::ErrorCode code, autan::MeshInput input) {
  try {
    (void)autan::assemble(std::move(input));
    FAIL("expected " << autan::error_code_string(code));
  } catch (const autan::Error& e) {
    CHECK(e.code() == code);
  }
}

// Each face of zone a, translated, is its partner: same centroid, opposite
// area vector (both point out of the domain).
void check_partners(const autan::Mesh& mesh) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto off = mesh.boundary_zone_offsets();
  for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
    const auto& pair = mesh.periodic_pairs()[p];
    const auto partners = mesh.periodic_partners(p);
    REQUIRE(partners.size() == at(off[at(pair.zone_a) + 1] - off[at(pair.zone_a)]));
    for (std::size_t i = 0; i < partners.size(); ++i) {
      const auto a = at(off[at(pair.zone_a)]) + i;
      const auto b = at(partners[i]);
      CHECK(mesh.face_zone()[b] == pair.zone_b);
      CHECK_THAT(g.face_centroid_x[a] + pair.translation[0],
                 WithinAbs(g.face_centroid_x[b], 1e-14));
      CHECK_THAT(g.face_centroid_y[a] + pair.translation[1],
                 WithinAbs(g.face_centroid_y[b], 1e-14));
      CHECK_THAT(g.face_centroid_z[a] + pair.translation[2],
                 WithinAbs(g.face_centroid_z[b], 1e-14));
      CHECK_THAT(g.face_area_x[a], WithinAbs(-g.face_area_x[b], 1e-15));
      CHECK_THAT(g.face_area_y[a], WithinAbs(-g.face_area_y[b], 1e-15));
      CHECK_THAT(g.face_area_z[a], WithinAbs(-g.face_area_z[b], 1e-15));
    }
  }
}

}  // namespace

TEST_CASE("The faces of a periodic pair match one to one", "[periodic]") {
  using autan::test::BoxFill;
  const BoxFill fill = GENERATE(BoxFill::hexa, BoxFill::tetra, BoxFill::prism, BoxFill::pyramid);
  CAPTURE(autan::test::to_string(fill));
  const autan::Mesh mesh = autan::assemble(periodic_box(3, fill, {0, 1, 2}));
  REQUIRE(mesh.periodic_pairs().size() == 3);
  check_partners(mesh);
  // Partners survive renumbering, which re-sorts every face.
  check_partners(autan::renumber_cells(mesh, autan::rcm_order(mesh)));
}

TEST_CASE("A periodic declaration that the mesh contradicts is a coded error", "[periodic]") {
  using autan::ErrorCode;
  using autan::test::BoxFill;
  SECTION("faces that do not coincide under the translation") {
    require_code(ErrorCode::mesh_periodic_unmatched,
                 periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                   m.periodic[0].translation = {1.0 + 1e-3, 0.0, 0.0};
                 }));
  }
  SECTION("zones with different face counts") {
    require_code(ErrorCode::mesh_periodic_unmatched,
                 periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                   // One face of xmax moves to ymin.
                   for (auto& z : m.labelled_face_zone) {
                     if (z == 1) {
                       z = 2;
                       break;
                     }
                   }
                 }));
  }
  SECTION("a zero translation") {
    require_code(ErrorCode::mesh_periodic_invalid,
                 periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                   m.periodic[0].translation = {0.0, 0.0, 0.0};
                 }));
  }
  SECTION("a zone paired with itself") {
    require_code(ErrorCode::mesh_periodic_invalid,
                 periodic_box(2, BoxFill::hexa, {0},
                              [](autan::CellwiseMesh& m) { m.periodic[0].zone_b = 0; }));
  }
  SECTION("a zone in two pairs") {
    require_code(ErrorCode::mesh_periodic_invalid,
                 periodic_box(2, BoxFill::hexa, {0},
                              [](autan::CellwiseMesh& m) { m.periodic.push_back(m.periodic[0]); }));
  }
  SECTION("a periodic zone in no pair") {
    require_code(ErrorCode::mesh_periodic_invalid,
                 periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                   m.face_zones[2].kind = autan::BcKind::periodic;
                 }));
  }
  SECTION("a pair of zones not of kind periodic") {
    require_code(ErrorCode::mesh_periodic_invalid,
                 periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                   m.periodic.push_back({.zone_a = 2, .zone_b = 3, .translation = {0.0, 1.0, 0.0}});
                 }));
  }
}
