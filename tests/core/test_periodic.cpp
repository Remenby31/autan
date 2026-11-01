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

TEST_CASE("Periodic nodes match within 1e-4 of the local shortest edge", "[periodic]") {
  // n = 3: every edge on the sides is 1/3 long, so the tolerance is 3.3e-5.
  using autan::test::BoxFill;
  const auto shifted = [](double dy, bool single) {
    return periodic_box(3, BoxFill::hexa, {0}, [=](autan::CellwiseMesh& m) {
      for (std::size_t i = 0; i < m.x.size(); ++i) {
        if (m.x[i] != 1.0) continue;
        m.y[i] += dy;
        if (single) {
          m.y[i] = static_cast<float>(m.y[i]);
          m.z[i] = static_cast<float>(m.z[i]);
        }
      }
    });
  };
  CHECK(autan::assemble(shifted(1e-5, false)).periodic_pairs().size() == 1);
  CHECK(autan::assemble(shifted(0.0, true)).periodic_pairs().size() == 1);
  // Edges so short that the tolerance underflows: still a coded outcome.
  auto tiny = periodic_box(3, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
    for (auto* v : {&m.x, &m.y, &m.z}) {
      for (double& c : *v) c *= 1e-318;
    }
    m.periodic[0].translation = {1e-318, 0.0, 0.0};
  });
  try {
    (void)autan::assemble(std::move(tiny));
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_periodic_unmatched);
  }
  require_code(autan::ErrorCode::mesh_periodic_unmatched, shifted(1e-4, false));
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
  SECTION("partners that point the same way") {
    // Two unit cubes, [0,1]^3 and [2,3]x[0,1]^2: their x = 0 and x = 2 faces
    // coincide under (2, 0, 0) but both point towards -x.
    autan::CellwiseMesh m;
    for (const double x0 : {0.0, 2.0}) {
      std::array<index_t, 8> nodes{};
      for (int k = 0; k < 8; ++k) {
        static constexpr std::array<std::array<int, 3>, 8> corner{{{0, 0, 0},
                                                                   {1, 0, 0},
                                                                   {1, 1, 0},
                                                                   {0, 1, 0},
                                                                   {0, 0, 1},
                                                                   {1, 0, 1},
                                                                   {1, 1, 1},
                                                                   {0, 1, 1}}};
        const auto& c = corner[static_cast<std::size_t>(k)];
        m.x.push_back(x0 + c[0]);
        m.y.push_back(c[1]);
        m.z.push_back(c[2]);
        nodes[static_cast<std::size_t>(k)] = static_cast<index_t>(m.x.size() - 1);
      }
      m.add_cell(autan::CellShape::hexa, nodes, 0);
      const std::array<index_t, 4> left{nodes[0], nodes[4], nodes[7], nodes[3]};
      m.add_labelled_face(left, x0 == 0.0 ? 0 : 1);
    }
    m.cell_zones.push_back(
        {.name = "fluid", .type = "", .kind = autan::CellKind::fluid, .provenance = {}});
    for (const char* name : {"a", "b"}) {
      m.face_zones.push_back(
          {.name = name, .bc_type = "", .kind = autan::BcKind::periodic, .provenance = {}});
    }
    m.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {2.0, 0.0, 0.0}});
    require_code(ErrorCode::mesh_periodic_unmatched, autan::to_face_based(std::move(m)));
  }
  SECTION("an empty zone") {
    require_code(
        ErrorCode::mesh_periodic_invalid,
        periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
          // Two declared zones that label no face.
          for (const char* name : {"left", "right"}) {
            m.face_zones.push_back(
                {.name = name, .bc_type = "", .kind = autan::BcKind::periodic, .provenance = {}});
          }
          m.periodic.push_back({.zone_a = 6, .zone_b = 7, .translation = {1.0, 0.0, 0.0}});
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
