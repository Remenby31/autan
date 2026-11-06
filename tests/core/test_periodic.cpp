#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
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
  // A node of xmin 1e-321 away from the corner: its shortest edge, hence
  // its tolerance and the bins it searches, underflow. A coded mismatch.
  require_code(autan::ErrorCode::mesh_periodic_unmatched,
               periodic_box(3, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                 for (std::size_t i = 0; i < m.x.size(); ++i) {
                   if (m.x[i] == 0.0 && m.z[i] == 0.0 && m.y[i] > 0.3 && m.y[i] < 0.4)
                     m.y[i] = 1e-321;
                 }
               }));
  require_code(autan::ErrorCode::mesh_periodic_unmatched, shifted(1e-4, false));
  // The same node moved on both sides: exact translates with subnormal bins.
  const auto both = periodic_box(3, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
    for (std::size_t i = 0; i < m.x.size(); ++i) {
      if ((m.x[i] == 0.0 || m.x[i] == 1.0) && m.z[i] == 0.0 && m.y[i] > 0.3 && m.y[i] < 0.4)
        m.y[i] = 1e-305;
    }
  });
  CHECK(autan::assemble(both).periodic_partners(0).size() == 9);
}

TEST_CASE("Periodic nodes matched within the tolerance become exact translates", "[periodic]") {
  // n = 3, periodic in x and y: the tolerance on the sides is 1e-4 / 3. Zone b of the x pair
  // (x = 1) is moved outward by half of it; assembly moves it back onto zone a translated, to the
  // bit, and records what it moved: the 16 nodes of that side, by the offset, and the volume
  // change of the cells beside it, against the same mesh assembled without its pairs.
  using autan::test::BoxFill;
  const double tolerance = 1e-4 / 3.0;
  const double delta = 0.5 * tolerance;
  const autan::Mesh exact = autan::assemble(periodic_box(3, BoxFill::hexa, {0, 1}));
  CHECK(exact.periodic_snap().nodes == 0);
  const auto outward = [&](autan::CellwiseMesh& m) {
    for (double& x : m.x) {
      if (x == 1.0) x += delta;
    }
  };
  const autan::Mesh moved = autan::assemble(periodic_box(3, BoxFill::hexa, {0, 1}, outward));
  const autan::Geometry before =
      autan::compute_geometry(autan::assemble(periodic_box(3, BoxFill::hexa, {}, outward)));
  const autan::Geometry after = autan::compute_geometry(moved);
  double change = 0.0;
  for (std::size_t c = 0; c < before.cell_volume.size(); ++c) {
    change = std::max(
        change, std::abs(after.cell_volume[c] - before.cell_volume[c]) / before.cell_volume[c]);
  }
  REQUIRE(change > 0.0);
  const auto& snap = moved.periodic_snap();
  CHECK(snap.nodes == 16);
  CHECK_THAT(snap.max_distance, WithinAbs(delta, 1e-15));
  CHECK_THAT(snap.max_relative, WithinAbs(0.5, 1e-9));
  const auto x = moved.x(), y = moved.y(), z = moved.z();
  for (std::size_t p = 0; p < moved.periodic_pairs().size(); ++p) {
    const auto& t = moved.periodic_pairs()[p].translation;
    const auto off = moved.boundary_zone_offsets();
    const auto partners = moved.periodic_partners(p);
    for (std::size_t i = 0; i < partners.size(); ++i) {
      const index_t a = off[at(moved.periodic_pairs()[p].zone_a)] + static_cast<index_t>(i);
      for (const index_t na : moved.face_nodes(a)) {
        bool found = false;
        for (const index_t nb : moved.face_nodes(partners[i])) {
          found = found || (x[at(na)] + t[0] == x[at(nb)] && y[at(na)] + t[1] == y[at(nb)] &&
                            z[at(na)] + t[2] == z[at(nb)]);
        }
        CHECK(found);
      }
    }
  }
  check_partners(moved);
  CHECK_THAT(snap.max_volume_change, Catch::Matchers::WithinRel(change, 1e-12));
  CHECK(exact.periodic_snap().max_volume_change == 0.0);

  // The x pair declared the other way round, xmin moved outward: zone b is xmin, zone 0, whose
  // faces come first among the boundary faces, so the first boundary face holds moved nodes.
  const auto xmin_outward = [&](autan::CellwiseMesh& m) {
    m.face_zones[0].kind = autan::BcKind::periodic;
    m.face_zones[1].kind = autan::BcKind::periodic;
    m.periodic.push_back({.zone_a = 1, .zone_b = 0, .translation = {-1.0, 0.0, 0.0}});
    for (double& coordinate : m.x) {
      if (coordinate == 0.0) coordinate -= delta;
    }
  };
  const autan::Mesh first = autan::assemble(periodic_box(3, BoxFill::hexa, {}, xmin_outward));
  CHECK(first.periodic_snap().nodes == 16);
  CHECK_THAT(first.periodic_snap().max_distance, WithinAbs(delta, 1e-15));
  REQUIRE(first.face_zone()[at(first.n_interior_faces())] == 0);
  for (const index_t p : first.face_nodes(first.n_interior_faces())) CHECK(first.x()[at(p)] == 0.0);
  check_partners(first);
}

TEST_CASE("Moving periodic nodes that would make a valid cell degenerate is a coded error",
          "[periodic]") {
  // n = 2, x-periodic: the tolerance on the sides is 1e-4 / 2. The interior plane moves to
  // x = 1 + 1.25e-5 and xmax to x = 1 + 2.5e-5, half the tolerance from xmin translated: every
  // cell is valid, and moving xmax back onto x = 1 would pass the interior plane.
  using autan::test::BoxFill;
  const auto input = periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
    for (double& x : m.x) {
      if (x == 1.0) {
        x = 1.0 + 2.5e-5;
      } else if (x != 0.0) {
        x = 1.0 + 1.25e-5;
      }
    }
  });
  try {
    (void)autan::assemble(input);
    FAIL("expected E0119");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_periodic_unmatched);
    CHECK(e.data_json().find("\"volume_after\"") != std::string::npos);
  }
}

TEST_CASE("A cell already degenerate keeps its own error whatever the periodic move",
          "[periodic]") {
  // As above with the planes inside out: x = 1 - 1.25e-5 inside, xmax at 1 - 2.5e-5. The cells
  // beside xmax have negative volume before any move; moving xmax back onto x = 1 would right
  // them, and must not: E0121, as compute_geometry reports it without the pair.
  using autan::test::BoxFill;
  const auto inside_out = [](autan::CellwiseMesh& m) {
    for (double& x : m.x) {
      if (x == 1.0) {
        x = 1.0 - 2.5e-5;
      } else if (x != 0.0) {
        x = 1.0 - 1.25e-5;
      }
    }
  };
  require_code(autan::ErrorCode::geometry_non_positive_volume,
               periodic_box(2, BoxFill::hexa, {0}, inside_out));
}

TEST_CASE("A class with several roots is placed when its pairs agree", "[periodic]") {
  // n = 2, periodic in x on the whole of xmin -> xmax and in y only on the half x > 1/2 of ymin
  // -> ymax, walls elsewhere. The corner classes {(0,0), (1,0), (0,1), (1,1)} have two roots,
  // (0,0) and (0,1): each reaches only part of the class, and the pairs agree.
  using autan::test::BoxFill;
  const autan::Mesh mesh =
      autan::assemble(periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
        const auto first = static_cast<index_t>(m.face_zones.size());
        for (const index_t side : {2, 3}) {
          autan::FaceZone half = m.face_zones[at(side)];
          half.name += "-half";
          half.kind = autan::BcKind::periodic;
          m.face_zones.push_back(half);
        }
        for (std::size_t f = 0; f < m.labelled_face_zone.size(); ++f) {
          const index_t zone = m.labelled_face_zone[f];
          if (zone != 2 && zone != 3) continue;
          double x = 0.0;
          const auto begin = m.labelled_face_node_offsets[f];
          const auto end = m.labelled_face_node_offsets[f + 1];
          for (auto k = begin; k < end; ++k)
            x += m.x[at(m.labelled_face_nodes[static_cast<std::size_t>(k)])];
          if (x / static_cast<double>(end - begin) > 0.5)
            m.labelled_face_zone[f] = first + zone - 2;
        }
        m.periodic.push_back(
            {.zone_a = first, .zone_b = first + 1, .translation = {0.0, 1.0, 0.0}});
      }));
  REQUIRE(mesh.periodic_pairs().size() == 2);
  CHECK(mesh.periodic_snap().nodes == 0);
  check_partners(mesh);
}

TEST_CASE("Periodic pairs that place a node in two ways are a coded error", "[periodic]") {
  // n = 2: the upper halves of xmin and xmax become a second pair, translated by (1, 0, 2e-5),
  // within the tolerance (5e-5). Their seam nodes are matched by both pairs, which place them
  // 2e-5 apart.
  using autan::test::BoxFill;
  require_code(autan::ErrorCode::mesh_periodic_unmatched,
               periodic_box(2, BoxFill::hexa, {0}, [](autan::CellwiseMesh& m) {
                 const auto first = static_cast<index_t>(m.face_zones.size());
                 for (const index_t side : {0, 1}) {
                   autan::FaceZone upper = m.face_zones[at(side)];
                   upper.name += "-upper";
                   m.face_zones.push_back(upper);
                 }
                 for (std::size_t f = 0; f < m.labelled_face_zone.size(); ++f) {
                   const index_t zone = m.labelled_face_zone[f];
                   if (zone != 0 && zone != 1) continue;
                   double z = 0.0;
                   const auto begin = m.labelled_face_node_offsets[f];
                   const auto end = m.labelled_face_node_offsets[f + 1];
                   for (auto k = begin; k < end; ++k)
                     z += m.z[at(m.labelled_face_nodes[static_cast<std::size_t>(k)])];
                   if (z / static_cast<double>(end - begin) > 0.5)
                     m.labelled_face_zone[f] = first + zone;
                 }
                 m.periodic.push_back(
                     {.zone_a = first, .zone_b = first + 1, .translation = {1.0, 0.0, 2e-5}});
               }));
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
