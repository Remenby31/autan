#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "cfd/core/error.hpp"
#include "cfd/core/geometry.hpp"
#include "cfd/core/mesh.hpp"
#include "fixtures.hpp"

using cfd::index_t;
using cfd::offset_t;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

void require_error(cfd::ErrorCode expected, const std::function<void()>& action) {
  try {
    action();
    FAIL("expected " << cfd::error_code_string(expected));
  } catch (const cfd::Error& e) {
    CHECK(e.code() == expected);
  }
}

// Two unit cubes side by side along x, written directly in face-based form.
cfd::MeshInput two_cubes() {
  cfd::MeshInput in;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 3; ++i) {
        in.x.push_back(i);
        in.y.push_back(j);
        in.z.push_back(k);
      }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(i + 3 * (j + 2 * k)); };
  auto face = [&](std::array<index_t, 4> nodes, index_t c0, index_t c1, index_t zone) {
    in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(c0);
    in.face_cell1.push_back(c1);
    in.face_zone.push_back(zone);
  };
  in.face_node_offsets.push_back(0);
  // Interior face listed from cell 1 so that assemble must flip it.
  face({p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(1, 1, 0)}, 1, 0, 1);
  for (int c = 0; c < 2; ++c) {
    const auto ci = static_cast<index_t>(c);
    if (c == 0) face({p(0, 0, 0), p(0, 0, 1), p(0, 1, 1), p(0, 1, 0)}, ci, cfd::no_cell, 0);
    if (c == 1) face({p(2, 0, 0), p(2, 1, 0), p(2, 1, 1), p(2, 0, 1)}, ci, cfd::no_cell, 0);
    face({p(c, 0, 0), p(c + 1, 0, 0), p(c + 1, 0, 1), p(c, 0, 1)}, ci, cfd::no_cell, 0);
    face({p(c, 1, 0), p(c, 1, 1), p(c + 1, 1, 1), p(c + 1, 1, 0)}, ci, cfd::no_cell, 0);
    face({p(c, 0, 0), p(c, 1, 0), p(c + 1, 1, 0), p(c + 1, 0, 0)}, ci, cfd::no_cell, 0);
    face({p(c, 0, 1), p(c + 1, 0, 1), p(c + 1, 1, 1), p(c, 1, 1)}, ci, cfd::no_cell, 0);
  }
  in.cell_zone = {0, 0};
  in.face_zones = {cfd::FaceZone{.name = "walls", .bc_type = "wall", .kind = cfd::BcKind::wall, .provenance = {}},
                   cfd::FaceZone{.name = "interior", .bc_type = "", .kind = cfd::BcKind::unspecified, .provenance = {}}};
  in.cell_zones = {cfd::CellZone{.name = "fluid", .type = "", .kind = cfd::CellKind::unspecified, .provenance = {}}};
  in.source = cfd::MeshSource{.format = "memory", .path = ""};
  return in;
}

}  // namespace

TEST_CASE("Assembled meshes satisfy the canonical ordering of the mesh contract", "[mesh]") {
  const auto fill = GENERATE(cfd::test::BoxFill::hexa, cfd::test::BoxFill::tetra,
                             cfd::test::BoxFill::prism, cfd::test::BoxFill::pyramid,
                             cfd::test::BoxFill::hybrid);
  CAPTURE(cfd::test::to_string(fill));
  const cfd::Mesh mesh = cfd::assemble(cfd::to_face_based(cfd::test::box(4, fill, 0.0)));
  const cfd::Geometry g = cfd::compute_geometry(mesh);
  const auto owner = mesh.owner();
  const auto neighbour = mesh.neighbour();
  const index_t n_interior = mesh.n_interior_faces();

  for (index_t f = 0; f < n_interior; ++f) {
    CHECK(owner[at(f)] < neighbour[at(f)]);
    if (f > 0) {
      const auto prev = std::array{owner[at(f - 1)], neighbour[at(f - 1)]};
      CHECK(prev <= (std::array{owner[at(f)], neighbour[at(f)]}));
    }
    // Convex cells: the area vector points from the owner's centroid to the
    // neighbour's.
    const std::size_t o = at(owner[at(f)]);
    const std::size_t n = at(neighbour[at(f)]);
    const double d = g.face_area_x[at(f)] * (g.cell_centroid_x[n] - g.cell_centroid_x[o]) +
                     g.face_area_y[at(f)] * (g.cell_centroid_y[n] - g.cell_centroid_y[o]) +
                     g.face_area_z[at(f)] * (g.cell_centroid_z[n] - g.cell_centroid_z[o]);
    CHECK(d > 0.0);
  }

  const auto zone_offsets = mesh.boundary_zone_offsets();
  REQUIRE(zone_offsets.size() == mesh.face_zones().size() + 1);
  CHECK(zone_offsets.front() == n_interior);
  CHECK(zone_offsets.back() == mesh.n_faces());
  for (std::size_t z = 0; z + 1 < zone_offsets.size(); ++z) {
    for (index_t f = zone_offsets[z]; f < zone_offsets[z + 1]; ++f) {
      CHECK(mesh.face_zone()[at(f)] == static_cast<index_t>(z));
      if (f > zone_offsets[z]) CHECK(owner[at(f - 1)] <= owner[at(f)]);
      const std::size_t o = at(owner[at(f)]);
      const double d = g.face_area_x[at(f)] * (g.face_centroid_x[at(f)] - g.cell_centroid_x[o]) +
                       g.face_area_y[at(f)] * (g.face_centroid_y[at(f)] - g.cell_centroid_y[o]) +
                       g.face_area_z[at(f)] * (g.face_centroid_z[at(f)] - g.cell_centroid_z[o]);
      CHECK(d > 0.0);
    }
  }

  // Sides of the box keep their labels; interior faces get the synthesised zone.
  const std::array<std::string, 6> sides{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
  for (std::size_t z = 0; z < sides.size(); ++z) CHECK(mesh.face_zones()[z].name == sides[z]);
  REQUIRE(mesh.face_zones().size() == 7);
  CHECK(mesh.face_zones()[6].name == "interior");
  CHECK(mesh.face_zones()[6].provenance.source_id == -1);
  for (index_t f = 0; f < n_interior; ++f) CHECK(mesh.face_zone()[at(f)] == 6);

  // Cell->face CSR lists exactly the faces whose owner or neighbour is the cell.
  std::vector<int> seen(at(mesh.n_faces()), 0);
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    const auto faces = mesh.cell_faces(c);
    CHECK(std::is_sorted(faces.begin(), faces.end()));
    for (const index_t f : faces) {
      CHECK((owner[at(f)] == c || (f < n_interior && neighbour[at(f)] == c)));
      ++seen[at(f)];
    }
  }
  for (index_t f = 0; f < mesh.n_faces(); ++f) CHECK(seen[at(f)] == (f < n_interior ? 2 : 1));
}

TEST_CASE("Analytic entity counts of box meshes", "[mesh]") {
  struct Expected {
    cfd::test::BoxFill fill;
    int cells_per_cube;
    int faces_per_cell;
    int boundary_faces_per_side_square;
  };
  const auto e = GENERATE(Expected{cfd::test::BoxFill::hexa, 1, 6, 1},
                          Expected{cfd::test::BoxFill::tetra, 6, 4, 2},
                          Expected{cfd::test::BoxFill::pyramid, 6, 5, 1});
  CAPTURE(cfd::test::to_string(e.fill));
  constexpr int n = 3;
  const cfd::Mesh mesh = cfd::assemble(cfd::to_face_based(cfd::test::box(n, e.fill)));
  const int cells = e.cells_per_cube * n * n * n;
  const int boundary = 6 * n * n * e.boundary_faces_per_side_square;
  CHECK(mesh.n_cells() == cells);
  CHECK(mesh.n_boundary_faces() == boundary);
  CHECK(mesh.n_interior_faces() == (e.faces_per_cell * cells - boundary) / 2);
  const int corner_points = (n + 1) * (n + 1) * (n + 1);
  CHECK(mesh.n_points() == corner_points + (e.fill == cfd::test::BoxFill::pyramid ? n * n * n : 0));
}

TEST_CASE("Assemble flips interior faces to owner < neighbour and keeps zones", "[mesh]") {
  const cfd::Mesh mesh = cfd::assemble(two_cubes());
  REQUIRE(mesh.n_interior_faces() == 1);
  CHECK(mesh.owner()[0] == 0);
  CHECK(mesh.neighbour()[0] == 1);
  const cfd::Geometry g = cfd::compute_geometry(mesh);
  CHECK(g.face_area_x[0] == 1.0);
  CHECK(mesh.face_zones()[0].kind == cfd::BcKind::wall);
  CHECK(mesh.face_zones()[0].bc_type == "wall");
  CHECK(mesh.boundary_zone_offsets()[1] - mesh.boundary_zone_offsets()[0] == 10);
}

TEST_CASE("Points referenced by no face are removed", "[mesh]") {
  cfd::MeshInput in = two_cubes();
  in.x.insert(in.x.begin(), 42.0);
  in.y.insert(in.y.begin(), 42.0);
  in.z.insert(in.z.begin(), 42.0);
  for (index_t& node : in.face_nodes) ++node;
  const cfd::Mesh mesh = cfd::assemble(in);
  CHECK(mesh.n_points() == 12);
  CHECK(std::find(mesh.x().begin(), mesh.x().end(), 42.0) == mesh.x().end());
}

TEST_CASE("Invalid meshes are rejected with their documented code", "[mesh]") {
  using cfd::ErrorCode;
  SECTION("E0101 inconsistent sizes") {
    auto in = two_cubes();
    in.face_zone.pop_back();
    require_error(ErrorCode::mesh_inconsistent_sizes, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0102 node index out of range") {
    auto in = two_cubes();
    in.face_nodes[5] = 12;
    require_error(ErrorCode::mesh_index_out_of_range, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0103 face with fewer than 3 nodes") {
    auto in = two_cubes();
    in.face_nodes.erase(in.face_nodes.begin() + 2, in.face_nodes.begin() + 4);
    for (std::size_t f = 1; f < in.face_node_offsets.size(); ++f) in.face_node_offsets[f] -= 2;
    require_error(ErrorCode::mesh_face_too_few_nodes, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0104 repeated node") {
    auto in = two_cubes();
    in.face_nodes[1] = in.face_nodes[0];
    require_error(ErrorCode::mesh_face_repeated_node, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0105 same cell on both sides") {
    auto in = two_cubes();
    in.face_cell1[0] = 1;
    require_error(ErrorCode::mesh_face_same_cell_both_sides, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0106 cell with fewer than 4 faces") {
    auto in = two_cubes();
    in.cell_zone.push_back(0);
    require_error(ErrorCode::mesh_cell_too_few_faces, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0107 open cell") {
    auto in = two_cubes();
    in.face_node_offsets.pop_back();
    in.face_nodes.resize(static_cast<std::size_t>(in.face_node_offsets.back()));
    in.face_cell0.pop_back();
    in.face_cell1.pop_back();
    in.face_zone.pop_back();
    require_error(ErrorCode::mesh_cell_not_closed, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0107 inconsistently oriented cell") {
    auto in = two_cubes();
    std::reverse(in.face_nodes.end() - 4, in.face_nodes.end());
    require_error(ErrorCode::mesh_cell_not_closed, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0108 face shared by three cells") {
    auto cw = cfd::test::affine_cell(cfd::CellShape::tetra, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}).mesh;
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-1.0);
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-2.0);
    cw.add_cell(cfd::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 4}, 0);
    cw.add_cell(cfd::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 5}, 0);
    require_error(ErrorCode::mesh_face_shared_by_more_than_two_cells, [&] { (void)cfd::to_face_based(cw); });
  }
  SECTION("E0109 zone with interior and boundary faces") {
    auto in = two_cubes();
    in.face_zone[0] = 0;
    require_error(ErrorCode::mesh_zone_mixes_interior_and_boundary, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0110 duplicate zone name") {
    auto in = two_cubes();
    in.face_zones[1].name = "walls";
    require_error(ErrorCode::mesh_duplicate_zone_name, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0111 labelled face matching no cell face") {
    auto cw = cfd::test::box(1, cfd::test::BoxFill::hexa, 0.0, false);
    cw.face_zones.push_back(cfd::FaceZone{.name = "stray", .bc_type = "", .kind = cfd::BcKind::unspecified, .provenance = {}});
    cw.add_labelled_face(std::array<index_t, 3>{0, 1, 2}, 0);
    require_error(ErrorCode::mesh_labelled_face_not_found, [&] { (void)cfd::to_face_based(cw); });
  }
  SECTION("E0112 empty mesh") {
    cfd::MeshInput in;
    in.face_node_offsets.push_back(0);
    require_error(ErrorCode::mesh_empty, [&] { (void)cfd::assemble(in); });
  }
  SECTION("E0114 inverted neighbour in a cellwise mesh") {
    auto cw = cfd::test::affine_cell(cfd::CellShape::tetra, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0}).mesh;
    cw.x.push_back(0.3);
    cw.y.push_back(0.3);
    cw.z.push_back(-1.0);
    // Inverted tetra: it lists the shared face (0, 1, 2) with the same
    // orientation as the first one.
    cw.add_cell(cfd::CellShape::tetra, std::array<index_t, 4>{0, 1, 2, 4}, 0);
    require_error(ErrorCode::mesh_shared_face_orientation, [&] { (void)cfd::to_face_based(cw); });
  }
  SECTION("E0115 face labelled twice") {
    auto cw = cfd::test::box(1, cfd::test::BoxFill::hexa);
    cw.add_labelled_face(std::array<index_t, 4>{0, 2, 6, 4}, 1);
    require_error(ErrorCode::mesh_face_labelled_twice, [&] { (void)cfd::to_face_based(cw); });
  }
}

TEST_CASE("Unlabelled boundary faces land in the synthesised zone 'unassigned'", "[mesh]") {
  const cfd::Mesh mesh = cfd::assemble(cfd::to_face_based(cfd::test::hanging_node_box()));
  const auto zones = mesh.face_zones();
  REQUIRE(zones.size() == 3);
  CHECK(zones[0].name == "inlet");
  CHECK(zones[0].kind == cfd::BcKind::inlet);
  const auto offsets = mesh.boundary_zone_offsets();
  CHECK(offsets[1] - offsets[0] == 1);
  const auto unassigned = std::find_if(zones.begin(), zones.end(),
                                       [](const cfd::FaceZone& z) { return z.name == "unassigned"; });
  REQUIRE(unassigned != zones.end());
  CHECK(unassigned->provenance.source_id == -1);
  const auto u = static_cast<std::size_t>(unassigned - zones.begin());
  // 4 sides of the big cell, 100 hex faces at x = 2 and 4 x 10 hex sides.
  CHECK(offsets[u + 1] - offsets[u] == 4 + 100 + 40);
  CHECK(mesh.n_interior_faces() == 100 + 2 * 10 * 9);
  CHECK(mesh.cell_zones()[at(mesh.cell_zone()[0])].name == "coarse");
}
