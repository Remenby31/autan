#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

namespace fs = std::filesystem;

const fs::path data_dir{AUTAN_TEST_MESH_DIR};

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Format-independent identity of a mesh: points by coordinates (rounded to
// 1e-7, coarser than the 10 significant digits Fluent files carry), faces by
// their point set, cells by their face set.
struct Canonical {
  std::vector<std::array<std::int64_t, 3>> point_key;
  std::map<std::vector<std::array<std::int64_t, 3>>, index_t> face_of_key;
  std::vector<std::vector<std::array<std::int64_t, 3>>> face_key;
  std::map<std::vector<index_t>, index_t> cell_of_faces;  // faces as indices into `faces`
};

std::vector<std::array<std::int64_t, 3>> sorted_face_key(
    const autan::Mesh& mesh, index_t f, const std::vector<std::array<std::int64_t, 3>>& pk) {
  std::vector<std::array<std::int64_t, 3>> key;
  for (const index_t p : mesh.face_nodes(f)) key.push_back(pk[at(p)]);
  std::sort(key.begin(), key.end());
  return key;
}

Canonical canonical(const autan::Mesh& mesh) {
  Canonical out;
  for (index_t p = 0; p < mesh.n_points(); ++p) {
    out.point_key.push_back({std::llround(mesh.x()[at(p)] * 1e7),
                             std::llround(mesh.y()[at(p)] * 1e7),
                             std::llround(mesh.z()[at(p)] * 1e7)});
  }
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    out.face_key.push_back(sorted_face_key(mesh, f, out.point_key));
    out.face_of_key[out.face_key.back()] = f;
  }
  return out;
}

// Checks that `b` describes the same mesh as `a`: same points, faces, cells,
// boundary zones per face and orientation; and cell zones per cell unless
// `b` comes from a format that does not keep them.
void require_same_mesh(const autan::Mesh& a, const autan::Mesh& b, bool cell_zones = true) {
  REQUIRE(a.n_points() == b.n_points());
  REQUIRE(a.n_faces() == b.n_faces());
  REQUIRE(a.n_interior_faces() == b.n_interior_faces());
  REQUIRE(a.n_cells() == b.n_cells());
  const Canonical ca = canonical(a);
  const Canonical cb = canonical(b);
  REQUIRE(ca.face_of_key.size() == static_cast<std::size_t>(a.n_faces()));

  std::vector<index_t> face_map(at(a.n_faces()));
  for (index_t f = 0; f < a.n_faces(); ++f) {
    const auto it = cb.face_of_key.find(ca.face_key[at(f)]);
    REQUIRE(it != cb.face_of_key.end());
    face_map[at(f)] = it->second;
  }
  // Cells match when their (mapped) face sets match.
  std::map<std::vector<index_t>, index_t> b_cells;
  for (index_t c = 0; c < b.n_cells(); ++c) {
    std::vector<index_t> faces(b.cell_faces(c).begin(), b.cell_faces(c).end());
    std::sort(faces.begin(), faces.end());
    b_cells[faces] = c;
  }
  std::vector<index_t> cell_map(at(a.n_cells()));
  for (index_t c = 0; c < a.n_cells(); ++c) {
    std::vector<index_t> faces;
    for (const index_t f : a.cell_faces(c)) faces.push_back(face_map[at(f)]);
    std::sort(faces.begin(), faces.end());
    const auto it = b_cells.find(faces);
    REQUIRE(it != b_cells.end());
    cell_map[at(c)] = it->second;
  }
  const autan::Geometry ga = autan::compute_geometry(a);
  const autan::Geometry gb = autan::compute_geometry(b);
  for (index_t f = 0; f < a.n_faces(); ++f) {
    const index_t g = face_map[at(f)];
    const bool same_owner = cell_map[at(a.owner()[at(f)])] == b.owner()[at(g)];
    const double dot = ga.face_area_x[at(f)] * gb.face_area_x[at(g)] +
                       ga.face_area_y[at(f)] * gb.face_area_y[at(g)] +
                       ga.face_area_z[at(f)] * gb.face_area_z[at(g)];
    CHECK((dot > 0) == same_owner);
    if (f >= a.n_interior_faces()) {
      CHECK(a.face_zones()[at(a.face_zone()[at(f)])].name ==
            b.face_zones()[at(b.face_zone()[at(g)])].name);
    }
  }
  for (index_t c = 0; c < a.n_cells(); ++c) {
    const index_t d = cell_map[at(c)];
    CHECK_THAT(gb.cell_volume[at(d)], WithinRel(ga.cell_volume[at(c)], 1e-8));
    if (cell_zones) {
      CHECK(a.cell_zones()[at(a.cell_zone()[at(c)])].name ==
            b.cell_zones()[at(b.cell_zone()[at(d)])].name);
    }
  }
}

autan::FaceZone zone_named(const autan::Mesh& mesh, const std::string& name) {
  const auto zones = mesh.face_zones();
  const auto it = std::find_if(zones.begin(), zones.end(),
                               [&](const autan::FaceZone& z) { return z.name == name; });
  REQUIRE(it != zones.end());
  return *it;
}

void require_valid_box(const autan::Mesh& mesh, double volume) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto closure = autan::cell_closure_error(mesh, g);
  CHECK(*std::max_element(closure.begin(), closure.end()) < 1e-12);
  CHECK_THAT(std::accumulate(g.cell_volume.begin(), g.cell_volume.end(), 0.0),
             WithinRel(volume, 1e-12));
}

// One directory per test process: ctest runs test cases in parallel.
const fs::path& temp_dir() {
  static const fs::path dir = [] {
    std::random_device random;
    return fs::temp_directory_path() / ("autan_io_tests-" + std::to_string(random()));
  }();
  return dir;
}

fs::path write_temp(const std::string& name, const std::string& content) {
  const fs::path dir = temp_dir();
  fs::create_directories(dir);
  const fs::path path = dir / name;
  std::ofstream(path, std::ios::binary) << content;
  return path;
}

void require_error(autan::ErrorCode expected, const fs::path& path) {
  try {
    (void)autan::io::read_mesh(path);
    FAIL("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

}  // namespace

// Facts of the unit-cube test meshes known from their construction
// (tests/data/mesh/generate/generate.py), independent of every reader: each
// boundary face lies on the plane its zone names, and structured cells have
// the volume of their share of a cube of side 1/n.
void require_constructed(const autan::Mesh& mesh, const std::string& name) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::map<std::string, std::pair<int, double>> planes{
      {"xmin", {0, 0.0}}, {"xmax", {0, 1.0}}, {"ymin", {1, 0.0}},
      {"ymax", {1, 1.0}}, {"zmin", {2, 0.0}}, {"zmax", {2, 1.0}}};
  for (index_t f = mesh.n_interior_faces(); f < mesh.n_faces(); ++f) {
    const auto it = planes.find(mesh.face_zones()[at(mesh.face_zone()[at(f)])].name);
    REQUIRE(it != planes.end());
    const std::array<double, 3> c{g.face_centroid_x[at(f)], g.face_centroid_y[at(f)],
                                  g.face_centroid_z[at(f)]};
    CHECK_THAT(c[static_cast<std::size_t>(it->second.first)], WithinAbs(it->second.second, 1e-9));
  }
  const std::map<std::string, int> cubes_per_side{
      {"hex", 3}, {"prism", 3}, {"pyramid", 2}, {"tet", 2}, {"hybrid", 4}};
  const auto n = cubes_per_side.find(name);
  if (n == cubes_per_side.end()) return;
  const double cube = std::pow(1.0 / n->second, 3);
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    int triangles = 0;
    for (const index_t f : mesh.cell_faces(c)) triangles += mesh.face_nodes(f).size() == 3 ? 1 : 0;
    const std::size_t faces = mesh.cell_faces(c).size();
    // hexahedron: whole cube; prism (2 triangles): half; pyramid (4
    // triangles, 5 faces) and Kuhn tetrahedron: a sixth.
    const double share = faces == 6 ? 1.0 : (faces == 5 && triangles == 2) ? 0.5 : 1.0 / 6.0;
    CHECK_THAT(g.cell_volume[at(c)], WithinRel(share * cube, 1e-8));
  }
}

TEST_CASE("Every format of a test mesh reads to the same mesh", "[io]") {
  const std::string name = GENERATE("hex", "prism", "pyramid", "tet", "hybrid", "tet-gmsh");
  CAPTURE(name);
  const fs::path dir = data_dir / name;
  const autan::Mesh reference = autan::io::read_mesh(dir / "polyMesh");
  require_valid_box(reference, 1.0);
  require_constructed(reference, name);
  CHECK(reference.source().format == "polymesh");

  const std::array<std::string, 6> sides{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
  for (const auto& side : sides) CHECK(zone_named(reference, side).boundary);
  CHECK(zone_named(reference, "ymin").kind == autan::BcKind::wall);
  CHECK(zone_named(reference, "zmax").kind == autan::BcKind::symmetry);
  CHECK(zone_named(reference, "xmin").kind == autan::BcKind::unspecified);
  if (fs::exists(dir / "polyMesh-binary"))
    require_same_mesh(reference, autan::io::read_mesh(dir / "polyMesh-binary"));

  for (const char* file : {".ascii.msh", ".binary.msh"}) {
    CAPTURE(file);
    const autan::Mesh gmsh = autan::io::read_mesh(dir / (name + file));
    CHECK(gmsh.source().format == "gmsh");
    require_constructed(gmsh, name);
    require_same_mesh(reference, gmsh);
  }
  for (const char* file : {".fluent.msh", ".fluent-binary.msh"}) {
    CAPTURE(file);
    const autan::Mesh fluent = autan::io::read_mesh(dir / (name + file));
    CHECK(fluent.source().format == "fluent");
    require_constructed(fluent, name);
    require_same_mesh(reference, fluent, false);  // foamMeshToFluent writes one cell zone
    CHECK(zone_named(fluent, "ymin").bc_type == "wall");
    CHECK(zone_named(fluent, "ymin").kind == autan::BcKind::wall);
    CHECK(zone_named(fluent, "zmax").kind == autan::BcKind::symmetry);
    CHECK_FALSE(zone_named(fluent, "interior-1").boundary);
    REQUIRE(fluent.cell_zones().size() == 1);
    CHECK(fluent.cell_zones()[0].kind == autan::CellKind::fluid);
  }
}

TEST_CASE("Polyhedral meshes read identically from polyMesh and Fluent", "[io]") {
  const std::string name = GENERATE("poly", "hanging");
  CAPTURE(name);
  const fs::path dir = data_dir / name;
  const autan::Mesh reference = autan::io::read_mesh(dir / "polyMesh");
  require_valid_box(reference, name == "hanging" ? 2.0 : 1.0);
  if (fs::exists(dir / "polyMesh-binary"))
    require_same_mesh(reference, autan::io::read_mesh(dir / "polyMesh-binary"));
  for (const char* file : {".fluent.msh", ".fluent-binary.msh"}) {
    CAPTURE(file);
    require_same_mesh(reference, autan::io::read_mesh(dir / (name + file)), false);
  }
  index_t max_faces = 0;
  for (index_t c = 0; c < reference.n_cells(); ++c) {
    max_faces = std::max(max_faces, static_cast<index_t>(reference.cell_faces(c).size()));
  }
  if (name == "hanging") {
    CHECK(max_faces == 105);
    REQUIRE(reference.cell_zones().size() == 2);
    CHECK(reference.cell_zones()[0].name == "coarse");
    CHECK(zone_named(reference, "walls").kind == autan::BcKind::wall);
  } else {
    CHECK(max_faces > 6);
  }
}

namespace {

// Written by hand from the Fluent file format specification: one unit
// tetrahedron; one boundary face lists c0 = 0 (normal leaves c1), the zone
// section 39 overrides the code of its face section, comments in between.
const std::string fluent_tet = R"((0 "hand-written single tetrahedron")
(2 3)
(10 (0 1 4 0 3))
(12 (0 1 1 0))
(13 (0 1 4 0))
(10 (1 1 4 1 3)(
0 0 0
1 0 0
0 1 0
0 0 1
))
(12 (2 1 1 1 2))
(0 "faces")
(13 (3 1 2 3 3)(
1 2 3 1 0
1 4 2 1 0
))
(13 (4 3 4 3 0)(
3 1 3 4 1 0
3 2 3 4 0 1
))
(45 (2 fluid tet-cell)())
(45 (3 wall bottom-and-front)())
(39 (4 velocity-inlet sides 1)())
)";

// Written by hand from the MSH 4.1 specification: sparse node tags, a
// parametric node block, a point element, a surface outside every physical
// group and an unknown section.
const std::string gmsh_tet = R"($MeshFormat
4.1 0 8
$EndMeshFormat
$Comments
not part of the mesh
$EndComments
$PhysicalNames
2
2 7 "base"
3 9 "solid-block"
$EndPhysicalNames
$Entities
1 0 2 1
1 0 0 0 0
1 0 0 0 1 1 0 1 7 0
2 0 0 0 1 1 1 0 0
1 0 0 0 1 1 1 1 9 0
$EndEntities
$Nodes
2 4 10 40
2 1 1 3
10
20
30
0 0 0 0 0
1 0 0 1 0
0 1 0 0 1
3 1 0 1
40
0 0 1
$EndNodes
$Elements
3 3 1 3
0 1 15 1
1 10
2 1 2 1
2 10 30 20
3 1 4 1
3 10 20 30 40
$EndElements
)";

std::string replace(std::string text, const std::string& from, const std::string& to) {
  const auto at_pos = text.find(from);
  REQUIRE(at_pos != std::string::npos);
  return text.replace(at_pos, from.size(), to);
}

}  // namespace

TEST_CASE("Hand-written files from the format specifications read as specified", "[io]") {
  SECTION("Fluent") {
    const autan::Mesh mesh = autan::io::read_mesh(write_temp("tet.msh", fluent_tet));
    const autan::Geometry g = autan::compute_geometry(mesh);
    REQUIRE(mesh.n_cells() == 1);
    CHECK_THAT(g.cell_volume[0], WithinRel(1.0 / 6.0, 1e-15));
    const auto closure = autan::cell_closure_error(mesh, g);
    CHECK(closure[0] < 1e-12);
    const autan::FaceZone sides = zone_named(mesh, "sides");
    CHECK(sides.bc_type == "velocity-inlet");
    CHECK(sides.kind == autan::BcKind::inlet);
    CHECK(sides.provenance.source_id == 4);
    CHECK(zone_named(mesh, "bottom-and-front").kind == autan::BcKind::wall);
    REQUIRE(mesh.cell_zones().size() == 1);
    CHECK(mesh.cell_zones()[0].name == "tet-cell");
    CHECK(mesh.cell_zones()[0].kind == autan::CellKind::fluid);
  }
  SECTION("Gmsh") {
    const autan::Mesh mesh = autan::io::read_mesh(write_temp("tet-gmsh.msh", gmsh_tet));
    const autan::Geometry g = autan::compute_geometry(mesh);
    REQUIRE(mesh.n_cells() == 1);
    CHECK(mesh.n_points() == 4);
    CHECK_THAT(g.cell_volume[0], WithinRel(1.0 / 6.0, 1e-15));
    const auto offsets = mesh.boundary_zone_offsets();
    const auto zones = mesh.face_zones();
    for (std::size_t z = 0; z < zones.size(); ++z) {
      const index_t count = offsets[z + 1] - offsets[z];
      CAPTURE(zones[z].name);
      CHECK(count == (zones[z].name == "base" ? 1 : 3));
    }
    CHECK(zone_named(mesh, "base").provenance.source_id == 7);
    CHECK(mesh.cell_zones()[0].name == "solid-block");
  }
}

TEST_CASE("Readers reject what the formats forbid or the core does not support", "[io]") {
  using autan::ErrorCode;
  SECTION("E0130 missing file") {
    require_error(ErrorCode::io_cannot_open, data_dir / "missing.msh");
  }
  SECTION("E0131 unknown content") {
    require_error(ErrorCode::io_unknown_format, write_temp("x.msh", "hello"));
  }
  SECTION("E0132 truncated Fluent file") {
    require_error(ErrorCode::io_malformed,
                  write_temp("cut.msh", fluent_tet.substr(0, fluent_tet.size() / 2)));
  }
  SECTION("E0132 Fluent cell whose faces contradict its element type") {
    require_error(
        ErrorCode::io_malformed,
        write_temp("hex.msh", replace(fluent_tet, "(12 (2 1 1 1 2))", "(12 (2 1 1 1 4))")));
  }
  SECTION("E0133 Fluent periodic shadow faces") {
    require_error(
        ErrorCode::io_unsupported,
        write_temp("periodic.msh", replace(fluent_tet, "(2 3)", "(2 3)\n(18 (1 2 3 4)())")));
  }
  SECTION("E0133 MSH 2.2") {
    require_error(ErrorCode::io_unsupported,
                  write_temp("v2.msh", "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"));
  }
  SECTION("E0133 second-order element") {
    require_error(ErrorCode::io_unsupported,
                  write_temp("tet10.msh", replace(gmsh_tet, "3 1 4 1\n", "3 1 11 1\n")));
  }
  SECTION("E0135 entity in two physical groups") {
    require_error(
        ErrorCode::io_ambiguous_zone,
        write_temp("two.msh", replace(gmsh_tet, "1 0 0 0 1 1 1 1 9 0", "1 0 0 0 1 1 1 2 9 10 0")));
  }
  SECTION("E0110 source zone named like a synthesised zone") {
    require_error(ErrorCode::mesh_duplicate_zone_name,
                  write_temp("clash.msh", replace(gmsh_tet, "2 7 \"base\"", "2 7 \"unassigned\"")));
  }
  SECTION("polyMesh") {
    const fs::path dir = temp_dir() / "polyMesh";
    fs::create_directories(temp_dir());
    fs::remove_all(dir);
    fs::copy(data_dir / "hex" / "polyMesh", dir);
    SECTION("E0133 decomposed mesh") {
      std::ifstream in(dir / "boundary");
      std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      std::ofstream(dir / "boundary")
          << replace(text, "type            patch;", "type            processor;");
      require_error(ErrorCode::io_unsupported, dir);
    }
    SECTION("E0135 cell in two cell zones") {
      std::ofstream(dir / "cellZones")
          << "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
             "2 ( a { type cellZone; cellLabels List<label> 2(0 1); }"
             " b { type cellZone; cellLabels List<label> 1(1); } )\n";
      require_error(ErrorCode::io_ambiguous_zone, dir);
    }
  }
}

namespace {

std::string foam_header(const char* format, const char* object_class, const char* object) {
  return std::string(
             "/* banner */\nFoamFile\n{\n    version 2.0; // trailing comment\n    format ") +
         format + ";\n    class " + object_class + ";\n    object " + object + ";\n}\n";
}

// One unit hexahedron as an OpenFOAM polyMesh using legal but less common
// forms: comments everywhere, ASCII faceCompactList with nested faces, a
// binary neighbour list holding a bare 0, an unsized label list and the
// Foundation class name of cellZones.
fs::path single_hex_polymesh() {
  const fs::path dir = temp_dir() / "single-hex";
  fs::create_directories(dir);
  write_temp("single-hex/points", foam_header("ascii", "vectorField", "points") +
                                      "8 // points\n(\n(0 0 0) (1 0 0) /* c */ (1 1 0) (0 1 0)\n"
                                      "(0 0 1) (1 0 1) (1 1 1) (0 1 1)\n)\n");
  write_temp(
      "single-hex/faces",
      foam_header("ascii", "faceCompactList", "faces") +
          "6\n(\n4(0 3 2 1)\n4(4 5 6 7)\n4(0 1 5 4)\n4(1 2 6 5)\n4(2 3 7 6)\n4(3 0 4 7)\n)\n");
  write_temp("single-hex/owner", foam_header("ascii", "labelList", "owner") + "6{0}\n");
  write_temp("single-hex/neighbour", foam_header("binary", "labelList", "neighbour") + "0\n");
  write_temp("single-hex/boundary",
             foam_header("ascii", "polyBoundaryMesh", "boundary") +
                 "1\n(\n    walls // the only patch\n    {\n        type wall /* no slip */;\n"
                 "        inGroups List<word> 1(wall);\n        nFaces 6;\n        startFace 0;\n  "
                 "  }\n)\n");
  write_temp("single-hex/cellZones",
             foam_header("ascii", "cellZoneList", "cellZones") +
                 "1\n(\nblock\n{\n    type cellZone;\n    cellLabels List<label> (0);\n}\n)\n");
  return dir;
}

}  // namespace

TEST_CASE("Legal variants of each format read as specified", "[io]") {
  SECTION("polyMesh") {
    const autan::Mesh mesh = autan::io::read_mesh(single_hex_polymesh());
    const autan::Geometry g = autan::compute_geometry(mesh);
    REQUIRE(mesh.n_cells() == 1);
    CHECK_THAT(g.cell_volume[0], WithinRel(1.0, 1e-15));
    CHECK(zone_named(mesh, "walls").kind == autan::BcKind::wall);
    CHECK(zone_named(mesh, "walls").bc_type == "wall");
    REQUIRE(mesh.cell_zones().size() == 1);
    CHECK(mesh.cell_zones()[0].name == "block");
  }
  SECTION("Gmsh $PhysicalNames after $Elements, and a group without elements") {
    std::string text = replace(
        gmsh_tet, "$PhysicalNames\n2\n2 7 \"base\"\n3 9 \"solid-block\"\n$EndPhysicalNames\n", "");
    text +=
        "$PhysicalNames\n3\n2 7 \"base\"\n2 8 \"spare\"\n3 9 \"solid-block\"\n$EndPhysicalNames\n";
    const autan::Mesh mesh = autan::io::read_mesh(write_temp("late-names.msh", text));
    CHECK(mesh.cell_zones()[0].name == "solid-block");
    CHECK(zone_named(mesh, "base").provenance.source_id == 7);
    const auto zones = mesh.face_zones();
    const auto spare = static_cast<std::size_t>(
        std::find_if(zones.begin(), zones.end(),
                     [](const autan::FaceZone& z) { return z.name == "spare"; }) -
        zones.begin());
    REQUIRE(spare < zones.size());
    CHECK(zones[spare].boundary);
    CHECK(mesh.boundary_zone_offsets()[spare + 1] == mesh.boundary_zone_offsets()[spare]);
  }
  SECTION("Fluent zone declared without faces") {
    const autan::Mesh mesh =
        autan::io::read_mesh(write_temp("spare.msh", fluent_tet + "(45 (11 wall spare)())\n"));
    CHECK(zone_named(mesh, "spare").kind == autan::BcKind::wall);
    CHECK(zone_named(mesh, "spare").boundary);
  }
}

TEST_CASE("Hostile or contradictory sizes are coded errors, not crashes", "[io]") {
  using autan::ErrorCode;
  SECTION("Gmsh node count") {
    require_error(ErrorCode::io_malformed,
                  write_temp("count.msh",
                             replace(gmsh_tet, "2 4 10 40\n", "2 18446744073709551615 10 40\n")));
  }
  SECTION("Gmsh non-positive physical tag") {
    require_error(ErrorCode::io_malformed,
                  write_temp("negative.msh",
                             replace(gmsh_tet, "1 0 0 0 1 1 1 1 9 0", "1 0 0 0 1 1 1 1 -9 0")));
  }
  SECTION("Fluent node range") {
    require_error(ErrorCode::io_malformed,
                  write_temp("range.msh", replace(fluent_tet, "(10 (1 1 4 1 3)(",
                                                  "(10 (1 1 7fffffffffff 1 3)(")));
  }
  SECTION("polyMesh point count") {
    const fs::path dir = single_hex_polymesh();
    write_temp("single-hex/points",
               foam_header("ascii", "vectorField", "points") + "6148914691236517206((0 0 0))\n");
    require_error(ErrorCode::io_malformed, dir);
  }
  SECTION("polyMesh compressed cell zones") {
    const fs::path dir = single_hex_polymesh();
    fs::rename(dir / "cellZones", dir / "cellZones.gz");
    require_error(ErrorCode::io_unsupported, dir);
  }
}
