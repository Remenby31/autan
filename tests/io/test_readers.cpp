#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "support.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace autan_test;  // NOLINT(google-build-using-namespace)

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

  // Files Gmsh wrote: MSH 4.1 (ASCII, binary) and, when built with CGNS,
  // its CGNS export (standard elements, physical groups as families).
  std::vector<std::pair<std::string, std::string>> gmsh_files{{".ascii.msh", "gmsh"},
                                                              {".binary.msh", "gmsh"}};
#ifdef AUTAN_HAVE_CGNS
  gmsh_files.emplace_back(".cgns", "cgns");
#endif
  for (const auto& [file, format] : gmsh_files) {
    CAPTURE(file);
    const autan::Mesh gmsh = autan::io::read_mesh(dir / (name + file));
    CHECK(gmsh.source().format == format);
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
    CHECK_FALSE(zone_named(fluent, "(interior 2)").boundary);  // named by its id (mesh.md, Readers)
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
    const std::string solid_tet =
        replace(fluent_tet, "(45 (2 fluid tet-cell)())", "(45 (2 solid tet-cell)())");
    const autan::Mesh solid = autan::io::read_mesh(write_temp("solid.msh", solid_tet));
    CHECK(solid.cell_zones()[0].kind == autan::CellKind::solid);
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
    const autan::Mesh ungrouped = autan::io::read_mesh(
        write_temp("ungrouped.msh", replace(gmsh_tet, "1 0 0 0 1 1 1 1 9 0", "1 0 0 0 1 1 1 0 0")));
    REQUIRE(ungrouped.cell_zones().size() == 1);
    CHECK(ungrouped.cell_zones()[0].name == "fluid");
    CHECK(ungrouped.cell_zones()[0].provenance.source_id == -1);
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
  SECTION("E0132 Fluent node declared without coordinates") {
    require_error(
        ErrorCode::io_malformed,
        write_temp("node5.msh", replace(fluent_tet, "(10 (0 1 4 0 3))", "(10 (0 1 5 0 3))")));
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
// forms: comments everywhere, one right after a word, ASCII faceCompactList
// with nested faces, a binary neighbour list holding a bare 0, a
// sub-dictionary holding a quoted brace, an unsized label list and the
// Foundation class name of cellZones.
fs::path single_hex_polymesh() {
  const fs::path dir = temp_dir() / "single-hex";
  fs::create_directories(dir);
  write_temp("single-hex/points",
             foam_header("ascii", "vectorField", "points") +
                 "8 // points\n(\n(0 0 0) (1/* x */ 0 0) /* c */ (1 1 0) (0 1 0)\n"
                 "(0 0 1) (1 0 1) (1 1 1) (0 1 1)\n)\n");
  write_temp(
      "single-hex/faces",
      foam_header("ascii", "faceCompactList", "faces") +
          "6\n(\n4(0 3 2 1)\n4(4 5 6 7)\n4(0 1 5 4)\n4(1 2 6 5)\n4(2 3 7 6)\n4(3 0 4 7)\n)\n");
  write_temp("single-hex/owner", foam_header("ascii", "labelList", "owner") + "6{0}\n");
  write_temp("single-hex/neighbour", foam_header("ascii", "labelList", "neighbour") + "0{}\n");
  write_temp("single-hex/boundary",
             foam_header("ascii", "polyBoundaryMesh", "boundary") +
                 "1\n(\n    walls // the only patch\n    {\n        type wall/* no slip */;\n"
                 "        inGroups List<word> 1(wall);\n        meta { note \"} not the end\"; }\n"
                 "        nFaces 6;\n        startFace 0;\n    }\n)\n");
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
    // OpenFOAM writes an empty binary list as a bare size.
    write_temp("single-hex/neighbour", foam_header("binary", "labelList", "neighbour") + "0\n");
    CHECK(autan::io::read_mesh(temp_dir() / "single-hex").n_cells() == 1);
    // arch label=64: binary labels take 8 bytes.
    write_temp("single-hex/owner", replace(foam_header("binary", "labelList", "owner"), "    class",
                                           "    arch \"LSB;label=64;scalar=64\";\n    class") +
                                       "6(" + std::string(48, '\0') + ")\n");
    CHECK(autan::io::read_mesh(temp_dir() / "single-hex").n_cells() == 1);
    // A directory holding `points` is the polyMesh, even beside a constant/polyMesh.
    fs::create_directories(temp_dir() / "single-hex" / "constant");
    fs::copy(data_dir / "hex" / "polyMesh", temp_dir() / "single-hex" / "constant" / "polyMesh");
    CHECK(autan::io::read_mesh(temp_dir() / "single-hex").n_cells() == 1);
    fs::remove_all(temp_dir() / "single-hex" / "constant");
  }
  SECTION("Gmsh $PhysicalNames after $Elements, groups without elements, an empty name") {
    std::string text = replace(
        gmsh_tet, "$PhysicalNames\n2\n2 7 \"base\"\n3 9 \"solid-block\"\n$EndPhysicalNames\n", "");
    text +=
        "$PhysicalNames\n4\n2 6 \"\"\n2 7 \"base\"\n2 8 \"spare\"\n3 9 "
        "\"solid-block\"\n$EndPhysicalNames\n";
    const autan::Mesh mesh = autan::io::read_mesh(write_temp("late-names.msh", text));
    CHECK(mesh.cell_zones()[0].name == "solid-block");
    CHECK(zone_named(mesh, "base").provenance.source_id == 7);
    CHECK(zone_named(mesh, "6").provenance.source_id == 6);
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
  SECTION("Gmsh element count") {
    require_error(
        ErrorCode::io_malformed,
        write_temp("elements.msh", replace(gmsh_tet, "3 3 1 3\n", "3 18446744073709551615 1 3\n")));
  }
  SECTION("Gmsh physical name tag 0") {
    require_error(ErrorCode::io_malformed,
                  write_temp("name.msh", replace(gmsh_tet, "2 7 \"base\"", "2 0 \"base\"")));
  }
  SECTION("Gmsh non-positive physical tag") {
    require_error(ErrorCode::io_malformed,
                  write_temp("negative.msh",
                             replace(gmsh_tet, "1 0 0 0 1 1 1 1 9 0", "1 0 0 0 1 1 1 1 -9 0")));
    require_error(
        ErrorCode::io_malformed,
        write_temp("zero.msh", replace(gmsh_tet, "1 0 0 0 1 1 1 1 9 0", "1 0 0 0 1 1 1 1 0 0")));
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

TEST_CASE("Gmsh $Periodic surface links become periodic pairs", "[io][periodic]") {
  // periodic-box (generate.py): unstructured tetrahedra in the unit cube,
  // xmax meshed as xmin translated by (1, 0, 0), ymax as ymin by (0, 1, 0).
  const std::string file = GENERATE("periodic-box.ascii.msh", "periodic-box.binary.msh");
  CAPTURE(file);
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "periodic-box" / file);
  require_valid_box(mesh, 1.0);
  REQUIRE(mesh.periodic_pairs().size() == 2);
  const std::map<std::pair<std::string, std::string>, std::array<double, 3>> expected{
      {{"xmin", "xmax"}, {1.0, 0.0, 0.0}}, {{"ymin", "ymax"}, {0.0, 1.0, 0.0}}};
  for (const auto& pair : mesh.periodic_pairs()) {
    const auto key =
        std::pair{mesh.face_zones()[at(pair.zone_a)].name, mesh.face_zones()[at(pair.zone_b)].name};
    REQUIRE(expected.contains(key));
    CHECK(pair.translation == expected.at(key));
  }
  for (const char* name : {"xmin", "xmax", "ymin", "ymax"})
    CHECK(zone_named(mesh, name).kind == autan::BcKind::periodic);
  for (const char* name : {"zmin", "zmax"})
    CHECK(zone_named(mesh, name).kind == autan::BcKind::unspecified);

  std::ifstream in(data_dir / "periodic-box" / "periodic-box.ascii.msh", std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto edited = [&](const std::string& from, const std::string& to) {
    std::string copy = text;
    const auto at_link = copy.find(from);
    REQUIRE(at_link != std::string::npos);
    copy.replace(at_link, from.size(), to);
    return write_temp("periodic-edited.msh", copy);
  };
  // The xmin -> xmax link with a rotation by 90 degrees about z.
  require_error(autan::ErrorCode::io_unsupported,
                edited("2 2 1\n16 1 0 0 1 0 1 0 0", "2 2 1\n16 0 -1 0 1 1 0 0 0"));
  // The same link with a translation its node pairs do not follow.
  require_error(autan::ErrorCode::io_malformed,
                edited("2 2 1\n16 1 0 0 1 0", "2 2 1\n16 1 0 0 1.01 0"));
  // A node pair of that link swapped for another: it contradicts the link.
  require_error(autan::ErrorCode::io_malformed, edited("58\n5 1\n6 2\n", "58\n5 2\n6 1\n"));
  // A link of a dimension the format does not have.
  require_error(autan::ErrorCode::io_malformed, edited("2 2 1\n16 1 0 0 1", "4 2 1\n16 1 0 0 1"));
  // A curve link to entity tag 0: entity tags are positive.
  require_error(autan::ErrorCode::io_malformed, edited("\n1 3 1\n", "\n1 0 1\n"));
  // The xmin -> xmax link without its affine transform.
  require_error(autan::ErrorCode::io_unsupported,
                edited("2 2 1\n16 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1\n", "2 2 1\n0\n"));
  // A first link between the same groups, by (2, 0, 0) and without node pairs.
  require_error(
      autan::ErrorCode::io_malformed,
      edited("$Periodic\n15\n", "$Periodic\n16\n2 2 1\n16 1 0 0 2 0 1 0 0 0 0 1 0 0 0 0 1\n0\n"));
  // Node pairs off the translation by exactly the tolerance, 1e-9 of its length.
  const autan::Mesh shifted =
      autan::io::read_mesh(edited("2 2 1\n16 1 0 0 1 0 1 0 0", "2 2 1\n16 1 0 0 1 0 1 0 1e-09"));
  CHECK(shifted.periodic_pairs().size() == 2);
}

TEST_CASE("A Gmsh section name longer than 4096 bytes is E0122 before it is copied", "[io]") {
  // mesh.md, E0122: an unknown section is skipped, its name read whole first.
  for (const std::size_t bytes : {std::size_t{4096}, std::size_t{4097}}) {
    CAPTURE(bytes);
    const std::string name(bytes, 'S');
    const fs::path file = write_temp("section-" + std::to_string(bytes) + ".msh",
                                     gmsh_tet + "$" + name + "\nignored\n$End" + name + "\n");
    if (bytes == 4096) {
      CHECK(autan::io::read_mesh(file).n_cells() == 1);
    } else {
      require_error(autan::ErrorCode::mesh_text_too_long, file);
    }
  }
}

TEST_CASE("A zone name longer than 4096 bytes is E0122 in the text readers", "[io]") {
  // mesh.md, E0122: tests/data/mesh/hex with its zone xmin renamed to 4096 bytes (read) and to
  // 4097 (refused), in each format whose names are text: the Gmsh physical name, the Fluent
  // zone name, the polyMesh patch name.
  const std::string format = GENERATE("gmsh", "fluent", "polymesh");
  CAPTURE(format);
  const auto slurp = [](const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  };
  for (const std::size_t bytes : {std::size_t{4096}, std::size_t{4097}}) {
    CAPTURE(bytes);
    const std::string name(bytes, 'x');
    const std::string tag = format + "-" + std::to_string(bytes);
    fs::path path;
    if (format == "polymesh") {
      path = temp_dir() / tag / "polyMesh";
      fs::create_directories(path);
      for (const auto& entry : fs::directory_iterator(data_dir / "hex" / "polyMesh"))
        fs::copy_file(entry.path(), path / entry.path().filename());
      std::string boundary = slurp(path / "boundary");
      const auto at = boundary.find("    xmin\n");
      REQUIRE(at != std::string::npos);
      boundary.replace(at + 4, 4, name);
      std::ofstream(path / "boundary", std::ios::binary) << boundary;
    } else {
      const bool gmsh = format == "gmsh";
      std::string text = slurp(data_dir / "hex" / (gmsh ? "hex.ascii.msh" : "hex.fluent.msh"));
      const std::string old = gmsh ? "\"xmin\"" : " xmin)";
      const auto at = text.find(old);
      REQUIRE(at != std::string::npos);
      text.replace(at, old.size(), gmsh ? "\"" + name + "\"" : " " + name + ")");
      path = write_temp(tag + ".msh", text);
    }
    if (bytes == 4096) {
      CHECK(zone_named(autan::io::read_mesh(path), name).boundary);
    } else {
      require_error(autan::ErrorCode::mesh_text_too_long, path);
    }
  }
}

#if defined(__linux__)
TEST_CASE("Records a reader does not use cost their text, not a copy each", "[io]") {
  // mesh.md, Readers: Gmsh physical names no zone takes, element blocks that label nothing and
  // polyMesh dictionary entries the reader does not use are read past, so 2^19 of them cost the
  // file's text, held once, and the mesh reads the same.
  constexpr int n = 1 << 19;
  SECTION("Gmsh physical names of points and of volumes no element uses") {
    std::string names;
    for (int i = 0; i < n; ++i) {
      names += "0 " + std::to_string(i + 1) + " \"p\"\n3 " + std::to_string(i + 100) + " \"v\"\n";
    }
    const fs::path plain = write_temp("names-plain.msh", gmsh_tet);
    const std::string text = replace(gmsh_tet, "$PhysicalNames\n2\n",
                                     "$PhysicalNames\n" + std::to_string(2 + 2 * n) + "\n" + names);
    const fs::path padded = write_temp("names-padded.msh", text);
    const auto added = static_cast<std::int64_t>(text.size() - gmsh_tet.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(std::ranges::any_of(mesh->face_zones(), [](const auto& z) { return z.name == "base"; }));
    CHECK(std::ranges::any_of(mesh->cell_zones(),
                              [](const auto& z) { return z.name == "solid-block"; }));
  }
  SECTION("polyMesh boundary entries the reader does not use") {
    const fs::path dir = single_hex_polymesh();
    (void)autan::io::read_mesh(dir);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(dir); });
    std::string entries;
    for (int i = 0; i < n; ++i) entries += "        k" + std::to_string(i) + " 0;\n";
    const std::string boundary = foam_header("ascii", "polyBoundaryMesh", "boundary") +
                                 "1\n(\n    walls\n    {\n        type wall;\n" + entries +
                                 "        nFaces 6;\n        startFace 0;\n    }\n)\n";
    write_temp("single-hex/boundary", boundary);
    const auto added = static_cast<std::int64_t>(entries.size());
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(dir); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(std::ranges::any_of(mesh->face_zones(), [](const auto& z) { return z.name == "walls"; }));
  }
  SECTION("Gmsh volume entities in no physical group") {
    std::string entities;
    for (int i = 0; i < n; ++i) entities += std::to_string(i + 2) + " 0 0 0 1 1 1 0 0\n";
    const fs::path plain = write_temp("entities-plain.msh", gmsh_tet);
    const std::string text = replace(replace(gmsh_tet, "$Entities\n1 0 2 1\n",
                                             "$Entities\n1 0 2 " + std::to_string(1 + n) + "\n"),
                                     "$EndEntities", entities + "$EndEntities");
    const fs::path padded = write_temp("entities-padded.msh", text);
    const auto added = static_cast<std::int64_t>(text.size() - gmsh_tet.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(std::ranges::any_of(mesh->cell_zones(),
                              [](const auto& z) { return z.name == "solid-block"; }));
  }
  SECTION("Gmsh curve blocks and surface blocks of entities in no group") {
    // Each block one element: a curve, or a triangle of its own entity, which no group holds.
    std::string blocks;
    for (int i = 0; i < n / 2; ++i) {
      blocks += "1 1 1 1\n" + std::to_string(i + 4) + " 10 20\n2 " + std::to_string(i + 2) +
                " 2 1\n" + std::to_string(n / 2 + i + 4) + " 10 20 30\n";
    }
    const fs::path plain = write_temp("blocks-plain.msh", gmsh_tet);
    const std::string text =
        replace(replace(gmsh_tet, "$Elements\n3 3 1 3\n",
                        "$Elements\n" + std::to_string(3 + n) + " " + std::to_string(3 + n) +
                            " 1 " + std::to_string(3 + n) + "\n"),
                "$EndElements", blocks + "$EndElements");
    const fs::path padded = write_temp("blocks-padded.msh", text);
    const auto added = static_cast<std::int64_t>(text.size() - gmsh_tet.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(mesh->n_cells() == 1);
    CHECK(std::ranges::any_of(mesh->face_zones(), [](const auto& z) { return z.name == "base"; }));
  }
  SECTION("Gmsh volume entities in a physical group without elements") {
    std::string entities;
    for (int i = 0; i < n; ++i) entities += std::to_string(i + 2) + " 0 0 0 1 1 1 1 9 0\n";
    const fs::path plain = write_temp("grouped-plain.msh", gmsh_tet);
    const std::string text = replace(replace(gmsh_tet, "$Entities\n1 0 2 1\n",
                                             "$Entities\n1 0 2 " + std::to_string(1 + n) + "\n"),
                                     "$EndEntities", entities + "$EndEntities");
    const fs::path padded = write_temp("grouped-padded.msh", text);
    const auto added = static_cast<std::int64_t>(text.size() - gmsh_tet.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(std::ranges::any_of(mesh->cell_zones(),
                              [](const auto& z) { return z.name == "solid-block"; }));
  }
  SECTION("Fluent zone records repeated") {
    // The last record of an id names its zone: the others are read past, not kept.
    std::string records;
    for (int i = 0; i < n; ++i) records += "(45 (3 wall bottom-and-front)())\n";
    const fs::path plain = write_temp("repeated-plain.msh", fluent_tet);
    const fs::path padded = write_temp("repeated-padded.msh", fluent_tet + records);
    const auto added = static_cast<std::int64_t>(records.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    CHECK(std::ranges::count(mesh->face_zones(), std::string("bottom-and-front"),
                             &autan::FaceZone::name) == 1);
  }
  SECTION("Fluent fluid zones no cell is in") {
    std::string zones;
    // The file's own fluid zone and n / 32 - 1 more: the 16 384 records a file may hold.
    for (int i = 0; i < n / 32 - 1; ++i) {
      zones += "(45 (" + std::to_string(100 + i) + " fluid " + std::string(1024, 'f') + ")())\n";
    }
    const fs::path plain = write_temp("fluid-plain.msh", fluent_tet);
    const fs::path padded = write_temp("fluid-padded.msh", fluent_tet + zones);
    const auto added = static_cast<std::int64_t>(zones.size());
    (void)autan::io::read_mesh(plain);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    REQUIRE(mesh->cell_zones().size() == 1);
    CHECK(mesh->cell_zones()[0].name == "tet-cell");
  }
  SECTION("polyMesh cell zones without cells") {
    const fs::path dir = single_hex_polymesh();
    (void)autan::io::read_mesh(dir);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(dir); });
    std::string zones;
    for (int i = 0; i < n / 32; ++i) {
      zones += "e" + std::to_string(i) + std::string(1024, 'z') + "\n{\n    type cellZone;\n" +
               "    cellLabels List<label> 0();\n}\n";
    }
    const std::string cell_zones = foam_header("ascii", "cellZoneList", "cellZones") +
                                   std::to_string(1 + n / 32) +
                                   "\n(\nblock\n{\n    type cellZone;\n    cellLabels List<label> "
                                   "(0);\n}\n" +
                                   zones + ")\n";
    write_temp("single-hex/cellZones", cell_zones);
    const auto added = static_cast<std::int64_t>(zones.size());
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(dir); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));
    REQUIRE(mesh->cell_zones().size() == 1);
    CHECK(mesh->cell_zones()[0].name == "block");
  }
  SECTION("polyMesh cell labels given again") {
    // The last occurrence of a keyword wins (OpenFOAM User Guide, 4.2.14): the list it replaces
    // is read past, not held.
    const fs::path dir = single_hex_polymesh();
    (void)autan::io::read_mesh(dir);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(dir); });
    std::string replaced = std::to_string(n) + "(";
    for (int i = 0; i < n; ++i) replaced += "0 ";
    replaced += ")";
    const std::string cell_zones = foam_header("ascii", "cellZoneList", "cellZones") +
                                   "1\n(\nblock\n{\n    type cellZone;\n    cellLabels "
                                   "List<label> " +
                                   replaced + ";\n    cellLabels List<label> 1(0);\n}\n)\n";
    write_temp("single-hex/cellZones", cell_zones);
    const auto added = static_cast<std::int64_t>(replaced.size());
    std::optional<autan::Mesh> mesh;
    const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(dir); });
    CAPTURE(before, after, added);
    CHECK(after - before <= added + (std::int64_t{1} << 20));  // held, 8 bytes a label
    REQUIRE(mesh->cell_zones().size() == 1);
    CHECK(mesh->cell_zones()[0].name == "block");
  }
}
namespace {

// A column of `cells` unit hexahedra along z, written as Fluent (hexadecimal indices, normals
// toward c0): its boundary faces in one wall section, each interior face in a section and zone of
// its own.
std::string fluent_column(int cells) {
  const auto hex = [](long long v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%llx", v);
    return std::string(buffer);
  };
  const auto node = [](int layer, int corner) { return 4LL * layer + corner + 1; };
  const auto face = [&](std::array<long long, 4> nodes, long long c0, long long c1) {
    std::string row;
    for (const long long v : nodes) row += hex(v) + " ";
    return row + hex(c0) + " " + hex(c1) + "\n";
  };
  const long long n_nodes = 4LL * (cells + 1);
  const long long n_boundary = 4LL * cells + 2;
  const long long n_faces = n_boundary + cells - 1;
  std::string out = "(0 \"column\")\n(2 3)\n(10 (0 1 " + hex(n_nodes) + " 0 3))\n(10 (1 1 " +
                    hex(n_nodes) + " 1 3)(\n";
  const std::array<std::array<int, 2>, 4> corners{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
  for (int layer = 0; layer <= cells; ++layer) {
    for (const auto& [x, y] : corners) {
      out += std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(layer) + "\n";
    }
  }
  out += "))\n(12 (0 1 " + hex(cells) + " 0))\n(12 (2 1 " + hex(cells) + " 1 4))\n";
  out += "(13 (0 1 " + hex(n_faces) + " 0))\n(13 (3 1 " + hex(n_boundary) + " 3 4)(\n";
  out += face({node(0, 0), node(0, 1), node(0, 2), node(0, 3)}, 1, 0);
  out += face({node(cells, 3), node(cells, 2), node(cells, 1), node(cells, 0)}, cells, 0);
  for (int c = 1; c <= cells; ++c) {
    const int lo = c - 1;
    for (int side = 0; side < 4; ++side) {
      const int a = side, b = (side + 1) % 4;
      out += face({node(lo, a), node(c, a), node(c, b), node(lo, b)}, c, 0);
    }
  }
  out += "))\n";
  for (int c = 1; c < cells; ++c) {
    const long long f = n_boundary + c;
    out += "(13 (" + hex(9 + c) + " " + hex(f) + " " + hex(f) + " 2 4)(\n" +
           face({node(c, 0), node(c, 1), node(c, 2), node(c, 3)}, c + 1, c) + "))\n";
  }
  return out + "(45 (2 fluid column)())\n(45 (3 wall walls)())\n";
}

}  // namespace

TEST_CASE("A Fluent file holds at most 16 384 fluid, solid or interior zone records", "[io]") {
  // mesh.md, Readers: a record each read again by offset for the cells that use it.
  std::string zones;
  for (int i = 0; i < 16383; ++i) zones += "(45 (" + std::to_string(100 + i) + " fluid f)())\n";
  CHECK(autan::io::read_mesh(write_temp("fluid-limit.msh", fluent_tet + zones)).n_cells() == 1);
  require_error(autan::ErrorCode::io_unsupported,
                write_temp("fluid-over.msh", fluent_tet + zones + "(45 (99999 solid s)())\n"));
  // Likewise its sections of interior faces, each a zone MeshInfo does not show: a column whose
  // 16 384 interior faces each have their own reads, one more is refused.
  const autan::Mesh column =
      autan::io::read_mesh(write_temp("sections-limit.msh", fluent_column(16385)));
  CHECK(column.n_cells() == 16385);
  CHECK(column.n_interior_faces() == 16384);
  require_error(autan::ErrorCode::io_unsupported,
                write_temp("sections-over.msh", fluent_column(16386)));
}
TEST_CASE("Fluent zone records of short names cost little beside their text", "[io]") {
  // mesh.md, Readers, and reader-memory.md: 16 383 empty interior zones of one-letter names cost
  // their text and their records, 16 bytes each, at most twice over: the reader's own bookkeeping
  // fits the 1 MiB beside the text.
  std::string zones;
  for (int i = 0; i < 16383; ++i) zones += "(45 (" + std::to_string(100 + i) + " interior i)())\n";
  const fs::path plain = write_temp("records-plain.msh", fluent_tet);
  const fs::path padded = write_temp("records-padded.msh", fluent_tet + zones);
  const auto added = static_cast<std::int64_t>(zones.size());
  (void)autan::io::read_mesh(plain);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  std::optional<autan::Mesh> mesh;
  const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
  CAPTURE(before, after, added);
  CHECK(after - before <= added + (std::int64_t{1} << 20));
  CHECK(mesh->face_zones().size() == 2);  // fluent_tet's zones 3 and 4
}

TEST_CASE("Fluent interior zones no face is in are not kept", "[io]") {
  // mesh.md, Readers: 16 383 empty interior zones of 1024-byte names cost their text.
  std::string zones;
  for (int i = 0; i < 16383; ++i) {
    zones += "(45 (" + std::to_string(100 + i) + " interior " + std::string(1024, 'i') + ")())\n";
  }
  const fs::path plain = write_temp("interior-plain.msh", fluent_tet);
  const fs::path padded = write_temp("interior-padded.msh", fluent_tet + zones);
  const auto added = static_cast<std::int64_t>(zones.size());
  (void)autan::io::read_mesh(plain);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  std::optional<autan::Mesh> mesh;
  const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(padded); });
  CAPTURE(before, after, added);
  CHECK(after - before <= added + (std::int64_t{1} << 20));
  CHECK(mesh->face_zones().size() == autan::io::read_mesh(plain).face_zones().size());
}

TEST_CASE("A Fluent zone of interior faces keeps no name of its own", "[io]") {
  // mesh.md, Readers: a zone whose faces are interior, whatever its type (a fan here), which
  // nothing exposes, is named by its id, so its record's name, here 4096 bytes and the same as a
  // boundary zone's, is neither kept nor checked.
  const std::string type = GENERATE("interior", "fan");
  CAPTURE(type);
  const std::string name(4096, 'n');
  std::ifstream in(data_dir / "hex" / "hex.fluent.msh", std::ios::binary);
  std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  text = replace(
      replace(text, "(39 (2 interior interior-1)())", "(39 (2 " + type + " " + name + ")())"),
      "(39 (10 pressure-outlet xmin)())", "(39 (10 pressure-outlet " + name + ")())");
  const autan::Mesh mesh = autan::io::read_mesh(write_temp("interior-named.msh", text));
  const auto named = std::ranges::count(mesh.face_zones(), name, &autan::FaceZone::name);
  CHECK(named == 1);
  CHECK(zone_named(mesh, name).boundary);
  CHECK_FALSE(zone_named(mesh, "(interior 2)").boundary);
}
TEST_CASE("Sizes a text reader is told are checked before they are allocated", "[io]") {
  SECTION("a polyMesh uniform list longer than the mesh's faces") {
    // mesh.md, Readers: N{v} expands to N labels from a few bytes of text; owner holds as many
    // as the faces file has faces. 2^28 is refused before the 2 GiB are allocated: anything
    // sized by N costs at least N bytes, sixteen times the bound.
    const fs::path dir = single_hex_polymesh();
    write_temp("single-hex/owner", foam_header("ascii", "labelList", "owner") + "268435456{0}\n");
    const std::int64_t peak = peak_of([&] { require_error(autan::ErrorCode::io_malformed, dir); });
    CHECK(peak <= std::int64_t{1} << 24U);
  }
  SECTION("a polyMesh header key longer than 4096 bytes") {
    const fs::path dir = single_hex_polymesh();
    write_temp("single-hex/owner", "FoamFile\n{\n    " + std::string(4097, 'k') +
                                       " 1;\n    format ascii;\n    class labelList;\n"
                                       "    object owner;\n}\n6{0}\n");
    require_error(autan::ErrorCode::mesh_text_too_long, dir);
  }
  SECTION("a Fluent section header of more than 8 fields") {
    // The node, cell and face headers hold five (Ansys Fluent legacy file formats).
    require_error(autan::ErrorCode::io_malformed,
                  write_temp("header.msh",
                             replace(fluent_tet, "(10 (0 1 4 0 3))", "(10 (0 1 4 0 3 0 0 0 0))")));
  }
}
#endif
