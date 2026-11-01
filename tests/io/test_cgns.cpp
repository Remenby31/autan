#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <cgnslib.h>

#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "fixtures.hpp"
#include "support.hpp"

using autan::index_t;
using namespace autan_test;  // NOLINT(google-build-using-namespace)

namespace {

// A solution with distinct values everywhere, so that any permutation or
// truncation shows; BC types by zone name.
autan::io::Solution solution_for(const autan::Mesh& mesh) {
  autan::io::Solution s;
  s.run_metadata = R"({"identity": "test"})";
  const auto zones = autan::io::meshinfo_zones(mesh);
  const auto offsets = mesh.boundary_zone_offsets();
  for (const index_t z : zones.boundary) {
    const auto& name = mesh.face_zones()[at(z)].name;
    s.boundary_bc_types.push_back(name == "ymin" || name == "walls" ? "BCWallViscousHeatFlux"
                                  : name == "zmax"                  ? "BCSymmetryPlane"
                                                                    : "BCFarfield");
    if (s.boundary_bc_types.back() == "BCWallViscousHeatFlux") {
      for (index_t f = offsets[at(z)]; f < offsets[at(z) + 1]; ++f) s.wall_faces.push_back(f);
    }
  }
  std::ranges::sort(s.wall_faces);
  for (const char* name : {"Density", "Pressure"}) {
    autan::io::Field f{.name = name, .values = {}};
    for (index_t c = 0; c < mesh.n_cells(); ++c) f.values.push_back(name[0] + 0.25 * c);
    s.fields.push_back(f);
  }
  autan::io::Field p{.name = "Pressure", .values = {}};
  for (std::size_t i = 0; i < s.wall_faces.size(); ++i)
    p.values.push_back(1e5 + static_cast<double>(i));
  s.wall_fields.push_back(p);
  return s;
}

std::vector<double> read_field(const fs::path& file, const char* name, cgsize_t n) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  std::vector<double> values(static_cast<std::size_t>(n));
  const cgsize_t first = 1;
  REQUIRE(cg_field_read(fn, 1, 1, 1, name, CGNS_ENUMV(RealDouble), &first, &n, values.data()) ==
          CG_OK);
  cg_close(fn);
  return values;
}

// Writes a minimal CGNS file with the library and lets `edit` add to it.
template <typename Edit>
fs::path cgns_file(const std::string& name, Edit edit) {
  const fs::path path = temp_dir() / name;
  fs::create_directories(temp_dir());
  int fn = 0, base = 0;
  REQUIRE(cg_set_file_type(CG_FILE_HDF5) == CG_OK);
  REQUIRE(cg_open(path.string().c_str(), CG_MODE_WRITE, &fn) == CG_OK);
  REQUIRE(cg_base_write(fn, "Base", 3, 3, &base) == CG_OK);
  edit(fn, base);
  REQUIRE(cg_close(fn) == CG_OK);
  return path;
}

void unstructured_tet(int fn, int base, const char* zone_name, cgsize_t cells = 1) {
  cgsize_t size[3] = {4, cells, 0};
  int zone = 0, coord = 0, section = 0;
  REQUIRE(cg_zone_write(fn, base, zone_name, size, CGNS_ENUMV(Unstructured), &zone) == CG_OK);
  const std::array<double, 4> x{0, 1, 0, 0}, y{0, 0, 1, 0}, z{0, 0, 0, 1};
  REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(), &coord) ==
          CG_OK);
  REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateY", y.data(), &coord) ==
          CG_OK);
  REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateZ", z.data(), &coord) ==
          CG_OK);
  const std::array<cgsize_t, 4> tet{1, 2, 3, 4};
  REQUIRE(cg_section_write(fn, base, zone, "Cells", CGNS_ENUMV(TETRA_4), 1, 1, 0, tet.data(),
                           &section) == CG_OK);
}

}  // namespace

TEST_CASE("solution.cgns round-trips the mesh and passes cgnscheck", "[cgns]") {
  const std::string name = GENERATE("hybrid", "poly", "hanging");
  CAPTURE(name);
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / name / "polyMesh");
  const autan::io::Solution solution = solution_for(mesh);
  const fs::path file = temp_dir() / (name + "-solution.cgns");
  fs::create_directories(temp_dir());
  autan::io::write_solution(file, mesh, solution);

  // An independent checker: the CGNS distribution's cgnscheck.
  const std::string command = std::string(AUTAN_CGNSCHECK) + " " + file.string() + " > " +
                              (file.string() + ".check") + " 2>&1";
  CHECK(std::system(command.c_str()) == 0);
  std::ifstream report(file.string() + ".check");
  const std::string text((std::istreambuf_iterator<char>(report)),
                         std::istreambuf_iterator<char>());
  CHECK(text.find("ERROR") == std::string::npos);

  const autan::Mesh back = autan::io::read_mesh(file);
  CHECK(back.source().format == "cgns");
  require_same_mesh(mesh, back);
  const auto zones = autan::io::meshinfo_zones(mesh);
  for (std::size_t i = 0; i < zones.boundary.size(); ++i) {
    const auto& zone = zone_named(back, mesh.face_zones()[at(zones.boundary[i])].name);
    CHECK(zone.bc_type == solution.boundary_bc_types[i]);
  }
  for (const auto& field : solution.fields) {
    CHECK(read_field(file, field.name.c_str(), mesh.n_cells()) == field.values);
  }
}

TEST_CASE("Zone node names follow the results.md rule", "[cgns]") {
  const auto names =
      autan::io::zone_node_names({"wall", "shared", "Zone", std::string(33, 'a'), "a/b",
                                  "boundary-0005", ".", "ZoneBC", "z\xc3\xb4ne"},
                                 {"shared", "fluid", "ZoneBC-cells", "FlowSolution"});
  const std::vector<std::string> boundary{"wall",          "boundary-0001", "boundary-0002",
                                          "boundary-0003", "boundary-0004", "boundary-0005",
                                          "boundary-0006", "ZoneBC",        "boundary-0008"};
  CHECK(names.boundary == boundary);
  CHECK(names.cell ==
        std::vector<std::string>{"cell-zone-0000", "fluid", "ZoneBC-cells", "cell-zone-0003"});
}

TEST_CASE("CGNS files outside the supported layout are coded errors", "[cgns]") {
  using autan::ErrorCode;
  SECTION("two zones") {
    require_error(ErrorCode::io_unsupported, cgns_file("two-zones.cgns", [](int fn, int base) {
                    unstructured_tet(fn, base, "A");
                    unstructured_tet(fn, base, "B");
                  }));
  }
  SECTION("structured zone") {
    require_error(
        ErrorCode::io_unsupported, cgns_file("structured.cgns", [](int fn, int base) {
          cgsize_t size[9] = {2, 2, 2, 1, 1, 1, 0, 0, 0};
          int zone = 0;
          REQUIRE(cg_zone_write(fn, base, "Block", size, CGNS_ENUMV(Structured), &zone) == CG_OK);
        }));
  }
  SECTION("link to another file") {
    require_error(ErrorCode::io_external_reference, cgns_file("linked.cgns", [](int fn, int base) {
                    unstructured_tet(fn, base, "Zone");
                    REQUIRE(cg_goto(fn, base, "Zone_t", 1, "end") == CG_OK);
                    REQUIRE(cg_link_write("GridCoordinates2", "other.cgns",
                                          "/Base/Zone/GridCoordinates") == CG_OK);
                  }));
  }
  SECTION("a tetrahedron reads") {
    const autan::Mesh mesh = autan::io::read_mesh(
        cgns_file("tet.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); }));
    CHECK(mesh.n_cells() == 1);
    CHECK(zone_named(mesh, "unassigned").boundary);
  }
}

TEST_CASE("The writer rejects arrays that do not match the mesh", "[cgns]") {
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "hex" / "polyMesh");
  autan::io::Solution solution = solution_for(mesh);
  solution.fields[0].values.pop_back();
  try {
    autan::io::write_solution(temp_dir() / "bad.cgns", mesh, solution);
    FAIL("expected E0101");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_inconsistent_sizes);
  }
}

TEST_CASE("The writer rejects a periodic mesh until CGNS records its pairs", "[cgns]") {
  autan::CellwiseMesh cells = autan::test::box(2, autan::test::BoxFill::hexa);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  try {
    autan::io::write_solution(temp_dir() / "periodic.cgns", mesh, solution_for(mesh));
    FAIL("expected E0133");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::io_unsupported);
  }
  CHECK(!fs::exists(temp_dir() / "periodic.cgns"));
}

namespace {

// `mesh` with its zones renamed by `names` and an empty boundary zone added.
autan::Mesh renamed(const autan::Mesh& mesh, const std::map<std::string, std::string>& names) {
  autan::MeshInput in;
  in.x.assign(mesh.x().begin(), mesh.x().end());
  in.y.assign(mesh.y().begin(), mesh.y().end());
  in.z.assign(mesh.z().begin(), mesh.z().end());
  in.face_node_offsets.assign(mesh.face_node_offsets().begin(), mesh.face_node_offsets().end());
  in.face_nodes.assign(mesh.face_nodes().begin(), mesh.face_nodes().end());
  in.face_zone.assign(mesh.face_zone().begin(), mesh.face_zone().end());
  in.cell_zone.assign(mesh.cell_zone().begin(), mesh.cell_zone().end());
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    in.face_cell0.push_back(mesh.owner()[at(f)]);
    in.face_cell1.push_back(f < mesh.n_interior_faces() ? mesh.neighbour()[at(f)] : autan::no_cell);
  }
  in.face_zones.assign(mesh.face_zones().begin(), mesh.face_zones().end());
  in.cell_zones.assign(mesh.cell_zones().begin(), mesh.cell_zones().end());
  for (auto& z : in.face_zones) {
    if (names.contains(z.name)) z.name = names.at(z.name);
  }
  for (auto& z : in.cell_zones) {
    if (names.contains(z.name)) z.name = names.at(z.name);
  }
  in.face_zones.push_back(autan::FaceZone{
      .name = "spare", .bc_type = "", .kind = {}, .boundary = true, .provenance = {}});
  in.source = mesh.source();
  return autan::assemble(std::move(in));
}

}  // namespace

TEST_CASE("Zone names outside the node-name rule survive a write and read", "[cgns]") {
  const autan::Mesh mesh =
      renamed(autan::io::read_mesh(data_dir / "hybrid" / "polyMesh"), {{"xmin",
                                                                        "z\xc3\xb4ne d'entr\xc3\xa9"
                                                                        "e"},
                                                                       {"xmax", "a/b"},
                                                                       {"lower", "Zone"}});
  autan::io::Solution solution = solution_for(mesh);
  solution.boundary_bc_types.back() = "BCOutflowSubsonic";  // "spare" has no faces
  fs::create_directories(temp_dir());
  const fs::path file = temp_dir() / "names.cgns";
  autan::io::write_solution(file, mesh, solution);

  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
  int count = 0;
  REQUIRE(cg_ndescriptors(&count) == CG_OK);
  for (int d = 1; d <= count; ++d) {
    char name[33];
    char* text = nullptr;
    REQUIRE(cg_descriptor_read(d, name, &text) == CG_OK);
    const std::string content(text);
    cg_free(text);
    CHECK(
        std::ranges::all_of(content, [](char c) { return static_cast<unsigned char>(c) < 0x80; }));
  }
  cg_close(fn);

  const autan::Mesh back = autan::io::read_mesh(file);
  require_same_mesh(mesh, back);
  CHECK(zone_named(back, "a/b").bc_type == "BCFarfield");
  const auto spare = zone_named(back, "spare");
  CHECK(spare.boundary);
  CHECK(spare.kind == autan::BcKind::outlet);
  CHECK(std::ranges::any_of(back.cell_zones(),
                            [](const autan::CellZone& z) { return z.name == "Zone"; }));
}

TEST_CASE("CGNS content that contradicts itself is E0132", "[cgns]") {
  using autan::ErrorCode;
  SECTION("a zone declaring more cells than it holds") {
    require_error(ErrorCode::io_malformed, cgns_file("count.cgns", [](int fn, int base) {
                    unstructured_tet(fn, base, "Zone", 2);
                  }));
  }
  SECTION("a FamilySpecified BC naming no family") {
    require_error(
        ErrorCode::io_malformed, cgns_file("family.cgns", [](int fn, int base) {
          unstructured_tet(fn, base, "Zone");
          const std::array<cgsize_t, 3> tri{1, 3, 2};
          int section = 0, bc = 0;
          REQUIRE(cg_section_write(fn, base, 1, "Face", CGNS_ENUMV(TRI_3), 2, 2, 0, tri.data(),
                                   &section) == CG_OK);
          const cgsize_t face = 2;
          REQUIRE(cg_boco_write(fn, base, 1, "wall", CGNS_ENUMV(FamilySpecified),
                                CGNS_ENUMV(PointList), 1, &face, &bc) == CG_OK);
          REQUIRE(cg_boco_gridlocation_write(fn, base, 1, bc, CGNS_ENUMV(FaceCenter)) == CG_OK);
          REQUIRE(cg_goto(fn, base, "Zone_t", 1, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK);
          REQUIRE(cg_famname_write("Nowhere") == CG_OK);
        }));
  }
  SECTION("a cell-centred subregion selecting a face") {
    require_error(
        ErrorCode::io_malformed, cgns_file("subregion.cgns", [](int fn, int base) {
          unstructured_tet(fn, base, "Zone");
          const std::array<cgsize_t, 3> tri{1, 3, 2};
          int section = 0, region = 0;
          REQUIRE(cg_section_write(fn, base, 1, "Face", CGNS_ENUMV(TRI_3), 2, 2, 0, tri.data(),
                                   &section) == CG_OK);
          const cgsize_t face = 2;
          REQUIRE(cg_subreg_ptset_write(fn, base, 1, "cells", 3, CGNS_ENUMV(CellCenter),
                                        CGNS_ENUMV(PointList), 1, &face, &region) == CG_OK);
        }));
  }
}
