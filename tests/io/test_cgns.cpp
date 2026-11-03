#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cgns_io.h>
#include <cgnslib.h>
#include <hdf5.h>
#include <nlohmann/json.hpp>

#include "autan/core/geometry.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "fixtures.hpp"
#include "outside_data.hpp"
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
  autan::io::Solution solution = solution_for(mesh);
  solution.mean_fields.push_back(solution.fields[0]);
  autan::io::Field previous{.name = "DensityPreviousLevel", .values = {}};
  for (index_t c = 0; c < mesh.n_cells(); ++c) previous.values.push_back(-1.0 / (c + 1));
  solution.restart_fields.push_back(previous);
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

  const auto stored = autan::io::read_solution(file);
  CHECK(stored.cells == mesh.n_cells());
  CHECK(stored.run_metadata == solution.run_metadata);
  const auto same = [](const std::vector<autan::io::Field>& a,
                       const std::vector<autan::io::Field>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].name == b[i].name);
      CHECK(a[i].values == b[i].values);
    }
  };
  same(stored.fields, solution.fields);
  same(stored.mean_fields, solution.mean_fields);
  same(stored.restart_fields, solution.restart_fields);
  // A warm start reads FlowSolution alone.
  const auto fields_only = autan::io::read_solution(file, false);
  same(fields_only.fields, solution.fields);
  CHECK(fields_only.mean_fields.empty());
  CHECK(fields_only.restart_fields.empty());
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
  SECTION("structured zone, as a mesh and as a solution") {
    const auto file = cgns_file("structured.cgns", [](int fn, int base) {
      cgsize_t size[9] = {2, 2, 2, 1, 1, 1, 0, 0, 0};
      int zone = 0;
      REQUIRE(cg_zone_write(fn, base, "Block", size, CGNS_ENUMV(Structured), &zone) == CG_OK);
    });
    require_error(ErrorCode::io_unsupported, file);
    try {
      (void)autan::io::read_solution(file);
      FAIL("expected E0133");
    } catch (const autan::Error& e) {
      CHECK(e.code() == ErrorCode::io_unsupported);
    }
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

TEST_CASE("A CGNS file whose data lives in another file is refused before any read", "[cgns]") {
  // mesh.md, E0134: a CGNS link, an HDF5 external link, a dataset with external storage or a
  // virtual dataset, in a mesh or a solution. HDF5 first reads the donor's values through the
  // file (the control that the form is real); with the donor gone, a read of it would fail
  // with another code, so E0134 and its node come from the file's structure alone.
  const auto form = GENERATE(Outside::cgns_link, Outside::external_link, Outside::external_storage,
                             Outside::virtual_dataset);
  CAPTURE(outside_name(form));
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "hex" / "polyMesh");
  const fs::path dir = temp_dir() / ("outside-" + std::to_string(static_cast<int>(form)));
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path file = dir / "solution.cgns";
  autan::io::write_solution(file, mesh, solution_for(mesh));
  std::string control;
  const std::string node = move_outside(file, form, dir / "donor", 7.25, control);
  for (const double v : read_doubles(file, control)) CHECK(v == 7.25);
  fs::remove(dir / "donor");
  const auto refused = [&](auto read) {
    try {
      read();
      FAIL("expected E0134");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::io_external_reference);
      CHECK(nlohmann::json::parse(e.data_json()) == nlohmann::json{{"node", node}});
    }
  };
  refused([&] { (void)autan::io::read_mesh(file); });
  refused([&] { (void)autan::io::read_solution(file); });
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

TEST_CASE("Periodic pairs round-trip through solution.cgns", "[cgns][periodic]") {
  // periodic-box, written by Gmsh: xmin -> xmax by (1, 0, 0), ymin -> ymax by
  // (0, 1, 0). The pairs are written as reciprocal SIDS GridConnectivity_t
  // with Periodic_t, which cgnscheck accepts, and read back as they were.
  const autan::Mesh mesh =
      autan::io::read_mesh(data_dir / "periodic-box" / "periodic-box.ascii.msh");
  REQUIRE(mesh.periodic_pairs().size() == 2);
  autan::io::Solution solution;
  solution.run_metadata = "{}";
  for (const index_t z : autan::io::meshinfo_zones(mesh).boundary) {
    solution.boundary_bc_types.emplace_back(
        mesh.face_zones()[at(z)].kind == autan::BcKind::periodic ? "" : "BCSymmetryPlane");
  }
  autan::io::Field density{.name = "Density", .values = {}};
  density.values.assign(static_cast<std::size_t>(mesh.n_cells()), 1.0);
  solution.fields.push_back(density);
  const fs::path file = temp_dir() / "periodic-box-solution.cgns";
  fs::create_directories(temp_dir());
  autan::io::write_solution(file, mesh, solution);

  const std::string command = std::string(AUTAN_CGNSCHECK) + " " + file.string() + " > " +
                              (file.string() + ".check") + " 2>&1";
  CHECK(std::system(command.c_str()) == 0);
  std::ifstream report(file.string() + ".check");
  const std::string text((std::istreambuf_iterator<char>(report)),
                         std::istreambuf_iterator<char>());
  CHECK(text.find("ERROR") == std::string::npos);

  // Independently of the reader: every listed face, translated, is its
  // donor (face j is element j + 1, results.md), and the reciprocal mirrors it.
  {
    const autan::Geometry g = autan::compute_geometry(mesh);
    int fn = 0;
    REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
    int conns = 0;
    REQUIRE(cg_nconns(fn, 1, 1, &conns) == CG_OK);
    CHECK(conns == 4);
    for (int i = 1; i <= conns; ++i) {
      char name[33], donor_name[33];
      CGNS_ENUMT(GridLocation_t) location{};
      CGNS_ENUMT(GridConnectivityType_t) type{};
      CGNS_ENUMT(PointSetType_t) set{}, donor_set{};
      CGNS_ENUMT(ZoneType_t) donor_zone{};
      CGNS_ENUMT(DataType_t) donor_type{};
      cgsize_t count = 0, donor_count = 0;
      REQUIRE(cg_conn_info(fn, 1, 1, i, name, &location, &type, &set, &count, donor_name,
                           &donor_zone, &donor_set, &donor_type, &donor_count) == CG_OK);
      std::vector<cgsize_t> points(static_cast<std::size_t>(count)),
          donor(static_cast<std::size_t>(donor_count));
      REQUIRE(cg_conn_read(fn, 1, 1, i, points.data(), CGNS_ENUMV(LongInteger), donor.data()) ==
              CG_OK);
      std::array<float, 3> center{}, angle{}, t{};
      REQUIRE(cg_conn_periodic_read(fn, 1, 1, i, center.data(), angle.data(), t.data()) == CG_OK);
      for (std::size_t k = 0; k < points.size(); ++k) {
        const auto a = static_cast<std::size_t>(points[k] - 1),
                   d = static_cast<std::size_t>(donor[k] - 1);
        CHECK_THAT(g.face_centroid_x[a] + t[0],
                   Catch::Matchers::WithinAbs(g.face_centroid_x[d], 1e-7));
        CHECK_THAT(g.face_centroid_y[a] + t[1],
                   Catch::Matchers::WithinAbs(g.face_centroid_y[d], 1e-7));
        CHECK_THAT(g.face_centroid_z[a] + t[2],
                   Catch::Matchers::WithinAbs(g.face_centroid_z[d], 1e-7));
      }
    }
    cg_close(fn);
  }

  const autan::Mesh back = autan::io::read_mesh(file);
  require_same_mesh(mesh, back);
  REQUIRE(back.periodic_pairs().size() == mesh.periodic_pairs().size());
  for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
    const auto& a = mesh.periodic_pairs()[p];
    const auto& b = back.periodic_pairs()[p];
    CHECK(back.face_zones()[at(b.zone_a)].name == mesh.face_zones()[at(a.zone_a)].name);
    CHECK(back.face_zones()[at(b.zone_b)].name == mesh.face_zones()[at(a.zone_b)].name);
    CHECK(b.translation == a.translation);
    CHECK(back.face_zones()[at(b.zone_a)].kind == autan::BcKind::periodic);
  }
}

TEST_CASE("The writer refuses a BCType on a periodic zone", "[cgns][periodic]") {
  const autan::Mesh mesh =
      autan::io::read_mesh(data_dir / "periodic-box" / "periodic-box.ascii.msh");
  autan::io::Solution solution;
  solution.run_metadata = "{}";
  solution.boundary_bc_types.assign(autan::io::meshinfo_zones(mesh).boundary.size(),
                                    "BCSymmetryPlane");
  try {
    autan::io::write_solution(temp_dir() / "periodic-bctype.cgns", mesh, solution);
    FAIL("expected E0101");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_inconsistent_sizes);
  }
}

TEST_CASE("A periodic translation keeps its double precision through solution.cgns",
          "[cgns][periodic]") {
  // A box 0.1 wide in x, periodic by (0.1, 0, 0): 0.1 has no float value, so
  // the R4 translation of Periodic_t alone could not give it back.
  autan::CellwiseMesh cells = autan::test::box(2, autan::test::BoxFill::hexa);
  for (double& x : cells.x) x *= 0.1;
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {0.1, 0.0, 0.0}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  autan::io::Solution solution;
  solution.run_metadata = "{}";
  for (const index_t z : autan::io::meshinfo_zones(mesh).boundary) {
    solution.boundary_bc_types.emplace_back(
        mesh.face_zones()[at(z)].kind == autan::BcKind::periodic ? "" : "BCSymmetryPlane");
  }
  const fs::path file = temp_dir() / "thin-periodic.cgns";
  fs::create_directories(temp_dir());
  autan::io::write_solution(file, mesh, solution);
  const autan::Mesh back = autan::io::read_mesh(file);
  REQUIRE(back.periodic_pairs().size() == 1);
  CHECK(back.periodic_pairs()[0].translation == std::array<double, 3>{0.1, 0.0, 0.0});
}

namespace {

// Two hexahedra stacked in y, written with standard elements and no
// descriptor, as another tool would: cells 1-2, the x = 0 quads 3-4 and the
// x = 1 quads 5-6, joined by a periodic connection and its reciprocal.
// `reverse`: the reciprocal lists its pairs in the other order; `cross`:
// both connections pair face 3 with 6 and 4 with 5; `rotate`: a non-zero
// RotationAngle.
fs::path periodic_slab(const std::string& name, bool reverse, bool cross, bool rotate,
                       const std::string& descriptor = "", bool duplicate = false) {
  return cgns_file(name, [=](int fn, int base) {
    cgsize_t size[3] = {12, 2, 0};
    int zone = 0, coord = 0, section = 0;
    REQUIRE(cg_zone_write(fn, base, "Zone", size, CGNS_ENUMV(Unstructured), &zone) == CG_OK);
    std::array<double, 12> x{}, y{}, z{};
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 2; ++i) {
          const auto p = static_cast<std::size_t>(i + 2 * j + 6 * k);
          x[p] = i;
          y[p] = j;
          z[p] = k;
        }
      }
    }
    REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateX", x.data(),
                           &coord) == CG_OK);
    REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateY", y.data(),
                           &coord) == CG_OK);
    REQUIRE(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), "CoordinateZ", z.data(),
                           &coord) == CG_OK);
    const auto p = [](int i, int j, int k) { return static_cast<cgsize_t>(1 + i + 2 * j + 6 * k); };
    std::vector<cgsize_t> hexes;
    for (int j = 0; j < 2; ++j) {
      for (const cgsize_t n : {p(0, j, 0), p(1, j, 0), p(1, j + 1, 0), p(0, j + 1, 0), p(0, j, 1),
                               p(1, j, 1), p(1, j + 1, 1), p(0, j + 1, 1)})
        hexes.push_back(n);
    }
    REQUIRE(cg_section_write(fn, base, zone, "Cells", CGNS_ENUMV(HEXA_8), 1, 2, 0, hexes.data(),
                             &section) == CG_OK);
    std::vector<cgsize_t> quads;
    for (const int i : {0, 1}) {
      for (int j = 0; j < 2; ++j) {
        for (const cgsize_t n : {p(i, j, 0), p(i, j + 1, 0), p(i, j + 1, 1), p(i, j, 1)})
          quads.push_back(n);
      }
    }
    REQUIRE(cg_section_write(fn, base, zone, "Sides", CGNS_ENUMV(QUAD_4), 3, 6, 0, quads.data(),
                             &section) == CG_OK);
    std::vector<cgsize_t> left{3, 4};
    std::vector<cgsize_t> right = cross ? std::vector<cgsize_t>{6, 5} : std::vector<cgsize_t>{5, 6};
    if (duplicate) {
      left.push_back(3);
      right.push_back(5);
    }
    std::vector<cgsize_t> back_points = right, back_donor = left;
    if (reverse) {
      std::ranges::reverse(back_points);
      std::ranges::reverse(back_donor);
    }
    const std::array<float, 3> zero{}, plus{1.0F, 0.0F, 0.0F}, minus{-1.0F, 0.0F, 0.0F};
    const std::array<float, 3> angle{0.0F, 0.0F, rotate ? 0.5F : 0.0F};
    for (const auto& [name_, points, donor, t] :
         {std::tuple{"left", left, right, plus},
          std::tuple{"right", back_points, back_donor, minus}}) {
      int conn = 0;
      REQUIRE(cg_conn_write(fn, base, zone, name_, CGNS_ENUMV(FaceCenter), CGNS_ENUMV(Abutting1to1),
                            CGNS_ENUMV(PointList), static_cast<cgsize_t>(points.size()),
                            points.data(), "Zone", CGNS_ENUMV(Unstructured),
                            CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger),
                            static_cast<cgsize_t>(donor.size()), donor.data(), &conn) == CG_OK);
      REQUIRE(cg_conn_periodic_write(fn, base, zone, conn, zero.data(), angle.data(), t.data()) ==
              CG_OK);
      if (!descriptor.empty()) {
        REQUIRE(cg_goto(fn, base, "Zone_t", zone, "ZoneGridConnectivity_t", 1, "GridConnectivity_t",
                        conn, "end") == CG_OK);
        REQUIRE(cg_descriptor_write("PeriodicTranslation", descriptor.c_str()) == CG_OK);
      }
    }
  });
}

}  // namespace

TEST_CASE("Periodic connections from another writer read as pairs", "[cgns][periodic]") {
  // Standard elements, no PeriodicTranslation descriptor, the reciprocal in
  // either order: one pair, left -> right by (1, 0, 0).
  const bool reverse = GENERATE(false, true);
  const autan::Mesh mesh = autan::io::read_mesh(periodic_slab("slab.cgns", reverse, false, false));
  REQUIRE(mesh.periodic_pairs().size() == 1);
  const auto& pair = mesh.periodic_pairs()[0];
  CHECK(mesh.face_zones()[at(pair.zone_a)].name == "left");
  CHECK(mesh.face_zones()[at(pair.zone_b)].name == "right");
  CHECK(pair.translation == std::array<double, 3>{1.0, 0.0, 0.0});
}

TEST_CASE("A connectivity CGNS cannot express as a periodic pair is a coded error",
          "[cgns][periodic]") {
  using autan::ErrorCode;
  // Faces paired crosswise: each connection is the other's reciprocal, but
  // face 3 does not translate onto face 6.
  require_error(ErrorCode::io_malformed, periodic_slab("crossed.cgns", false, true, false));
  require_error(ErrorCode::io_unsupported, periodic_slab("rotated.cgns", false, false, true));
  // A descriptor that is not three numbers, and a face listed twice.
  require_error(ErrorCode::io_malformed,
                periodic_slab("four.cgns", false, false, false, "[1.0, 0.0, 0.0, 123]"));
  require_error(ErrorCode::io_malformed,
                periodic_slab("twice.cgns", false, false, false, "", true));
  // A one-sided periodic connection.
  require_error(
      ErrorCode::io_malformed, cgns_file("one-sided.cgns", [](int fn, int base) {
        unstructured_tet(fn, base, "Zone");
        const std::array<cgsize_t, 12> faces{1, 3, 2, 1, 2, 4, 2, 3, 4, 1, 4, 3};
        int section = 0;
        REQUIRE(cg_section_write(fn, base, 1, "Faces", CGNS_ENUMV(TRI_3), 2, 5, 0, faces.data(),
                                 &section) == CG_OK);
        const cgsize_t face = 2, donor = 3;
        int conn = 0;
        REQUIRE(cg_conn_write(fn, base, 1, "side", CGNS_ENUMV(FaceCenter), CGNS_ENUMV(Abutting1to1),
                              CGNS_ENUMV(PointList), 1, &face, "Zone", CGNS_ENUMV(Unstructured),
                              CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger), 1, &donor,
                              &conn) == CG_OK);
        const std::array<float, 3> t{1.0F, 0.0F, 0.0F}, zero{};
        REQUIRE(cg_conn_periodic_write(fn, base, 1, conn, zero.data(), zero.data(), t.data()) ==
                CG_OK);
      }));
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

TEST_CASE("Point and edge sections are skipped, and the ids after them still resolve", "[cgns]") {
  // mesh.md, Readers (CGNS): BAR_2 and NODE elements carry nothing the mesh uses. A tet whose
  // face (1, 2, 3) a BC "wall" names by its element id, without and with three elements before
  // that face, edges in a BAR_2 section or a point and two edges in a MIXED one: the same mesh,
  // the face still found by its id.
  enum class Edges : std::uint8_t { none, section, mixed };
  const auto write = [](Edges edges) {
    const char* name = edges == Edges::none      ? "no-edges.cgns"
                       : edges == Edges::section ? "edges.cgns"
                                                 : "mixed-edges.cgns";
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      int section = 0, bc = 0;
      cgsize_t next = 2;
      if (edges == Edges::section) {
        const std::array<cgsize_t, 6> bars{1, 2, 2, 3, 3, 1};
        REQUIRE(cg_section_write(fn, base, 1, "Edges", CGNS_ENUMV(BAR_2), next, next + 2, 0,
                                 bars.data(), &section) == CG_OK);
        next += 3;
      } else if (edges == Edges::mixed) {
        const cgsize_t bar = CGNS_ENUMV(BAR_2), node = CGNS_ENUMV(NODE);
        const std::array<cgsize_t, 8> bars{node, 1, bar, 2, 3, bar, 3, 1};
        const std::array<cgsize_t, 4> offsets{0, 2, 5, 8};
        REQUIRE(cg_poly_section_write(fn, base, 1, "Edges", CGNS_ENUMV(MIXED), next, next + 2, 0,
                                      bars.data(), offsets.data(), &section) == CG_OK);
        next += 3;
      }
      const std::array<cgsize_t, 3> tri{1, 3, 2};
      REQUIRE(cg_section_write(fn, base, 1, "Wall", CGNS_ENUMV(TRI_3), next, next, 0, tri.data(),
                               &section) == CG_OK);
      REQUIRE(cg_boco_write(fn, base, 1, "wall", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointList), 1,
                            &next, &bc) == CG_OK);
      REQUIRE(cg_boco_gridlocation_write(fn, base, 1, bc, CGNS_ENUMV(FaceCenter)) == CG_OK);
    });
  };
  const autan::Mesh plain = autan::io::read_mesh(write(Edges::none));
  const autan::Mesh with_edges = autan::io::read_mesh(write(Edges::section));
  const autan::Mesh mixed = autan::io::read_mesh(write(Edges::mixed));
  for (const autan::Mesh* m : {&plain, &with_edges, &mixed}) {
    CHECK(m->n_cells() == 1);
    CHECK(m->n_faces() == 4);
    const auto zones = m->face_zones();
    const auto wall = std::ranges::find(zones, std::string("wall"), &autan::FaceZone::name);
    REQUIRE(wall != zones.end());
    const auto z = static_cast<std::size_t>(wall - zones.begin());
    CHECK(m->boundary_zone_offsets()[z + 1] - m->boundary_zone_offsets()[z] == 1);
  }
  CHECK(std::ranges::equal(plain.face_nodes(), with_edges.face_nodes()));
  CHECK(std::ranges::equal(plain.face_nodes(), mixed.face_nodes()));
}

namespace {

// `file`'s 64-bit integer dataset at `path` replaced by `values` of extent `dims`.
void replace_dataset(const fs::path& file, const char* path, const std::vector<hsize_t>& dims,
                     const std::vector<std::int64_t>& values) {
  const hid_t h5 = H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(h5 >= 0);
  REQUIRE(H5Ldelete(h5, path, H5P_DEFAULT) >= 0);
  const hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
  const hid_t dataset =
      H5Dcreate2(h5, path, H5T_STD_I64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(h5);
}

}  // namespace

TEST_CASE("CGNS content that contradicts itself is E0132", "[cgns]") {
  using autan::ErrorCode;
  fs::create_directories(temp_dir());
  SECTION("a MIXED section whose offsets run past its connectivity") {
    // Past the first chunk of offsets the reader reads, before it sizes anything from them.
    constexpr std::int64_t edges = 4097;
    const fs::path file = cgns_file("offsets.cgns", [](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      std::vector<cgsize_t> connectivity, offsets{0};
      for (std::int64_t e = 0; e < edges; ++e) {
        connectivity.insert(connectivity.end(), {CGNS_ENUMV(BAR_2), 1, 2});
        offsets.push_back(offsets.back() + 3);
      }
      int section = 0;
      REQUIRE(cg_poly_section_write(fn, base, 1, "Edges", CGNS_ENUMV(MIXED), 2, edges + 1, 0,
                                    connectivity.data(), offsets.data(), &section) == CG_OK);
    });
    std::vector<std::int64_t> offsets;
    for (std::int64_t e = 0; e <= edges; ++e) offsets.push_back(3 * e);
    offsets[4096] = std::numeric_limits<std::int64_t>::max();
    replace_dataset(file, "/Base/Zone/Edges/ElementStartOffset/ data",
                    {static_cast<hsize_t>(edges + 1)}, offsets);
    require_error(ErrorCode::io_malformed, file);
  }
  SECTION("a connectivity that is not one-dimensional, or not of its element type's size") {
    const auto [dims, values] =
        GENERATE(std::pair{std::vector<hsize_t>{1, 4}, std::vector<std::int64_t>{1, 2, 3, 4}},
                 std::pair{std::vector<hsize_t>{3}, std::vector<std::int64_t>{1, 2, 3}});
    const fs::path file = cgns_file("connectivity.cgns",
                                    [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
    replace_dataset(file, "/Base/Zone/Cells/ElementConnectivity/ data", dims, values);
    require_error(ErrorCode::io_malformed, file);
  }
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

namespace {

// `json` padded with spaces after its first character to `bytes` bytes: the same document.
std::string padded(const std::string& json, std::size_t bytes) {
  REQUIRE(bytes >= json.size());
  return json.substr(0, 1) + std::string(bytes - json.size(), ' ') + json.substr(1);
}

// Overwrites a base descriptor of `file`.
void write_base_descriptor(const fs::path& file, const char* name, const std::string& text) {
  int fn = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
  REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
  REQUIRE(cg_descriptor_write(name, text.c_str()) == CG_OK);
  REQUIRE(cg_close(fn) == CG_OK);
}

int families_of(const fs::path& file) {
  int fn = 0, families = 0;
  REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
  REQUIRE(cg_nfamilies(fn, 1, &families) == CG_OK);
  REQUIRE(cg_close(fn) == CG_OK);
  return families;
}

}  // namespace

TEST_CASE("The CGNS reader caps the texts it keeps and counts every descriptor", "[cgns]") {
  // mesh.md, E0122: a zone name of at most 4096 bytes; PeriodicTranslation of at most 4096;
  // ZoneNames of at most (families + 1) x 25 600. Any other descriptor is accepted whatever its
  // size, and counted in source.descriptor_bytes, since the CGNS library loads it.
  fs::create_directories(temp_dir());
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "hybrid" / "polyMesh");
  SECTION("a zone name of 4096 bytes, not 4097") {
    for (const std::size_t bytes : {std::size_t{4096}, std::size_t{4097}}) {
      CAPTURE(bytes);
      const std::string name(bytes, 'z');  // longer than a CGNS name: it goes to ZoneNames
      const autan::Mesh named = renamed(mesh, {{"xmin", name}});
      const fs::path file = temp_dir() / ("zone-name-" + std::to_string(bytes) + ".cgns");
      autan::io::write_solution(file, named, solution_for(named));
      if (bytes == 4096) {
        CHECK(zone_named(autan::io::read_mesh(file), name).boundary);
      } else {
        require_error(autan::ErrorCode::mesh_text_too_long, file);
      }
    }
  }
  SECTION("ZoneNames up to (families + 1) x 25 600 bytes") {
    const autan::Mesh named = renamed(mesh, {{"xmin",
                                              "entr\xc3\xa9"
                                              "e"}});
    const fs::path file = temp_dir() / "zone-names-cap.cgns";
    autan::io::write_solution(file, named, solution_for(named));
    const auto cap = static_cast<std::size_t>(families_of(file) + 1) * 25600;
    const std::string names = R"({"boundary": {}, "cell": {}})";
    write_base_descriptor(file, "ZoneNames", padded(names, cap));
    CHECK(autan::io::read_mesh(file).n_cells() == mesh.n_cells());
    write_base_descriptor(file, "ZoneNames", padded(names, cap + 1));
    require_error(autan::ErrorCode::mesh_text_too_long, file);
  }
  SECTION("a few hundred zones whose ZoneNames exceeds 4 KiB") {
    // Each boundary face of a 6^3 box in a zone of its own, its name outside the CGNS rule.
    autan::CellwiseMesh cells = autan::test::box(6, autan::test::BoxFill::hexa);
    const std::size_t faces = cells.labelled_face_zone.size();
    cells.face_zones.resize(faces, cells.face_zones[0]);
    for (std::size_t f = 0; f < faces; ++f) {
      cells.face_zones[f].name = "c\xc3\xb4t\xc3\xa9 " + std::to_string(f);
      cells.labelled_face_zone[f] = static_cast<index_t>(f);
    }
    const autan::Mesh many = autan::assemble(autan::to_face_based(std::move(cells)));
    const fs::path file = temp_dir() / "many-zones.cgns";
    autan::io::write_solution(file, many, solution_for(many));
    int fn = 0;
    REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
    REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
    int count = 0;
    REQUIRE(cg_ndescriptors(&count) == CG_OK);
    std::size_t zone_names = 0;
    for (int d = 1; d <= count; ++d) {
      char name[33];
      char* text = nullptr;
      REQUIRE(cg_descriptor_read(d, name, &text) == CG_OK);
      if (std::string_view(name) == "ZoneNames") zone_names = std::string_view(text).size();
      cg_free(text);
    }
    cg_close(fn);
    CHECK(zone_names > 4096);
    const autan::Mesh back = autan::io::read_mesh(file);
    CHECK(faces >= 200);
    CHECK(zone_named(back, "c\xc3\xb4t\xc3\xa9 199").boundary);
  }
  SECTION("PeriodicTranslation of 4096 bytes, not 4097") {
    for (const std::size_t bytes : {std::size_t{4096}, std::size_t{4097}}) {
      CAPTURE(bytes);
      // The two reciprocal connections translate by (1, 0, 0) and (-1, 0, 0).
      const fs::path file =
          periodic_slab("translation-" + std::to_string(bytes) + ".cgns", false, false, false);
      int fn = 0;
      REQUIRE(cg_open(file.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
      for (const auto& [conn, json] : {std::pair{1, "[1, 0, 0]"}, std::pair{2, "[-1, 0, 0]"}}) {
        REQUIRE(cg_goto(fn, 1, "Zone_t", 1, "ZoneGridConnectivity_t", 1, "GridConnectivity_t", conn,
                        "end") == CG_OK);
        REQUIRE(cg_descriptor_write("PeriodicTranslation", padded(json, bytes).c_str()) == CG_OK);
      }
      REQUIRE(cg_close(fn) == CG_OK);
      if (bytes == 4096) {
        CHECK(autan::io::read_mesh(file).periodic_pairs().size() == 1);
      } else {
        require_error(autan::ErrorCode::mesh_text_too_long, file);
      }
    }
  }
  SECTION("another descriptor is accepted and counted, whatever its size") {
    const auto write = [](bool about) {
      return cgns_file(about ? "about.cgns" : "no-about.cgns", [&](int fn, int base) {
        unstructured_tet(fn, base, "Zone");
        if (about) {
          REQUIRE(cg_goto(fn, base, "end") == CG_OK);
          REQUIRE(cg_descriptor_write("About", std::string(std::size_t{1} << 20, 'a').c_str()) ==
                  CG_OK);
        }
      });
    };
    const fs::path plain = write(false), about = write(true);
    CHECK(autan::io::read_mesh(about).n_cells() == 1);
    const auto with = autan::io::cgns_logical_bytes(about);
    const auto without = autan::io::cgns_logical_bytes(plain);
    CHECK(with.descriptors - without.descriptors == std::int64_t{1} << 20);
    CHECK(with.total - without.total == std::int64_t{1} << 20);
    CHECK(with.zone_names == 0);
  }
  SECTION("ten thousand families and a small ZoneNames") {
    const fs::path file = cgns_file("families.cgns", [](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      for (int f = 0; f < 10000; ++f) {
        int family = 0;
        REQUIRE(cg_family_write(fn, base, ("F" + std::to_string(f)).c_str(), &family) == CG_OK);
      }
      REQUIRE(cg_goto(fn, base, "end") == CG_OK);
      REQUIRE(cg_descriptor_write("ZoneNames", R"({"boundary": {}, "cell": {}})") == CG_OK);
    });
    CHECK(autan::io::read_mesh(file).n_cells() == 1);
  }
}

TEST_CASE("A start solution holds only what the writer writes for its mesh", "[cgns]") {
  // results.md, Sizes a reader accepts: a solution written for a mesh passes; with one of its
  // arrays compressed, which the writer never does, it is refused before the library opens it.
  fs::create_directories(temp_dir());
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "hybrid" / "polyMesh");
  const fs::path file = temp_dir() / "start.cgns";
  autan::io::write_solution(file, mesh, solution_for(mesh));
  const auto zones = autan::io::meshinfo_zones(mesh);
  const autan::io::StartLimits limits{
      .zones = static_cast<std::int64_t>(zones.boundary.size() + zones.cell.size()),
      .periodic_pairs = static_cast<std::int64_t>(mesh.periodic_pairs().size())};
  autan::io::check_start_solution(file, limits);
  {
    // More nodes than a mesh file may hold, as user data the writer never writes.
    const fs::path crowded = temp_dir() / "start-crowded.cgns";
    fs::copy_file(file, crowded, fs::copy_options::overwrite_existing);
    const std::int64_t n = autan::io::cgns_logical_bytes(crowded).nodes;
    int fn = 0;
    REQUIRE(cg_open(crowded.string().c_str(), CG_MODE_MODIFY, &fn) == CG_OK);
    for (std::int64_t i = n; i <= 16384; ++i) {
      REQUIRE(cg_goto(fn, 1, "end") == CG_OK);
      REQUIRE(cg_user_data_write(("u" + std::to_string(i)).c_str()) == CG_OK);
    }
    REQUIRE(cg_close(fn) == CG_OK);
    try {
      autan::io::check_start_solution(crowded, limits);
      FAIL("expected E0132");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::io_malformed);
    }
  }
  const hid_t h5 = H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(h5 >= 0);
  const char* path = "/Base/Zone/GridCoordinates/CoordinateX/ data";
  const hid_t old = H5Dopen2(h5, path, H5P_DEFAULT);
  REQUIRE(old >= 0);
  const hid_t space = H5Dget_space(old);
  std::vector<double> values(static_cast<std::size_t>(H5Sget_simple_extent_npoints(space)));
  REQUIRE(H5Dread(old, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(old);
  REQUIRE(H5Ldelete(h5, path, H5P_DEFAULT) >= 0);
  const hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
  const hsize_t chunk = values.size();
  H5Pset_chunk(properties, 1, &chunk);
  H5Pset_deflate(properties, 1);
  const hid_t dataset =
      H5Dcreate2(h5, path, H5T_IEEE_F64LE, space, H5P_DEFAULT, properties, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
  H5Dclose(dataset);
  H5Pclose(properties);
  H5Sclose(space);
  H5Fclose(h5);
  try {
    autan::io::check_start_solution(file, limits);
    FAIL("expected E0132");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::io_malformed);
  }
}

TEST_CASE("A CGNS file's logical bytes do not depend on its compression", "[cgns]") {
  // mesh-info.md, source.logical_bytes: each dataset at its element count times its element
  // size. The coordinates of a 12^3 box rewritten as deflated chunks: a smaller file, the same
  // logical bytes, the same mesh.
  fs::create_directories(temp_dir());
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(12, autan::test::BoxFill::hexa)));
  const fs::path plain = temp_dir() / "plain.cgns", packed = temp_dir() / "packed.cgns";
  autan::io::write_solution(plain, mesh, solution_for(mesh));
  fs::copy_file(plain, packed, fs::copy_options::overwrite_existing);
  const hid_t file = H5Fopen(packed.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  std::int64_t largest = 0;  // logical + stored of the largest deflated coordinate array
  for (const char* axis : {"CoordinateX", "CoordinateY", "CoordinateZ"}) {
    const std::string path = std::string("/Base/Zone/GridCoordinates/") + axis + "/ data";
    const hid_t old = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    REQUIRE(old >= 0);
    const hid_t space = H5Dget_space(old);
    const auto n = static_cast<std::size_t>(H5Sget_simple_extent_npoints(space));
    std::vector<double> values(n);
    REQUIRE(H5Dread(old, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0);
    H5Dclose(old);
    REQUIRE(H5Ldelete(file, path.c_str(), H5P_DEFAULT) >= 0);
    const hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t chunk = n;
    H5Pset_chunk(properties, 1, &chunk);
    H5Pset_deflate(properties, 9);
    const hid_t dataset =
        H5Dcreate2(file, path.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT, properties, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >=
            0);
    largest =
        std::max(largest, static_cast<std::int64_t>(2 * 8 * n + H5Dget_storage_size(dataset)));
    H5Dclose(dataset);
    H5Pclose(properties);
    H5Sclose(space);
  }
  H5Fclose(file);
  CHECK(autan::io::cgns_logical_bytes(packed).total == autan::io::cgns_logical_bytes(plain).total);
  CHECK(autan::io::cgns_logical_bytes(plain).filtered == 0);
  CHECK(autan::io::cgns_logical_bytes(packed).filtered == largest);
  CHECK(autan::io::mesh_logical_bytes(packed).total == autan::io::cgns_logical_bytes(plain).total);
  CHECK(autan::io::read_mesh(packed).n_points() == mesh.n_points());
  // The coordinates alone are 3 x 13^3 doubles.
  CHECK(autan::io::cgns_logical_bytes(plain).total > 3 * 13 * 13 * 13 * 8);
}

#if defined(__linux__)
namespace {

// The peak resident memory cg_open of `file` adds, until cg_close.
std::int64_t open_peak(const fs::path& file) {
  return peak_of([&] {
    int fn = 0;
    REQUIRE(cg_open(file.string().c_str(), CG_MODE_READ, &fn) == CG_OK);
    REQUIRE(cg_close(fn) == CG_OK);
  });
}

enum class Random { none, all, half };

// `file`'s dataset at `path` rewritten as one deflated chunk, its bytes, all or the first half,
// replaced by random ones, which deflate cannot shrink; with `chunk` values beyond its extent, in
// a chunk that large, the dataset's extent unlimited. Returns its stored size.
std::int64_t deflate(const fs::path& file, const char* path, Random random, hsize_t chunk = 0) {
  const hid_t h5 = H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(h5 >= 0);
  const hid_t old = H5Dopen2(h5, path, H5P_DEFAULT);
  REQUIRE(old >= 0);
  const hid_t type = H5Dget_type(old);
  const hid_t space = H5Dget_space(old);
  const auto n = static_cast<std::size_t>(H5Sget_simple_extent_npoints(space));
  std::vector<unsigned char> bytes(n * H5Tget_size(type));
  REQUIRE(H5Dread(old, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data()) >= 0);
  H5Dclose(old);
  if (random != Random::none) {
    std::mt19937_64 engine(411);
    const std::size_t end = random == Random::all ? bytes.size() : bytes.size() / 2;
    for (std::size_t i = 0; i < end; ++i) bytes[i] = static_cast<unsigned char>(engine() & 0xffU);
  }
  REQUIRE(H5Ldelete(h5, path, H5P_DEFAULT) >= 0);
  const hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
  if (chunk == 0) chunk = n;
  H5Pset_chunk(properties, 1, &chunk);
  H5Pset_deflate(properties, 9);
  const hsize_t extent = n, unlimited = H5S_UNLIMITED;
  const hid_t layout = H5Screate_simple(1, &extent, chunk > n ? &unlimited : nullptr);
  const hid_t dataset = H5Dcreate2(h5, path, type, layout, H5P_DEFAULT, properties, H5P_DEFAULT);
  REQUIRE(dataset >= 0);
  REQUIRE(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes.data()) >= 0);
  const auto stored = static_cast<std::int64_t>(H5Dget_storage_size(dataset));
  H5Dclose(dataset);
  H5Pclose(properties);
  H5Sclose(layout);
  H5Sclose(space);
  H5Tclose(type);
  H5Fclose(h5);
  return stored;
}

// A one-cell file with a MIXED section of `edges` BAR_2 [3, 3]: its connectivity a chunked
// dataset never written, which reads as its fill value 3, its offsets deflated. The file holds
// little more than the offsets, however many edges.
fs::path stored_edges(const std::string& name, hsize_t edges) {
  const fs::path file = cgns_file(name, [](int fn, int base) {
    unstructured_tet(fn, base, "Zone");
    const cgsize_t connectivity[] = {CGNS_ENUMV(BAR_2), 3, 3}, offsets[] = {0, 3};
    int section = 0;
    REQUIRE(cg_poly_section_write(fn, base, 1, "Edges", CGNS_ENUMV(MIXED), 2, 2, 0, connectivity,
                                  offsets, &section) == CG_OK);
  });
  const hid_t h5 = H5Fopen(file.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(h5 >= 0);
  const auto recreate = [&](const char* path, hsize_t n, bool filled) {
    const hid_t old = H5Dopen2(h5, path, H5P_DEFAULT);
    REQUIRE(old >= 0);
    const hid_t type = H5Dget_type(old);
    H5Dclose(old);
    REQUIRE(H5Ldelete(h5, path, H5P_DEFAULT) >= 0);
    const hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t chunk = std::min<hsize_t>(n, hsize_t{1} << 17);
    H5Pset_chunk(properties, 1, &chunk);
    if (filled) {
      const std::int64_t fill = 3;
      H5Pset_fill_value(properties, H5T_NATIVE_INT64, &fill);
    } else {
      H5Pset_shuffle(properties);
      H5Pset_deflate(properties, 1);
    }
    const hid_t space = H5Screate_simple(1, &n, nullptr);
    const hid_t dataset = H5Dcreate2(h5, path, type, space, H5P_DEFAULT, properties, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    if (!filled) {
      std::vector<std::int64_t> values;
      for (hsize_t first = 0; first < n; first += hsize_t{1} << 20) {
        const hsize_t count = std::min<hsize_t>(n - first, hsize_t{1} << 20);
        values.resize(count);
        for (hsize_t i = 0; i < count; ++i) values[i] = static_cast<std::int64_t>(3 * (first + i));
        const hid_t memory = H5Screate_simple(1, &count, nullptr);
        REQUIRE(H5Sselect_hyperslab(space, H5S_SELECT_SET, &first, nullptr, &count, nullptr) >= 0);
        REQUIRE(H5Dwrite(dataset, H5T_NATIVE_INT64, memory, space, H5P_DEFAULT, values.data()) >=
                0);
        H5Sclose(memory);
      }
    }
    H5Dclose(dataset);
    H5Sclose(space);
    H5Pclose(properties);
    H5Tclose(type);
  };
  recreate("/Base/Zone/Edges/ElementConnectivity/ data", 3 * edges, true);
  recreate("/Base/Zone/Edges/ElementStartOffset/ data", edges + 1, false);
  const hid_t range = H5Dopen2(h5, "/Base/Zone/Edges/ElementRange/ data", H5P_DEFAULT);
  REQUIRE(range >= 0);
  const std::int64_t bounds[] = {2, static_cast<std::int64_t>(edges) + 1};
  REQUIRE(H5Dwrite(range, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, bounds) >= 0);
  H5Dclose(range);
  H5Fclose(h5);
  return file;
}

}  // namespace

TEST_CASE(
    "Reading a CGNS file costs its descriptors, one filtered read and the HDF5 caches at most",
    "[cgns]") {
  // mesh.md, "Memory of a CGNS read": descriptor_bytes + filtered_bytes + the HDF5 caches.
  fs::create_directories(temp_dir());
  constexpr std::int64_t caches = (std::int64_t{4} << 20) + (std::int64_t{8} << 20) + (64 << 10);
  const auto tet = [](const std::string& name, std::size_t about) {
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      if (about > 0) {
        REQUIRE(cg_goto(fn, base, "end") == CG_OK);
        REQUIRE(cg_descriptor_write("About", std::string(about, 'a').c_str()) == CG_OK);
      }
    });
  };
  SECTION("a descriptor of 8 or 128 MiB, in one deflated chunk") {
    const std::int64_t baseline = open_peak(tet("about-none.cgns", 0));
    // Deflate's output starts at the stored size and doubles (H5Zdeflate.c:85, :133): 'a' grows
    // it many times from almost nothing, random bytes not at all, half random once from L/2.
    const auto random = GENERATE(Random::none, Random::all, Random::half);
    const std::size_t size = GENERATE(std::size_t{8} << 20, std::size_t{128} << 20);
    CAPTURE(random, size);
    const auto bytes = static_cast<std::int64_t>(size);
    const fs::path file = tet(
        "about-" + std::to_string(static_cast<int>(random)) + "-" + std::to_string(size) + ".cgns",
        size);
    const std::int64_t stored = deflate(file, "/Base/About/ data", random);
    const auto logical = autan::io::cgns_logical_bytes(file);
    CHECK(logical.descriptors == bytes);
    CHECK(logical.filtered == 2 * bytes + stored);
    const std::int64_t added = open_peak(file) - baseline;
    CAPTURE(added, stored);
    CHECK(added >= bytes);  // loaded, as the bound assumes
    CHECK(added <= logical.descriptors + logical.filtered + caches);
    // Without the filtered read, the bound would not hold for an incompressible descriptor.
    if (random == Random::all && size == (std::size_t{128} << 20)) {
      CHECK(added > logical.descriptors + caches);
    }
  }
  SECTION("a short descriptor in one large deflated chunk") {
    // HDF5 inflates the whole chunk, 128 MiB, for the descriptor's 15 bytes.
    const std::int64_t baseline = open_peak(tet("about-none.cgns", 0));
    constexpr hsize_t chunk = hsize_t{1} << 27;
    const fs::path file = tet("about-chunk.cgns", 15);
    const std::int64_t stored = deflate(file, "/Base/About/ data", Random::none, chunk);
    const auto logical = autan::io::cgns_logical_bytes(file);
    CHECK(logical.descriptors == 15);
    CHECK(logical.filtered == 2 * static_cast<std::int64_t>(chunk) + stored);
    const std::int64_t added = open_peak(file) - baseline;
    CAPTURE(added, stored);
    CHECK(added <= logical.descriptors + logical.filtered + caches);
    // The dataset's extent alone would not bound it.
    CHECK(added > logical.descriptors + 2 * 15 + stored + caches);
  }
  SECTION("a deflated coordinate array, read with the mesh") {
    const autan::Mesh mesh = autan::assemble(
        autan::to_face_based(autan::test::box(80, autan::test::BoxFill::hexa, 0.2)));
    const fs::path plain = temp_dir() / "coordinates-plain.cgns",
                   packed = temp_dir() / "coordinates-packed.cgns";
    autan::io::write_solution(plain, mesh, solution_for(mesh));
    fs::copy_file(plain, packed, fs::copy_options::overwrite_existing);
    // Perturbed coordinates, which deflate barely shrinks.
    const std::int64_t stored =
        deflate(packed, "/Base/Zone/GridCoordinates/CoordinateX/ data", Random::none);
    const auto logical = autan::io::cgns_logical_bytes(packed);
    CHECK(logical.filtered == static_cast<std::int64_t>(2 * 8 * mesh.n_points()) + stored);
    const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
    const std::int64_t after = peak_of([&] { (void)autan::io::read_mesh(packed); });
    CAPTURE(before, after, stored);
    CHECK(after - before <= logical.filtered + caches);
  }
}
TEST_CASE("Reading a MIXED section costs the elements it keeps, not its points and edges",
          "[cgns]") {
  // mesh.md, Readers: a section is read in chunks of at most 4096 offsets and 2^17 entries, so a
  // million edges cost those buffers and HDF5's caches at most, not their 32 MB.
  fs::create_directories(temp_dir());
  constexpr cgsize_t edges = 1'000'000;
  const fs::path plain =
      cgns_file("mixed-none.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
  const fs::path padded = cgns_file("mixed-edges.cgns", [](int fn, int base) {
    unstructured_tet(fn, base, "Zone");
    std::vector<cgsize_t> connectivity, offsets{0};
    for (cgsize_t e = 0; e < edges; ++e) {
      connectivity.insert(connectivity.end(), {CGNS_ENUMV(BAR_2), 1, 2});
      offsets.push_back(offsets.back() + 3);
    }
    int section = 0;
    REQUIRE(cg_poly_section_write(fn, base, 1, "Edges", CGNS_ENUMV(MIXED), 2, edges + 1, 0,
                                  connectivity.data(), offsets.data(), &section) == CG_OK);
  });
  constexpr std::int64_t buffers = (4096 + 1) * 8 + (std::int64_t{1} << 17) * 8;
  constexpr std::int64_t caches = (std::int64_t{4} << 20) + (std::int64_t{8} << 20) + (64 << 10);
  (void)autan::io::read_mesh(plain);  // the libraries' first-use setup, outside both peaks
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(padded).n_cells() == 1); });
  CAPTURE(before, after);
  CHECK(after - before <= buffers + caches);
  // Nor anything per edge beyond them: 2^26 edges cost what 2^25 do, where a bit each would be
  // 4 MiB more.
  const fs::path half = stored_edges("edges-half.cgns", hsize_t{1} << 25);
  const fs::path full = stored_edges("edges-full.cgns", hsize_t{1} << 26);
  (void)autan::io::read_mesh(half);
  const std::int64_t at_half = peak_of([&] { CHECK(autan::io::read_mesh(half).n_cells() == 1); });
  const std::int64_t at_full = peak_of([&] { CHECK(autan::io::read_mesh(full).n_cells() == 1); });
  CAPTURE(at_half, at_full);
  CHECK(at_full - at_half <= std::int64_t{1} << 20);
}
TEST_CASE("Reading a CGNS file costs at most 1 MiB and 16 KiB a node, up to 16 384 nodes",
          "[cgns]") {
  // mesh.md, Memory of a CGNS read: cg_open allocates for every node it parses; families,
  // sections, BCs, user data and descriptors mixed up to the limit stay within the bound, and one
  // node more is refused before cg_open, at the cost of the walk alone.
  fs::create_directories(temp_dir());
  constexpr std::int64_t limit = 16384;
  const fs::path plain =
      cgns_file("nodes-none.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
  const std::int64_t n0 = autan::io::cgns_logical_bytes(plain).nodes;
  const auto build = [&](const std::string& name, std::int64_t nodes) {
    const fs::path file = cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      const std::array<cgsize_t, 2> bar{1, 2};
      int section = 0, bc = 0, family = 0;
      REQUIRE(cg_section_write(fn, base, 1, "Edge", CGNS_ENUMV(BAR_2), 2, 2, 0, bar.data(),
                               &section) == CG_OK);
      // Round robin of the kinds, 9 nodes a round (a section and a BC are 3 each), then
      // families for the rest.
      std::int64_t n = n0 + 4, i = 0;  // the edge section and ZoneBC_t's first BC
      for (; n + 9 <= nodes; n += 9, ++i) {
        const std::string tag = std::to_string(i);
        REQUIRE(cg_family_write(fn, base, ("f" + tag).c_str(), &family) == CG_OK);
        REQUIRE(cg_goto(fn, base, "end") == CG_OK);
        REQUIRE(cg_descriptor_write(("d" + tag).c_str(), "a") == CG_OK);
        REQUIRE(cg_goto(fn, base, "end") == CG_OK);
        REQUIRE(cg_user_data_write(("u" + tag).c_str()) == CG_OK);
        REQUIRE(cg_section_write(fn, base, 1, ("s" + tag).c_str(), CGNS_ENUMV(BAR_2), 3 + i, 3 + i,
                                 0, bar.data(), &section) == CG_OK);
        const std::array<cgsize_t, 2> range{2, 2};
        REQUIRE(cg_boco_write(fn, base, 1, ("b" + tag).c_str(), CGNS_ENUMV(BCWall),
                              CGNS_ENUMV(ElementRange), 2, range.data(), &bc) == CG_OK);
      }
      for (; n < nodes; ++n, ++i) {
        REQUIRE(cg_family_write(fn, base, ("g" + std::to_string(i)).c_str(), &family) == CG_OK);
      }
    });
    return file;
  };
  const fs::path full = build("nodes-limit.cgns", limit);
  const std::int64_t nodes = autan::io::cgns_logical_bytes(full).nodes;
  REQUIRE(nodes == limit);
  (void)autan::io::read_mesh(plain);  // the libraries' first-use setup, outside both peaks
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(full).n_cells() == 1); });
  CAPTURE(before, after, nodes);
  CHECK(after - before <= (std::int64_t{1} << 20) + (std::int64_t{16} << 10) * (nodes - n0));
  const fs::path over = build("nodes-over.cgns", limit + 1);
  try {
    (void)autan::io::cgns_logical_bytes(over);  // MeshInfo refuses it too
    FAIL("expected E0133");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::io_unsupported);
  }
  constexpr std::int64_t caches = (std::int64_t{4} << 20) + (std::int64_t{8} << 20) + (64 << 10);
  const std::int64_t refused =
      peak_of([&] { require_error(autan::ErrorCode::io_unsupported, over); });
  CHECK(refused <= (std::int64_t{1} << 20) + caches);
}

TEST_CASE("A BC range over points and edges is never expanded", "[cgns]") {
  // mesh.md, Readers: a range of points or edges labels nothing; the reader judges it id by id.
  // 2^22 skipped edges under one BC cost what one does, not 8 bytes each.
  fs::create_directories(temp_dir());
  constexpr cgsize_t edges = cgsize_t{1} << 22;
  const auto with_bc = [&](const std::string& name, cgsize_t last) {
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      std::vector<cgsize_t> bar(2 * static_cast<std::size_t>(edges));
      for (std::size_t e = 0; e < bar.size(); e += 2) {
        bar[e] = 1;
        bar[e + 1] = 2;
      }
      int section = 0, bc = 0;
      REQUIRE(cg_section_write(fn, base, 1, "Edges", CGNS_ENUMV(BAR_2), 2, edges + 1, 0, bar.data(),
                               &section) == CG_OK);
      const std::array<cgsize_t, 2> range{2, last};
      REQUIRE(cg_boco_write(fn, base, 1, "edges", CGNS_ENUMV(BCWall), CGNS_ENUMV(ElementRange), 2,
                            range.data(), &bc) == CG_OK);
    });
  };
  const fs::path one = with_bc("range-one.cgns", 2);
  const fs::path all = with_bc("range-all.cgns", edges + 1);
  (void)autan::io::read_mesh(one);  // the libraries' first-use setup, outside both peaks
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(one); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(all).n_cells() == 1); });
  CAPTURE(before, after);
  CHECK(after - before <= std::int64_t{1} << 20);  // the ids would be 32 MiB
}
TEST_CASE("A CGNS tree is read within its per-node bound up to 16 levels, refused deeper",
          "[cgns]") {
  // mesh.md, Memory of a CGNS read: HDF5 keeps every open node's path, so a node costs more the
  // deeper it is. 15 000 user data nodes of 32-character names at the deepest level allowed stay
  // within 1 MiB + 16 KiB a node; 1024 levels are refused before cg_open, at the walk's cost.
  fs::create_directories(temp_dir());
  const fs::path plain =
      cgns_file("deep-none.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
  const auto nested = [&](const std::string& name, int chain, int leaves) {
    const fs::path file = temp_dir() / name;
    fs::copy_file(plain, file, fs::copy_options::overwrite_existing);
    int cgio = 0;
    REQUIRE(cgio_open_file(file.string().c_str(), CGIO_MODE_MODIFY, CGIO_FILE_NONE, &cgio) ==
            CG_OK);
    double root = 0, parent = 0;
    REQUIRE(cgio_get_root_id(cgio, &root) == CG_OK);
    REQUIRE(cgio_get_node_id(cgio, root, "/Base", &parent) == CG_OK);
    const auto add = [&](double under, int i) {
      std::string tag = std::to_string(i);
      tag.insert(0, 32 - tag.size(), 'u');
      double id = 0;
      REQUIRE(cgio_create_node(cgio, under, tag.c_str(), &id) == CG_OK);
      REQUIRE(cgio_set_label(cgio, id, "UserDefinedData_t") == CG_OK);
      return id;
    };
    for (int i = 0; i < chain; ++i) parent = add(parent, i);
    for (int i = 0; i < leaves; ++i) (void)add(parent, chain + i);
    REQUIRE(cgio_close_file(cgio) == CG_OK);
    return file;
  };
  constexpr std::int64_t caches = (std::int64_t{4} << 20) + (std::int64_t{8} << 20) + (64 << 10);
  // The base at level 1, a chain of 14 under it, the leaves at level 16.
  const fs::path deepest = nested("deep-16.cgns", 14, 15000);
  const std::int64_t n0 = autan::io::cgns_logical_bytes(plain).nodes;
  const std::int64_t nodes = autan::io::cgns_logical_bytes(deepest).nodes;
  REQUIRE(nodes == n0 + 15014);
  (void)autan::io::read_mesh(plain);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(deepest).n_cells() == 1); });
  CAPTURE(before, after);
  CHECK(after - before <= (std::int64_t{1} << 20) + (std::int64_t{16} << 10) * (nodes - n0));
  const fs::path deeper = nested("deep-17.cgns", 15, 1);
  require_error(autan::ErrorCode::io_unsupported, deeper);
  const fs::path deep = nested("deep-1024.cgns", 1024, 15000);
  const std::int64_t refused =
      peak_of([&] { require_error(autan::ErrorCode::io_unsupported, deep); });
  CHECK(refused <= (std::int64_t{1} << 20) + caches);
}

TEST_CASE("A BC point list over points and edges is read in chunks, never whole", "[cgns]") {
  // mesh.md, Readers: a point list of points or edges labels nothing; the reader judges it in
  // chunks of 2^17 ids. 2^22 listed edges cost what one does, not 8 bytes each.
  fs::create_directories(temp_dir());
  constexpr cgsize_t edges = cgsize_t{1} << 22;
  const auto with_bc = [&](const std::string& name, cgsize_t listed) {
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      std::vector<cgsize_t> bar(2 * static_cast<std::size_t>(edges));
      for (std::size_t e = 0; e < bar.size(); e += 2) {
        bar[e] = 1;
        bar[e + 1] = 2;
      }
      int section = 0, bc = 0;
      REQUIRE(cg_section_write(fn, base, 1, "Edges", CGNS_ENUMV(BAR_2), 2, edges + 1, 0, bar.data(),
                               &section) == CG_OK);
      std::vector<cgsize_t> ids(static_cast<std::size_t>(listed));
      std::iota(ids.begin(), ids.end(), cgsize_t{2});
      REQUIRE(cg_boco_write(fn, base, 1, "edges", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointList), listed,
                            ids.data(), &bc) == CG_OK);
      REQUIRE(cg_boco_gridlocation_write(fn, base, 1, bc, CGNS_ENUMV(EdgeCenter)) == CG_OK);
    });
  };
  const fs::path one = with_bc("list-one.cgns", 1);
  const fs::path all = with_bc("list-all.cgns", edges);
  (void)autan::io::read_mesh(one);  // the libraries' first-use setup, outside both peaks
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(one); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(all).n_cells() == 1); });
  CAPTURE(before, after);
  CHECK(after - before <= std::int64_t{2} << 20);  // a 1 MiB chunk; the list would be 32 MiB
}
TEST_CASE("A section of boundary faces nothing names is skipped", "[cgns]") {
  // mesh.md, Readers: the faces come from the cells; 2^20 triangles no family, BC, subregion or
  // connectivity names cost nothing kept, not 88 bytes each.
  fs::create_directories(temp_dir());
  constexpr cgsize_t triangles = cgsize_t{1} << 20;
  const fs::path plain =
      cgns_file("faces-none.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
  const fs::path padded = cgns_file("faces-unused.cgns", [&](int fn, int base) {
    unstructured_tet(fn, base, "Zone");
    std::vector<cgsize_t> tri(3 * static_cast<std::size_t>(triangles));
    for (std::size_t t = 0; t < tri.size(); t += 3) {
      tri[t] = 1;
      tri[t + 1] = 3;
      tri[t + 2] = 2;
    }
    int section = 0;
    REQUIRE(cg_section_write(fn, base, 1, "Faces", CGNS_ENUMV(TRI_3), 2, triangles + 1, 0,
                             tri.data(), &section) == CG_OK);
  });
  (void)autan::io::read_mesh(plain);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(padded).n_cells() == 1); });
  CAPTURE(before, after);
  CHECK(after - before <= std::int64_t{2} << 20);  // kept, the triangles would be 88 MiB
}

TEST_CASE("A CGNS file reads at most 1 MiB of other data when it opens", "[cgns]") {
  // mesh.md, Memory of a CGNS read: cg_open reads a BaseIterativeData_t's TimeValues whole; the
  // largest within 1 MiB is read, one value more is refused before cg_open.
  fs::create_directories(temp_dir());
  const auto with_times = [&](const std::string& name, cgsize_t steps) {
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      REQUIRE(cg_biter_write(fn, base, "Iterations", static_cast<int>(steps)) == CG_OK);
      REQUIRE(cg_goto(fn, base, "BaseIterativeData_t", 1, "end") == CG_OK);
      const std::vector<double> times(static_cast<std::size_t>(steps), 1.0);
      REQUIRE(cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &steps, times.data()) ==
              CG_OK);
    });
  };
  constexpr std::int64_t limit = std::int64_t{1} << 20;
  const std::int64_t one = autan::io::cgns_logical_bytes(with_times("times-1.cgns", 1)).eager;
  const auto steps = static_cast<cgsize_t>(1 + (limit - one) / 8);
  const fs::path at = with_times("times-at.cgns", steps);
  CHECK(autan::io::cgns_logical_bytes(at).eager <= limit);
  CHECK(autan::io::read_mesh(at).n_cells() == 1);
  require_error(autan::ErrorCode::io_unsupported, with_times("times-over.cgns", steps + 1));
}
TEST_CASE("Faces of a section nothing names are skipped, one named is kept", "[cgns]") {
  // mesh.md, Readers: 2^20 triangles in a MIXED or a TRI_3 section, the first named by a BC: it
  // labels its face, the others cost nothing kept, not 88 to 92 bytes each.
  fs::create_directories(temp_dir());
  constexpr cgsize_t triangles = cgsize_t{1} << 20;
  const bool mixed_section = GENERATE(true, false);
  CAPTURE(mixed_section);
  const auto mixed = [&](const std::string& name, cgsize_t count) {
    return cgns_file(name, [&](int fn, int base) {
      unstructured_tet(fn, base, "Zone");
      std::vector<cgsize_t> data, offsets{0};
      for (cgsize_t t = 0; t < count; ++t) {
        if (mixed_section) data.push_back(CGNS_ENUMV(TRI_3));
        data.insert(data.end(), {1, 3, 2});
        offsets.push_back(static_cast<cgsize_t>(data.size()));
      }
      int section = 0, bc = 0;
      if (mixed_section) {
        REQUIRE(cg_poly_section_write(fn, base, 1, "Faces", CGNS_ENUMV(MIXED), 2, count + 1, 0,
                                      data.data(), offsets.data(), &section) == CG_OK);
      } else {
        REQUIRE(cg_section_write(fn, base, 1, "Faces", CGNS_ENUMV(TRI_3), 2, count + 1, 0,
                                 data.data(), &section) == CG_OK);
      }
      const cgsize_t first = 2;
      REQUIRE(cg_boco_write(fn, base, 1, "wall", CGNS_ENUMV(BCWall), CGNS_ENUMV(PointList), 1,
                            &first, &bc) == CG_OK);
      REQUIRE(cg_boco_gridlocation_write(fn, base, 1, bc, CGNS_ENUMV(FaceCenter)) == CG_OK);
    });
  };
  const fs::path one = mixed("mixed-faces-1.cgns", 1);
  const fs::path many = mixed("mixed-faces-many.cgns", triangles);
  (void)autan::io::read_mesh(one);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(one); });
  std::optional<autan::Mesh> mesh;
  const std::int64_t after = peak_of([&] { mesh = autan::io::read_mesh(many); });
  CAPTURE(before, after);
  CHECK(after - before <= std::int64_t{2} << 20);  // kept, the triangles would be 92 MiB
  const auto zones = mesh->face_zones();
  const auto wall = std::ranges::find(zones, std::string("wall"), &autan::FaceZone::name);
  REQUIRE(wall != zones.end());
  const auto z = static_cast<std::size_t>(wall - zones.begin());
  CHECK(mesh->boundary_zone_offsets()[z + 1] - mesh->boundary_zone_offsets()[z] == 1);
  // mesh-info.md, source.point_list_bytes: the BC's list of one 8-byte id.
  CHECK(autan::io::cgns_logical_bytes(many).point_lists == 8);
}

TEST_CASE("Looking up many BCs among many families keeps no HDF5 handle", "[cgns]") {
  // mesh.md, Memory of a CGNS read: 8000 unused families and 2000 BCs, each a point list of the
  // one boundary face in a family: read within 1 MiB + 16 KiB a node, the walks releasing what
  // they open.
  fs::create_directories(temp_dir());
  const fs::path plain =
      cgns_file("handles-none.cgns", [](int fn, int base) { unstructured_tet(fn, base, "Zone"); });
  const fs::path file = cgns_file("handles.cgns", [](int fn, int base) {
    unstructured_tet(fn, base, "Zone");
    const std::array<cgsize_t, 3> tri{1, 3, 2};
    int section = 0, family = 0, bc = 0;
    REQUIRE(cg_section_write(fn, base, 1, "Face", CGNS_ENUMV(TRI_3), 2, 2, 0, tri.data(),
                             &section) == CG_OK);
    REQUIRE(cg_family_write(fn, base, "wall", &family) == CG_OK);
    for (int i = 0; i < 8000; ++i) {
      REQUIRE(cg_family_write(fn, base, ("f" + std::to_string(i)).c_str(), &family) == CG_OK);
    }
    const cgsize_t face = 2;
    for (int i = 0; i < 2000; ++i) {
      REQUIRE(cg_boco_write(fn, base, 1, ("b" + std::to_string(i)).c_str(),
                            CGNS_ENUMV(FamilySpecified), CGNS_ENUMV(PointList), 1, &face,
                            &bc) == CG_OK);
      REQUIRE(cg_boco_gridlocation_write(fn, base, 1, bc, CGNS_ENUMV(FaceCenter)) == CG_OK);
      REQUIRE(cg_goto(fn, base, "Zone_t", 1, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK);
      REQUIRE(cg_famname_write("wall") == CG_OK);
    }
  });
  const std::int64_t n0 = autan::io::cgns_logical_bytes(plain).nodes;
  const std::int64_t nodes = autan::io::cgns_logical_bytes(file).nodes;
  REQUIRE(nodes <= 16384);
  (void)autan::io::read_mesh(plain);
  const std::int64_t before = peak_of([&] { (void)autan::io::read_mesh(plain); });
  const std::int64_t after = peak_of([&] { CHECK(autan::io::read_mesh(file).n_cells() == 1); });
  CAPTURE(before, after, nodes);
  CHECK(after - before <= (std::int64_t{1} << 20) + (std::int64_t{16} << 10) * (nodes - n0));
}
#endif
