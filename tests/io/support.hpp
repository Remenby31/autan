#pragma once

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

// Comparisons and fixtures shared by the reader tests.
namespace autan_test {

namespace fs = std::filesystem;
using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

inline const fs::path data_dir{AUTAN_TEST_MESH_DIR};

inline std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Format-independent identity of a mesh: points by coordinates (rounded to
// 1e-7, coarser than the 10 significant digits Fluent files carry), faces by
// their point set, cells by their face set.
struct Canonical {
  std::vector<std::array<std::int64_t, 3>> point_key;
  std::map<std::vector<std::array<std::int64_t, 3>>, index_t> face_of_key;
  std::vector<std::vector<std::array<std::int64_t, 3>>> face_key;
  std::map<std::vector<index_t>, index_t> cell_of_faces;  // faces as indices into `faces`
};

inline std::vector<std::array<std::int64_t, 3>> sorted_face_key(
    const autan::Mesh& mesh, index_t f, const std::vector<std::array<std::int64_t, 3>>& pk) {
  std::vector<std::array<std::int64_t, 3>> key;
  for (const index_t p : mesh.face_nodes(f)) key.push_back(pk[at(p)]);
  std::sort(key.begin(), key.end());
  return key;
}

inline Canonical canonical(const autan::Mesh& mesh) {
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
inline void require_same_mesh(const autan::Mesh& a, const autan::Mesh& b, bool cell_zones = true) {
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

inline autan::FaceZone zone_named(const autan::Mesh& mesh, const std::string& name) {
  const auto zones = mesh.face_zones();
  const auto it = std::find_if(zones.begin(), zones.end(),
                               [&](const autan::FaceZone& z) { return z.name == name; });
  REQUIRE(it != zones.end());
  return *it;
}

inline void require_valid_box(const autan::Mesh& mesh, double volume) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto closure = autan::cell_closure_error(mesh, g);
  CHECK(*std::max_element(closure.begin(), closure.end()) < 1e-12);
  CHECK_THAT(std::accumulate(g.cell_volume.begin(), g.cell_volume.end(), 0.0),
             WithinRel(volume, 1e-12));
}

// One directory per test process: ctest runs test cases in parallel.
inline const fs::path& temp_dir() {
  static const fs::path dir = [] {
    std::random_device random;
    return fs::temp_directory_path() / ("autan_io_tests-" + std::to_string(random()));
  }();
  return dir;
}

inline fs::path write_temp(const std::string& name, const std::string& content) {
  const fs::path dir = temp_dir();
  fs::create_directories(dir);
  const fs::path path = dir / name;
  std::ofstream(path, std::ios::binary) << content;
  return path;
}

inline void require_error(autan::ErrorCode expected, const fs::path& path) {
  try {
    (void)autan::io::read_mesh(path);
    FAIL("expected " << autan::error_code_string(expected));
  } catch (const autan::Error& e) {
    CHECK(e.code() == expected);
  }
}

// Facts of the unit-cube test meshes known from their construction
// (tests/data/mesh/generate/generate.py), independent of every reader: each
// boundary face lies on the plane its zone names, and structured cells have
// the volume of their share of a cube of side 1/n.
inline void require_constructed(const autan::Mesh& mesh, const std::string& name) {
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
  // hybrid: cells below z = 1/2 form the volume group "lower", the others
  // "upper" (Fluent files from foamMeshToFluent keep a single cell zone).
  if (name == "hybrid" && mesh.cell_zones().size() == 2) {
    for (index_t c = 0; c < mesh.n_cells(); ++c) {
      CHECK(mesh.cell_zones()[at(mesh.cell_zone()[at(c)])].name ==
            (g.cell_centroid_z[at(c)] < 0.5 ? "lower" : "upper"));
    }
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

}  // namespace autan_test
