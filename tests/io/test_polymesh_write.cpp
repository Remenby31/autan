#include <filesystem>
#include <random>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/io/write_polymesh.hpp"

namespace fs = std::filesystem;

TEST_CASE("A written polyMesh reads back as the mesh it was written from", "[polymesh]") {
  // Mixed cells and zones (hybrid, Gmsh), and a 105-face polyhedron with a wall (hanging, Fluent).
  const std::string source = GENERATE("hybrid/hybrid.ascii.msh", "hanging/hanging.fluent.msh");
  CAPTURE(source);
  const autan::Mesh mesh = autan::io::read_mesh(fs::path(AUTAN_TEST_MESH_DIR) / source);
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_polymesh_write-" + std::to_string(random())) / "polyMesh";
  autan::io::write_polymesh(dir, mesh);
  const autan::Mesh back = autan::io::read_polymesh(dir);
  CHECK(std::ranges::equal(back.x(), mesh.x()));
  CHECK(std::ranges::equal(back.y(), mesh.y()));
  CHECK(std::ranges::equal(back.z(), mesh.z()));
  CHECK(std::ranges::equal(back.face_node_offsets(), mesh.face_node_offsets()));
  CHECK(std::ranges::equal(back.face_nodes(), mesh.face_nodes()));
  CHECK(std::ranges::equal(back.owner(), mesh.owner()));
  CHECK(std::ranges::equal(back.neighbour(), mesh.neighbour()));
  CHECK(std::ranges::equal(back.cell_zone(), mesh.cell_zone()));
  REQUIRE(back.cell_zones().size() == mesh.cell_zones().size());
  for (std::size_t z = 0; z < mesh.cell_zones().size(); ++z)
    CHECK(back.cell_zones()[z].name == mesh.cell_zones()[z].name);
  // Boundary zones: same names, kinds of wall and symmetry, faces.
  const auto zones = [](const autan::Mesh& m) {
    std::vector<std::tuple<std::string, autan::BcKind, autan::index_t>> out;
    const auto offsets = m.boundary_zone_offsets();
    for (std::size_t z = 0; z < m.face_zones().size(); ++z) {
      const auto& zone = m.face_zones()[z];
      if (!zone.boundary) continue;
      const bool kept = zone.kind == autan::BcKind::wall || zone.kind == autan::BcKind::symmetry;
      out.emplace_back(zone.name, kept ? zone.kind : autan::BcKind::unspecified,
                       offsets[z + 1] - offsets[z]);
    }
    return out;
  };
  CHECK(zones(back) == zones(mesh));
  CHECK(autan::compute_geometry(back).cell_volume == autan::compute_geometry(mesh).cell_volume);
}
