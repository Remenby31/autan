#include <filesystem>
#include <map>
#include <random>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/io/write_polymesh.hpp"
#include "fixtures.hpp"

namespace fs = std::filesystem;

TEST_CASE("A written polyMesh reads back as the mesh it was written from", "[polymesh]") {
  // Mixed cells and zones (hybrid, Gmsh), and a 105-face polyhedron with a wall (hanging, Fluent).
  const std::string source = GENERATE("hybrid/hybrid.ascii.msh", "hanging/hanging.fluent.msh");
  CAPTURE(source);
  const autan::Mesh mesh = autan::io::read_mesh(fs::path(AUTAN_TEST_MESH_DIR) / source);
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_polymesh_write-" + std::to_string(random())) / "polyMesh";
  CHECK(autan::io::write_polymesh(dir, mesh).empty());
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

TEST_CASE("Zone names a polyMesh cannot hold are written as words, and reported", "[polymesh]") {
  // OpenFOAM patch and zone names are words: VGRID's "Patch   1" is written "Patch_1".
  auto cells = autan::test::box(2, autan::test::BoxFill::hexa);
  cells.face_zones[0].name = "Patch   1";
  cells.cell_zones[0].name = "fluid (core)";
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_polymesh_write-" + std::to_string(random())) / "polyMesh";
  const auto renamed = autan::io::write_polymesh(dir, mesh);
  CHECK(renamed == std::map<std::string, std::string>{{"Patch   1", "Patch_1"},
                                                      {"fluid (core)", "fluid_core_"}});
  const autan::Mesh back = autan::io::read_polymesh(dir);
  CHECK(back.face_zones()[0].name == "Patch_1");
  CHECK(back.cell_zones()[0].name == "fluid_core_");
  // A leading digit (a number to OpenFOAM's tokenizer), `$`, and the reader's own `interior`
  // (a patch of a single cell, whose mesh has no internal face to take that name).
  auto odd = autan::test::box(1, autan::test::BoxFill::hexa);
  odd.face_zones[0].name = "1";
  odd.face_zones[1].name = "a$b";
  odd.face_zones[2].name = "interior";
  const auto odd_names = autan::io::write_polymesh(
      dir.parent_path() / "odd", autan::assemble(autan::to_face_based(std::move(odd))));
  CHECK(odd_names ==
        std::map<std::string, std::string>{{"1", "_1"}, {"a$b", "a_b"}, {"interior", "interior_"}});
  CHECK(autan::io::read_polymesh(dir.parent_path() / "odd").face_zones()[2].name == "interior_");
  // Two names that become one word are refused.
  auto clash = autan::test::box(2, autan::test::BoxFill::hexa);
  clash.face_zones[0].name = "a b";
  clash.face_zones[1].name = "a_b";
  try {
    (void)autan::io::write_polymesh(dir.parent_path() / "clash",
                                    autan::assemble(autan::to_face_based(std::move(clash))));
    FAIL("wrote two zones under one name");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_duplicate_zone_name);
  }
}
