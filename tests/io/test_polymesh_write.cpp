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
  // A leading digit (a number to OpenFOAM's tokenizer) and `$`.
  auto odd = autan::test::box(1, autan::test::BoxFill::hexa);
  odd.face_zones[0].name = "1";
  odd.face_zones[1].name = "a$b";
  CHECK(autan::io::write_polymesh(dir.parent_path() / "odd",
                                  autan::assemble(autan::to_face_based(std::move(odd)))) ==
        std::map<std::string, std::string>{{"1", "_1"}, {"a$b", "a_b"}});
  // A patch named `interior` beside internal faces of another zone (as Fluent's `int_fluid`): the
  // reader gives the internal faces its own zone `interior`, so the patch is written `interior_`.
  const autan::Mesh box =
      autan::assemble(autan::to_face_based(autan::test::box(2, autan::test::BoxFill::hexa)));
  autan::MeshInput in;
  in.x.assign(box.x().begin(), box.x().end());
  in.y.assign(box.y().begin(), box.y().end());
  in.z.assign(box.z().begin(), box.z().end());
  in.face_node_offsets.assign(box.face_node_offsets().begin(), box.face_node_offsets().end());
  in.face_nodes.assign(box.face_nodes().begin(), box.face_nodes().end());
  in.face_cell0.assign(box.owner().begin(), box.owner().end());
  in.cell_zone.assign(box.cell_zone().begin(), box.cell_zone().end());
  in.face_zones.assign(box.face_zones().begin(), box.face_zones().end());
  in.cell_zones.assign(box.cell_zones().begin(), box.cell_zones().end());
  const auto inner = static_cast<autan::index_t>(in.face_zones.size());
  in.face_zones.push_back(autan::FaceZone{.name = "inner",
                                          .bc_type = "",
                                          .kind = autan::BcKind::unspecified,
                                          .boundary = false,
                                          .provenance = {}});
  for (autan::index_t f = 0; f < box.n_faces(); ++f) {
    const auto i = static_cast<std::size_t>(f);
    const bool interior = f < box.n_interior_faces();
    in.face_cell1.push_back(interior ? box.neighbour()[i] : autan::no_cell);
    in.face_zone.push_back(interior ? inner : box.face_zone()[i]);
  }
  for (auto& zone : in.face_zones) {
    if (zone.name == "interior") zone.name = "unused";  // assembly's zone, emptied above
  }
  in.face_zones[static_cast<std::size_t>(box.face_zone().back())].name = "interior";
  CHECK(autan::io::write_polymesh(dir.parent_path() / "inner", autan::assemble(std::move(in))) ==
        std::map<std::string, std::string>{{"interior", "interior_"}});
  CHECK(autan::io::read_polymesh(dir.parent_path() / "inner").n_interior_faces() ==
        box.n_interior_faces());
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
