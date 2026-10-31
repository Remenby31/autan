#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"

namespace {

namespace fs = std::filesystem;
const fs::path data_dir{AUTAN_TEST_MESH_DIR};

fs::path fresh_dir(const std::string& name) {
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_mesh_info-" + std::to_string(random())) / name;
  fs::create_directories(dir);
  return dir;
}

void write(const fs::path& path, const std::string& content) {
  std::ofstream(path, std::ios::binary) << content;
}

}  // namespace

TEST_CASE("The mesh hash follows the MeshInfo contract", "[mesh-info]") {
  const fs::path dir = fresh_dir("hash");
  SECTION("file: SHA-256 of its bytes (FIPS 180-2 test vector)") {
    write(dir / "abc.msh", "abc");
    CHECK(autan::io::mesh_hash(dir / "abc.msh") ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  }
  // Expected digests computed with Python's hashlib over the framed
  // concatenation "<name>\n<length>\n<bytes>" in contract order.
  SECTION("polyMesh directory without cellZones") {
    write(dir / "points", "p");
    write(dir / "faces", "ff");
    write(dir / "owner", "");
    write(dir / "neighbour", "n");
    write(dir / "boundary", "b");
    CHECK(autan::io::mesh_hash(dir) ==
          "a60f55a0d1f0eedb2434572069b71f36320c3d949b334dc1eadbd136651e1ea8");
    write(dir / "cellZones", "cz");
    CHECK(autan::io::mesh_hash(dir) ==
          "3d0fa5c0ad6c894d6f887b1a3ad8ba0418a4b06e5208125c29f0ae91073ee9df");
  }
}

TEST_CASE("MeshInfo of the hybrid test mesh has its constructed counts", "[mesh-info]") {
  // tests/data/mesh/generate/generate.py, hybrid with n = 4: 8 hexahedra and
  // 8 cubes of 6 pyramids (lower corner block), 2 prisms per cube in the 16
  // cubes of each of two blocks, 6 tetrahedra per cube in the last 16. Zones
  // lower (k < 2) and upper share the plane z = 1/2: 8 quadrilaterals and 16
  // triangles. Side x = 0: 8 quadrilaterals and 16 triangles.
  const std::string source = GENERATE("hybrid/hybrid.ascii.msh", "hybrid/polyMesh");
  CAPTURE(source);
  const fs::path path = data_dir / source;
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(path), autan::io::mesh_hash(path)));

  CHECK(doc["schema"] == "autan.mesh-info/1");
  CHECK(doc["source"]["sha256"] == autan::io::mesh_hash(path));
  CHECK(doc["counts"]["cells"] == 216);
  const auto& types = doc["counts"]["cells_by_type"];
  CHECK(types["tetra"] == 96);
  CHECK(types["pyramid"] == 48);
  CHECK(types["prism"] == 64);
  CHECK(types["hexa"] == 8);
  CHECK(types["polyhedron"] == 0);
  REQUIRE(doc["boundary_zones"].size() == 6);
  CHECK(doc["boundary_zones"][0]["name"] == "xmin");
  CHECK(doc["boundary_zones"][0]["faces"] == 24);
  REQUIRE(doc["cell_zones"].size() == 2);
  REQUIRE(doc["interfaces"].size() == 1);
  CHECK(doc["interfaces"][0]["zones"] == nlohmann::json::array({"lower", "upper"}));
  CHECK(doc["interfaces"][0]["faces"] == 24);
  CHECK(doc["quality"]["max_faces_per_cell"] == 6);
  int cells = 0;
  for (const auto& zone : doc["cell_zones"]) cells += zone["cells"].get<int>();
  CHECK(cells == 216);
}

TEST_CASE("MeshInfo types follow the normalisation table", "[mesh-info]") {
  const fs::path path = data_dir / "hanging" / "hanging.fluent.msh";
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(path), autan::io::mesh_hash(path)));
  CHECK(doc["source"]["format"] == "fluent");
  CHECK(doc["counts"]["cells_by_type"]["polyhedron"] == 1);
  CHECK(doc["counts"]["cells_by_type"]["hexa"] == 100);
  CHECK(doc["quality"]["max_faces_per_cell"] == 105);
  for (const auto& zone : doc["boundary_zones"]) {
    CAPTURE(zone.dump());
    if (zone["name"] == "walls") {
      CHECK(zone["type"] == "wall");
    } else {
      CHECK(zone["type"] == "outlet");  // foamMeshToFluent writes patches as pressure-outlet
      CHECK(zone["native_type"] == "pressure-outlet");
    }
  }
  CHECK(doc["cell_zones"][0]["type"] == "fluid");
}
