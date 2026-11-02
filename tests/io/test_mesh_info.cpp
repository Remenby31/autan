#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <nlohmann/json.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "fixtures.hpp"

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
  // The unit cube of generate.py, in the file's units.
  CHECK(doc["bounds"]["min"] == nlohmann::json::array({0.0, 0.0, 0.0}));
  CHECK(doc["bounds"]["max"] == nlohmann::json::array({1.0, 1.0, 1.0}));
  REQUIRE(doc["boundary_zones"].size() == 6);
  CHECK(doc["boundary_zones"][0]["name"] == "xmin");
  CHECK(doc["boundary_zones"][0]["faces"] == 24);
  REQUIRE(doc["cell_zones"].size() == 2);
  REQUIRE(doc["interfaces"].size() == 1);
  CHECK(doc["interfaces"][0]["zones"] == nlohmann::json::array({"lower", "upper"}));
  CHECK(doc["interfaces"][0]["faces"] == 24);
  CHECK(doc["periodic"] == nlohmann::json::array());
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

TEST_CASE("Only cells with the topology of a standard shape count as that shape", "[mesh-info]") {
  // A triangular prism with one rectangular side split by a diagonal and the
  // two cap edges it ends on split at their midpoints: six quadrilaterals and
  // eight vertices, but a vertex of degree 4, so not a hexahedron.
  autan::CellwiseMesh cw;
  const std::array<std::array<double, 3>, 8> points{
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {0.5, 0, 0}, {0.5, 0, 1}}};
  for (const auto& p : points) {
    cw.x.push_back(p[0]);
    cw.y.push_back(p[1]);
    cw.z.push_back(p[2]);
  }
  const std::vector<autan::offset_t> offsets{0, 4, 8, 12, 16, 20, 24};
  const std::vector<autan::index_t> faces{0, 2, 1, 6, 3, 7, 4, 5, 0, 6, 1, 4,
                                          0, 4, 7, 3, 1, 2, 5, 4, 2, 0, 3, 5};
  cw.add_polyhedron(offsets, faces, 0);
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const auto doc = nlohmann::json::parse(autan::io::mesh_info_json(mesh, std::string(64, '0')));
  CHECK(doc["counts"]["cells_by_type"]["hexa"] == 0);
  CHECK(doc["counts"]["cells_by_type"]["polyhedron"] == 1);
}

TEST_CASE("MeshInfo omits cell zones without cells", "[mesh-info]") {
  const fs::path dir = fresh_dir("polyMesh");
  for (const auto& entry : fs::directory_iterator(data_dir / "hex" / "polyMesh")) {
    fs::copy(entry.path(), dir / entry.path().filename());
  }
  std::string labels;
  for (int c = 0; c < 27; ++c) labels += std::to_string(c) + " ";
  write(dir / "cellZones",
        "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
        "2 ( fluid { type cellZone; cellLabels List<label> 27(" +
            labels + "); } unused { type cellZone; cellLabels List<label> 0(); } )\n");
  const auto doc = nlohmann::json::parse(
      autan::io::mesh_info_json(autan::io::read_mesh(dir), autan::io::mesh_hash(dir)));
  REQUIRE(doc["cell_zones"].size() == 1);
  CHECK(doc["cell_zones"][0]["name"] == "fluid");
}

TEST_CASE("MeshInfo lists periodic pairs and types their zones periodic", "[mesh-info]") {
  // The unit box cut in 3^3 hexahedra, periodic along x: 9 faces a side.
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::hexa);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const auto doc = nlohmann::json::parse(autan::io::mesh_info_json(
      autan::assemble(autan::to_face_based(std::move(cells))), std::string(64, '0')));
  REQUIRE(doc["periodic"].size() == 1);
  CHECK(doc["periodic"][0] ==
        nlohmann::json{{"zones", {"xmin", "xmax"}},
                       {"transform", {{"kind", "translation"}, {"vector", {1.0, 0.0, 0.0}}}},
                       {"faces", 9}});
  for (const auto& zone : doc["boundary_zones"]) {
    CAPTURE(zone.dump());
    const bool periodic = zone["name"] == "xmin" || zone["name"] == "xmax";
    CHECK((zone["type"] == "periodic") == periodic);
  }
}

TEST_CASE("The boundary surface closes the domain, outward, zone by zone", "[mesh-info][surface]") {
  // Every family through OpenFOAM, and two meshes through every reader: the
  // file parsed here by its layout alone (mesh.md, Boundary surface).
  std::vector<std::string> files;
  for (const char* family : {"tet", "hex", "prism", "pyramid", "poly", "hybrid"})
    files.push_back(std::string(family) + "/polyMesh");
  for (const char* name : {"hybrid", "tet-gmsh"}) {
    for (const char* suffix : {".ascii.msh", ".binary.msh", ".fluent.msh", ".fluent-binary.msh"})
      files.push_back(std::string(name) + "/" + name + suffix);
#ifdef AUTAN_HAVE_CGNS
    files.push_back(std::string(name) + "/" + name + ".cgns");
#endif
  }
  files.emplace_back("periodic-box/periodic-box.ascii.msh");
  for (const auto& file : files) {
    CAPTURE(file);
    const fs::path path = data_dir / file;
    const autan::Mesh mesh = autan::io::read_mesh(path);
    const std::string hash = autan::io::mesh_hash(path);
    const std::string bytes = autan::io::surface_bytes(mesh, hash);

    std::uint32_t n = 0;
    REQUIRE(bytes.size() >= 4);
    std::memcpy(&n, bytes.data(), 4);
    REQUIRE((4 + n) % 8 == 0);
    const std::string text = bytes.substr(4, n);
    const auto end = text.find_last_of('}');
    REQUIRE(end != std::string::npos);
    CHECK(text.find_first_not_of(' ', end + 1) == std::string::npos);  // spaces only
    const auto header = nlohmann::json::parse(text);
    CHECK(header["schema"] == "autan.surface/1");
    CHECK(header["source"]["sha256"] == hash);
    const auto buffer = [&](const char* name, auto element) {
      using T = decltype(element);
      for (const auto& b : header["buffers"]) {
        if (b["name"] != name) continue;
        CHECK(b["dtype"] == (std::is_same_v<T, double> ? "f64" : "u32"));
        const auto offset = b["offset"].get<std::size_t>();
        const auto length = b["length"].get<std::size_t>();
        CHECK(offset % 8 == 0);
        REQUIRE(4 + n + offset + length * sizeof(T) <= bytes.size());
        std::vector<T> out(length);
        if (length > 0) std::memcpy(out.data(), bytes.data() + 4 + n + offset, length * sizeof(T));
        return out;
      }
      FAIL("no buffer " << name);
      return std::vector<T>{};
    };
    const auto positions = buffer("positions", double{});
    const auto offsets = buffer("face_offsets", std::uint32_t{});
    const auto nodes = buffer("face_nodes", std::uint32_t{});
    const auto n_faces = header["faces"].get<std::size_t>();
    REQUIRE(positions.size() == 3 * header["nodes"].get<std::size_t>());
    REQUIRE(offsets.size() == n_faces + 1);
    CHECK(offsets.front() == 0);
    REQUIRE(offsets.back() == nodes.size());
    // Nodes numbered in order of first use.
    std::uint32_t next = 0;
    for (const auto node : nodes) {
      REQUIRE(node <= next);
      if (node == next) ++next;
    }
    CHECK(next == header["nodes"]);

    // Zones: MeshInfo's boundary zones, their faces in a row.
    const auto info = nlohmann::json::parse(autan::io::mesh_info_json(mesh, hash));
    const auto& zones = header["zones"];
    REQUIRE(zones.size() == info["boundary_zones"].size());
    std::size_t first = 0;
    for (std::size_t k = 0; k < zones.size(); ++k) {
      CHECK(zones[k]["name"] == info["boundary_zones"][k]["name"]);
      CHECK(zones[k]["type"] == info["boundary_zones"][k]["type"]);
      CHECK(zones[k]["native_type"] == info["boundary_zones"][k]["native_type"]);
      CHECK(zones[k]["faces"] == info["boundary_zones"][k]["faces"]);
      CHECK(zones[k]["first_face"] == first);
      first += zones[k]["faces"].get<std::size_t>();
    }
    CHECK(first == n_faces);
    // Each side of the box (zones xmin ... zmax) holds faces on its plane only.
    std::array<double, 3> low{}, high{};
    for (std::size_t d = 0; d < 3; ++d) {
      low[d] = high[d] = positions[static_cast<std::size_t>(d)];
      for (std::size_t i = static_cast<std::size_t>(d); i < positions.size(); i += 3) {
        low[d] = std::min(low[d], positions[i]);
        high[d] = std::max(high[d], positions[i]);
      }
    }
    for (const auto& zone : zones) {
      const std::string name = zone["name"];
      if (name.size() != 4 || name[0] < 'x' || name[0] > 'z') continue;
      const auto d = static_cast<std::size_t>(name[0] - 'x');
      const double plane = name.substr(1) == "min" ? low[d] : high[d];
      const auto from = zone["first_face"].get<std::size_t>();
      for (std::size_t f = from; f < from + zone["faces"].get<std::size_t>(); ++f) {
        for (auto i = offsets[f]; i < offsets[f + 1]; ++i)
          CHECK(std::abs(positions[3 * std::size_t{nodes[i]} + d] - plane) <=
                1e-12 * (high[d] - low[d]));
      }
    }

    // Closed: every edge in exactly two faces, once each way. Outward: the
    // area vectors (Newell) sum to zero, and the divergence theorem gives the
    // mesh's volume with a positive sign.
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> edges;
    std::array<double, 3> sum{};
    double area = 0.0, volume = 0.0;
    for (std::size_t f = 0; f < n_faces; ++f) {
      std::array<double, 3> s{}, c{};
      const auto k = offsets[f + 1] - offsets[f];
      for (auto i = offsets[f]; i < offsets[f + 1]; ++i) {
        const auto a = nodes[i], b = nodes[i + 1 < offsets[f + 1] ? i + 1 : offsets[f]];
        REQUIRE(3 * std::size_t{a} + 2 < positions.size());
        ++edges[{a, b}];
        const double* p = &positions[3 * std::size_t{a}];
        const double* q = &positions[3 * std::size_t{b}];
        s[0] += (p[1] - q[1]) * (p[2] + q[2]);
        s[1] += (p[2] - q[2]) * (p[0] + q[0]);
        s[2] += (p[0] - q[0]) * (p[1] + q[1]);
        for (std::size_t d = 0; d < 3; ++d) c[d] += p[d] / k;
      }
      for (std::size_t d = 0; d < 3; ++d) {
        s[d] *= 0.5;
        sum[d] += s[d];
        volume += c[d] * s[d] / 3.0;
      }
      area += std::hypot(s[0], s[1], s[2]);
    }
    for (const auto& [edge, count] : edges) {
      CHECK(count == 1);  // these boundaries are manifolds
      CHECK(edges.contains({edge.second, edge.first}));
    }
    constexpr double eps = 0x1p-52;
    for (const double s : sum) CHECK(std::abs(s) <= 64 * eps * area);
    const autan::Geometry g = autan::compute_geometry(mesh);
    double cells = 0.0;
    for (const double v : g.cell_volume) cells += v;
    CHECK(std::abs(volume - cells) <= 1e-12 * cells);
  }
  // Outward normals need positive cells: an inverted tetrahedron is refused.
  autan::CellwiseMesh inverted;
  for (const auto& p :
       std::array<std::array<double, 3>, 4>{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}) {
    inverted.x.push_back(p[0]);
    inverted.y.push_back(p[1]);
    inverted.z.push_back(p[2]);
  }
  inverted.add_cell(autan::CellShape::tetra, std::array<autan::index_t, 4>{0, 2, 1, 3}, 0);
  inverted.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(inverted));
  try {
    (void)autan::io::surface_bytes(mesh, std::string(64, '0'));
    FAIL("expected E0121");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::geometry_non_positive_volume);
  }
}

TEST_CASE("MeshInfo counts the nodes of every face", "[mesh-info]") {
  // Tetrahedra have triangular faces only, hexahedra quadrilateral ones.
  for (const auto& [mesh, nodes_per_face] :
       {std::pair{"tet/polyMesh", 3}, std::pair{"hex/polyMesh", 4}}) {
    CAPTURE(mesh);
    const fs::path path = data_dir / mesh;
    const auto doc = nlohmann::json::parse(
        autan::io::mesh_info_json(autan::io::read_mesh(path), autan::io::mesh_hash(path)));
    CHECK(doc["counts"]["face_nodes"] ==
          nodes_per_face * doc["counts"]["faces"].get<std::int64_t>());
  }
}
