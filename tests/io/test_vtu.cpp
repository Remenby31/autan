#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/io/vtu.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;

const fs::path data_dir{AUTAN_TEST_MESH_DIR};

fs::path temp_dir() { return fs::temp_directory_path() / "autan_io_vtu_tests"; }

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// A minimal reader of the file docs/contracts/vtu.md describes, written from that text and not
// from the writer: the XML header's DataArray attributes, then each array of the raw appended
// block at its offset, a UInt64 byte count followed by that many bytes.
struct Vtu {
  struct Array {
    std::string type, section;
    int components = 1;
    std::uint64_t offset = 0;
  };
  std::string header;
  std::map<std::string, Array> arrays;
  std::string data;  // the appended block after its '_'

  explicit Vtu(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string mark = "<AppendedData encoding=\"raw\">";
    const auto block = text.find(mark);
    if (block == std::string::npos) throw std::runtime_error("no raw AppendedData");
    const auto underscore = text.find('_', block + mark.size());
    header = text.substr(0, block);
    data = text.substr(underscore + 1);
    const std::regex array(R"(<DataArray ([^>]*)/>)");
    const auto value = [](const std::string& attributes, const std::string& key) {
      std::smatch m;
      if (!std::regex_search(attributes, m, std::regex(" ?" + key + "=\"([^\"]*)\"")))
        return std::string{};
      return m[1].str();
    };
    for (auto it = std::sregex_iterator(header.begin(), header.end(), array);
         it != std::sregex_iterator(); ++it) {
      const std::string attributes = (*it)[1].str();
      const auto position = static_cast<std::size_t>(it->position());
      Array a;
      a.type = value(attributes, "type");
      const std::string components = value(attributes, "NumberOfComponents");
      a.components = components.empty() ? 1 : std::stoi(components);
      a.offset = std::stoull(value(attributes, "offset"));
      for (const char* section : {"Points", "Cells", "CellData", "PointData"}) {
        const auto open = header.rfind(std::string("<") + section + ">", position);
        const auto close = header.rfind(std::string("</") + section + ">", position);
        if (open != std::string::npos && (close == std::string::npos || close < open))
          a.section = section;
      }
      CHECK(value(attributes, "format") == "appended");
      arrays[value(attributes, "Name")] = a;
    }
  }

  std::string attribute(const std::string& element, const std::string& key) const {
    std::smatch m;
    const std::regex r("<" + element + "[^>]* " + key + "=\"([^\"]*)\"");
    if (!std::regex_search(header, m, r)) return {};
    return m[1].str();
  }

  template <class T>
  std::vector<T> read(const std::string& name) const {
    const Array& a = arrays.at(name);
    std::uint64_t bytes = 0;
    std::memcpy(&bytes, data.data() + a.offset, sizeof bytes);
    REQUIRE(bytes % sizeof(T) == 0);
    std::vector<T> out(bytes / sizeof(T));
    std::memcpy(out.data(), data.data() + a.offset + sizeof bytes, bytes);
    return out;
  }
};

autan::io::Solution solution_of(const autan::Mesh& mesh) {
  autan::io::Solution s;
  s.run_metadata = R"({"identity": "test"})";
  const auto zones = autan::io::meshinfo_zones(mesh);
  for (std::size_t z = 0; z < zones.boundary.size(); ++z)
    s.boundary_bc_types.push_back("BCFarfield");
  for (const char* name : {"Density", "Pressure"}) {
    autan::io::Field f{.name = name, .values = {}};
    // Values whose bits a decimal round trip would change.
    for (index_t c = 0; c < mesh.n_cells(); ++c)
      f.values.push_back(std::ldexp(1.0 / 3.0, static_cast<int>(c % 7)) + name[0]);
    s.fields.push_back(f);
  }
  return s;
}

// Volume of a polyhedron from its faces as read, each face a fan of triangles about its node
// average: V = sum over triangles of (centroid . area vector) / 3, the divergence theorem. Also the
// sum of the terms' magnitudes, the scale of its round-off.
std::array<double, 2> volume(const std::vector<std::int64_t>& faces, std::size_t begin,
                             const std::vector<double>& points) {
  const auto p = [&](std::int64_t i, int k) { return points[3 * static_cast<std::size_t>(i) + k]; };
  std::size_t k = begin;
  const auto n_faces = faces[k++];
  double v = 0.0, scale = 0.0;
  for (std::int64_t f = 0; f < n_faces; ++f) {
    const auto n = faces[k++];
    std::array<double, 3> c{};
    for (std::int64_t i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d) c[d] += p(faces[k + i], d) / static_cast<double>(n);
    for (std::int64_t i = 0; i < n; ++i) {
      const auto a = faces[k + i], b = faces[k + (i + 1) % n];
      std::array<double, 3> u{}, w{}, g{};
      for (int d = 0; d < 3; ++d) {
        u[d] = p(a, d) - c[d];
        w[d] = p(b, d) - c[d];
        g[d] = (c[d] + p(a, d) + p(b, d)) / 3.0;
      }
      const std::array<double, 3> s{0.5 * (u[1] * w[2] - u[2] * w[1]),
                                    0.5 * (u[2] * w[0] - u[0] * w[2]),
                                    0.5 * (u[0] * w[1] - u[1] * w[0])};
      const double term = (g[0] * s[0] + g[1] * s[1] + g[2] * s[2]) / 3.0;
      v += term;
      scale += std::abs(term);
    }
    k += static_cast<std::size_t>(n);
  }
  return {v, scale};
}

}  // namespace

TEST_CASE("A run's solution converts to a VTU of outward polyhedra with its cell fields", "[vtu]") {
  // docs/contracts/vtu.md, read back by the reader above, against the solution.cgns the runtime
  // writes (write_solution): points and fields bit for bit, every cell a polyhedron whose faces are
  // the mesh's faces oriented outward, CellZone the MeshInfo zone index. The volume the faces
  // enclose against compute_geometry's is a second check of the same orientation.
  const std::string name = GENERATE("tet", "hex", "hybrid", "poly", "hanging");
  CAPTURE(name);
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / name / "polyMesh");
  autan::io::Solution solution = solution_of(mesh);
  solution.mean_fields.push_back(solution.fields[0]);
  for (double& v : solution.mean_fields[0].values) v = -v;
  fs::create_directories(temp_dir());
  const fs::path cgns = temp_dir() / (name + ".cgns");
  const fs::path out = temp_dir() / (name + ".vtu");
  autan::io::write_solution(cgns, mesh, solution);
  autan::io::write_vtu(cgns, out);

  const Vtu vtu(out);
  CHECK(vtu.header.find("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
                        "byte_order=\"LittleEndian\" header_type=\"UInt64\">") !=
        std::string::npos);
  REQUIRE(vtu.attribute("Piece", "NumberOfPoints") == std::to_string(mesh.n_points()));
  REQUIRE(vtu.attribute("Piece", "NumberOfCells") == std::to_string(mesh.n_cells()));

  // Points: the solution's coordinates, which are the mesh's.
  CHECK(vtu.arrays.at("Points").section == "Points");
  CHECK(vtu.arrays.at("Points").components == 3);
  const auto points = vtu.read<double>("Points");
  REQUIRE(points.size() == 3 * at(mesh.n_points()));
  for (index_t i = 0; i < mesh.n_points(); ++i) {
    CHECK(points[3 * at(i)] == mesh.x()[at(i)]);
    CHECK(points[3 * at(i) + 1] == mesh.y()[at(i)]);
    CHECK(points[3 * at(i) + 2] == mesh.z()[at(i)]);
  }

  // Cells: every one a polyhedron (VTK type 42) whose nodes are its faces' nodes.
  CHECK(vtu.arrays.at("Points").type == "Float64");
  for (const char* array : {"connectivity", "offsets", "faces", "faceoffsets"}) {
    CAPTURE(array);
    CHECK(vtu.arrays.at(array).section == "Cells");
    CHECK(vtu.arrays.at(array).type == "Int64");
  }
  CHECK(vtu.arrays.at("types").type == "UInt8");
  const auto types = vtu.read<std::uint8_t>("types");
  const auto connectivity = vtu.read<std::int64_t>("connectivity");
  const auto offsets = vtu.read<std::int64_t>("offsets");
  const auto faces = vtu.read<std::int64_t>("faces");
  const auto faceoffsets = vtu.read<std::int64_t>("faceoffsets");
  REQUIRE(types.size() == at(mesh.n_cells()));
  REQUIRE(offsets.size() == at(mesh.n_cells()));
  REQUIRE(faceoffsets.size() == at(mesh.n_cells()));
  CHECK(std::ranges::all_of(types, [](std::uint8_t t) { return t == 42; }));
  REQUIRE(offsets.back() == static_cast<std::int64_t>(connectivity.size()));
  REQUIRE(faceoffsets.back() == static_cast<std::int64_t>(faces.size()));
  const auto geometry = autan::compute_geometry(mesh);
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    CAPTURE(c);
    const std::size_t begin = c == 0 ? 0 : static_cast<std::size_t>(faceoffsets[at(c) - 1]);
    REQUIRE(faces[begin] == static_cast<std::int64_t>(mesh.cell_faces(c).size()));
    std::set<std::int64_t> face_nodes;
    std::size_t k = begin + 1;
    std::size_t matched = 0;
    for (std::int64_t f = 0; f < faces[begin]; ++f) {
      const auto n = static_cast<std::size_t>(faces[k++]);
      const std::vector<std::int64_t> read(faces.begin() + static_cast<std::ptrdiff_t>(k),
                                           faces.begin() + static_cast<std::ptrdiff_t>(k + n));
      face_nodes.insert(read.begin(), read.end());
      k += n;
      // Outward: mesh.md orients a face from its owner to its neighbour, so the face as read is the
      // mesh face's node cycle for its owner and that cycle reversed for its neighbour.
      for (const index_t mf : mesh.cell_faces(c)) {
        const auto nodes = mesh.face_nodes(mf);
        std::vector<std::int64_t> cycle(nodes.begin(), nodes.end());
        if (std::set<std::int64_t>(cycle.begin(), cycle.end()) !=
            std::set<std::int64_t>(read.begin(), read.end()))
          continue;
        if (mesh.owner()[at(mf)] != c) std::ranges::reverse(cycle);
        const auto first = std::ranges::find(cycle, read[0]);
        std::ranges::rotate(cycle, first);
        CHECK(cycle == read);
        ++matched;
      }
    }
    CHECK(matched == mesh.cell_faces(c).size());
    CHECK(static_cast<std::int64_t>(k) == faceoffsets[at(c)]);
    const std::size_t first = c == 0 ? 0 : static_cast<std::size_t>(offsets[at(c) - 1]);
    const std::set<std::int64_t> nodes(connectivity.begin() + static_cast<std::ptrdiff_t>(first),
                                       connectivity.begin() + offsets[at(c)]);
    CHECK(nodes == face_nodes);
    CHECK(nodes.size() == static_cast<std::size_t>(offsets[at(c)]) - first);
    const auto [v, scale] = volume(faces, begin, points);
    CHECK(std::abs(v - geometry.cell_volume[at(c)]) <= 64 * eps * scale);
  }

  // Cell data: the FlowSolution fields by name, the means as <name>Mean, CellZone.
  for (const auto& field : solution.fields) {
    CAPTURE(field.name);
    CHECK(vtu.arrays.at(field.name).section == "CellData");
    CHECK(vtu.arrays.at(field.name).type == "Float64");
    CHECK(vtu.read<double>(field.name) == field.values);
  }
  CHECK(vtu.read<double>("DensityMean") == solution.mean_fields[0].values);
  CHECK(vtu.arrays.at("CellZone").type == "Int32");
  const auto zone = vtu.read<std::int32_t>("CellZone");
  const auto listed = autan::io::meshinfo_zones(mesh).cell;
  REQUIRE(zone.size() == at(mesh.n_cells()));
  std::set<std::int32_t> seen;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    const auto it = std::ranges::find(listed, mesh.cell_zone()[at(c)]);
    CHECK(zone[at(c)] == static_cast<std::int32_t>(it - listed.begin()));
    seen.insert(zone[at(c)]);
  }
  CHECK(seen.size() == listed.size());
  CHECK(std::ranges::none_of(vtu.arrays,
                             [](const auto& a) { return a.second.section == "PointData"; }));
  CHECK_FALSE(vtu.arrays.contains("DensityPreviousLevel"));
}

TEST_CASE("autan-vtu refuses what is not a run's solution", "[vtu]") {
  // A mesh file holds no FlowSolution (E0147); a missing input or an unwritable output is E0130.
  // None leaves an output.
  fs::create_directories(temp_dir());
  const fs::path out = temp_dir() / "refused.vtu";
  fs::remove(out);
  const auto code = [&](const fs::path& in) {
    try {
      autan::io::write_vtu(in, out);
    } catch (const autan::Error& e) {
      return e.code();
    }
    return autan::ErrorCode{};
  };
  const autan::Mesh mesh = autan::io::read_mesh(data_dir / "hex" / "polyMesh");
  autan::io::Solution bare = solution_of(mesh);
  bare.fields.clear();
  const fs::path mesh_only = temp_dir() / "no-fields.cgns";
  autan::io::write_solution(mesh_only, mesh, bare);
  CHECK(code(mesh_only) == autan::ErrorCode::io_not_a_solution);
  CHECK(code(temp_dir() / "missing.cgns") == autan::ErrorCode::io_cannot_open);
  CHECK_FALSE(fs::exists(out));
  // An output that cannot be written (here a directory) is E0130, and no temporary file remains.
  const fs::path good = temp_dir() / "good.cgns";
  autan::io::write_solution(good, mesh, solution_of(mesh));
  const fs::path directory = temp_dir() / "a-directory.vtu";
  fs::create_directories(directory);
  try {
    autan::io::write_vtu(good, directory);
    FAIL("a directory was accepted as the output");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::io_cannot_open);
  }
  CHECK(std::ranges::none_of(fs::directory_iterator(temp_dir()), [](const fs::directory_entry& e) {
    return e.path().filename().string().starts_with("a-directory.vtu.tmp");
  }));
}
