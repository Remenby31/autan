#include "autan/io/vtu.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"

namespace autan::io {
namespace {

static_assert(std::endian::native == std::endian::little,
              "vtu.md: the appended block is written in the host's byte order, little-endian");

constexpr std::uint8_t vtk_polyhedron = 42;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

std::string xml_escaped(std::string_view text) {
  std::string out;
  for (const char c : text) {
    switch (c) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      default:
        out += c;
    }
  }
  return out;
}

// A cell's distinct nodes in the order its faces first list them; `stamp` marks the last cell
// that listed each node.
void cell_nodes(const Mesh& mesh, index_t cell, std::vector<index_t>& stamp,
                std::vector<std::int64_t>& out) {
  out.clear();
  for (const index_t f : mesh.cell_faces(cell)) {
    for (const index_t n : mesh.face_nodes(f)) {
      if (stamp[at(n)] != cell) {
        stamp[at(n)] = cell;
        out.push_back(n);
      }
    }
  }
}

// vtu.md: the cell's face count, then each face's node count and nodes, its normal outward. The
// mesh orients each face from its owner to its neighbour, so the neighbour reverses it.
void cell_faces(const Mesh& mesh, index_t cell, std::vector<std::int64_t>& out) {
  out.clear();
  const auto faces = mesh.cell_faces(cell);
  out.push_back(static_cast<std::int64_t>(faces.size()));
  for (const index_t f : faces) {
    const auto nodes = mesh.face_nodes(f);
    out.push_back(static_cast<std::int64_t>(nodes.size()));
    if (mesh.owner()[at(f)] == cell) {
      out.insert(out.end(), nodes.begin(), nodes.end());
    } else {
      out.insert(out.end(), nodes.rbegin(), nodes.rend());
    }
  }
}

// The output file, created exclusively under a name of its own beside `out` ("x": fails on any
// existing file or link), so that no other writer's file and no link target is ever truncated.
class Output {
 public:
  explicit Output(const std::filesystem::path& out) {
    std::random_device random;
    for (int attempt = 0; attempt < 16 && file_ == nullptr; ++attempt) {
      path_ = out;
      path_ += ".tmp-" + std::to_string(random()) + std::to_string(random());
      file_ = std::fopen(path_.string().c_str(), "wbx");
    }
    if (file_ == nullptr) {
      throw Error(ErrorCode::io_cannot_open, out.string() + ": cannot be written");
    }
  }
  Output(const Output&) = delete;
  Output& operator=(const Output&) = delete;
  ~Output() {
    if (file_ != nullptr) {
      std::fclose(file_);
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }

  void text(std::string_view s) { put(s.data(), s.size()); }
  void begin(std::uint64_t bytes) { put(&bytes, sizeof bytes); }

  template <class T>
  void values(const T* data, std::size_t count) {
    put(data, count * sizeof(T));
  }

  // Closes the file and renames it to `out`; E0130 when either fails.
  void publish(const std::filesystem::path& out) {
    const bool written = !failed_ && std::fclose(file_) == 0;
    file_ = nullptr;
    std::error_code error;
    if (written) std::filesystem::rename(path_, out, error);
    if (!written || error) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
      throw Error(ErrorCode::io_cannot_open, out.string() + ": cannot be written");
    }
  }

 private:
  void put(const void* data, std::size_t bytes) {
    if (bytes > 0 && std::fwrite(data, 1, bytes, file_) != bytes) failed_ = true;
  }
  std::filesystem::path path_;
  std::FILE* file_ = nullptr;
  bool failed_ = false;
};

}  // namespace

void write_vtu(const std::filesystem::path& solution, const std::filesystem::path& out) {
  std::error_code status;
  if (!std::filesystem::is_regular_file(solution, status)) {
    throw Error(ErrorCode::io_cannot_open, solution.string() + ": no such file");
  }
  const Mesh mesh = read_cgns(solution);
  const StoredSolution stored = read_solution(solution);
  if (stored.fields.empty()) {
    throw Error(ErrorCode::io_not_a_solution,
                solution.string() + ": no FlowSolution, not a run's solution.cgns");
  }
  const index_t cells = mesh.n_cells();
  const index_t points = mesh.n_points();

  // First pass: the sizes of the cell arrays, so that every offset is known before writing.
  std::vector<index_t> stamp(at(points), -1);
  std::vector<std::int64_t> scratch;
  std::uint64_t connectivity = 0;
  std::uint64_t faces = 0;
  for (index_t c = 0; c < cells; ++c) {
    cell_nodes(mesh, c, stamp, scratch);
    connectivity += scratch.size();
    cell_faces(mesh, c, scratch);
    faces += scratch.size();
  }

  struct Array {
    std::string section, name, type;
    int components;
    std::uint64_t bytes;
  };
  const auto n = static_cast<std::uint64_t>(cells);
  std::vector<Array> arrays{
      {"Points", "Points", "Float64", 3, 3 * static_cast<std::uint64_t>(points) * 8},
      {"Cells", "connectivity", "Int64", 1, connectivity * 8},
      {"Cells", "offsets", "Int64", 1, n * 8},
      {"Cells", "types", "UInt8", 1, n},
      {"Cells", "faces", "Int64", 1, faces * 8},
      {"Cells", "faceoffsets", "Int64", 1, n * 8},
  };
  for (const Field& f : stored.fields) arrays.push_back({"CellData", f.name, "Float64", 1, n * 8});
  for (const Field& f : stored.mean_fields) {
    arrays.push_back({"CellData", f.name + "Mean", "Float64", 1, n * 8});
  }
  arrays.push_back({"CellData", "CellZone", "Int32", 1, n * 4});

  std::string header =
      "<?xml version=\"1.0\"?>\n"
      "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
      "header_type=\"UInt64\">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\"" +
      std::to_string(points) + "\" NumberOfCells=\"" + std::to_string(cells) + "\">\n";
  std::uint64_t offset = 0;
  std::string section;
  for (const Array& a : arrays) {
    if (a.section != section) {
      if (!section.empty()) header += "</" + section + ">\n";
      section = a.section;
      header += "<" + section + ">\n";
    }
    header += R"(<DataArray type=")" + a.type + R"(" Name=")" + xml_escaped(a.name) +
              R"(" NumberOfComponents=")" + std::to_string(a.components) +
              R"(" format="appended" offset=")" + std::to_string(offset) + "\"/>\n";
    offset += 8 + a.bytes;
  }
  header += "</" + section + ">\n</Piece>\n</UnstructuredGrid>\n<AppendedData encoding=\"raw\">\n_";

  Output file(out);
  file.text(header);
  file.begin(arrays[0].bytes);
  for (index_t i = 0; i < points; ++i) {
    const double p[3] = {mesh.x()[at(i)], mesh.y()[at(i)], mesh.z()[at(i)]};
    file.values(p, 3);
  }
  std::ranges::fill(stamp, -1);
  file.begin(arrays[1].bytes);
  for (index_t c = 0; c < cells; ++c) {
    cell_nodes(mesh, c, stamp, scratch);
    file.values(scratch.data(), scratch.size());
  }
  std::ranges::fill(stamp, -1);
  file.begin(arrays[2].bytes);
  std::int64_t end = 0;
  for (index_t c = 0; c < cells; ++c) {
    cell_nodes(mesh, c, stamp, scratch);
    end += static_cast<std::int64_t>(scratch.size());
    file.values(&end, 1);
  }
  file.begin(arrays[3].bytes);
  for (index_t c = 0; c < cells; ++c) file.values(&vtk_polyhedron, 1);
  file.begin(arrays[4].bytes);
  for (index_t c = 0; c < cells; ++c) {
    cell_faces(mesh, c, scratch);
    file.values(scratch.data(), scratch.size());
  }
  file.begin(arrays[5].bytes);
  end = 0;
  for (index_t c = 0; c < cells; ++c) {
    cell_faces(mesh, c, scratch);
    end += static_cast<std::int64_t>(scratch.size());
    file.values(&end, 1);
  }
  for (const auto* list : {&stored.fields, &stored.mean_fields}) {
    for (const Field& f : *list) {
      file.begin(n * 8);
      file.values(f.values.data(), f.values.size());
    }
  }
  // mesh-info.md: the cell zones in MeshInfo's order.
  const auto listed = meshinfo_zones(mesh).cell;
  std::vector<std::int32_t> position(mesh.cell_zones().size(), -1);
  for (std::size_t i = 0; i < listed.size(); ++i) {
    position[at(listed[i])] = static_cast<std::int32_t>(i);
  }
  file.begin(n * 4);
  for (const index_t z : mesh.cell_zone()) file.values(&position[at(z)], 1);
  file.text("\n</AppendedData>\n</VTKFile>\n");
  file.publish(out);
}

}  // namespace autan::io
