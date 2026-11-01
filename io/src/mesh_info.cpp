#include "autan/io/mesh_info.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <picosha2.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/quality.hpp"
#include "cursor.hpp"

namespace autan::io {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

void hash_file(picosha2::hash256_one_by_one& hasher, const fs::path& path, bool framed) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error(ErrorCode::io_cannot_open, "cannot open '" + path.string() + "'");
  if (framed) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) throw Error(ErrorCode::io_cannot_open, "cannot stat '" + path.string() + "'");
    const std::string frame = path.filename().string() + "\n" + std::to_string(size) + "\n";
    hasher.process(frame.begin(), frame.end());
  }
  std::vector<char> buffer(std::size_t{1} << 20U);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hasher.process(buffer.begin(), buffer.begin() + in.gcount());
  }
  if (in.bad()) throw Error(ErrorCode::io_cannot_open, "cannot read '" + path.string() + "'");
}

const char* bc_type_name(BcKind kind) {
  switch (kind) {
    case BcKind::wall:
      return "wall";
    case BcKind::thermal_wall:
      return "thermal-wall";
    case BcKind::slip:
      return "slip";
    case BcKind::symmetry:
      return "symmetry";
    case BcKind::far_field:
      return "farfield";
    case BcKind::inlet:
      return "inlet";
    case BcKind::outlet:
      return "outlet";
    case BcKind::periodic:
      return "periodic";
    case BcKind::unspecified:
      return nullptr;
  }
  return nullptr;
}

Json name_or_null(const char* name) { return name == nullptr ? Json(nullptr) : Json(name); }
Json string_or_null(const std::string& s) { return s.empty() ? Json(nullptr) : Json(s); }

// A cell is one of the four standard shapes when its face sizes, vertex
// count and vertex degrees (faces per vertex) are that shape's: the only
// simple polyhedra with six quadrilaterals, or two triangles and three
// quadrilaterals, are the hexahedron and the triangular prism; a pyramid's
// apex is its one vertex of degree 4, off its quadrilateral.
std::size_t shape_index(const Mesh& mesh, index_t cell, std::vector<index_t>& nodes) {
  const auto faces = mesh.cell_faces(cell);
  int triangles = 0, quads = 0;
  nodes.clear();
  index_t quad_face = -1;
  for (const index_t f : faces) {
    const auto fn = mesh.face_nodes(f);
    if (fn.size() == 3) ++triangles;
    if (fn.size() == 4) {
      ++quads;
      quad_face = f;
    }
    nodes.insert(nodes.end(), fn.begin(), fn.end());
  }
  std::ranges::sort(nodes);
  std::vector<int> degrees;
  for (std::size_t i = 0; i < nodes.size();) {
    std::size_t j = i;
    while (j < nodes.size() && nodes[j] == nodes[i]) ++j;
    degrees.push_back(static_cast<int>(j - i));
    i = j;
  }
  const auto count = [&](int d) { return std::count(degrees.begin(), degrees.end(), d); };
  const std::size_t n = faces.size();
  const std::size_t v = degrees.size();
  if (n == 4 && triangles == 4 && v == 4) return 0;
  if (n == 5 && triangles == 4 && quads == 1 && v == 5 && count(3) == 4 && count(4) == 1) {
    for (const index_t p : mesh.face_nodes(quad_face)) {
      const auto range = std::ranges::equal_range(nodes, p);
      if (range.size() != 3) return 4;
    }
    return 1;
  }
  if (n == 5 && triangles == 2 && quads == 3 && v == 6 && count(3) == 6) return 2;
  if (n == 6 && quads == 6 && v == 8 && count(3) == 8) return 3;
  return 4;
}

}  // namespace

MeshInfoZones meshinfo_zones(const Mesh& mesh) {
  MeshInfoZones zones;
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].boundary) zones.boundary.push_back(static_cast<index_t>(z));
  }
  std::vector<char> used(mesh.cell_zones().size(), 0);
  for (const index_t z : mesh.cell_zone()) used[at(z)] = 1;
  for (std::size_t z = 0; z < used.size(); ++z) {
    if (used[z] != 0) zones.cell.push_back(static_cast<index_t>(z));
  }
  return zones;
}

std::string mesh_hash(const fs::path& path) {
  picosha2::hash256_one_by_one hasher;
  std::error_code ec;
  if (fs::is_directory(path, ec)) {
    const fs::path dir = polymesh_directory(path);
    for (const char* name : {"points", "faces", "owner", "neighbour", "boundary", "cellZones"}) {
      const fs::path file = dir / name;
      if (std::string_view(name) == "cellZones" && !fs::exists(file, ec)) continue;
      hash_file(hasher, file, true);
    }
  } else {
    hash_file(hasher, path, false);
  }
  hasher.finish();
  return picosha2::get_hash_hex_string(hasher);
}

std::string mesh_info_json(const Mesh& mesh, const std::string& sha256) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "MeshInfo describes an unpartitioned mesh");
  }
  const Geometry geometry = compute_geometry(mesh);
  const MeshQuality quality = compute_quality(mesh, geometry);

  std::array<std::int64_t, 5> by_shape{};
  std::vector<index_t> scratch;
  std::size_t max_faces = 0;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    ++by_shape[shape_index(mesh, c, scratch)];
    max_faces = std::max(max_faces, mesh.cell_faces(c).size());
  }

  const MeshInfoZones zones = meshinfo_zones(mesh);
  Json boundary = Json::array();
  const auto offsets = mesh.boundary_zone_offsets();
  for (const index_t zi : zones.boundary) {
    const auto z = at(zi);
    const FaceZone& zone = mesh.face_zones()[z];
    boundary.push_back(Json{{"name", zone.name},
                            {"type", name_or_null(bc_type_name(zone.kind))},
                            {"native_type", string_or_null(zone.bc_type)},
                            {"faces", offsets[z + 1] - offsets[z]}});
  }

  std::vector<std::int64_t> cells_in_zone(mesh.cell_zones().size(), 0);
  for (const index_t z : mesh.cell_zone()) ++cells_in_zone[at(z)];
  Json cell_zones = Json::array();
  for (const index_t zi : zones.cell) {
    const auto z = at(zi);
    const CellZone& zone = mesh.cell_zones()[z];
    const char* type = zone.kind == CellKind::fluid   ? "fluid"
                       : zone.kind == CellKind::solid ? "solid"
                                                      : nullptr;
    cell_zones.push_back(Json{{"name", zone.name},
                              {"type", name_or_null(type)},
                              {"native_type", string_or_null(zone.type)},
                              {"cells", cells_in_zone[z]}});
  }

  std::map<std::pair<std::string, std::string>, std::int64_t> shared;
  for (index_t f = 0; f < mesh.n_interior_faces(); ++f) {
    const auto& a = mesh.cell_zones()[at(mesh.cell_zone()[at(mesh.owner()[at(f)])])].name;
    const auto& b = mesh.cell_zones()[at(mesh.cell_zone()[at(mesh.neighbour()[at(f)])])].name;
    if (a != b) ++shared[std::minmax(a, b)];
  }
  Json interfaces = Json::array();
  for (const auto& [pair, faces] : shared) {
    interfaces.push_back(Json{{"zones", {pair.first, pair.second}}, {"faces", faces}});
  }

  Json periodic = Json::array();
  for (const auto& pair : mesh.periodic_pairs()) {
    periodic.push_back(
        Json{{"zones",
              {mesh.face_zones()[at(pair.zone_a)].name, mesh.face_zones()[at(pair.zone_b)].name}},
             {"transform",
              {{"kind", "translation"},
               {"vector", {pair.translation[0], pair.translation[1], pair.translation[2]}}}},
             {"faces", offsets[at(pair.zone_a) + 1] - offsets[at(pair.zone_a)]}});
  }

  const double max_angle = *std::ranges::max_element(quality.non_orthogonality);
  const Json doc{
      {"schema", "autan.mesh-info/1"},
      {"source",
       {{"path", mesh.source().path}, {"format", mesh.source().format}, {"sha256", sha256}}},
      {"counts",
       {{"cells", mesh.n_cells()},
        {"faces", mesh.n_faces()},
        {"nodes", mesh.n_points()},
        {"cells_by_type",
         {{"tetra", by_shape[0]},
          {"pyramid", by_shape[1]},
          {"prism", by_shape[2]},
          {"hexa", by_shape[3]},
          {"polyhedron", by_shape[4]}}}}},
      {"boundary_zones", boundary},
      {"cell_zones", cell_zones},
      {"interfaces", interfaces},
      {"periodic", periodic},
      {"quality", {{"max_non_orthogonality_deg", max_angle}, {"max_faces_per_cell", max_faces}}},
  };
  return doc.dump(2);
}

std::string surface_bytes(const Mesh& mesh, const std::string& sha256) {
  static_assert(std::endian::native == std::endian::little, "autan.surface/1 is little-endian");
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "a surface describes an unpartitioned mesh");
  }
  const MeshInfoZones zones = meshinfo_zones(mesh);
  const auto offsets = mesh.boundary_zone_offsets();
  std::vector<std::int64_t> node_of(at(mesh.n_points()), -1);
  std::vector<double> positions;
  std::vector<std::uint64_t> face_offsets{0}, face_nodes;
  Json zone_list = Json::array();
  for (const index_t zi : zones.boundary) {
    const auto z = at(zi);
    const FaceZone& zone = mesh.face_zones()[z];
    zone_list.push_back(Json{{"name", zone.name},
                             {"type", name_or_null(bc_type_name(zone.kind))},
                             {"native_type", string_or_null(zone.bc_type)},
                             {"first_face", face_offsets.size() - 1},
                             {"faces", offsets[z + 1] - offsets[z]}});
    for (index_t f = offsets[z]; f < offsets[z + 1]; ++f) {
      for (const index_t p : mesh.face_nodes(f)) {
        auto& node = node_of[at(p)];
        if (node < 0) {
          node = static_cast<std::int64_t>(positions.size() / 3);
          const auto x = mesh.x()[at(p)], y = mesh.y()[at(p)], zc = mesh.z()[at(p)];
          positions.insert(positions.end(), {x, y, zc});
        }
        face_nodes.push_back(static_cast<std::uint64_t>(node));
      }
      face_offsets.push_back(face_nodes.size());
    }
  }
  constexpr std::uint64_t u32_max = std::numeric_limits<std::uint32_t>::max();
  if (face_nodes.size() > u32_max || positions.size() / 3 > u32_max) {
    throw Error(ErrorCode::mesh_too_large, "the boundary surface exceeds 32-bit indices");
  }
  const auto to_u32 = [](const std::vector<std::uint64_t>& v) {
    std::vector<std::uint32_t> out(v.begin(), v.end());
    return out;
  };
  const std::vector<std::uint32_t> offsets32 = to_u32(face_offsets), nodes32 = to_u32(face_nodes);

  struct Buffer {
    const char* name;
    const char* dtype;
    const void* data;
    std::size_t length, bytes;
  };
  const std::array<Buffer, 3> buffers{{
      {"positions", "f64", positions.data(), positions.size(), positions.size() * sizeof(double)},
      {"face_offsets", "u32", offsets32.data(), offsets32.size(),
       offsets32.size() * sizeof(std::uint32_t)},
      {"face_nodes", "u32", nodes32.data(), nodes32.size(), nodes32.size() * sizeof(std::uint32_t)},
  }};
  const auto aligned = [](std::size_t n) { return (n + 7) / 8 * 8; };
  Json layout = Json::array();
  std::size_t offset = 0;
  for (const Buffer& b : buffers) {
    layout.push_back(
        Json{{"name", b.name}, {"dtype", b.dtype}, {"offset", offset}, {"length", b.length}});
    offset = aligned(offset + b.bytes);
  }
  const Json header{{"schema", "autan.surface/1"},
                    {"source", {{"sha256", sha256}}},
                    {"nodes", positions.size() / 3},
                    {"faces", face_offsets.size() - 1},
                    {"zones", zone_list},
                    {"buffers", layout}};
  std::string text = header.dump();
  text.append(aligned(4 + text.size()) - 4 - text.size(), ' ');
  std::string out(4 + text.size() + offset, '\0');
  const auto length = static_cast<std::uint32_t>(text.size());
  std::memcpy(out.data(), &length, 4);
  std::memcpy(out.data() + 4, text.data(), text.size());
  std::size_t at_byte = 4 + text.size();
  for (const Buffer& b : buffers) {
    if (b.bytes > 0) std::memcpy(out.data() + at_byte, b.data, b.bytes);
    at_byte += aligned(b.bytes);
  }
  return out;
}

}  // namespace autan::io
