#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/io/write_polymesh.hpp"

namespace autan::io {
namespace {

namespace fs = std::filesystem;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

std::int32_t label(std::int64_t v) {
  if (v < 0 || v > std::numeric_limits<std::int32_t>::max()) {
    throw Error(ErrorCode::mesh_too_large,
                "a polyMesh label exceeds 2^31 - 1: " + std::to_string(v));
  }
  return static_cast<std::int32_t>(v);
}

// One polyMesh file: its FoamFile header, then the body `write` gives.
template <class Body>
void foam_file(const fs::path& path, const char* object_class, const char* object, Body write) {
  std::ofstream out(path, std::ios::binary);
  out << "FoamFile\n{\n    version     2.0;\n    format      binary;\n"
      << "    arch        \"LSB;label=32;scalar=64\";\n    class       " << object_class
      << ";\n    object      " << object << ";\n}\n\n";
  write(out);
  out.close();
  if (out.fail()) throw Error(ErrorCode::io_cannot_open, "cannot write '" + path.string() + "'");
}

// `N(<raw>)`: a binary list of 32-bit labels.
void labels(std::ofstream& out, std::span<const std::int32_t> v) {
  out << v.size() << "\n(";
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(std::int32_t)));
  out << ")\n";
}

// `name` as an OpenFOAM word: each run of characters a word cannot hold becomes one '_'.
std::string word(const std::string& name) {
  std::string out;
  bool run = false;
  for (const char c : name) {
    const bool bad = std::isspace(static_cast<unsigned char>(c)) != 0 || c == '"' || c == '\'' ||
                     c == '/' || c == ';' || c == '{' || c == '}' || c == '(' || c == ')';
    if (bad && !run) out.push_back('_');
    if (!bad) out.push_back(c);
    run = bad;
  }
  return out.empty() ? "_" : out;
}

}  // namespace

std::map<std::string, std::string> write_polymesh(const fs::path& directory, const Mesh& mesh) {
  if (mesh.n_owned_cells() != mesh.n_cells() || !mesh.periodic_pairs().empty()) {
    throw Error(ErrorCode::io_unsupported,
                "a polyMesh is written for an unpartitioned mesh without periodic pairs");
  }
  std::error_code ec;
  fs::create_directories(directory, ec);
  if (ec) throw Error(ErrorCode::io_cannot_open, "cannot create '" + directory.string() + "'");
  (void)label(mesh.n_cells());
  (void)label(static_cast<std::int64_t>(mesh.face_nodes().size()));

  foam_file(directory / "points", "vectorField", "points", [&](std::ofstream& out) {
    std::vector<double> xyz(3 * at(mesh.n_points()));
    for (std::size_t p = 0; p < at(mesh.n_points()); ++p) {
      xyz[3 * p] = mesh.x()[p];
      xyz[3 * p + 1] = mesh.y()[p];
      xyz[3 * p + 2] = mesh.z()[p];
    }
    out << mesh.n_points() << "\n(";
    out.write(reinterpret_cast<const char*>(xyz.data()),
              static_cast<std::streamsize>(xyz.size() * sizeof(double)));
    out << ")\n";
  });
  foam_file(directory / "faces", "faceCompactList", "faces", [&](std::ofstream& out) {
    std::vector<std::int32_t> offsets, nodes;
    for (const offset_t o : mesh.face_node_offsets()) offsets.push_back(label(o));
    for (const index_t n : mesh.face_nodes()) nodes.push_back(label(n));
    labels(out, offsets);
    out << "\n";
    labels(out, nodes);
  });
  foam_file(directory / "owner", "labelList", "owner", [&](std::ofstream& out) {
    std::vector<std::int32_t> owner;
    for (const index_t c : mesh.owner()) owner.push_back(label(c));
    labels(out, owner);
  });
  foam_file(directory / "neighbour", "labelList", "neighbour", [&](std::ofstream& out) {
    std::vector<std::int32_t> neighbour;
    for (const index_t c : mesh.neighbour()) neighbour.push_back(label(c));
    labels(out, neighbour);
  });
  // The names written, and those that changed.
  std::map<std::string, std::string> renamed;
  const auto names = [&](const auto& zones, const auto& keep) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto& zone : zones) {
      if (!keep(zone)) {
        out.emplace_back();
        continue;
      }
      out.push_back(word(zone.name));
      if (out.back() != zone.name) renamed[zone.name] = out.back();
      if (!seen.insert(out.back()).second) {
        throw Error(ErrorCode::mesh_duplicate_zone_name,
                    "two zones are written as '" + out.back() + "' in a polyMesh");
      }
    }
    return out;
  };
  const auto patch_names = names(mesh.face_zones(), [](const FaceZone& z) { return z.boundary; });
  const auto zone_names = names(mesh.cell_zones(), [](const CellZone&) { return true; });
  const auto offsets = mesh.boundary_zone_offsets();
  std::size_t patches = 0;
  for (const auto& zone : mesh.face_zones()) patches += zone.boundary ? 1 : 0;
  foam_file(directory / "boundary", "polyBoundaryMesh", "boundary", [&](std::ofstream& out) {
    out << patches << "\n(\n";
    for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
      const FaceZone& zone = mesh.face_zones()[z];
      if (!zone.boundary) continue;
      const char* type = zone.kind == BcKind::wall || zone.kind == BcKind::thermal_wall ? "wall"
                         : zone.kind == BcKind::symmetry ? "symmetryPlane"
                                                         : "patch";
      out << "    " << patch_names[z] << "\n    {\n        type            " << type
          << ";\n        nFaces          " << offsets[z + 1] - offsets[z]
          << ";\n        startFace       " << offsets[z] << ";\n    }\n";
    }
    out << ")\n";
  });
  foam_file(directory / "cellZones", "regIOobject", "cellZones", [&](std::ofstream& out) {
    out << mesh.cell_zones().size() << "\n(\n";
    for (std::size_t z = 0; z < mesh.cell_zones().size(); ++z) {
      std::vector<std::int32_t> cells;
      for (index_t c = 0; c < mesh.n_cells(); ++c) {
        if (at(mesh.cell_zone()[at(c)]) == z) cells.push_back(label(c));
      }
      out << zone_names[z] << "\n{\n    type cellZone;\n    cellLabels List<label> ";
      labels(out, cells);
      out << ";\n}\n";
    }
    out << ")\n";
  });
  return renamed;
}

}  // namespace autan::io
