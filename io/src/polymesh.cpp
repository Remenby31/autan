// OpenFOAM polyMesh reader (OpenFOAM User Guide, section 4.1 "Mesh
// description"): points, faces, owner, neighbour, boundary and optional
// cellZones, ASCII or binary, 32- or 64-bit labels, 32- or 64-bit scalars.
// faceZones, pointZones and sets are auxiliary selections, not zones, and are
// not read.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "autan/io/read_mesh.hpp"
#include "cursor.hpp"

namespace autan::io {
namespace {

namespace fs = std::filesystem;

void skip_blank(Cursor& c) {
  for (;;) {
    c.skip_space();
    if (c.accept("//")) {
      (void)c.line();
    } else if (c.accept("/*")) {
      while (!c.accept("*/")) c.advance();
    } else {
      return;
    }
  }
}

// Skips an entry value up to its ';', or a braced sub-dictionary.
void skip_entry(Cursor& c) {
  skip_blank(c);
  int depth = 0;
  bool quoted = false;
  for (;;) {
    if (c.at_end()) c.fail("unterminated entry");
    const char ch = c.peek_raw();
    c.advance();
    if (quoted) {
      quoted = ch != '"';
    } else if (ch == '"') {
      quoted = true;
    } else if (ch == '{' || ch == '(') {
      ++depth;
    } else if (ch == '}' || ch == ')') {
      --depth;
      if (depth == 0 && ch == '}') return;
    } else if (ch == ';' && depth == 0) {
      return;
    }
  }
}

struct Header {
  bool binary = false;
  int label_bytes = 4;
  int scalar_bytes = 8;
  std::string object_class;
};

std::string unquote(std::string_view v) {
  if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
  return std::string(v);
}

// Reads `key value;` entries of the FoamFile dictionary.
Header read_header(Cursor& c) {
  skip_blank(c);
  c.expect("FoamFile");
  c.expect('{');
  Header h;
  for (;;) {
    skip_blank(c);
    if (c.accept('}')) break;
    const std::string key(c.token("{};"));
    skip_blank(c);
    if (c.peek() == '{') {
      skip_entry(c);
      continue;
    }
    std::string value;
    bool quoted = false;
    while (!c.at_end() && (quoted || c.peek_raw() != ';')) {
      if (c.peek_raw() == '"') quoted = !quoted;
      value += c.peek_raw();
      c.advance();
    }
    c.expect(';');
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
    value = unquote(value);
    if (key == "format") {
      if (value != "ascii" && value != "binary") c.fail("unknown format '" + value + "'");
      h.binary = value == "binary";
    } else if (key == "class") {
      h.object_class = value;
    } else if (key == "arch") {
      if (value.find("label=64") != std::string::npos) h.label_bytes = 8;
      if (value.find("scalar=32") != std::string::npos) h.scalar_bytes = 4;
      if (value.find("MSB") != std::string::npos) c.unsupported("big-endian binary files");
    }
  }
  return h;
}

class FoamFile {
 public:
  FoamFile(const fs::path& path, std::string_view expected_class)
      : data_(read_file(path.string())),
        cursor_(data_, path.string()),
        header_(read_header(cursor_)) {
    if (!expected_class.empty() && header_.object_class != expected_class) {
      cursor_.fail("class '" + header_.object_class + "', expected '" +
                   std::string(expected_class) + "'");
    }
  }

  Cursor& cursor() { return cursor_; }
  [[nodiscard]] const Header& header() const { return header_; }

  std::size_t list_size() {
    skip_blank(cursor_);
    const auto n = cursor_.integer<std::size_t>(10, "({/");
    skip_blank(cursor_);
    return n;
  }

  // `N ( v0 v1 ... )`, `N { v }` or, in binary files, `N ( <raw> )`.
  std::vector<std::int64_t> labels() {
    skip_blank(cursor_);
    if (cursor_.accept("List<label>")) skip_blank(cursor_);
    const std::size_t n = list_size();
    std::vector<std::int64_t> out(n);
    if (cursor_.accept('{')) {
      std::fill(out.begin(), out.end(), cursor_.integer<std::int64_t>(10, "}"));
      cursor_.expect('}');
      return out;
    }
    cursor_.expect('(');
    if (header_.binary && n > 0) {
      if (header_.label_bytes == 4) {
        std::vector<std::int32_t> raw(n);
        cursor_.binary(raw.data(), n);
        std::copy(raw.begin(), raw.end(), out.begin());
      } else {
        cursor_.binary(out.data(), n);
      }
    } else {
      for (auto& v : out) v = cursor_.integer<std::int64_t>();
    }
    cursor_.expect(')');
    return out;
  }

  std::vector<double> points() {
    const std::size_t n = list_size();
    std::vector<double> out(3 * n);
    cursor_.expect('(');
    if (header_.binary && n > 0) {
      if (header_.scalar_bytes == 8) {
        cursor_.binary(out.data(), out.size());
      } else {
        std::vector<float> raw(out.size());
        cursor_.binary(raw.data(), raw.size());
        std::copy(raw.begin(), raw.end(), out.begin());
      }
    } else {
      for (std::size_t p = 0; p < n; ++p) {
        cursor_.expect('(');
        for (int k = 0; k < 3; ++k) out[3 * p + static_cast<std::size_t>(k)] = cursor_.real();
        cursor_.expect(')');
      }
    }
    cursor_.expect(')');
    return out;
  }

 private:
  std::string data_;
  Cursor cursor_;
  Header header_;
};

struct Faces {
  std::vector<offset_t> offsets;
  std::vector<index_t> nodes;
};

Faces read_faces(const fs::path& path) {
  FoamFile file(path, "");
  Cursor& c = file.cursor();
  Faces faces;
  auto narrow = [&](std::int64_t v) {
    if (v < 0 || v > std::numeric_limits<index_t>::max()) c.fail("point label out of range");
    return static_cast<index_t>(v);
  };
  if (file.header().object_class == "faceCompactList") {
    const auto offsets = file.labels();
    const auto nodes = file.labels();
    if (offsets.empty() || offsets.front() != 0 ||
        static_cast<std::size_t>(offsets.back()) != nodes.size()) {
      c.fail("faceCompactList offsets do not match its labels");
    }
    faces.offsets.assign(offsets.begin(), offsets.end());
    faces.nodes.reserve(nodes.size());
    for (const auto v : nodes) faces.nodes.push_back(narrow(v));
  } else if (file.header().object_class == "faceList") {
    if (file.header().binary) c.unsupported("binary faceList (OpenFOAM writes faceCompactList)");
    const std::size_t n = file.list_size();
    c.expect('(');
    faces.offsets.reserve(n + 1);
    faces.offsets.push_back(0);
    for (std::size_t f = 0; f < n; ++f) {
      const auto k = c.integer<std::size_t>();
      c.expect('(');
      for (std::size_t i = 0; i < k; ++i) faces.nodes.push_back(narrow(c.integer<std::int64_t>()));
      c.expect(')');
      faces.offsets.push_back(static_cast<offset_t>(faces.nodes.size()));
    }
    c.expect(')');
  } else {
    c.fail("class '" + file.header().object_class + "', expected faceList or faceCompactList");
  }
  return faces;
}

// Entries of one dictionary inside a list (boundary patch, cell zone).
struct NamedDict {
  std::string name;
  std::map<std::string, std::string> values;
  std::vector<std::int64_t> labels;
};

std::vector<NamedDict> read_dict_list(FoamFile& file, std::string_view label_key) {
  Cursor& c = file.cursor();
  const std::size_t n = file.list_size();
  c.expect('(');
  std::vector<NamedDict> out(n);
  for (auto& d : out) {
    skip_blank(c);
    d.name = std::string(c.token("{"));
    skip_blank(c);
    c.expect('{');
    for (;;) {
      skip_blank(c);
      if (c.accept('}')) break;
      const std::string key(c.token("{};"));
      skip_blank(c);
      if (key == label_key) {
        d.labels = file.labels();
        skip_blank(c);
        c.expect(';');
        continue;
      }
      if (c.peek() == '{') {
        skip_entry(c);
        continue;
      }
      std::string value;
      while (!c.at_end() && c.peek_raw() != ';') {
        value += c.peek_raw();
        c.advance();
      }
      c.expect(';');
      while (!value.empty() &&
             (value.back() == ' ' || value.back() == '\t' || value.back() == '\n')) {
        value.pop_back();
      }
      d.values[key] = value;
    }
  }
  skip_blank(c);
  c.expect(')');
  return out;
}

BcKind patch_kind(std::string_view type) {
  if (type == "wall") return BcKind::wall;
  if (type == "symmetry" || type == "symmetryPlane") return BcKind::symmetry;
  return BcKind::unspecified;
}

index_t to_index(const Cursor& c, std::int64_t v, std::int64_t limit, const char* what) {
  if (v < 0 || v >= limit) c.fail(std::string(what) + " out of range");
  return static_cast<index_t>(v);
}

}  // namespace

fs::path polymesh_directory(const fs::path& path) {
  std::error_code ec;
  if (fs::exists(path / "points", ec) || fs::exists(path / "points.gz", ec)) return path;
  if (fs::is_directory(path / "constant" / "polyMesh", ec)) return path / "constant" / "polyMesh";
  return path;
}

Mesh read_polymesh(const fs::path& directory) {
  const fs::path dir = polymesh_directory(directory);
  std::error_code ec;
  for (const char* name : {"points", "faces", "owner", "neighbour", "boundary"}) {
    if (!fs::exists(dir / name, ec)) {
      if (fs::exists(dir / (std::string(name) + ".gz"), ec)) {
        throw Error(ErrorCode::io_unsupported,
                    (dir / name).string() + ".gz: compressed polyMesh files");
      }
      throw Error(ErrorCode::io_cannot_open, (dir / name).string() + " does not exist");
    }
  }

  MeshInput in;
  in.source = MeshSource{.format = "polymesh", .path = directory.string()};
  {
    FoamFile file(dir / "points", "vectorField");
    const auto xyz = file.points();
    const std::size_t n = xyz.size() / 3;
    in.x.resize(n);
    in.y.resize(n);
    in.z.resize(n);
    for (std::size_t p = 0; p < n; ++p) {
      in.x[p] = xyz[3 * p];
      in.y[p] = xyz[3 * p + 1];
      in.z[p] = xyz[3 * p + 2];
    }
  }
  Faces faces = read_faces(dir / "faces");
  const std::size_t n_faces = faces.offsets.size() - 1;

  FoamFile owner_file(dir / "owner", "labelList");
  const auto owner = owner_file.labels();
  FoamFile neighbour_file(dir / "neighbour", "labelList");
  const auto neighbour = neighbour_file.labels();
  if (owner.size() != n_faces || neighbour.size() > n_faces) {
    owner_file.cursor().fail("owner and neighbour sizes do not match the faces");
  }
  std::int64_t n_cells = 0;
  for (const auto c : owner) n_cells = std::max(n_cells, c + 1);
  for (const auto c : neighbour) n_cells = std::max(n_cells, c + 1);
  if (n_cells > std::numeric_limits<index_t>::max()) {
    throw Error(ErrorCode::mesh_too_large, "cell count exceeds the 32-bit index range");
  }

  in.face_node_offsets = std::move(faces.offsets);
  in.face_nodes = std::move(faces.nodes);
  in.face_cell0.resize(n_faces);
  in.face_cell1.assign(n_faces, no_cell);
  in.face_zone.assign(n_faces, -1);
  for (std::size_t f = 0; f < n_faces; ++f) {
    in.face_cell0[f] = to_index(owner_file.cursor(), owner[f], n_cells, "owner");
  }
  for (std::size_t f = 0; f < neighbour.size(); ++f) {
    in.face_cell1[f] = to_index(neighbour_file.cursor(), neighbour[f], n_cells, "neighbour");
  }

  FoamFile boundary_file(dir / "boundary", "polyBoundaryMesh");
  Cursor& bc = boundary_file.cursor();
  std::size_t expected_start = neighbour.size();
  for (const auto& patch : read_dict_list(boundary_file, "")) {
    auto value = [&](const char* key) {
      const auto it = patch.values.find(key);
      if (it == patch.values.end()) bc.fail("patch '" + patch.name + "' has no '" + key + "'");
      return it->second;
    };
    const std::string type = value("type");
    if (type == "processor" || type == "processorCyclic") {
      bc.unsupported("decomposed mesh (processor patch '" + patch.name + "')");
    }
    std::size_t start = 0, count = 0;
    try {
      start = std::stoull(value("startFace"));
      count = std::stoull(value("nFaces"));
    } catch (const std::logic_error&) {
      bc.fail("patch '" + patch.name + "' has a non-integer startFace or nFaces");
    }
    if (start != expected_start || start + count > n_faces) {
      bc.fail("patch '" + patch.name + "' does not continue the boundary face range");
    }
    expected_start = start + count;
    const auto zone = static_cast<index_t>(in.face_zones.size());
    in.face_zones.push_back(FaceZone{
        .name = patch.name,
        .bc_type = type,
        .kind = patch_kind(type),
        .provenance =
            Provenance{.source_id = zone, .detail = "polyMesh patch " + std::to_string(zone)}});
    std::fill(in.face_zone.begin() + static_cast<std::ptrdiff_t>(start),
              in.face_zone.begin() + static_cast<std::ptrdiff_t>(start + count), zone);
  }
  if (expected_start != n_faces) bc.fail("patches do not cover every boundary face");
  if (!neighbour.empty()) {
    const auto interior = static_cast<index_t>(in.face_zones.size());
    in.face_zones.push_back(FaceZone{
        .name = "interior",
        .bc_type = "",
        .kind = BcKind::unspecified,
        .boundary = false,
        .provenance =
            Provenance{.source_id = -1, .detail = "synthesised: polyMesh internal faces"}});
    std::fill(in.face_zone.begin(),
              in.face_zone.begin() + static_cast<std::ptrdiff_t>(neighbour.size()), interior);
  }

  in.cell_zone.assign(static_cast<std::size_t>(n_cells), -1);
  if (fs::exists(dir / "cellZones", ec)) {
    FoamFile zones_file(dir / "cellZones", "regIOobject");
    for (const auto& zone : read_dict_list(zones_file, "cellLabels")) {
      const auto z = static_cast<index_t>(in.cell_zones.size());
      in.cell_zones.push_back(CellZone{
          .name = zone.name,
          .type = "",
          .kind = CellKind::unspecified,
          .provenance =
              Provenance{.source_id = z, .detail = "polyMesh cellZone " + std::to_string(z)}});
      for (const auto cell : zone.labels) {
        const index_t c = to_index(zones_file.cursor(), cell, n_cells, "cell label");
        if (in.cell_zone[static_cast<std::size_t>(c)] != -1) {
          throw Error(ErrorCode::io_ambiguous_zone, (dir / "cellZones").string() + ": cell " +
                                                        std::to_string(c) +
                                                        " is in two cell zones");
        }
        in.cell_zone[static_cast<std::size_t>(c)] = z;
      }
    }
  }
  if (std::find(in.cell_zone.begin(), in.cell_zone.end(), -1) != in.cell_zone.end()) {
    const auto fluid = static_cast<index_t>(in.cell_zones.size());
    in.cell_zones.push_back(CellZone{
        .name = "fluid",
        .type = "",
        .kind = CellKind::unspecified,
        .provenance =
            Provenance{.source_id = -1, .detail = "synthesised: cells outside every cellZone"}});
    std::replace(in.cell_zone.begin(), in.cell_zone.end(), index_t{-1}, fluid);
  }
  return assemble(std::move(in));
}

}  // namespace autan::io
