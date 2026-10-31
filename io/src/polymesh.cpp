// OpenFOAM polyMesh reader (OpenFOAM User Guide, section 4.1 "Mesh
// description"): points, faces, owner, neighbour, boundary and optional
// cellZones, ASCII or binary, 32- or 64-bit labels, 32- or 64-bit scalars.
// faceZones, pointZones and sets are auxiliary selections, not zones, and are
// not read.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
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

bool at_comment(std::string_view rest) { return rest.starts_with("//") || rest.starts_with("/*"); }

// Value of a `key value;` entry: its tokens, comments removed, joined by
// single spaces; quoted strings keep their content.
std::string entry_value(Cursor& c, std::string_view data) {
  std::string value;
  for (;;) {
    skip_blank(c);
    if (c.at_end()) c.fail("unterminated entry");
    const char ch = c.peek_raw();
    if (ch == ';') {
      c.advance();
      return value;
    }
    if (!value.empty()) value += ' ';
    if (ch == '"') {
      c.advance();
      while (!c.at_end() && c.peek_raw() != '"') {
        value += c.peek_raw();
        c.advance();
      }
      c.expect('"');
      continue;
    }
    while (!c.at_end()) {
      const char t = c.peek_raw();
      if (t == ';' || t == ' ' || t == '\t' || t == '\n' || t == '\r' ||
          at_comment(data.substr(c.position()))) {
        break;
      }
      value += t;
      c.advance();
    }
  }
}

// Skips a braced sub-dictionary, comments and strings included.
void skip_dict(Cursor& c, std::string_view data) {
  c.expect('{');
  int depth = 1;
  while (depth > 0) {
    skip_blank(c);
    if (c.at_end()) c.fail("unterminated dictionary");
    const char ch = c.peek_raw();
    if (ch == '"') {
      c.advance();
      while (!c.at_end() && c.peek_raw() != '"') c.advance();
      c.expect('"');
      continue;
    }
    if (ch == '{') ++depth;
    if (ch == '}') --depth;
    if (!at_comment(data.substr(c.position()))) c.advance();
  }
}

struct Header {
  bool binary = false;
  int label_bytes = 4;
  int scalar_bytes = 8;
  std::string object_class;
};

class FoamFile {
 public:
  explicit FoamFile(const fs::path& path)
      : data_(read_file(path.string())), cursor_(data_, path.string()) {
    read_header();
  }

  Cursor& cursor() { return cursor_; }
  [[nodiscard]] const Header& header() const { return header_; }
  [[nodiscard]] std::string_view data() const { return data_; }

  void require_class(std::initializer_list<std::string_view> classes) {
    for (const auto expected : classes) {
      if (header_.object_class == expected) return;
    }
    cursor_.fail("unexpected class '" + header_.object_class + "'");
  }

  std::string value() { return entry_value(cursor_, data_); }
  void skip_sub_dict() { skip_dict(cursor_, data_); }

  // Optional leading size of a list; nullopt when the list starts with '('.
  std::optional<std::size_t> list_size() {
    skip_blank(cursor_);
    if (cursor_.peek() == '(') return std::nullopt;
    const auto n = cursor_.integer<std::size_t>(10, "({/");
    skip_blank(cursor_);
    return n;
  }

  std::int64_t label() {
    skip_blank(cursor_);
    return cursor_.integer<std::int64_t>(10, "(){}/;");
  }

  // `N(v ...)`, `(v ...)`, `N{v}`; in binary files `N(<raw>)`, or a bare `0`.
  std::vector<std::int64_t> labels() {
    skip_blank(cursor_);
    if (cursor_.accept("List<label>")) skip_blank(cursor_);
    const auto n = list_size();
    std::vector<std::int64_t> out;
    if (!n) {
      cursor_.expect('(');
      while (skip_blank(cursor_), cursor_.peek() != ')') out.push_back(label());
      cursor_.expect(')');
      return out;
    }
    if (cursor_.accept('{')) {
      skip_blank(cursor_);
      if (*n == 0 && cursor_.accept('}')) return out;
      const std::int64_t v = label();
      skip_blank(cursor_);
      cursor_.expect('}');
      out.assign(*n, v);
      return out;
    }
    if (header_.binary) {
      if (*n == 0) {
        if (cursor_.accept('(')) cursor_.expect(')');
        return out;
      }
      cursor_.expect('(');
      cursor_.check_fits(*n, static_cast<std::uint64_t>(header_.label_bytes));
      out.resize(*n);
      if (header_.label_bytes == 4) {
        std::vector<std::int32_t> raw(*n);
        cursor_.binary(raw.data(), *n);
        std::copy(raw.begin(), raw.end(), out.begin());
      } else {
        cursor_.binary(out.data(), *n);
      }
      skip_blank(cursor_);
      cursor_.expect(')');
      return out;
    }
    cursor_.expect('(');
    cursor_.check_fits(*n, 2);
    out.resize(*n);
    for (auto& v : out) v = label();
    skip_blank(cursor_);
    cursor_.expect(')');
    return out;
  }

  std::vector<double> points() {
    const auto n = list_size();
    if (!n) cursor_.fail("points list without a size");
    std::vector<double> out;
    if (header_.binary && *n == 0) {
      if (cursor_.accept('(')) cursor_.expect(')');
      return out;
    }
    cursor_.expect('(');
    cursor_.check_fits(*n,
                       header_.binary ? 3 * static_cast<std::uint64_t>(header_.scalar_bytes) : 7);
    out.resize(3 * *n);
    if (header_.binary) {
      if (header_.scalar_bytes == 8) {
        cursor_.binary(out.data(), out.size());
      } else {
        std::vector<float> raw(out.size());
        cursor_.binary(raw.data(), raw.size());
        std::copy(raw.begin(), raw.end(), out.begin());
      }
    } else {
      for (std::size_t p = 0; p < *n; ++p) {
        skip_blank(cursor_);
        cursor_.expect('(');
        for (std::size_t k = 0; k < 3; ++k) {
          skip_blank(cursor_);
          out[3 * p + k] = cursor_.real("()/");
        }
        skip_blank(cursor_);
        cursor_.expect(')');
      }
    }
    skip_blank(cursor_);
    cursor_.expect(')');
    return out;
  }

 private:
  // `FoamFile { key value; ... }`.
  void read_header() {
    skip_blank(cursor_);
    cursor_.expect("FoamFile");
    skip_blank(cursor_);
    cursor_.expect('{');
    for (;;) {
      skip_blank(cursor_);
      if (cursor_.accept('}')) break;
      const std::string key(cursor_.token("{};/"));
      skip_blank(cursor_);
      if (cursor_.peek() == '{') {
        skip_sub_dict();
        continue;
      }
      const std::string v = value();
      if (key == "format") {
        if (v != "ascii" && v != "binary") cursor_.fail("unknown format '" + v + "'");
        header_.binary = v == "binary";
      } else if (key == "class") {
        header_.object_class = v;
      } else if (key == "arch") {
        if (v.find("label=64") != std::string::npos) header_.label_bytes = 8;
        if (v.find("scalar=32") != std::string::npos) header_.scalar_bytes = 4;
        if (v.find("MSB") != std::string::npos) cursor_.unsupported("big-endian binary files");
      }
    }
  }

  std::string data_;
  Cursor cursor_;
  Header header_;
};

struct Faces {
  std::vector<offset_t> offsets;
  std::vector<index_t> nodes;
};

Faces read_faces(const fs::path& path) {
  FoamFile file(path);
  file.require_class({"faceList", "faceCompactList"});
  Cursor& c = file.cursor();
  Faces faces;
  auto narrow = [&](std::int64_t v) {
    if (v < 0 || v > std::numeric_limits<index_t>::max()) c.fail("point label out of range");
    return static_cast<index_t>(v);
  };
  if (file.header().binary) {
    // Binary polyMesh faces are written as faceCompactList: offsets, labels.
    if (file.header().object_class != "faceCompactList") {
      c.unsupported("binary faceList (OpenFOAM writes binary faces as faceCompactList)");
    }
    const auto offsets = file.labels();
    const auto nodes = file.labels();
    if (offsets.empty() || offsets.front() != 0 ||
        static_cast<std::size_t>(offsets.back()) != nodes.size() ||
        std::adjacent_find(offsets.begin(), offsets.end(), std::greater<>()) != offsets.end()) {
      c.fail("faceCompactList offsets do not match its labels");
    }
    faces.offsets.assign(offsets.begin(), offsets.end());
    faces.nodes.reserve(nodes.size());
    for (const auto v : nodes) faces.nodes.push_back(narrow(v));
    return faces;
  }
  // ASCII faces are nested lists `k(a b c)` whichever the class.
  const auto n = file.list_size();
  if (!n) c.fail("faces list without a size");
  c.expect('(');
  c.check_fits(*n, 4);
  faces.offsets.reserve(*n + 1);
  faces.offsets.push_back(0);
  for (std::size_t f = 0; f < *n; ++f) {
    for (const auto v : file.labels()) faces.nodes.push_back(narrow(v));
    faces.offsets.push_back(static_cast<offset_t>(faces.nodes.size()));
  }
  skip_blank(c);
  c.expect(')');
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
  const auto n = file.list_size();
  c.expect('(');
  std::vector<NamedDict> out;
  for (;;) {
    skip_blank(c);
    if (c.accept(')')) break;
    NamedDict d;
    d.name = std::string(c.token("{/"));
    skip_blank(c);
    c.expect('{');
    for (;;) {
      skip_blank(c);
      if (c.accept('}')) break;
      const std::string key(c.token("{};/"));
      skip_blank(c);
      if (!label_key.empty() && key == label_key) {
        d.labels = file.labels();
        skip_blank(c);
        c.expect(';');
      } else if (c.peek() == '{') {
        file.skip_sub_dict();
      } else {
        d.values[key] = file.value();
      }
    }
    out.push_back(std::move(d));
  }
  if (n && *n != out.size()) c.fail("list size does not match its entries");
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
    FoamFile file(dir / "points");
    file.require_class({"vectorField"});
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

  FoamFile owner_file(dir / "owner");
  owner_file.require_class({"labelList"});
  const auto owner = owner_file.labels();
  FoamFile neighbour_file(dir / "neighbour");
  neighbour_file.require_class({"labelList"});
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

  FoamFile boundary_file(dir / "boundary");
  boundary_file.require_class({"polyBoundaryMesh"});
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
  if (!fs::exists(dir / "cellZones", ec) && fs::exists(dir / "cellZones.gz", ec)) {
    throw Error(ErrorCode::io_unsupported,
                (dir / "cellZones").string() + ".gz: compressed polyMesh files");
  }
  if (fs::exists(dir / "cellZones", ec)) {
    FoamFile zones_file(dir / "cellZones");
    zones_file.require_class({"regIOobject", "cellZoneList"});
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
