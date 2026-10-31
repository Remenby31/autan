// Fluent mesh file reader (ANSYS Fluent User's Guide, appendix "Mesh File
// Format"). Sections: 0/1 comments and header, 2 dimension, 10/2010/3010
// nodes, 12/2012 cells, 13/2013 faces, 39/45 zones. Indices in sections 10,
// 12 and 13 are hexadecimal; binary sections hold little-endian values.
// The right-hand-rule normal of a face points toward c0 (OpenFOAM's
// fluentMeshToFoam reads c1 as owner), and a boundary face has c1 = 0.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "autan/io/read_mesh.hpp"
#include "cursor.hpp"

namespace autan::io {
namespace {

struct ZoneInfo {
  std::string type;
  std::string name;
};

struct FaceBlock {
  std::int64_t zone;
  std::int64_t bc_code;
};

// Fluent boundary-condition codes of section 13 headers, used when no zone
// section names the type.
std::string bc_type_from_code(std::int64_t code) {
  switch (code) {
    case 2:
      return "interior";
    case 3:
      return "wall";
    case 4:
      return "pressure-inlet";
    case 5:
      return "pressure-outlet";
    case 7:
      return "symmetry";
    case 8:
      return "periodic-shadow";
    case 9:
      return "pressure-far-field";
    case 10:
      return "velocity-inlet";
    case 12:
      return "periodic";
    case 14:
      return "fan";
    case 20:
      return "mass-flow-inlet";
    case 24:
      return "interface";
    case 31:
      return "parent";
    case 36:
      return "outflow";
    case 37:
      return "axis";
    default:
      return "";
  }
}

// docs/contracts/mesh-info.md, "Normalized boundary types".
BcKind bc_kind(std::string_view type) {
  if (type == "wall") return BcKind::wall;
  if (type == "symmetry") return BcKind::symmetry;
  if (type == "pressure-far-field") return BcKind::far_field;
  if (type == "velocity-inlet" || type == "pressure-inlet" || type == "mass-flow-inlet") {
    return BcKind::inlet;
  }
  if (type == "pressure-outlet" || type == "outflow") return BcKind::outlet;
  return BcKind::unspecified;
}

class FluentReader {
 public:
  FluentReader(std::string data, const std::string& path)
      : data_(std::move(data)), c_(data_, path) {}

  MeshInput read() {
    for (;;) {
      if (c_.peek() == '\0') break;
      c_.expect('(');
      const int id = c_.integer<int>(10, "()");
      section(id);
    }
    return build();
  }

 private:
  void section(int id) {
    switch (id) {
      case 0:
      case 1:
      case 4:
      case 37:
      case 38:
      case 40:
        skip_rest();
        return;
      case 2: {
        const int dim = c_.integer<int>(10, "()");
        if (dim != 3) c_.unsupported("2D mesh");
        c_.expect(')');
        return;
      }
      case 10:
      case 2010:
      case 3010:
        nodes(id);
        return;
      case 12:
      case 2012:
        cells(id);
        return;
      case 13:
      case 2013:
        faces(id);
        return;
      case 39:
      case 45:
        zone();
        return;
      case 18:
      case 2018:
        c_.unsupported("periodic shadow faces (section 18)");
      case 58:
      case 59:
      case 2058:
      case 2059:
      case 61:
      case 62:
      case 63:
        c_.unsupported("hanging-node trees and interface parents (sections 58, 59, 61-63)");
      default:
        c_.unsupported("section " + std::to_string(id));
    }
  }

  // Skips to the parenthesis closing the current section, honouring strings.
  void skip_rest() {
    int depth = 1;
    bool quoted = false;
    while (depth > 0) {
      if (c_.at_end()) c_.fail("unterminated section");
      const char ch = c_.peek_raw();
      c_.advance();
      if (quoted) {
        quoted = ch != '"';
      } else if (ch == '"') {
        quoted = true;
      } else if (ch == '(') {
        ++depth;
      } else if (ch == ')') {
        --depth;
      }
    }
  }

  std::vector<std::int64_t> header() {
    c_.expect('(');
    std::vector<std::int64_t> values;
    while (c_.peek() != ')') values.push_back(c_.integer<std::int64_t>(16, "()"));
    c_.expect(')');
    if (values.size() < 4) c_.fail("section header has fewer than 4 fields");
    return values;
  }

  // "...<payload>)End of Binary Section <id>"; the caller consumes the ')'
  // that closes the section.
  void end_binary(int id) {
    c_.expect(')');
    c_.expect("End of Binary Section");
    if (c_.integer<int>(10, "()") != id) c_.fail("binary section ends with another id");
  }

  static std::size_t span_of(std::int64_t first, std::int64_t last) {
    return static_cast<std::size_t>(last - first + 1);
  }

  void check_range(std::int64_t first, std::int64_t last, const char* what) {
    if (first < 1 || last < first) c_.fail(std::string("invalid ") + what + " index range");
  }

  void nodes(int id) {
    const auto h = header();
    const std::int64_t zone = h[0], first = h[1], last = h[2];
    if (zone == 0) {
      n_nodes_ = std::max(n_nodes_, last);
      c_.expect(')');
      return;
    }
    check_range(first, last, "node");
    const std::int64_t dimension = h.size() > 4 ? h[4] : 3;
    if (dimension != 3) c_.unsupported("nodes with " + std::to_string(dimension) + " dimensions");
    n_nodes_ = std::max(n_nodes_, last);
    if (xyz_.size() < static_cast<std::size_t>(3 * last)) {
      xyz_.resize(static_cast<std::size_t>(3 * last));
      node_seen_.resize(static_cast<std::size_t>(last), 0);
    }
    const std::size_t n = span_of(first, last);
    double* out = xyz_.data() + 3 * (first - 1);
    c_.expect('(');
    if (id == 10) {
      for (std::size_t i = 0; i < 3 * n; ++i) out[i] = c_.real();
      c_.expect(')');
    } else if (id == 3010) {
      c_.binary(out, 3 * n);
      end_binary(id);
    } else {
      std::vector<float> raw(3 * n);
      c_.binary(raw.data(), raw.size());
      std::copy(raw.begin(), raw.end(), out);
      end_binary(id);
    }
    std::fill(node_seen_.begin() + first - 1, node_seen_.begin() + last, 1);
    c_.expect(')');
  }

  void cells(int id) {
    const auto h = header();
    const std::int64_t zone = h[0], first = h[1], last = h[2];
    if (zone == 0) {
      n_cells_ = std::max(n_cells_, last);
      if (c_.peek() == '(') skip_balanced_body();
      c_.expect(')');
      return;
    }
    check_range(first, last, "cell");
    const std::int64_t element = h.size() > 4 ? h[4] : 0;
    n_cells_ = std::max(n_cells_, last);
    if (cell_fluent_zone_.size() < static_cast<std::size_t>(last)) {
      cell_fluent_zone_.resize(static_cast<std::size_t>(last), 0);
      cell_element_.resize(static_cast<std::size_t>(last), 0);
    }
    const std::size_t n = span_of(first, last);
    std::fill(cell_fluent_zone_.begin() + first - 1, cell_fluent_zone_.begin() + last, zone);
    if (element == 0) {
      c_.expect('(');
      std::vector<std::int32_t> types(n);
      if (id == 12) {
        for (auto& t : types) t = c_.integer<std::int32_t>(16);
        c_.expect(')');
      } else {
        c_.binary(types.data(), n);
        end_binary(id);
      }
      std::copy(types.begin(), types.end(), cell_element_.begin() + first - 1);
    } else {
      std::fill(cell_element_.begin() + first - 1, cell_element_.begin() + last,
                static_cast<std::int32_t>(element));
      if (c_.peek() == '(') skip_balanced_body();
    }
    c_.expect(')');
    if (std::find(cell_zone_order_.begin(), cell_zone_order_.end(), zone) ==
        cell_zone_order_.end()) {
      cell_zone_order_.push_back(zone);
    }
  }

  void skip_balanced_body() {
    c_.expect('(');
    skip_rest();
  }

  void faces(int id) {
    const auto h = header();
    const std::int64_t zone = h[0], first = h[1], last = h[2];
    if (zone == 0) {
      c_.expect(')');
      return;
    }
    check_range(first, last, "face");
    if (h.size() < 5) c_.fail("face section header has fewer than 5 fields");
    const std::int64_t bc_code = h[3], face_type = h[4];
    if (face_type == 2) c_.unsupported("2D (linear) faces");
    if (face_type != 0 && face_type != 3 && face_type != 4 && face_type != 5) {
      c_.fail("unknown face type " + std::to_string(face_type));
    }
    const bool binary = id == 2013;
    auto next = [&]() -> std::int64_t {
      return binary ? c_.binary<std::int32_t>() : c_.integer<std::int64_t>(16);
    };
    const std::size_t n = span_of(first, last);
    const auto block = static_cast<index_t>(face_blocks_.size());
    face_blocks_.push_back(FaceBlock{.zone = zone, .bc_code = bc_code});
    c_.expect('(');
    std::vector<std::int64_t> nodes;
    for (std::size_t f = 0; f < n; ++f) {
      const std::int64_t count = face_type == 0 || face_type == 5 ? next() : face_type;
      if (count < 3) c_.fail("face with fewer than 3 nodes");
      nodes.resize(static_cast<std::size_t>(count));
      for (auto& v : nodes) v = next();
      const std::int64_t c0 = next();
      const std::int64_t c1 = next();
      for (const auto v : nodes) {
        if (v < 1) c_.fail("node index must be at least 1");
      }
      if (c0 < 0 || c1 < 0 || (c0 == 0 && c1 == 0)) c_.fail("face without a cell");
      // Normal points toward c0: c1 is the cell it leaves. On the boundary
      // (c1 = 0) the face is reversed so that it leaves c0.
      const bool reverse = c1 == 0;
      if (reverse) std::reverse(nodes.begin() + 1, nodes.end());
      for (const auto v : nodes) face_nodes_.push_back(v);
      face_offsets_.push_back(static_cast<offset_t>(face_nodes_.size()));
      face_cell0_.push_back(reverse ? c0 : c1);
      face_cell1_.push_back(reverse ? 0 : c0);
      face_block_.push_back(block);
    }
    if (binary) {
      end_binary(id);
    } else {
      c_.expect(')');
    }
    c_.expect(')');
  }

  void zone() {
    c_.expect('(');
    const auto id = c_.integer<std::int64_t>(10, "()");
    ZoneInfo info;
    info.type = std::string(c_.token("()"));
    info.name = std::string(c_.token("()"));
    while (c_.peek() != ')') (void)c_.token("()");
    c_.expect(')');
    skip_rest();
    zones_[id] = std::move(info);
  }

  index_t cell_index(std::int64_t cell) {
    if (cell < 1 || cell > n_cells_) c_.fail("cell index out of range");
    return static_cast<index_t>(cell - 1);
  }

  MeshInput build() {
    MeshInput in;
    in.source = MeshSource{.format = "fluent", .path = c_.path()};
    if (n_nodes_ == 0 || face_blocks_.empty()) c_.fail("no nodes or no faces");
    for (std::int64_t p = 0; p < n_nodes_; ++p) {
      if (static_cast<std::size_t>(p) >= node_seen_.size() ||
          node_seen_[static_cast<std::size_t>(p)] == 0) {
        c_.fail("node " + std::to_string(p + 1) + " is declared but has no coordinates");
      }
    }
    const auto n = static_cast<std::size_t>(n_nodes_);
    in.x.resize(n);
    in.y.resize(n);
    in.z.resize(n);
    for (std::size_t p = 0; p < n; ++p) {
      in.x[p] = xyz_[3 * p];
      in.y[p] = xyz_[3 * p + 1];
      in.z[p] = xyz_[3 * p + 2];
    }

    in.face_node_offsets = std::move(face_offsets_);
    in.face_nodes.reserve(face_nodes_.size());
    for (const auto v : face_nodes_) {
      if (v > n_nodes_) c_.fail("face node index out of range");
      in.face_nodes.push_back(static_cast<index_t>(v - 1));
    }
    std::vector<std::int64_t> zone_of_block;
    std::map<std::int64_t, index_t> face_zone_index;
    for (std::size_t b = 0; b < face_blocks_.size(); ++b) {
      const FaceBlock& block = face_blocks_[b];
      if (face_zone_index.contains(block.zone)) continue;
      const auto it = zones_.find(block.zone);
      const std::string type =
          it != zones_.end() ? it->second.type : bc_type_from_code(block.bc_code);
      const std::string name =
          it != zones_.end() ? it->second.name : "zone-" + std::to_string(block.zone);
      face_zone_index[block.zone] = static_cast<index_t>(in.face_zones.size());
      in.face_zones.push_back(FaceZone{
          .name = name,
          .bc_type = type,
          .kind = bc_kind(type),
          .boundary = true,
          .provenance = Provenance{.source_id = block.zone,
                                   .detail = "Fluent zone " + std::to_string(block.zone)}});
    }
    const std::size_t n_faces = face_cell0_.size();
    in.face_cell0.resize(n_faces);
    in.face_cell1.resize(n_faces);
    in.face_zone.resize(n_faces);
    for (std::size_t f = 0; f < n_faces; ++f) {
      in.face_cell0[f] = cell_index(face_cell0_[f]);
      in.face_cell1[f] = face_cell1_[f] == 0 ? no_cell : cell_index(face_cell1_[f]);
      const index_t zone =
          face_zone_index[face_blocks_[static_cast<std::size_t>(face_block_[f])].zone];
      in.face_zone[f] = zone;
      // A zone's role is that of its faces; assemble() rejects a zone that
      // mixes interior and boundary faces.
      if (in.face_cell1[f] != no_cell)
        in.face_zones[static_cast<std::size_t>(zone)].boundary = false;
    }

    const auto n_cells = static_cast<std::size_t>(n_cells_);
    in.cell_zone.assign(n_cells, -1);
    std::map<std::int64_t, index_t> cell_zone_index;
    for (const std::int64_t zone : cell_zone_order_) {
      const auto it = zones_.find(zone);
      const std::string type = it != zones_.end() ? it->second.type : "";
      cell_zone_index[zone] = static_cast<index_t>(in.cell_zones.size());
      in.cell_zones.push_back(CellZone{
          .name = it != zones_.end() ? it->second.name : "zone-" + std::to_string(zone),
          .type = type,
          .kind = type == "fluid"   ? CellKind::fluid
                  : type == "solid" ? CellKind::solid
                                    : CellKind::unspecified,
          .provenance =
              Provenance{.source_id = zone, .detail = "Fluent zone " + std::to_string(zone)}});
    }
    for (std::size_t c = 0; c < std::min(n_cells, cell_fluent_zone_.size()); ++c) {
      if (cell_fluent_zone_[c] != 0) in.cell_zone[c] = cell_zone_index[cell_fluent_zone_[c]];
    }
    if (std::find(in.cell_zone.begin(), in.cell_zone.end(), -1) != in.cell_zone.end()) {
      if (!cell_zone_order_.empty()) c_.fail("some cells belong to no cell section");
      in.cell_zones.push_back(CellZone{
          .name = "fluid",
          .type = "",
          .kind = CellKind::unspecified,
          .provenance = Provenance{.source_id = -1, .detail = "synthesised: no cell section"}});
      std::fill(in.cell_zone.begin(), in.cell_zone.end(), 0);
    }
    check_element_types(in);
    return in;
  }

  // Declared standard shapes must have their number of faces.
  void check_element_types(const MeshInput& in) {
    std::vector<int> count(static_cast<std::size_t>(n_cells_), 0);
    for (std::size_t f = 0; f < in.face_cell0.size(); ++f) {
      ++count[static_cast<std::size_t>(in.face_cell0[f])];
      if (in.face_cell1[f] != no_cell) ++count[static_cast<std::size_t>(in.face_cell1[f])];
    }
    for (std::size_t c = 0; c < cell_element_.size(); ++c) {
      int expected = 0;
      switch (cell_element_[c]) {
        case 0:
        case 7:
          continue;
        case 2:
          expected = 4;
          break;
        case 4:
          expected = 6;
          break;
        case 5:
        case 6:
          expected = 5;
          break;
        case 1:
        case 3:
          c_.unsupported("2D cells");
        default:
          c_.fail("unknown cell element type " + std::to_string(cell_element_[c]));
      }
      if (count[c] != expected) {
        c_.fail("cell " + std::to_string(c + 1) + " has " + std::to_string(count[c]) +
                " faces but its element type needs " + std::to_string(expected));
      }
    }
  }

  std::string data_;
  Cursor c_;
  std::int64_t n_nodes_ = 0;
  std::int64_t n_cells_ = 0;
  std::vector<double> xyz_;
  std::vector<unsigned char> node_seen_;
  std::vector<std::int64_t> cell_fluent_zone_;
  std::vector<std::int32_t> cell_element_;
  std::vector<std::int64_t> cell_zone_order_;
  std::vector<offset_t> face_offsets_{0};
  std::vector<std::int64_t> face_nodes_;
  std::vector<std::int64_t> face_cell0_;
  std::vector<std::int64_t> face_cell1_;
  std::vector<index_t> face_block_;
  std::vector<FaceBlock> face_blocks_;
  std::map<std::int64_t, ZoneInfo> zones_;
};

}  // namespace

Mesh read_fluent(const std::filesystem::path& file) {
  FluentReader reader(read_file(file.string()), file.string());
  MeshInput in = reader.read();
  return assemble(std::move(in));
}

}  // namespace autan::io
