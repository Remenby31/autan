// Gmsh MSH 4.1 reader, ASCII and binary (Gmsh reference manual, section
// "MSH file format", version 4.1). Volume elements become cells, surface
// elements label the cell faces they coincide with; physical groups become
// zones. Sections the reader does not need are skipped, as the format
// prescribes for unknown sections; sections that change the meaning of the
// mesh (partitions, periodicity, ghosts) are rejected.

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/cellwise.hpp"
#include "autan/io/read_mesh.hpp"
#include "cursor.hpp"

namespace autan::io {
namespace {

struct ElementType {
  int dim;
  std::size_t nodes;
};

// Linear elements only; higher-order types are rejected.
bool element_type(int type, ElementType& out) {
  switch (type) {
    case 15:
      out = {0, 1};
      return true;
    case 1:
      out = {1, 2};
      return true;
    case 2:
      out = {2, 3};
      return true;
    case 3:
      out = {2, 4};
      return true;
    case 4:
      out = {3, 4};
      return true;
    case 5:
      out = {3, 8};
      return true;
    case 6:
      out = {3, 6};
      return true;
    case 7:
      out = {3, 5};
      return true;
    default:
      return false;
  }
}

CellShape shape_of(int type) {
  switch (type) {
    case 4:
      return CellShape::tetra;
    case 5:
      return CellShape::hexa;
    case 6:
      return CellShape::prism;
    default:
      return CellShape::pyramid;
  }
}

class GmshReader {
 public:
  GmshReader(std::string data, const std::string& path) : data_(std::move(data)), c_(data_, path) {}

  MeshInput read() {
    bool seen_format = false, seen_nodes = false, seen_elements = false;
    while (c_.peek() != '\0') {
      const std::string_view name = c_.token();
      if (name.empty() || name.front() != '$') c_.fail("expected a section header");
      const std::string section(name.substr(1));
      if (section == "MeshFormat") {
        mesh_format();
        seen_format = true;
      } else if (!seen_format) {
        c_.fail("the file must start with $MeshFormat");
      } else if (section == "PhysicalNames") {
        physical_names();
      } else if (section == "Entities") {
        entities();
      } else if (section == "Nodes") {
        nodes();
        seen_nodes = true;
      } else if (section == "Elements") {
        if (!seen_nodes) c_.fail("$Elements before $Nodes");
        elements();
        seen_elements = true;
      } else if (section == "Periodic") {
        periodic();
      } else if (section == "PartitionedEntities" || section == "GhostElements") {
        c_.unsupported("$" + section + " section");
      } else {
        skip_section(section);
        continue;
      }
      c_.expect("$End" + section);
    }
    if (!seen_nodes || !seen_elements) c_.fail("missing $Nodes or $Elements");
    return build();
  }

 private:
  void mesh_format() {
    const double version = c_.real("");
    if (version != 4.1) c_.unsupported("MSH version " + std::to_string(version) + " (4.1 only)");
    binary_ = c_.integer<int>(10, "") == 1;
    if (c_.integer<int>(10, "") != 8) c_.unsupported("data-size other than 8");
    if (binary_) {
      (void)c_.line();
      if (c_.binary<std::int32_t>() != 1) c_.unsupported("big-endian binary file");
    }
  }

  // Surface links become periodic pairs: master group -> zone a, slave group
  // -> zone b, x_slave = A x_master. Point and curve links follow from them.
  void periodic() {
    if (binary_) (void)c_.line();
    const std::uint64_t n_links = next_size();
    c_.check_fits(n_links, min_bytes(5));
    for (std::uint64_t l = 0; l < n_links; ++l) {
      const int dim = next_int();
      const int tag = next_int();
      const int master = next_int();
      const std::uint64_t n_affine = next_size();
      if (n_affine != 0 && n_affine != 16)
        c_.fail("a periodic affine transform of " + std::to_string(n_affine) + " values");
      std::array<double, 16> a{};
      for (std::uint64_t k = 0; k < n_affine; ++k) a[k] = next_double();
      const std::uint64_t n_nodes = next_size();
      c_.check_fits(n_nodes, min_bytes(2));
      for (std::uint64_t k = 0; k < 2 * n_nodes; ++k) (void)next_size();
      if (dim != 2) continue;
      if (n_affine == 0) c_.unsupported("a periodic surface link without its affine transform");
      const std::array<double, 12> identity{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
      const std::array<double, 12> linear{a[0], a[1], a[2],  a[4],  a[5],  a[6],
                                          a[8], a[9], a[10], a[12], a[13], a[14]};
      if (linear != identity || a[15] != 1.0) c_.unsupported("rotational periodicity");
      const auto group = [&](int entity) {
        const auto it = entity_physical_.find({2, entity});
        if (it == entity_physical_.end() || it->second == 0)
          c_.fail("periodic surface " + std::to_string(entity) + " is in no physical group");
        return it->second;
      };
      const std::array<double, 3> t{a[3], a[7], a[11]};
      const auto [it, added] = periodic_links_.try_emplace({group(master), group(tag)}, t);
      if (!added && it->second != t) c_.fail("two translations for one periodic pair of groups");
    }
    if (binary_) c_.skip_space();
  }

  void skip_section(const std::string& section) {
    const std::string end = "\n$End" + section;
    const std::size_t at = data_.find(end, c_.position());
    if (at == std::string::npos) c_.fail("unterminated $" + section);
    c_.advance(at + end.size() - c_.position());
  }

  void physical_names() {
    const auto n = c_.integer<std::size_t>(10, "");
    for (std::size_t i = 0; i < n; ++i) {
      const int dim = c_.integer<int>(10, "");
      const int tag = c_.integer<int>(10, "");
      if (tag <= 0) c_.fail("non-positive physical tag " + std::to_string(tag));
      c_.skip_space();
      const std::string_view name = c_.line();
      if (name.size() < 2 || name.front() != '"' || name.back() != '"')
        c_.fail("unquoted physical name");
      names_[{dim, tag}] = std::string(name.substr(1, name.size() - 2));
    }
  }

  // Integers stored as int in binary files, ASCII text otherwise.
  int next_int() { return binary_ ? c_.binary<std::int32_t>() : c_.integer<int>(10, ""); }
  std::uint64_t next_size() {
    return binary_ ? c_.binary<std::uint64_t>() : c_.integer<std::uint64_t>(10, "");
  }
  double next_double() { return binary_ ? c_.binary<double>() : c_.real(""); }
  // Smallest encoding of `values` numbers: 4 bytes in binary files, a digit
  // and a separator in ASCII ones.
  [[nodiscard]] std::uint64_t min_bytes(std::uint64_t values) const {
    return values * (binary_ ? 4 : 2);
  }

  void entities() {
    if (binary_) (void)c_.line();
    std::uint64_t counts[4];
    for (auto& n : counts) n = next_size();
    for (int dim = 0; dim < 4; ++dim) {
      c_.check_fits(counts[dim], min_bytes(1));
      for (std::uint64_t e = 0; e < counts[dim]; ++e) {
        const int tag = next_int();
        for (int k = 0; k < (dim == 0 ? 3 : 6); ++k) (void)next_double();
        const std::uint64_t n_physicals = next_size();
        c_.check_fits(n_physicals, min_bytes(1));
        std::vector<int> physicals(n_physicals);
        for (auto& p : physicals) {
          p = next_int();
          // Physical tags are positive; signs only orient bounding entities.
          if (p <= 0) c_.fail("non-positive physical tag " + std::to_string(p));
        }
        if (dim > 0) {
          const std::uint64_t n_bounding = next_size();
          c_.check_fits(n_bounding, min_bytes(1));
          for (std::uint64_t b = 0; b < n_bounding; ++b) (void)next_int();
        }
        if (dim >= 2) {
          if (physicals.size() > 1) {
            c_.fail("entity (" + std::to_string(dim) + ", " + std::to_string(tag) +
                        ") belongs to several physical groups",
                    ErrorCode::io_ambiguous_zone);
          }
          entity_physical_[{dim, tag}] = physicals.empty() ? 0 : physicals.front();
        }
      }
    }
    if (binary_) c_.skip_space();
  }

  void nodes() {
    if (binary_) (void)c_.line();
    const std::uint64_t n_blocks = next_size();
    const std::uint64_t n_nodes = next_size();
    (void)next_size();
    (void)next_size();
    c_.check_fits(n_blocks, min_bytes(4));
    c_.check_fits(n_nodes, min_bytes(4));
    tags_.reserve(n_nodes);
    xyz_.reserve(3 * n_nodes);
    std::vector<std::uint64_t> block_tags;
    for (std::uint64_t b = 0; b < n_blocks; ++b) {
      const int dim = next_int();
      (void)next_int();
      const int parametric = next_int();
      const std::uint64_t n = next_size();
      c_.check_fits(n, min_bytes(4));
      if (tags_.size() + n > n_nodes) c_.fail("$Nodes blocks hold more nodes than declared");
      block_tags.resize(n);
      for (auto& t : block_tags) t = next_size();
      const int extra = parametric != 0 ? dim : 0;
      for (std::uint64_t i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) xyz_.push_back(next_double());
        for (int k = 0; k < extra; ++k) (void)next_double();
      }
      tags_.insert(tags_.end(), block_tags.begin(), block_tags.end());
    }
    if (tags_.size() != n_nodes) c_.fail("$Nodes count does not match its blocks");
    if (binary_) c_.skip_space();
    // Contiguous tags (what Gmsh writes) index directly; sparse ones go
    // through a sorted table. Both detect duplicate tags.
    const auto [lo, hi] = std::ranges::minmax_element(tags_);
    dense_ = !tags_.empty() && *hi - *lo + 1 == tags_.size();
    order_.assign(tags_.size(), -1);
    if (dense_) {
      first_tag_ = *lo;
      for (std::size_t i = 0; i < tags_.size(); ++i) {
        index_t& slot = order_[static_cast<std::size_t>(tags_[i] - first_tag_)];
        if (slot != -1) c_.fail("node tag " + std::to_string(tags_[i]) + " is defined twice");
        slot = static_cast<index_t>(i);
      }
      return;
    }
    for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<index_t>(i);
    std::ranges::sort(order_, [&](index_t a, index_t b) {
      return tags_[static_cast<std::size_t>(a)] < tags_[static_cast<std::size_t>(b)];
    });
    for (std::size_t i = 1; i < order_.size(); ++i) {
      if (tags_[static_cast<std::size_t>(order_[i])] ==
          tags_[static_cast<std::size_t>(order_[i - 1])]) {
        c_.fail("node tag " + std::to_string(tags_[static_cast<std::size_t>(order_[i])]) +
                " is defined twice");
      }
    }
  }

  index_t node_index(std::uint64_t tag) {
    if (dense_) {
      if (tag < first_tag_ || tag - first_tag_ >= order_.size()) {
        c_.fail("element refers to undefined node " + std::to_string(tag));
      }
      return order_[static_cast<std::size_t>(tag - first_tag_)];
    }
    const auto it = std::ranges::lower_bound(order_, tag, [&](index_t i, std::uint64_t t) {
      return tags_[static_cast<std::size_t>(i)] < t;
    });
    if (it == order_.end() || tags_[static_cast<std::size_t>(*it)] != tag) {
      c_.fail("element refers to undefined node " + std::to_string(tag));
    }
    return *it;
  }

  void elements() {
    if (binary_) (void)c_.line();
    const std::uint64_t n_blocks = next_size();
    const std::uint64_t n_elements = next_size();
    (void)next_size();
    (void)next_size();
    c_.check_fits(n_blocks, min_bytes(4));
    c_.check_fits(n_elements, min_bytes(2));
    std::uint64_t read = 0;
    std::vector<index_t> nodes;
    for (std::uint64_t b = 0; b < n_blocks; ++b) {
      const int dim = next_int();
      const int entity = next_int();
      const int type = next_int();
      const std::uint64_t n = next_size();
      ElementType et{};
      c_.check_fits(n, min_bytes(2));
      read += n;
      if (read > n_elements) c_.fail("$Elements blocks hold more elements than declared");
      if (!element_type(type, et))
        c_.unsupported("element type " + std::to_string(type) + " (linear elements only)");
      if (et.dim != dim) c_.fail("element type does not match its entity dimension");
      const auto physical_it = entity_physical_.find({dim, entity});
      const int physical = physical_it == entity_physical_.end() ? 0 : physical_it->second;
      nodes.resize(et.nodes);
      for (std::uint64_t e = 0; e < n; ++e) {
        (void)next_size();
        for (auto& v : nodes) v = node_index(next_size());
        if (dim == 3) {
          mesh_.add_cell(shape_of(type), nodes, cell_zone(physical));
        } else if (dim == 2 && physical != 0) {
          mesh_.add_labelled_face(nodes, face_zone(physical));
        }
      }
    }
    if (binary_) c_.skip_space();
  }

  std::string zone_name(int dim, int physical) {
    const auto it = names_.find({dim, physical});
    return it != names_.end() ? it->second : std::to_string(physical);
  }

  index_t cell_zone(int physical) {
    const auto it = cell_zone_index_.find(physical);
    if (it != cell_zone_index_.end()) return it->second;
    const auto index = static_cast<index_t>(mesh_.cell_zones.size());
    mesh_.cell_zones.push_back(CellZone{
        .name = physical == 0 ? "fluid" : std::to_string(physical),
        .type = "",
        .kind = CellKind::unspecified,
        .provenance =
            physical == 0
                ? Provenance{.source_id = -1,
                             .detail = "synthesised: volume elements outside every physical group"}
                : Provenance{.source_id = physical,
                             .detail = "Gmsh physical volume " + std::to_string(physical)}});
    cell_zone_index_[physical] = index;
    return index;
  }

  index_t face_zone(int physical) {
    const auto it = face_zone_index_.find(physical);
    if (it != face_zone_index_.end()) return it->second;
    const auto index = static_cast<index_t>(mesh_.face_zones.size());
    mesh_.face_zones.push_back(FaceZone{
        .name = std::to_string(physical),
        .bc_type = "",
        .kind = BcKind::unspecified,
        .boundary = true,
        .provenance = Provenance{.source_id = physical,
                                 .detail = "Gmsh physical surface " + std::to_string(physical)}});
    face_zone_index_[physical] = index;
    return index;
  }

  MeshInput build() {
    if (mesh_.cell_zone.empty()) c_.fail("no volume elements");
    // Declared surface groups are zones even without elements, so that their
    // names stay visible; names are resolved once every section is read.
    for (const auto& [key, name] : names_) {
      if (key.first == 2) (void)face_zone(key.second);
    }
    for (const auto& [key, physical] : entity_physical_) {
      if (key.first == 2 && physical != 0) (void)face_zone(physical);
    }
    for (const auto& [groups, t] : periodic_links_) {
      const index_t a = face_zone(groups.first), b = face_zone(groups.second);
      mesh_.face_zones[static_cast<std::size_t>(a)].kind = BcKind::periodic;
      mesh_.face_zones[static_cast<std::size_t>(b)].kind = BcKind::periodic;
      mesh_.periodic.push_back({.zone_a = a, .zone_b = b, .translation = t});
    }
    for (auto& zone : mesh_.cell_zones) {
      if (zone.provenance.source_id > 0)
        zone.name = zone_name(3, static_cast<int>(zone.provenance.source_id));
    }
    for (auto& zone : mesh_.face_zones) {
      zone.name = zone_name(2, static_cast<int>(zone.provenance.source_id));
    }
    const std::size_t n = tags_.size();
    mesh_.x.resize(n);
    mesh_.y.resize(n);
    mesh_.z.resize(n);
    for (std::size_t p = 0; p < n; ++p) {
      mesh_.x[p] = xyz_[3 * p];
      mesh_.y[p] = xyz_[3 * p + 1];
      mesh_.z[p] = xyz_[3 * p + 2];
    }
    mesh_.source = MeshSource{.format = "gmsh", .path = c_.path()};
    return to_face_based(std::move(mesh_));
  }

  std::string data_;
  Cursor c_;
  bool binary_ = false;
  std::map<std::pair<int, int>, std::string> names_;
  std::map<std::pair<int, int>, int> entity_physical_;
  std::map<std::pair<int, int>, std::array<double, 3>> periodic_links_;  // (master, slave) groups
  std::vector<std::uint64_t> tags_;
  std::vector<index_t> order_;
  bool dense_ = false;
  std::uint64_t first_tag_ = 0;
  std::vector<double> xyz_;
  std::map<int, index_t> cell_zone_index_;
  std::map<int, index_t> face_zone_index_;
  CellwiseMesh mesh_;
};

}  // namespace

Mesh read_gmsh(const std::filesystem::path& file) {
  GmshReader reader(read_file(file.string()), file.string());
  return assemble(reader.read());
}

}  // namespace autan::io
