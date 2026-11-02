// CGNS mesh reader through the CGNS mid-level library (SIDS 4.x). One base,
// one unstructured zone; cells as NFACE_n over NGON_n faces, or as linear
// standard elements. See docs/contracts/mesh.md, "Readers".

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cgnslib.h>
#include <nlohmann/json.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "cgns_util.hpp"
#include "self_contained.hpp"
#include "text_limit.hpp"

namespace autan::io {

static_assert(sizeof(cgsize_t) == 8, "CGNS must be built with 64-bit cgsize_t (CG_BUILD_64BIT)");

namespace {

namespace fs = std::filesystem;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

// Once the file is open, a library error means its content breaks the SIDS
// (E0132); opening it is checked by CgnsFile (E0130).
void ok(int status, const std::string& what, const std::string& name = "") {
  if (status != CG_OK) {
    throw Error(
        ErrorCode::io_malformed,
        "CGNS: cannot " + what + (name.empty() ? "" : " '" + name + "'") + ": " + cg_get_error());
  }
}

[[noreturn]] void unsupported(const std::string& path, const std::string& what) {
  throw Error(ErrorCode::io_unsupported, path + ": " + what);
}
[[noreturn]] void malformed(const std::string& path, const std::string& what) {
  throw Error(ErrorCode::io_malformed, path + ": " + what);
}

// docs/contracts/mesh-info.md, "Normalized boundary types", CGNS column.
BcKind bc_kind(std::string_view type) {
  if (type == "BCWall" || type == "BCWallViscous") return BcKind::wall;
  if (type == "BCWallViscousHeatFlux" || type == "BCWallViscousIsothermal") {
    return BcKind::thermal_wall;
  }
  if (type == "BCWallInviscid") return BcKind::slip;
  if (type == "BCSymmetryPlane") return BcKind::symmetry;
  if (type == "BCFarfield") return BcKind::far_field;
  if (type == "BCInflow" || type == "BCInflowSubsonic" || type == "BCInflowSupersonic" ||
      type == "BCTunnelInflow") {
    return BcKind::inlet;
  }
  if (type == "BCOutflow" || type == "BCOutflowSubsonic" || type == "BCOutflowSupersonic" ||
      type == "BCTunnelOutflow") {
    return BcKind::outlet;
  }
  return BcKind::unspecified;
}

struct Family {
  std::string name;
  std::string parent;   // value of a FamilyName_t child: the group it belongs to
  std::string bc_type;  // FamilyBC, empty if none
};

// Linear element shapes; nullopt for 2D/1D types and anything else.
std::optional<CellShape> cell_shape(CGNS_ENUMT(ElementType_t) type) {
  switch (type) {
    case CGNS_ENUMV(TETRA_4):
      return CellShape::tetra;
    case CGNS_ENUMV(PYRA_5):
      return CellShape::pyramid;
    case CGNS_ENUMV(PENTA_6):
      return CellShape::prism;
    case CGNS_ENUMV(HEXA_8):
      return CellShape::hexa;
    default:
      return std::nullopt;
  }
}

bool is_face_type(CGNS_ENUMT(ElementType_t) type) {
  return type == CGNS_ENUMV(TRI_3) || type == CGNS_ENUMV(QUAD_4);
}

// An element section is read in chunks of at most chunk_elements offsets (32 KiB) and
// chunk_entries connectivity entries (1 MiB), whatever the section's size: what reading a section
// costs beside the elements kept is a constant, and nothing for the points and edges skipped.
constexpr std::size_t chunk_elements = 4096;
constexpr std::int64_t chunk_entries = std::int64_t{1} << 17;

bool is_ignored_type(CGNS_ENUMT(ElementType_t) type) {
  return type == CGNS_ENUMV(NODE) || type == CGNS_ENUMV(BAR_2);
}

// One element: its type and its connectivity (nodes, or signed faces for
// NFACE_n), 1-based as in the file.
struct Element {
  CGNS_ENUMT(ElementType_t) type = CGNS_ENUMV(ElementTypeNull);
  std::vector<cgsize_t> connectivity;
  std::string family;
};

class Reader {
 public:
  explicit Reader(const fs::path& path)
      : path_(path.string()), file_((check_links(path), path), CG_MODE_READ) {}

  // E0134 before the library resolves any link or reads any data.
  static void check_links(const fs::path& path) { reject_outside_data(path); }

  MeshInput read() {
    fn_ = file_.handle();
    int bases = 0;
    ok(cg_nbases(fn_, &bases), "count bases");
    if (bases != 1) unsupported(path_, std::to_string(bases) + " bases (one is supported)");
    char name[33];
    int cell_dim = 0, phys_dim = 0;
    ok(cg_base_read(fn_, 1, name, &cell_dim, &phys_dim), "read base");
    if (cell_dim != 3) unsupported(path_, "a base of cell dimension " + std::to_string(cell_dim));
    int zones = 0;
    ok(cg_nzones(fn_, 1, &zones), "count zones");
    if (zones != 1) unsupported(path_, std::to_string(zones) + " zones (one is supported)");
    CGNS_ENUMT(ZoneType_t) zone_type{};
    ok(cg_zone_type(fn_, 1, 1, &zone_type), "read zone type");
    if (zone_type != CGNS_ENUMV(Unstructured)) unsupported(path_, "a structured zone");
    cgsize_t size[3] = {0, 0, 0};
    ok(cg_zone_read(fn_, 1, 1, name, size), "read zone");
    n_nodes_ = size[0];
    n_cells_ = size[1];

    read_families();
    read_coordinates();
    read_sections();
    read_bcs();
    read_connectivity();
    read_subregions();
    restore_zone_names();
    return has_nface_ ? face_based() : element_based();
  }

 private:
  struct Section;
  void read_families() {
    int n = 0;
    ok(cg_nfamilies(fn_, 1, &n), "count families");
    for (int f = 1; f <= n; ++f) {
      char name[33];
      int bcs = 0, geos = 0;
      ok(cg_family_read(fn_, 1, f, name, &bcs, &geos), "read family");
      Family family{.name = name, .parent = "", .bc_type = ""};
      if (bcs > 0) {
        char bc_name[33];
        CGNS_ENUMT(BCType_t) type{};
        ok(cg_fambc_read(fn_, 1, f, 1, bc_name, &type), "read FamilyBC", name);
        family.bc_type = BCTypeName[type];
      }
      int names = 0;
      ok(cg_nfamily_names(fn_, 1, f, &names), "count family names", name);
      if (names > 0) {
        char node[33];
        char parent[CG_MAX_GOTO_DEPTH * 33 + 1];
        ok(cg_family_name_read(fn_, 1, f, 1, node, parent), "read family name", name);
        family.parent = parent;
      }
      families_[family.name] = family;
    }
  }

  // A FamilyName may be a path (/Base/Wall); families are children of the
  // base, so its last component names the family.
  [[nodiscard]] std::string family_key(const std::string& reference) const {
    const auto slash = reference.rfind('/');
    std::string key = slash == std::string::npos ? reference : reference.substr(slash + 1);
    if (!families_.contains(key))
      malformed(path_, "reference to family '" + reference + "' that does not exist");
    return key;
  }

  // The group a family belongs to: the root of its FamilyName_t chain.
  [[nodiscard]] std::string group_of(std::string family) const {
    for (int depth = 0; depth < 16; ++depth) {
      const auto it = families_.find(family);
      if (it == families_.end() || it->second.parent.empty()) return family;
      family = it->second.parent;
    }
    malformed(path_, "family hierarchy deeper than 16 levels");
  }

  [[nodiscard]] std::string family_bc(std::string family) const {
    for (int depth = 0; depth < 16; ++depth) {
      const auto it = families_.find(family);
      if (it == families_.end()) return "";
      if (!it->second.bc_type.empty()) return it->second.bc_type;
      family = it->second.parent;
    }
    return "";
  }

  void read_coordinates() {
    const cgsize_t first = 1;
    const cgsize_t last = n_nodes_;
    for (const auto& [axis, out] : {std::pair{"CoordinateX", &x_}, std::pair{"CoordinateY", &y_},
                                    std::pair{"CoordinateZ", &z_}}) {
      out->resize(at(n_nodes_));
      ok(cg_coord_read(fn_, 1, 1, axis, CGNS_ENUMV(RealDouble), &first, &last, out->data()), "read",
         axis);
    }
  }

  void read_sections() {
    int n = 0;
    ok(cg_nsections(fn_, 1, 1, &n), "count sections");
    const CgioTree tree(fn_);
    std::map<std::string, double> section_nodes;
    if (const auto base = tree.child(tree.root(), "CGNSBase_t")) {
      if (const auto zone = tree.child(*base, "Zone_t")) {
        for (const double id : tree.children(*zone)) {
          if (tree.label(id) == "Elements_t") section_nodes.emplace(tree.name(id), id);
        }
      }
    }
    for (int s = 1; s <= n; ++s) {
      char name[33];
      CGNS_ENUMT(ElementType_t) type{};
      cgsize_t start = 0, end = 0;
      int boundary = 0, parent_flag = 0;
      ok(cg_section_read(fn_, 1, 1, s, name, &type, &start, &end, &boundary, &parent_flag),
         "read section");
      if (start < 1 || end < start) malformed(path_, std::string("section '") + name + "' range");
      std::string family;
      ok(cg_goto(fn_, 1, "Zone_t", 1, "Elements_t", s, "end"), "go to section", name);
      char family_name[CG_MAX_GOTO_DEPTH * 33 + 1];
      if (cg_famname_read(family_name) == CG_OK) family = family_name;
      const auto count = at(end - start + 1);
      if (is_ignored_type(type)) {
        // Points and edges the reader never uses: their ids resolve, their nodes are not read.
        sections_.push_back(
            Section{.start = start,
                    .count = count,
                    .elements = {},
                    .skipped = {},
                    .ignored = Element{.type = type, .connectivity = {}, .family = family}});
        continue;
      }

      int fixed_nodes = 0;
      const bool poly =
          type == CGNS_ENUMV(MIXED) || type == CGNS_ENUMV(NGON_n) || type == CGNS_ENUMV(NFACE_n);
      if (!poly) {
        ok(cg_npe(type, &fixed_nodes), "size element", name);
        if (fixed_nodes <= 0)
          unsupported(path_, std::string("element type ") + ElementTypeName[type]);
      }
      const auto node = section_nodes.find(name);
      if (node == section_nodes.end()) malformed(path_, std::string("section '") + name + "'");
      const auto connectivity = tree.child(node->second, "DataArray_t", "ElementConnectivity");
      const auto offsets = poly ? tree.child(node->second, "DataArray_t", "ElementStartOffset")
                                : std::optional<double>{};
      if (!connectivity || (poly && !offsets)) {
        // ElementStartOffset came with CGNS 4.0; earlier MIXED and NGON sections lack it.
        unsupported(path_, std::string("section '") + name + "' without ElementStartOffset");
      }
      const std::int64_t entries = tree.values(*connectivity);
      if (!poly && entries != std::int64_t{fixed_nodes} * static_cast<std::int64_t>(count)) {
        malformed(path_, std::string("section '") + name + "' connectivity size");
      }
      if (poly && tree.values(*offsets) != static_cast<std::int64_t>(count) + 1) {
        inconsistent_offsets(name);
      }
      Section& section = sections_.emplace_back(Section{
          .start = start,
          .count = count,
          .elements = {},
          .skipped = {},
          .ignored = Element{.type = CGNS_ENUMV(BAR_2), .connectivity = {}, .family = family}});
      if (type != CGNS_ENUMV(MIXED)) section.elements.resize(count);
      // The section in chunks of at most chunk_elements elements and chunk_entries connectivity
      // entries, so that reading costs a constant beside what is kept; a larger element is read
      // into its own storage.
      std::vector<cgsize_t> bounds(chunk_elements + 1), data;
      std::int64_t previous = 0;  // the offset that ends the elements read so far
      for (std::size_t first = 0; first < count; first += chunk_elements) {
        const std::size_t size = std::min(chunk_elements, count - first);
        const std::span<cgsize_t> b(bounds.data(), size + 1);
        if (poly) {
          tree.read_integers(*offsets, static_cast<std::int64_t>(first), b);
        } else {
          for (std::size_t e = 0; e <= size; ++e)
            b[e] = static_cast<std::int64_t>(first + e) * fixed_nodes;
        }
        // ElementStartOffset starts at 0, gives every element at least one entry and ends at
        // the connectivity size, checked before anything is allocated from it.
        if (b[0] != previous || (first + size == count && b[size] != entries))
          inconsistent_offsets(name);
        for (std::size_t e = 0; e < size; ++e) {
          if (b[e + 1] <= b[e] || b[e + 1] > entries) inconsistent_offsets(name);
        }
        previous = b[size];
        for (std::size_t lo = 0; lo < size;) {
          std::size_t hi = lo + 1;
          while (hi < size && b[hi + 1] - b[lo] <= chunk_entries) ++hi;
          if (b[hi] - b[lo] > chunk_entries) {
            std::vector<cgsize_t> own(static_cast<std::size_t>(b[hi] - b[lo]));
            tree.read_integers(*connectivity, b[lo], own);
            read_element(section, type, first + lo, own, &own, family, name);
            lo = hi;
            continue;
          }
          data.resize(static_cast<std::size_t>(b[hi] - b[lo]));
          tree.read_integers(*connectivity, b[lo], data);
          for (std::size_t e = lo; e < hi; ++e) {
            read_element(
                section, type, first + e,
                std::span<const cgsize_t>(data).subspan(static_cast<std::size_t>(b[e] - b[lo]),
                                                        static_cast<std::size_t>(b[e + 1] - b[e])),
                nullptr, family, name);
          }
          lo = hi;
        }
      }
    }
    std::ranges::sort(sections_, {}, &Section::start);
    for (std::size_t i = 1; i < sections_.size(); ++i) {
      const auto& previous = sections_[i - 1];
      if (sections_[i].start < previous.start + static_cast<cgsize_t>(previous.count)) {
        malformed(path_, "two sections share element ids");
      }
    }
    if (has_nface_ && !has_ngon_) malformed(path_, "NFACE_n cells without NGON_n faces");
    if (has_nface_ && has_standard_cells_) {
      unsupported(path_, "NFACE_n cells mixed with standard volume elements");
    }
  }

  [[noreturn]] void inconsistent_offsets(const char* section) const {
    malformed(path_,
              std::string("ElementStartOffset of section '") + section + "' is inconsistent");
  }

  // Element `e` of `section`, its entries `entries`: kept, or a point or an edge of a MIXED
  // section, skipped. With `owner`, the vector `entries` views, whose storage the element takes.
  void read_element(Section& section, CGNS_ENUMT(ElementType_t) type, std::size_t e,
                    std::span<const cgsize_t> entries, std::vector<cgsize_t>* owner,
                    const std::string& family, const char* name) {
    CGNS_ENUMT(ElementType_t) element_type = type;
    auto first = entries.begin();
    const auto last = entries.end();
    if (type == CGNS_ENUMV(MIXED)) {
      const cgsize_t code = *first;
      ++first;
      int nodes = 0;
      if (code <= CGNS_ENUMV(NODE) || code >= CGNS_ENUMV(MIXED) ||
          cg_npe(static_cast<CGNS_ENUMT(ElementType_t)>(code), &nodes) != CG_OK || nodes <= 0 ||
          last - first != nodes) {
        malformed(path_, std::string("MIXED element of section '") + name + "' is invalid");
      }
      element_type = static_cast<CGNS_ENUMT(ElementType_t)>(code);
      // Points and edges of a MIXED section: their index alone, as in a section of them.
      if (is_ignored_type(element_type)) {
        section.skip(e);
        return;
      }
      section.elements.emplace_back();
    }
    Element& element = type == CGNS_ENUMV(MIXED) ? section.elements.back() : section.elements[e];
    element.type = element_type;
    if (owner != nullptr) {
      owner->erase(owner->begin(), owner->begin() + (first - entries.begin()));
      element.connectivity = std::move(*owner);
    } else {
      element.connectivity.assign(first, last);
    }
    element.family = family;
    classify(element, name);
  }

  // The element of id `id`, or null when no section holds it.
  [[nodiscard]] const Element* element(cgsize_t id) const {
    const auto it = std::ranges::upper_bound(sections_, id, {}, &Section::start);
    if (it == sections_.begin()) return nullptr;
    const Section& section = *std::prev(it);
    const cgsize_t index = id - section.start;
    if (!std::cmp_less(index, section.count)) return nullptr;
    return section.at(at(index));
  }

  void classify(const Element& element, const char* section) {
    const auto type = element.type;
    if (type == CGNS_ENUMV(NGON_n)) {
      has_ngon_ = true;
    } else if (type == CGNS_ENUMV(NFACE_n)) {
      has_nface_ = true;
    } else if (cell_shape(type)) {
      has_standard_cells_ = true;
    } else if (!is_face_type(type) && !is_ignored_type(type)) {
      unsupported(path_, std::string("element type ") + ElementTypeName[type] + " in section '" +
                             section + "' (linear elements only)");
    }
  }

  // Elements a point set designates: a range or a list of 1-based ids.
  std::vector<cgsize_t> point_set(CGNS_ENUMT(PointSetType_t) type, std::vector<cgsize_t> points,
                                  const std::string& owner) {
    if (type == CGNS_ENUMV(PointRange) || type == CGNS_ENUMV(ElementRange)) {
      if (points.size() != 2 || points[1] < points[0] || points[0] < 1) {
        malformed(path_, "invalid range in '" + owner + "'");
      }
      std::vector<cgsize_t> ids;
      for (cgsize_t i = points[0]; i <= points[1]; ++i) ids.push_back(i);
      return ids;
    }
    if (type != CGNS_ENUMV(PointList) && type != CGNS_ENUMV(ElementList)) {
      unsupported(path_, "point set type of '" + owner + "'");
    }
    return points;
  }

  void read_bcs() {
    int n = 0;
    ok(cg_nbocos(fn_, 1, 1, &n), "count BCs");
    for (int b = 1; b <= n; ++b) {
      char name[33];
      CGNS_ENUMT(BCType_t) type{};
      CGNS_ENUMT(PointSetType_t) set{};
      cgsize_t points = 0, normal_size = 0;
      int normal_index[3];
      CGNS_ENUMT(DataType_t) normal_type{};
      int datasets = 0;
      ok(cg_boco_info(fn_, 1, 1, b, name, &type, &set, &points, normal_index, &normal_size,
                      &normal_type, &datasets),
         "read BC");
      CGNS_ENUMT(GridLocation_t) location{};
      ok(cg_boco_gridlocation_read(fn_, 1, 1, b, &location), "read BC location", name);
      std::vector<cgsize_t> ids(at(points));
      ok(cg_boco_read(fn_, 1, 1, b, ids.data(), nullptr), "read BC", name);
      ids = point_set(set, std::move(ids), name);
      ok(cg_goto(fn_, 1, "Zone_t", 1, "ZoneBC_t", 1, "BC_t", b, "end"), "go to BC", name);
      char family[CG_MAX_GOTO_DEPTH * 33 + 1];
      const std::string family_name = cg_famname_read(family) == CG_OK ? family : "";
      // A BC that names a family belongs to that family's group; the group,
      // not the BC node, is the zone the user sets a condition on.
      const std::string family_ref = family_name.empty() ? "" : family_key(family_name);
      if (type == CGNS_ENUMV(FamilySpecified) && family_ref.empty()) {
        malformed(path_, std::string("BC '") + name + "' is FamilySpecified without a family");
      }
      const std::string zone = family_ref.empty() ? name : group_of(family_ref);
      std::string bc_type = BCTypeName[type];
      if (type == CGNS_ENUMV(FamilySpecified)) bc_type = family_bc(family_ref);
      if (location == CGNS_ENUMV(Vertex) && set != CGNS_ENUMV(ElementList) &&
          set != CGNS_ENUMV(ElementRange)) {
        unsupported(path_,
                    std::string("BC '") + name + "' given by vertices (element point sets only)");
      }
      // The ids name elements; their dimension says whether the BC labels
      // faces or cells (some writers mark surface BCs as CellCenter).
      switch (dimension_of(ids, name)) {
        case 2:
          add_face_zone(zone, bc_type, ids, b, name);
          break;
        case 3:
          add_cell_zone(zone, ids, std::string("CGNS BC_t '") + name + "' over cells");
          break;
        default:
          break;  // a BC on lower-dimensional elements labels no face or cell
      }
    }
  }

  // Dimension shared by the elements `ids` designate: 3 for cells, 2 for
  // faces, 1 or less for points and edges.
  int dimension_of(const std::vector<cgsize_t>& ids, const char* owner) const {
    int dimension = -1;
    for (const cgsize_t id : ids) {
      if (element(id) == nullptr) {
        malformed(path_, std::string("'") + owner + "' refers to element " + std::to_string(id) +
                             " that does not exist");
      }
      const auto type = element(id)->type;
      const int d = (type == CGNS_ENUMV(NFACE_n) || cell_shape(type))    ? 3
                    : (type == CGNS_ENUMV(NGON_n) || is_face_type(type)) ? 2
                                                                         : 1;
      if (dimension != -1 && d != dimension) {
        malformed(path_, std::string("'") + owner + "' mixes elements of different dimensions");
      }
      dimension = d;
    }
    return dimension;
  }

  void read_subregions() {
    int n = 0;
    ok(cg_nsubregs(fn_, 1, 1, &n), "count subregions");
    for (int s = 1; s <= n; ++s) {
      char name[33];
      int dimension = 0;
      CGNS_ENUMT(GridLocation_t) location{};
      CGNS_ENUMT(PointSetType_t) set{};
      cgsize_t points = 0;
      int bc_name_length = 0, gc_name_length = 0;
      ok(cg_subreg_info(fn_, 1, 1, s, name, &dimension, &location, &set, &points, &bc_name_length,
                        &gc_name_length),
         "read subregion");
      if (location != CGNS_ENUMV(CellCenter) || points == 0) continue;
      std::vector<cgsize_t> ids(at(points));
      ok(cg_subreg_ptset_read(fn_, 1, 1, s, ids.data()), "read subregion", name);
      ids = point_set(set, std::move(ids), name);
      ok(cg_goto(fn_, 1, "Zone_t", 1, "ZoneSubRegion_t", s, "end"), "go to subregion", name);
      char family[CG_MAX_GOTO_DEPTH * 33 + 1];
      const std::string zone =
          cg_famname_read(family) == CG_OK ? group_of(family_key(family)) : name;
      if (dimension_of(ids, name) != 3)
        malformed(path_, std::string("subregion '") + name + "' does not select cells");
      add_cell_zone(zone, ids, std::string("CGNS ZoneSubRegion_t '") + name + "'");
    }
  }

  // Files Autan writes (results.md) map node names to zone names in the
  // ZoneNames descriptor of the base; zones keep their own names and those
  // declared without faces stay visible.
  // mesh.md, Periodic pairs and Readers: only ZoneNames is read among the base's descriptors,
  // sized from its node first; the others are never read.
  void restore_zone_names() {
    const CgioTree tree(fn_);
    const auto base = tree.child(tree.root(), "CGNSBase_t");
    const auto descriptor = base ? tree.child(*base, "Descriptor_t", "ZoneNames") : std::nullopt;
    if (descriptor) {
      // Each entry: a node name of at most 32 bytes and a zone name of at most 4096, both escaped
      // by the writer's ensure_ascii (6 bytes per byte at worst), and their framing.
      // Its keys are the families that results.md's writer declares, one per zone.
      check_text(path_, "ZoneNames descriptor",
                 static_cast<std::size_t>(tree.logical_bytes(*descriptor)),
                 (families_.size() + 1) * 25600);
      nlohmann::json names;
      try {
        names = nlohmann::json::parse(tree.text(*descriptor));
        for (const char* side : {"boundary", "cell"}) {
          for (const auto& [key, value] : names.at(side).items()) {
            if (value.is_string())
              check_text(path_, "zone name", value.get_ref<const std::string&>().size());
          }
        }
        for (auto& zone : face_zones_) {
          if (names.at("boundary").contains(zone.name))
            zone.name = names["boundary"][zone.name].get<std::string>();
        }
        for (auto& zone : cell_zones_) {
          if (names.at("cell").contains(zone.name))
            zone.name = names["cell"][zone.name].get<std::string>();
        }
        for (const auto& [node, zone] : names.at("boundary").items()) {
          if (face_zone_index_.contains(node)) continue;
          const std::string type = families_.contains(node) ? families_.at(node).bc_type : "";
          face_zone_index_[node] = static_cast<index_t>(face_zones_.size());
          face_zones_.push_back(
              FaceZone{.name = zone.get<std::string>(),
                       .bc_type = type,
                       .kind = bc_kind(type),
                       .boundary = true,
                       .provenance = Provenance{
                           .source_id = -1, .detail = "CGNS family '" + node + "' without faces"}});
        }
      } catch (const nlohmann::json::exception& e) {
        malformed(path_,
                  std::string("ZoneNames is not the descriptor results.md defines: ") + e.what());
      }
    }
  }

 public:
  // After assembly: the core matched the faces geometrically, and the pairs
  // the file declares must be the ones it found, face for face (compared by
  // their node sets), not replaced by them.
  void check_declared_pairs(const Mesh& mesh, const AssemblyMap& map) const {
    const auto key = [](std::vector<index_t> nodes) {
      std::ranges::sort(nodes);
      return nodes;
    };
    const auto mesh_face = [&](index_t f) {
      std::vector<index_t> nodes;
      for (const index_t p : mesh.face_nodes(f)) nodes.push_back(map.point_source[at(p)]);
      return key(std::move(nodes));
    };
    std::map<std::vector<index_t>, std::vector<index_t>> partner;
    const auto off = mesh.boundary_zone_offsets();
    for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
      const auto partners = mesh.periodic_partners(p);
      for (std::size_t k = 0; k < partners.size(); ++k) {
        partner[mesh_face(off[at(mesh.periodic_pairs()[p].zone_a)] + static_cast<index_t>(k))] =
            mesh_face(partners[k]);
      }
    }
    const auto element_face = [&](cgsize_t id) {
      std::vector<index_t> nodes;
      for (const cgsize_t q : element(id)->connectivity) nodes.push_back(node(q));
      return key(std::move(nodes));
    };
    for (const auto& [faces, donor] : declared_) {
      for (std::size_t k = 0; k < faces.size(); ++k) {
        const auto it = partner.find(element_face(faces[k]));
        if (it == partner.end() || it->second != element_face(donor[k])) {
          malformed(path_, "a periodic connectivity pairs face " + std::to_string(faces[k]) +
                               " with a face it does not translate onto");
        }
      }
    }
  }

 private:
  // Periodic pairs: 1-to-1 connections of the zone with itself carrying a
  // Periodic_t translation (SIDS GridConnectivity_t), as two reciprocal
  // connections, each naming the zone of its own faces. The first of a
  // reciprocal pair is zone a. The core matches the faces again.
  void read_connectivity() {
    int n = 0;
    ok(cg_nconns(fn_, 1, 1, &n), "count connectivities");
    struct Side {
      std::string name;
      std::vector<cgsize_t> faces, donor;
      std::array<double, 3> translation;
    };
    std::vector<Side> sides;
    char zone_name[33];
    cgsize_t size[3];
    ok(cg_zone_read(fn_, 1, 1, zone_name, size), "read zone");
    for (int i = 1; i <= n; ++i) {
      char name[33], donor_name[33];
      CGNS_ENUMT(GridLocation_t) location{};
      CGNS_ENUMT(GridConnectivityType_t) type{};
      CGNS_ENUMT(PointSetType_t) set{}, donor_set{};
      CGNS_ENUMT(ZoneType_t) donor_zone{};
      CGNS_ENUMT(DataType_t) donor_type{};
      cgsize_t points = 0, donor_points = 0;
      ok(cg_conn_info(fn_, 1, 1, i, name, &location, &type, &set, &points, donor_name, &donor_zone,
                      &donor_set, &donor_type, &donor_points),
         "read connectivity");
      if (type != CGNS_ENUMV(Abutting1to1) || location != CGNS_ENUMV(FaceCenter) ||
          set != CGNS_ENUMV(PointList) || donor_set != CGNS_ENUMV(PointListDonor) ||
          std::string_view(donor_name) != zone_name || points != donor_points) {
        unsupported(path_, std::string("connectivity '") + name +
                               "' other than a face-to-face connection of the zone with itself");
      }
      std::size_t elements = 0;
      for (const auto& section : sections_) elements += section.count;
      if (points < 1 || std::cmp_greater(points, elements)) {
        malformed(path_, std::string("connectivity '") + name + "' lists " +
                             std::to_string(points) + " faces");
      }
      Side side{.name = name,
                .faces = std::vector<cgsize_t>(at(points)),
                .donor = std::vector<cgsize_t>(at(points)),
                .translation = {}};
      ok(cg_conn_read(fn_, 1, 1, i, side.faces.data(), CGNS_ENUMV(LongInteger), side.donor.data()),
         "read connectivity", name);
      for (const auto* list : {&side.faces, &side.donor}) {
        std::vector<cgsize_t> sorted = *list;
        std::ranges::sort(sorted);
        if (std::ranges::adjacent_find(sorted) != sorted.end())
          malformed(path_, std::string("connectivity '") + name + "' lists a face twice");
      }
      if (dimension_of(side.faces, name) != 2 || dimension_of(side.donor, name) != 2) {
        malformed(path_,
                  std::string("connectivity '") + name + "' joins elements that are not faces");
      }
      if (cg_goto(fn_, 1, "Zone_t", 1, "ZoneGridConnectivity_t", 1, "GridConnectivity_t", i,
                  "GridConnectivityProperty_t", 1, "Periodic_t", 1, "end") != CG_OK) {
        unsupported(path_, std::string("connectivity '") + name + "' that is not periodic");
      }
      int arrays = 0;
      ok(cg_narrays(&arrays), "count Periodic arrays", name);
      bool translated = false;
      for (int a = 1; a <= arrays; ++a) {
        char array[33];
        CGNS_ENUMT(DataType_t) data_type{};
        int dims = 0;
        cgsize_t extent[12];
        ok(cg_array_info(a, array, &data_type, &dims, extent), "read Periodic array", name);
        if (dims != 1 || extent[0] != 3)
          malformed(path_, std::string("Periodic of '") + name + "' is not 3D");
        std::array<double, 3> v{};
        ok(cg_array_read_as(a, CGNS_ENUMV(RealDouble), v.data()), "read Periodic array", name);
        if (std::string_view(array) == "Translation") {
          side.translation = v;
          translated = true;
        } else if (std::string_view(array) == "RotationAngle" && v != std::array<double, 3>{}) {
          unsupported(path_, std::string("rotational periodicity in '") + name + "'");
        }
      }
      if (!translated)
        malformed(path_, std::string("Periodic of '") + name + "' has no Translation");
      // Files of results.md keep the translation to the bit in a descriptor;
      // it must round to the R4 one.
      // Only that descriptor is read, sized from its node first.
      const CgioTree tree(fn_);
      std::optional<double> node = tree.child(tree.root(), "CGNSBase_t");
      for (const auto& [label, child] : {std::pair<const char*, std::string>{"Zone_t", ""},
                                         {"ZoneGridConnectivity_t", ""},
                                         {"GridConnectivity_t", name},
                                         {"Descriptor_t", "PeriodicTranslation"}}) {
        if (node) node = tree.child(*node, label, child);
      }
      if (node) {
        check_text(path_, "PeriodicTranslation descriptor",
                   static_cast<std::size_t>(tree.logical_bytes(*node)));
        std::array<double, 3> exact{};
        const nlohmann::json value = nlohmann::json::parse(tree.text(*node), nullptr, false);
        if (!value.is_array() || value.size() != 3 ||
            !std::ranges::all_of(
                value, [](const nlohmann::json& v) { return v.is_number() && !v.is_boolean(); })) {
          malformed(path_,
                    std::string("PeriodicTranslation of '") + name + "' is not three numbers");
        }
        for (std::size_t k = 0; k < 3; ++k) exact[k] = value[k].get<double>();
        for (std::size_t k = 0; k < 3; ++k) {
          if (!std::isfinite(exact[k]) ||
              static_cast<double>(static_cast<float>(exact[k])) != side.translation[k]) {
            malformed(path_, std::string("PeriodicTranslation of '") + name +
                                 "' contradicts its Periodic_t translation");
          }
        }
        side.translation = exact;
      }
      sides.push_back(std::move(side));
    }
    // Reciprocal connections declare the same face pairs, in any order.
    const auto pairs_of = [](const Side& side, bool mirrored) {
      std::vector<std::pair<cgsize_t, cgsize_t>> pairs;
      pairs.reserve(side.faces.size());
      for (std::size_t k = 0; k < side.faces.size(); ++k) {
        pairs.emplace_back(mirrored ? side.donor[k] : side.faces[k],
                           mirrored ? side.faces[k] : side.donor[k]);
      }
      std::ranges::sort(pairs);
      return pairs;
    };
    std::vector<char> used(sides.size(), 0);
    for (std::size_t i = 0; i < sides.size(); ++i) {
      if (used[i] != 0) continue;
      const auto& s = sides[i];
      const auto declared = pairs_of(s, false);
      const std::array<double, 3> back{-s.translation[0], -s.translation[1], -s.translation[2]};
      std::size_t j = i + 1;
      while (j < sides.size() && (used[j] != 0 || sides[j].translation != back ||
                                  pairs_of(sides[j], true) != declared)) {
        ++j;
      }
      if (j == sides.size()) {
        malformed(path_, "periodic connectivity '" + s.name + "' has no reciprocal connection");
      }
      used[i] = used[j] = 1;
      declared_.emplace_back(s.faces, s.donor);
      const index_t a = add_face_zone(s.name, "", s.faces, -1, s.name.c_str(), BcKind::periodic);
      const index_t b = add_face_zone(sides[j].name, "", sides[j].faces, -1, sides[j].name.c_str(),
                                      BcKind::periodic);
      periodic_.push_back({.zone_a = a, .zone_b = b, .translation = s.translation});
    }
  }

  index_t add_face_zone(const std::string& zone, const std::string& bc_type,
                        const std::vector<cgsize_t>& ids, int bc, const char* bc_name,
                        std::optional<BcKind> kind = std::nullopt) {
    auto [it, inserted] =
        face_zone_index_.try_emplace(zone, static_cast<index_t>(face_zones_.size()));
    if (inserted) {
      face_zones_.push_back(FaceZone{
          .name = zone,
          .bc_type = bc_type,
          .kind = kind.value_or(bc_kind(bc_type)),
          .boundary = true,
          .provenance =
              Provenance{.source_id = bc, .detail = std::string("CGNS BC_t '") + bc_name + "'"}});
    } else if (face_zones_[at(it->second)].bc_type != bc_type) {
      throw Error(ErrorCode::io_ambiguous_zone,
                  path_ + ": BCs of zone '" + zone + "' declare different BC types");
    }
    for (const cgsize_t id : ids) {
      if (element(id) == nullptr)
        malformed(path_, std::string("BC '") + bc_name + "' element id out of range");
      auto [slot, fresh] = face_label_.try_emplace(id, it->second);
      if (!fresh && slot->second != it->second) {
        throw Error(ErrorCode::io_ambiguous_zone, path_ + ": a face belongs to two BCs");
      }
    }
    return it->second;
  }

  void add_cell_zone(const std::string& zone, const std::vector<cgsize_t>& ids,
                     const std::string& detail) {
    auto [it, inserted] =
        cell_zone_index_.try_emplace(zone, static_cast<index_t>(cell_zones_.size()));
    if (inserted) {
      cell_zones_.push_back(CellZone{.name = zone,
                                     .type = "",
                                     .kind = CellKind::unspecified,
                                     .provenance = Provenance{.source_id = -1, .detail = detail}});
    }
    for (const cgsize_t id : ids) {
      auto [slot, fresh] = cell_label_.try_emplace(id, it->second);
      if (!fresh && slot->second != it->second) {
        throw Error(ErrorCode::io_ambiguous_zone, path_ + ": a cell belongs to two cell zones");
      }
    }
  }

  // Cell zone of every cell element, in element order; families of the
  // sections name zones for cells no other source labels.
  std::vector<index_t> cell_zones_of(const std::vector<cgsize_t>& cells) {
    std::vector<index_t> zone(cells.size(), -1);
    for (std::size_t c = 0; c < cells.size(); ++c) {
      const auto it = cell_label_.find(cells[c]);
      if (it != cell_label_.end()) {
        zone[c] = it->second;
        continue;
      }
      const std::string& family = element(cells[c])->family;
      if (!family.empty()) {
        add_cell_zone(group_of(family_key(family)), {cells[c]},
                      "CGNS family '" + family + "' of an element section");
        zone[c] = cell_label_.at(cells[c]);
      }
    }
    if (std::ranges::find(zone, -1) != zone.end()) {
      const auto fluid = static_cast<index_t>(cell_zones_.size());
      cell_zones_.push_back(CellZone{
          .name = "fluid",
          .type = "",
          .kind = CellKind::unspecified,
          .provenance =
              Provenance{.source_id = -1, .detail = "synthesised: cells outside every cell zone"}});
      std::ranges::replace(zone, index_t{-1}, fluid);
    }
    return zone;
  }

  std::vector<cgsize_t> element_ids(bool (*keep)(const Element&)) const {
    std::vector<cgsize_t> ids;
    for (const Section& section : sections_) {
      for (std::size_t e = 0; e < section.count; ++e) {
        if (keep(*section.at(e))) ids.push_back(section.start + static_cast<cgsize_t>(e));
      }
    }
    return ids;
  }

  void check_cell_count(std::size_t cells) const {
    if (std::cmp_not_equal(cells, n_cells_)) {
      malformed(path_, "the zone declares " + std::to_string(n_cells_) + " cells and holds " +
                           std::to_string(cells));
    }
  }

  [[nodiscard]] index_t node(cgsize_t id) const {
    if (id < 1 || id > n_nodes_) malformed(path_, "node id out of range");
    return static_cast<index_t>(id - 1);
  }

  MeshInput face_based() {
    const auto faces = element_ids([](const Element& e) { return e.type == CGNS_ENUMV(NGON_n); });
    const auto cells = element_ids([](const Element& e) { return e.type == CGNS_ENUMV(NFACE_n); });
    check_cell_count(cells.size());
    std::map<cgsize_t, index_t> face_index;
    MeshInput in;
    in.face_node_offsets.push_back(0);
    for (const cgsize_t id : faces) {
      face_index[id] = static_cast<index_t>(face_index.size());
      for (const cgsize_t p : element(id)->connectivity) in.face_nodes.push_back(node(p));
      in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    }
    in.face_cell0.assign(faces.size(), no_cell);
    in.face_cell1.assign(faces.size(), no_cell);
    for (std::size_t c = 0; c < cells.size(); ++c) {
      for (const cgsize_t signed_id : element(cells[c])->connectivity) {
        const auto it = face_index.find(signed_id < 0 ? -signed_id : signed_id);
        if (it == face_index.end()) malformed(path_, "NFACE_n refers to a non-NGON element");
        // A positive id: the face normal points out of the cell (SIDS).
        index_t& side =
            signed_id > 0 ? in.face_cell0[at(it->second)] : in.face_cell1[at(it->second)];
        if (side != no_cell) {
          throw Error(ErrorCode::mesh_shared_face_orientation,
                      path_ + ": two cells list a face with the same orientation");
        }
        side = static_cast<index_t>(c);
      }
    }
    for (std::size_t f = 0; f < faces.size(); ++f) {
      if (in.face_cell0[f] == no_cell) {
        if (in.face_cell1[f] == no_cell) malformed(path_, "an NGON_n face belongs to no cell");
        // Boundary face listed inward: turn it outward.
        std::swap(in.face_cell0[f], in.face_cell1[f]);
        std::reverse(in.face_nodes.begin() + in.face_node_offsets[f] + 1,
                     in.face_nodes.begin() + in.face_node_offsets[f + 1]);
      }
    }
    in.face_zones = face_zones_;
    in.face_zone.assign(faces.size(), -1);
    index_t interior = -1, unassigned = -1;
    for (std::size_t f = 0; f < faces.size(); ++f) {
      const auto label = face_label_.find(faces[f]);
      const bool inner = in.face_cell1[f] != no_cell;
      if (label != face_label_.end()) {
        in.face_zone[f] = label->second;
        if (inner) in.face_zones[at(label->second)].boundary = false;
      } else {
        index_t& zone = inner ? interior : unassigned;
        if (zone == -1) {
          zone = static_cast<index_t>(in.face_zones.size());
          in.face_zones.push_back(FaceZone{
              .name = inner ? "interior" : "unassigned",
              .bc_type = "",
              .kind = BcKind::unspecified,
              .boundary = !inner,
              .provenance =
                  Provenance{.source_id = -1,
                             .detail = inner ? "synthesised: interior faces"
                                             : "synthesised: boundary faces without a BC"}});
        }
        in.face_zone[f] = zone;
      }
    }
    in.cell_zone = cell_zones_of(cells);
    in.cell_zones = cell_zones_;
    in.periodic = periodic_;
    in.x = x_;
    in.y = y_;
    in.z = z_;
    in.source = MeshSource{.format = "cgns", .path = path_, .nodes = 0};
    return in;
  }

  MeshInput element_based() {
    if (has_ngon_) unsupported(path_, "NGON_n faces without NFACE_n cells");
    const auto cells = element_ids([](const Element& e) { return cell_shape(e.type).has_value(); });
    if (cells.empty()) malformed(path_, "no volume elements");
    check_cell_count(cells.size());
    CellwiseMesh mesh;
    mesh.x = x_;
    mesh.y = y_;
    mesh.z = z_;
    const std::vector<index_t> zones = cell_zones_of(cells);
    std::vector<index_t> nodes;
    for (std::size_t c = 0; c < cells.size(); ++c) {
      const Element& e = *element(cells[c]);
      nodes.clear();
      for (const cgsize_t p : e.connectivity) nodes.push_back(node(p));
      const auto shape = cell_shape(e.type);
      if (!shape) malformed(path_, "a cell element lost its shape");
      mesh.add_cell(*shape, nodes, zones[c]);
    }
    for (const auto& [id, zone] : face_label_) {
      const Element& e = *element(id);
      if (!is_face_type(e.type)) malformed(path_, "a face-centred BC refers to a non-face element");
      nodes.clear();
      for (const cgsize_t p : e.connectivity) nodes.push_back(node(p));
      mesh.add_labelled_face(nodes, zone);
    }
    mesh.face_zones = face_zones_;
    mesh.cell_zones = cell_zones_;
    mesh.periodic = periodic_;
    mesh.source = MeshSource{.format = "cgns", .path = path_, .nodes = 0};
    return to_face_based(std::move(mesh));
  }

  std::string path_;
  CgnsFile file_;
  int fn_ = 0;
  cgsize_t n_nodes_ = 0;
  cgsize_t n_cells_ = 0;
  std::vector<double> x_, y_, z_;
  std::map<std::string, Family> families_;
  // A section's elements but its points and edges: none in a section of them, those of a
  // MIXED section listed as runs of consecutive indices in `skipped`, one element standing for
  // them all. A run follows a kept element, so there are at most as many runs as kept elements,
  // plus one.
  struct Run {
    std::size_t first;   // index of its first element
    std::size_t count;   // elements in it
    std::size_t before;  // elements skipped before it
  };
  struct Section {
    cgsize_t start;
    std::size_t count;
    std::vector<Element> elements;
    std::vector<Run> skipped;  // ascending
    Element ignored;

    void skip(std::size_t index) {
      if (!skipped.empty() && skipped.back().first + skipped.back().count == index) {
        ++skipped.back().count;
      } else {
        const std::size_t before =
            skipped.empty() ? 0 : skipped.back().before + skipped.back().count;
        skipped.push_back(Run{.first = index, .count = 1, .before = before});
      }
    }
    [[nodiscard]] const Element* at(std::size_t index) const {
      if (elements.empty()) return &ignored;
      const auto it = std::ranges::upper_bound(skipped, index, {}, &Run::first);
      if (it == skipped.begin()) return &elements[index];
      const Run& run = *std::prev(it);
      if (index < run.first + run.count) return &ignored;
      return &elements[index - run.before - run.count];
    }
  };
  std::vector<Section> sections_;
  bool has_ngon_ = false, has_nface_ = false, has_standard_cells_ = false;
  std::vector<FaceZone> face_zones_;
  std::map<std::string, index_t> face_zone_index_;
  std::vector<PeriodicPair> periodic_;
  std::vector<std::pair<std::vector<cgsize_t>, std::vector<cgsize_t>>> declared_;  // zone a -> b
  std::map<cgsize_t, index_t> face_label_;
  std::vector<CellZone> cell_zones_;
  std::map<std::string, index_t> cell_zone_index_;
  std::map<cgsize_t, index_t> cell_label_;
};

}  // namespace

Mesh read_cgns(const fs::path& file) {
  std::error_code ec;
  if (!fs::exists(file, ec))
    throw Error(ErrorCode::io_cannot_open, "'" + file.string() + "' does not exist");
  Reader reader(file);
  AssemblyMap map;
  Mesh mesh = assemble(reader.read(), &map);
  reader.check_declared_pairs(mesh, map);
  return mesh;
}

LogicalBytes cgns_logical_bytes(const fs::path& file) {
  reject_outside_data(file);
  const CgioTree tree(file);
  LogicalBytes out;
  // Every Descriptor_t, at any level: cg_open loads them all.
  const auto descriptors = [&](const auto& self, double node) -> void {
    for (const double child : tree.children(node)) {
      if (tree.label(child) == "Descriptor_t") out.descriptors += tree.logical_bytes(child);
      self(self, child);
    }
  };
  descriptors(descriptors, tree.root());
  // The datasets the reader reads, descriptors aside.
  std::int64_t arrays = 0;
  const auto subtree = [&](const auto& self, double node) -> void {
    arrays += tree.logical_bytes(node);
    for (const double child : tree.children(node)) {
      if (tree.label(child) != "Descriptor_t") self(self, child);
    }
  };
  for (const double base : tree.children(tree.root())) {
    if (tree.label(base) != "CGNSBase_t") continue;
    arrays += tree.logical_bytes(base);
    if (const auto names = tree.child(base, "Descriptor_t", "ZoneNames"))
      out.zone_names = tree.logical_bytes(*names);
    for (const double node : tree.children(base)) {
      const std::string label = tree.label(node);
      if (label == "Family_t") subtree(subtree, node);
      if (label != "Zone_t") continue;
      arrays += tree.logical_bytes(node);
      for (const double child : tree.children(node)) {
        const std::string kind = tree.label(child);
        if (kind == "ZoneType_t" || kind == "GridCoordinates_t" || kind == "ZoneBC_t" ||
            kind == "ZoneSubRegion_t" || kind == "FamilyName_t" ||
            kind == "ZoneGridConnectivity_t") {
          subtree(subtree, child);
        } else if (kind == "Elements_t") {
          // A section of points or edges: its header and range only (mesh.md, Readers).
          std::array<std::int64_t, 2> header{};
          tree.read_integers(child, header);
          if (header[0] == CGNS_ENUMV(NODE) || header[0] == CGNS_ENUMV(BAR_2)) {
            arrays += tree.logical_bytes(child);
            if (const auto range = tree.child(child, "IndexRange_t", "ElementRange"))
              arrays += tree.logical_bytes(*range);
          } else {
            subtree(subtree, child);
          }
        }
      }
    }
  }
  out.total = arrays + out.descriptors;
  out.filtered = largest_filtered_bytes(file);
  return out;
}

}  // namespace autan::io
