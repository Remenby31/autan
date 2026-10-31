// CGNS mesh reader through the CGNS mid-level library (SIDS 4.x). One base,
// one unstructured zone; cells as NFACE_n over NGON_n faces, or as linear
// standard elements. See docs/contracts/mesh.md, "Readers".

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cgns_io.h>
#include <cgnslib.h>

#include "autan/core/cellwise.hpp"
#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "cgns_util.hpp"

namespace autan::io {
namespace {

namespace fs = std::filesystem;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

[[noreturn]] void unsupported(const std::string& path, const std::string& what) {
  throw Error(ErrorCode::io_unsupported, path + ": " + what);
}
[[noreturn]] void malformed(const std::string& path, const std::string& what) {
  throw Error(ErrorCode::io_malformed, path + ": " + what);
}

// A mesh hash covers one file: links to other files are rejected (E0134).
void reject_external_links(int cgio, double root, const std::string& path) {
  std::vector<double> pending{root};
  std::vector<double> children;
  while (!pending.empty()) {
    const double node = pending.back();
    pending.pop_back();
    int length = 0;
    if (cgio_is_link(cgio, node, &length) != CG_OK) malformed(path, "unreadable node");
    if (length > 0) {
      int file_length = 0, name_length = 0;
      cgio_link_size(cgio, node, &file_length, &name_length);
      if (file_length > 0) {
        throw Error(ErrorCode::io_external_reference, path + ": link to another file");
      }
      continue;
    }
    int count = 0;
    cgio_number_children(cgio, node, &count);
    if (count == 0) continue;
    children.resize(at(count));
    int got = 0;
    cgio_children_ids(cgio, node, 1, count, &got, children.data());
    pending.insert(pending.end(), children.begin(), children.begin() + got);
  }
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

  // Scans the raw node tree before the library resolves any link.
  static void check_links(const fs::path& path) {
    int cgio = 0;
    if (cgio_open_file(path.string().c_str(), CGIO_MODE_READ, CGIO_FILE_NONE, &cgio) != CG_OK) {
      throw Error(ErrorCode::io_cannot_open, "cannot open '" + path.string() + "' as CGNS");
    }
    double root = 0;
    cgio_get_root_id(cgio, &root);
    try {
      reject_external_links(cgio, root, path.string());
    } catch (...) {
      cgio_close_file(cgio);
      throw;
    }
    cgio_close_file(cgio);
  }

  MeshInput read() {
    fn_ = file_.handle();
    int bases = 0;
    cgns_check(cg_nbases(fn_, &bases), "count bases");
    if (bases != 1) unsupported(path_, std::to_string(bases) + " bases (one is supported)");
    char name[33];
    int cell_dim = 0, phys_dim = 0;
    cgns_check(cg_base_read(fn_, 1, name, &cell_dim, &phys_dim), "read base");
    if (cell_dim != 3) unsupported(path_, "a base of cell dimension " + std::to_string(cell_dim));
    int zones = 0;
    cgns_check(cg_nzones(fn_, 1, &zones), "count zones");
    if (zones != 1) unsupported(path_, std::to_string(zones) + " zones (one is supported)");
    CGNS_ENUMT(ZoneType_t) zone_type{};
    cgns_check(cg_zone_type(fn_, 1, 1, &zone_type), "read zone type");
    if (zone_type != CGNS_ENUMV(Unstructured)) unsupported(path_, "a structured zone");
    cgsize_t size[3] = {0, 0, 0};
    cgns_check(cg_zone_read(fn_, 1, 1, name, size), "read zone");
    n_nodes_ = size[0];

    read_families();
    read_coordinates();
    read_sections();
    read_bcs();
    read_subregions();
    return has_nface_ ? face_based() : element_based();
  }

 private:
  void read_families() {
    int n = 0;
    cgns_check(cg_nfamilies(fn_, 1, &n), "count families");
    for (int f = 1; f <= n; ++f) {
      char name[33];
      int bcs = 0, geos = 0;
      cgns_check(cg_family_read(fn_, 1, f, name, &bcs, &geos), "read family");
      Family family{.name = name, .parent = "", .bc_type = ""};
      if (bcs > 0) {
        char bc_name[33];
        CGNS_ENUMT(BCType_t) type{};
        cgns_check(cg_fambc_read(fn_, 1, f, 1, bc_name, &type), "read FamilyBC", name);
        family.bc_type = BCTypeName[type];
      }
      int names = 0;
      cgns_check(cg_nfamily_names(fn_, 1, f, &names), "count family names", name);
      if (names > 0) {
        char node[33];
        char parent[CG_MAX_GOTO_DEPTH * 33 + 1];
        cgns_check(cg_family_name_read(fn_, 1, f, 1, node, parent), "read family name", name);
        family.parent = parent;
      }
      families_[family.name] = family;
    }
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
      cgns_check(cg_coord_read(fn_, 1, 1, axis, CGNS_ENUMV(RealDouble), &first, &last, out->data()),
                 "read", axis);
    }
  }

  void read_sections() {
    int n = 0;
    cgns_check(cg_nsections(fn_, 1, 1, &n), "count sections");
    for (int s = 1; s <= n; ++s) {
      char name[33];
      CGNS_ENUMT(ElementType_t) type{};
      cgsize_t start = 0, end = 0;
      int boundary = 0, parent_flag = 0;
      cgns_check(cg_section_read(fn_, 1, 1, s, name, &type, &start, &end, &boundary, &parent_flag),
                 "read section");
      if (start < 1 || end < start) malformed(path_, std::string("section '") + name + "' range");
      std::string family;
      cgns_check(cg_goto(fn_, 1, "Zone_t", 1, "Elements_t", s, "end"), "go to section", name);
      char family_name[CG_MAX_GOTO_DEPTH * 33 + 1];
      if (cg_famname_read(family_name) == CG_OK) family = family_name;

      cgsize_t data_size = 0;
      cgns_check(cg_ElementDataSize(fn_, 1, 1, s, &data_size), "size section", name);
      std::vector<cgsize_t> connectivity(at(data_size));
      std::vector<cgsize_t> offsets;
      const auto count = at(end - start + 1);
      const bool poly =
          type == CGNS_ENUMV(MIXED) || type == CGNS_ENUMV(NGON_n) || type == CGNS_ENUMV(NFACE_n);
      if (poly) {
        offsets.resize(count + 1);
        cgns_check(
            cg_poly_elements_read(fn_, 1, 1, s, connectivity.data(), offsets.data(), nullptr),
            "read section", name);
      } else {
        cgns_check(cg_elements_read(fn_, 1, 1, s, connectivity.data(), nullptr), "read section",
                   name);
        int nodes = 0;
        cgns_check(cg_npe(type, &nodes), "size element", name);
        if (nodes <= 0 || at(nodes) * count != connectivity.size()) {
          unsupported(path_, std::string("element type ") + ElementTypeName[type]);
        }
        offsets.resize(count + 1);
        for (std::size_t e = 0; e <= count; ++e) offsets[e] = static_cast<cgsize_t>(e) * nodes;
      }
      if (elements_.size() < at(end)) elements_.resize(at(end));
      for (std::size_t e = 0; e < count; ++e) {
        Element& element = elements_[at(start - 1) + e];
        if (element.type != CGNS_ENUMV(ElementTypeNull))
          malformed(path_, "two sections share element ids");
        auto first = connectivity.begin() + offsets[e];
        const auto last = connectivity.begin() + offsets[e + 1];
        element.type = type;
        if (type == CGNS_ENUMV(MIXED)) {
          if (first == last) malformed(path_, "empty MIXED element");
          element.type = static_cast<CGNS_ENUMT(ElementType_t)>(*first);
          ++first;
        }
        element.connectivity.assign(first, last);
        element.family = family;
        classify(element, name);
      }
    }
    if (has_nface_ && !has_ngon_) malformed(path_, "NFACE_n cells without NGON_n faces");
    if (has_nface_ && has_standard_cells_) {
      unsupported(path_, "NFACE_n cells mixed with standard volume elements");
    }
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
    cgns_check(cg_nbocos(fn_, 1, 1, &n), "count BCs");
    for (int b = 1; b <= n; ++b) {
      char name[33];
      CGNS_ENUMT(BCType_t) type{};
      CGNS_ENUMT(PointSetType_t) set{};
      cgsize_t points = 0, normal_size = 0;
      int normal_index[3];
      CGNS_ENUMT(DataType_t) normal_type{};
      int datasets = 0;
      cgns_check(cg_boco_info(fn_, 1, 1, b, name, &type, &set, &points, normal_index, &normal_size,
                              &normal_type, &datasets),
                 "read BC");
      CGNS_ENUMT(GridLocation_t) location{};
      cgns_check(cg_boco_gridlocation_read(fn_, 1, 1, b, &location), "read BC location", name);
      std::vector<cgsize_t> ids(at(points));
      cgns_check(cg_boco_read(fn_, 1, 1, b, ids.data(), nullptr), "read BC", name);
      ids = point_set(set, std::move(ids), name);
      cgns_check(cg_goto(fn_, 1, "Zone_t", 1, "ZoneBC_t", 1, "BC_t", b, "end"), "go to BC", name);
      char family[CG_MAX_GOTO_DEPTH * 33 + 1];
      const std::string family_name = cg_famname_read(family) == CG_OK ? family : "";
      // A BC that names a family belongs to that family's group; the group,
      // not the BC node, is the zone the user sets a condition on.
      const std::string zone = family_name.empty() ? name : group_of(family_name);
      std::string bc_type = BCTypeName[type];
      if (type == CGNS_ENUMV(FamilySpecified)) bc_type = family_bc(family_name);
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
      if (id < 1 || at(id) > elements_.size()) {
        malformed(path_, std::string("'") + owner + "' refers to element " + std::to_string(id) +
                             " that does not exist");
      }
      const auto type = elements_[at(id - 1)].type;
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
    cgns_check(cg_nsubregs(fn_, 1, 1, &n), "count subregions");
    for (int s = 1; s <= n; ++s) {
      char name[33];
      int dimension = 0;
      CGNS_ENUMT(GridLocation_t) location{};
      CGNS_ENUMT(PointSetType_t) set{};
      cgsize_t points = 0;
      int bc_name_length = 0, gc_name_length = 0;
      cgns_check(cg_subreg_info(fn_, 1, 1, s, name, &dimension, &location, &set, &points,
                                &bc_name_length, &gc_name_length),
                 "read subregion");
      if (location != CGNS_ENUMV(CellCenter) || points == 0) continue;
      std::vector<cgsize_t> ids(at(points));
      cgns_check(cg_subreg_ptset_read(fn_, 1, 1, s, ids.data()), "read subregion", name);
      ids = point_set(set, std::move(ids), name);
      cgns_check(cg_goto(fn_, 1, "Zone_t", 1, "ZoneSubRegion_t", s, "end"), "go to subregion",
                 name);
      char family[CG_MAX_GOTO_DEPTH * 33 + 1];
      const std::string zone = cg_famname_read(family) == CG_OK ? group_of(family) : name;
      add_cell_zone(zone, ids, std::string("CGNS ZoneSubRegion_t '") + name + "'");
    }
  }

  void add_face_zone(const std::string& zone, const std::string& bc_type,
                     const std::vector<cgsize_t>& ids, int bc, const char* bc_name) {
    auto [it, inserted] =
        face_zone_index_.try_emplace(zone, static_cast<index_t>(face_zones_.size()));
    if (inserted) {
      face_zones_.push_back(FaceZone{
          .name = zone,
          .bc_type = bc_type,
          .kind = bc_kind(bc_type),
          .boundary = true,
          .provenance =
              Provenance{.source_id = bc, .detail = std::string("CGNS BC_t '") + bc_name + "'"}});
    } else if (face_zones_[at(it->second)].bc_type != bc_type) {
      throw Error(ErrorCode::io_ambiguous_zone,
                  path_ + ": BCs of zone '" + zone + "' declare different BC types");
    }
    for (const cgsize_t id : ids) {
      if (id < 1 || at(id) > elements_.size())
        malformed(path_, std::string("BC '") + bc_name + "' element id out of range");
      auto [slot, fresh] = face_label_.try_emplace(id, it->second);
      if (!fresh && slot->second != it->second) {
        throw Error(ErrorCode::io_ambiguous_zone, path_ + ": a face belongs to two BCs");
      }
    }
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
      const std::string& family = elements_[at(cells[c] - 1)].family;
      if (!family.empty()) {
        add_cell_zone(group_of(family), {cells[c]},
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
    for (std::size_t e = 0; e < elements_.size(); ++e) {
      if (keep(elements_[e])) ids.push_back(static_cast<cgsize_t>(e + 1));
    }
    return ids;
  }

  [[nodiscard]] index_t node(cgsize_t id) const {
    if (id < 1 || id > n_nodes_) malformed(path_, "node id out of range");
    return static_cast<index_t>(id - 1);
  }

  MeshInput face_based() {
    const auto faces = element_ids([](const Element& e) { return e.type == CGNS_ENUMV(NGON_n); });
    const auto cells = element_ids([](const Element& e) { return e.type == CGNS_ENUMV(NFACE_n); });
    std::map<cgsize_t, index_t> face_index;
    MeshInput in;
    in.face_node_offsets.push_back(0);
    for (const cgsize_t id : faces) {
      face_index[id] = static_cast<index_t>(face_index.size());
      for (const cgsize_t p : elements_[at(id - 1)].connectivity) in.face_nodes.push_back(node(p));
      in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    }
    in.face_cell0.assign(faces.size(), no_cell);
    in.face_cell1.assign(faces.size(), no_cell);
    for (std::size_t c = 0; c < cells.size(); ++c) {
      for (const cgsize_t signed_id : elements_[at(cells[c] - 1)].connectivity) {
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
    in.x = x_;
    in.y = y_;
    in.z = z_;
    in.source = MeshSource{.format = "cgns", .path = path_};
    return in;
  }

  MeshInput element_based() {
    if (has_ngon_) unsupported(path_, "NGON_n faces without NFACE_n cells");
    const auto cells = element_ids([](const Element& e) { return cell_shape(e.type).has_value(); });
    if (cells.empty()) malformed(path_, "no volume elements");
    CellwiseMesh mesh;
    mesh.x = x_;
    mesh.y = y_;
    mesh.z = z_;
    const std::vector<index_t> zones = cell_zones_of(cells);
    std::vector<index_t> nodes;
    for (std::size_t c = 0; c < cells.size(); ++c) {
      const Element& e = elements_[at(cells[c] - 1)];
      nodes.clear();
      for (const cgsize_t p : e.connectivity) nodes.push_back(node(p));
      const auto shape = cell_shape(e.type);
      if (!shape) malformed(path_, "a cell element lost its shape");
      mesh.add_cell(*shape, nodes, zones[c]);
    }
    for (const auto& [id, zone] : face_label_) {
      const Element& e = elements_[at(id - 1)];
      if (!is_face_type(e.type)) malformed(path_, "a face-centred BC refers to a non-face element");
      nodes.clear();
      for (const cgsize_t p : e.connectivity) nodes.push_back(node(p));
      mesh.add_labelled_face(nodes, zone);
    }
    mesh.face_zones = face_zones_;
    mesh.cell_zones = cell_zones_;
    mesh.source = MeshSource{.format = "cgns", .path = path_};
    return to_face_based(std::move(mesh));
  }

  std::string path_;
  CgnsFile file_;
  int fn_ = 0;
  cgsize_t n_nodes_ = 0;
  std::vector<double> x_, y_, z_;
  std::map<std::string, Family> families_;
  std::vector<Element> elements_;
  bool has_ngon_ = false, has_nface_ = false, has_standard_cells_ = false;
  std::vector<FaceZone> face_zones_;
  std::map<std::string, index_t> face_zone_index_;
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
  return assemble(reader.read());
}

}  // namespace autan::io
