// solution.cgns writer: the layout of docs/contracts/results.md through the
// CGNS mid-level library (CGNS SIDS 4.x, HDF5 files).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

#include <cgnslib.h>
#include <nlohmann/json.hpp>

#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "autan/io/mesh_info.hpp"
#include "cgns_util.hpp"

namespace autan::io {
namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

CGNS_ENUMT(BCType_t) bc_type_from_name(const std::string& name) {
  for (int t = 0; t < NofValidBCTypes; ++t) {
    if (name == BCTypeName[t]) return static_cast<CGNS_ENUMT(BCType_t)>(t);
  }
  throw Error(ErrorCode::mesh_inconsistent_sizes, "unknown CGNS BCType '" + name + "'");
}

void write_array(const char* name, const std::vector<double>& values) {
  const auto size = static_cast<cgsize_t>(values.size());
  cgns_check(cg_array_write(name, CGNS_ENUMV(RealDouble), 1, &size, values.data()), "write array",
             name);
}

}  // namespace

ZoneNodeNames zone_node_names(const std::vector<std::string>& boundary,
                              const std::vector<std::string>& cell) {
  static const std::vector<std::string> base_children{"Zone", "ZoneNames", "RunMetadata",
                                                      "DataClass", "DimensionalUnits"};
  static const std::vector<std::string> zone_children{
      "ZoneType", "GridCoordinates", "NGonElements",     "NFaceElements",
      "ZoneBC",   "FlowSolution",    "FlowSolutionMean", "WallSolution"};
  static const std::regex generated("(boundary|cell-zone)-[0-9]{4}");
  std::vector<std::string> all(boundary);
  all.insert(all.end(), cell.begin(), cell.end());
  auto contains = [](const std::vector<std::string>& list, const std::string& name) {
    return std::ranges::find(list, name) != list.end();
  };
  const auto kept = [&](const std::string& name, bool cell_zone) {
    const bool printable = std::ranges::all_of(name, [](char c) { return c >= 0x20 && c < 0x7f; });
    return !name.empty() && name.size() <= 32 && printable && name.find('/') == std::string::npos &&
           name != "." && name != ".." && std::ranges::count(all, name) == 1 &&
           !contains(base_children, name) && !(cell_zone && contains(zone_children, name)) &&
           !std::regex_match(name, generated);
  };
  const auto numbered = [](const char* prefix, std::size_t i) {
    const std::string digits = std::to_string(i);
    return std::string(prefix) + std::string(4 - std::min<std::size_t>(4, digits.size()), '0') +
           digits;
  };
  ZoneNodeNames names;
  for (std::size_t i = 0; i < boundary.size(); ++i) {
    names.boundary.push_back(kept(boundary[i], false) ? boundary[i] : numbered("boundary-", i));
  }
  for (std::size_t i = 0; i < cell.size(); ++i) {
    names.cell.push_back(kept(cell[i], true) ? cell[i] : numbered("cell-zone-", i));
  }
  return names;
}

void write_solution(const std::filesystem::path& file, const Mesh& mesh, const Solution& solution) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw Error(ErrorCode::mesh_inconsistent_sizes,
                "solution.cgns describes an unpartitioned mesh");
  }
  const MeshInfoZones zones = meshinfo_zones(mesh);
  const auto n_cells = at(mesh.n_cells());
  const auto n_faces = at(mesh.n_faces());
  const auto check_size = [](std::size_t got, std::size_t expected, const std::string& what) {
    if (got != expected) {
      throw Error(
          ErrorCode::mesh_inconsistent_sizes,
          what + " has " + std::to_string(got) + " values, expected " + std::to_string(expected));
    }
  };
  check_size(solution.boundary_bc_types.size(), zones.boundary.size(), "boundary_bc_types");
  for (const auto& f : solution.fields) check_size(f.values.size(), n_cells, "field " + f.name);
  for (const auto& f : solution.mean_fields)
    check_size(f.values.size(), n_cells, "mean field " + f.name);
  for (const auto& f : solution.wall_fields) {
    check_size(f.values.size(), solution.wall_faces.size(), "wall field " + f.name);
  }
  if (!std::ranges::is_sorted(solution.wall_faces) ||
      std::ranges::adjacent_find(solution.wall_faces) != solution.wall_faces.end() ||
      (!solution.wall_faces.empty() && (solution.wall_faces.front() < mesh.n_interior_faces() ||
                                        at(solution.wall_faces.back()) >= n_faces))) {
    throw Error(ErrorCode::mesh_index_out_of_range, "wall faces must be ascending boundary faces");
  }

  std::vector<std::string> boundary_names, cell_names;
  for (const index_t z : zones.boundary) boundary_names.push_back(mesh.face_zones()[at(z)].name);
  for (const index_t z : zones.cell) cell_names.push_back(mesh.cell_zones()[at(z)].name);
  const ZoneNodeNames nodes = zone_node_names(boundary_names, cell_names);

  CgnsFile out(file, CG_MODE_WRITE);
  const int fn = out.handle();
  int base = 0;
  cgns_check(cg_base_write(fn, "Base", 3, 3, &base), "write base");
  cgns_check(cg_goto(fn, base, "end"), "go to base");
  cgns_check(cg_dataclass_write(CGNS_ENUMV(Dimensional)), "write DataClass");
  cgns_check(cg_units_write(CGNS_ENUMV(Kilogram), CGNS_ENUMV(Meter), CGNS_ENUMV(Second),
                            CGNS_ENUMV(Kelvin), CGNS_ENUMV(Radian)),
             "write DimensionalUnits");
  cgns_check(cg_descriptor_write("RunMetadata", solution.run_metadata.c_str()),
             "write RunMetadata");
  nlohmann::ordered_json zone_names{{"boundary", nlohmann::ordered_json::object()},
                                    {"cell", nlohmann::ordered_json::object()}};
  for (std::size_t i = 0; i < boundary_names.size(); ++i) {
    zone_names["boundary"][nodes.boundary[i]] = boundary_names[i];
  }
  for (std::size_t i = 0; i < cell_names.size(); ++i)
    zone_names["cell"][nodes.cell[i]] = cell_names[i];
  cgns_check(cg_descriptor_write("ZoneNames", zone_names.dump(-1, ' ', true).c_str()),
             "write ZoneNames");

  for (std::size_t i = 0; i < zones.boundary.size(); ++i) {
    int family = 0;
    cgns_check(cg_family_write(fn, base, nodes.boundary[i].c_str(), &family), "write family",
               nodes.boundary[i]);
    if (!solution.boundary_bc_types[i].empty()) {
      int fambc = 0;
      cgns_check(cg_fambc_write(fn, base, family, "FamilyBC",
                                bc_type_from_name(solution.boundary_bc_types[i]), &fambc),
                 "write FamilyBC", nodes.boundary[i]);
    }
  }
  for (const auto& name : nodes.cell) {
    int family = 0;
    cgns_check(cg_family_write(fn, base, name.c_str(), &family), "write family", name);
  }

  cgsize_t size[3] = {mesh.n_points(), mesh.n_cells(), 0};
  int zone = 0;
  cgns_check(cg_zone_write(fn, base, "Zone", size, CGNS_ENUMV(Unstructured), &zone), "write zone");
  const std::vector<double> xs(mesh.x().begin(), mesh.x().end());
  const std::vector<double> ys(mesh.y().begin(), mesh.y().end());
  const std::vector<double> zs(mesh.z().begin(), mesh.z().end());
  int coord = 0;
  for (const auto& [name, values] : {std::pair{"CoordinateX", &xs}, std::pair{"CoordinateY", &ys},
                                     std::pair{"CoordinateZ", &zs}}) {
    cgns_check(cg_coord_write(fn, base, zone, CGNS_ENUMV(RealDouble), name, values->data(), &coord),
               "write coordinates");
  }

  // Face j is element 1 + j; cell i is element F + 1 + i, its faces signed
  // positive when their normal points out of it (results.md).
  std::vector<cgsize_t> offsets(mesh.face_node_offsets().begin(), mesh.face_node_offsets().end());
  std::vector<cgsize_t> connectivity;
  connectivity.reserve(mesh.face_nodes().size());
  for (const index_t p : mesh.face_nodes()) connectivity.push_back(p + 1);
  int section = 0;
  const auto faces = static_cast<cgsize_t>(n_faces);
  cgns_check(cg_poly_section_write(fn, base, zone, "NGonElements", CGNS_ENUMV(NGON_n), 1, faces, 0,
                                   connectivity.data(), offsets.data(), &section),
             "write NGonElements");
  offsets.assign(mesh.cell_face_offsets().begin(), mesh.cell_face_offsets().end());
  connectivity.clear();
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    for (const index_t f : mesh.cell_faces(c)) {
      connectivity.push_back(mesh.owner()[at(f)] == c ? f + 1 : -(f + 1));
    }
  }
  cgns_check(cg_poly_section_write(fn, base, zone, "NFaceElements", CGNS_ENUMV(NFACE_n), faces + 1,
                                   faces + static_cast<cgsize_t>(n_cells), 0, connectivity.data(),
                                   offsets.data(), &section),
             "write NFaceElements");

  const auto boundary_offsets = mesh.boundary_zone_offsets();
  for (std::size_t i = 0; i < zones.boundary.size(); ++i) {
    const auto z = at(zones.boundary[i]);
    const index_t first = boundary_offsets[z];
    const index_t last = boundary_offsets[z + 1];
    if (first == last) continue;
    if (solution.boundary_bc_types[i].empty()) {
      throw Error(ErrorCode::mesh_inconsistent_sizes,
                  "boundary zone '" + boundary_names[i] + "' has faces but no BCType");
    }
    std::vector<cgsize_t> points;
    for (index_t f = first; f < last; ++f) points.push_back(f + 1);
    int bc = 0;
    cgns_check(
        cg_boco_write(fn, base, zone, nodes.boundary[i].c_str(),
                      bc_type_from_name(solution.boundary_bc_types[i]), CGNS_ENUMV(PointList),
                      static_cast<cgsize_t>(points.size()), points.data(), &bc),
        "write BC", nodes.boundary[i]);
    cgns_check(cg_boco_gridlocation_write(fn, base, zone, bc, CGNS_ENUMV(FaceCenter)),
               "write BC location", nodes.boundary[i]);
    cgns_check(cg_goto(fn, base, "Zone_t", zone, "ZoneBC_t", 1, "BC_t", bc, "end"), "go to BC");
    cgns_check(cg_famname_write(nodes.boundary[i].c_str()), "write BC family", nodes.boundary[i]);
  }

  for (std::size_t i = 0; i < zones.cell.size(); ++i) {
    std::vector<cgsize_t> points;
    for (index_t c = 0; c < mesh.n_cells(); ++c) {
      if (mesh.cell_zone()[at(c)] == zones.cell[i]) points.push_back(faces + c + 1);
    }
    int region = 0;
    cgns_check(cg_subreg_ptset_write(fn, base, zone, nodes.cell[i].c_str(), 3,
                                     CGNS_ENUMV(CellCenter), CGNS_ENUMV(PointList),
                                     static_cast<cgsize_t>(points.size()), points.data(), &region),
               "write cell zone", nodes.cell[i]);
    cgns_check(cg_goto(fn, base, "Zone_t", zone, "ZoneSubRegion_t", region, "end"),
               "go to cell zone");
    cgns_check(cg_famname_write(nodes.cell[i].c_str()), "write cell zone family", nodes.cell[i]);
  }

  for (const auto& [name, fields] : {std::pair{"FlowSolution", &solution.fields},
                                     std::pair{"FlowSolutionMean", &solution.mean_fields}}) {
    if (fields->empty()) continue;
    int sol = 0;
    cgns_check(cg_sol_write(fn, base, zone, name, CGNS_ENUMV(CellCenter), &sol), "write", name);
    for (const auto& f : *fields) {
      int field = 0;
      cgns_check(cg_field_write(fn, base, zone, sol, CGNS_ENUMV(RealDouble), f.name.c_str(),
                                f.values.data(), &field),
                 "write field", f.name);
    }
  }

  if (!solution.wall_faces.empty()) {
    std::vector<cgsize_t> points;
    points.reserve(solution.wall_faces.size());
    for (const index_t f : solution.wall_faces) points.push_back(f + 1);
    int region = 0;
    cgns_check(cg_subreg_ptset_write(fn, base, zone, "WallSolution", 2, CGNS_ENUMV(FaceCenter),
                                     CGNS_ENUMV(PointList), static_cast<cgsize_t>(points.size()),
                                     points.data(), &region),
               "write WallSolution");
    cgns_check(cg_goto(fn, base, "Zone_t", zone, "ZoneSubRegion_t", region, "end"),
               "go to WallSolution");
    for (const auto& f : solution.wall_fields) write_array(f.name.c_str(), f.values);
  }
  out.close();
}

}  // namespace autan::io
