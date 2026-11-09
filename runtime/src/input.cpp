#include "autan/runtime/input.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "autan/runtime/diagnostic.hpp"
#include "autan_diagnostics.hpp"

namespace autan::runtime {
namespace {

using Json = nlohmann::json;

[[noreturn]] void invalid(const std::string& path, const std::string& message) {
  throw InputError(Diagnostic{.code = "E0705",
                              .message = path + ": " + message,
                              .path = path,
                              .data_json = Json{{"path", path}}.dump()});
}

// A JSON object whose keys must be exactly `keys`.
class Object {
 public:
  Object(const Json& value, std::string path, std::initializer_list<std::string_view> keys)
      : value_(value), path_(std::move(path)) {
    check(std::span(keys.begin(), keys.size()));
  }
  Object(const Json& value, std::string path, const std::vector<std::string>& keys)
      : value_(value), path_(std::move(path)) {
    check(std::span(keys));
  }

  [[nodiscard]] std::string at(const std::string& key) const {
    return path_.empty() ? key : path_ + "." + key;
  }
  [[nodiscard]] const Json& get(const std::string& key) const { return value_.at(key); }
  [[nodiscard]] bool is_null(const std::string& key) const { return value_.at(key).is_null(); }

  [[nodiscard]] std::string string(const std::string& key) const {
    const Json& v = get(key);
    if (!v.is_string()) invalid(at(key), "expected a string");
    return v.get<std::string>();
  }
  [[nodiscard]] std::string tag(const std::string& key,
                                std::initializer_list<std::string_view> allowed) const {
    std::string v = string(key);
    if (std::ranges::find(allowed, std::string_view(v)) == allowed.end()) {
      invalid(at(key), "unexpected value '" + v + "'");
    }
    return v;
  }
  [[nodiscard]] double number(const std::string& key) const {
    const Json& v = get(key);
    if (!v.is_number() || !std::isfinite(v.get<double>()))
      invalid(at(key), "expected a finite number");
    return v.get<double>();
  }
  [[nodiscard]] double positive(const std::string& key) const {
    const double v = number(key);
    if (!(v > 0.0)) invalid(at(key), "expected a positive number");
    return v;
  }
  [[nodiscard]] std::int64_t integer(const std::string& key, std::int64_t minimum) const {
    const Json& v = get(key);
    if (!v.is_number_integer()) invalid(at(key), "expected an integer");
    const auto i = v.get<std::int64_t>();
    if (i < minimum) invalid(at(key), "must be at least " + std::to_string(minimum));
    return i;
  }
  [[nodiscard]] bool boolean(const std::string& key) const {
    const Json& v = get(key);
    if (!v.is_boolean()) invalid(at(key), "expected a boolean");
    return v.get<bool>();
  }
  [[nodiscard]] std::array<double, 3> vector3(const std::string& key) const {
    const Json& v = get(key);
    if (!v.is_array() || v.size() != 3) invalid(at(key), "expected three numbers");
    std::array<double, 3> out{};
    for (std::size_t i = 0; i < 3; ++i) {
      if (!v[i].is_number() || !std::isfinite(v[i].get<double>()))
        invalid(at(key), "expected three finite numbers");
      out[i] = v[i].get<double>();
    }
    return out;
  }
  // `required` true: must not be null; false: must be null.
  template <typename T, typename Read>
  [[nodiscard]] std::optional<T> maybe(const std::string& key, bool required, Read read) const {
    if (is_null(key)) {
      if (required) invalid(at(key), "must not be null here");
      return std::nullopt;
    }
    if (!required) invalid(at(key), "must be null here");
    return read(key);
  }
  [[nodiscard]] std::optional<double> number_if(const std::string& key, bool required) const {
    return maybe<double>(key, required, [this](const std::string& k) { return number(k); });
  }
  [[nodiscard]] std::optional<double> positive_if(const std::string& key, bool required) const {
    return maybe<double>(key, required, [this](const std::string& k) { return positive(k); });
  }
  // A key that may hold a value or null, whatever its neighbours.
  [[nodiscard]] std::optional<double> optional_number(const std::string& key) const {
    return is_null(key) ? std::nullopt : std::optional<double>(number(key));
  }

 private:
  template <typename Key>
  void check(std::span<const Key> keys) const {
    if (!value_.is_object()) invalid(path_, "expected an object");
    const std::set<std::string, std::less<>> expected(keys.begin(), keys.end());
    for (const auto& [key, item] : value_.items()) {
      if (!expected.contains(key)) invalid(at(key), "unknown key");
    }
    for (const auto& key : keys) {
      if (!value_.contains(key)) invalid(at(std::string(key)), "missing key");
    }
  }

  const Json& value_;
  std::string path_;
};

// A SHA-256 written as 64 lowercase hexadecimal characters.
std::string sha256(const Object& o, const std::string& key) {
  std::string v = o.string(key);
  const bool hex = v.size() == 64 && std::ranges::all_of(v, [](char c) {
                     return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                   });
  if (!hex) invalid(o.at(key), "expected 64 lowercase hexadecimal characters");
  return v;
}

std::string indexed(const std::string& path, std::size_t i) {
  return path + "[" + std::to_string(i) + "]";
}

// solver-input.md, mesh_info: the MeshInfo the input was planned with, which the run archives as
// its mesh-info.json, so the whole of mesh-info.md: its members, and the rules that tie them
// (the SDK's MeshInfo checks the same). Sets the mesh_info members of `in`.
void read_mesh_info(const Json& info, SolverInput& in) {
  std::vector<std::string> keys{"schema",         "source",        "counts",     "bounds",
                                "boundary_zones", "cell_zones",    "interfaces", "periodic",
                                "groups",         "groups_source", "quality"};
  if (info.is_object() && info.contains("partition")) keys.emplace_back("partition");
  const Object root(info, "mesh_info", keys);
  if (root.string("schema") != "autan.mesh-info/1")
    invalid(root.at("schema"), "expected autan.mesh-info/1");
  const auto count = [](const Object& o, const std::string& key) { return o.integer(key, 0); };
  const auto list = [](const Object& o, const std::string& key) -> const Json& {
    const Json& v = o.get(key);
    if (!v.is_array()) invalid(o.at(key), "expected a list");
    return v;
  };
  const auto name = [](const Object& o) {
    std::string n = o.string("name");
    if (n.empty()) invalid(o.at("name"), "expected a non-empty name");
    return n;
  };
  const auto null_or_tag = [](const Object& o, const std::string& key,
                              std::initializer_list<std::string_view> allowed) {
    return o.is_null(key) ? std::string() : o.tag(key, allowed);
  };

  const Object source(root.get("source"), root.at("source"),
                      {"path", "format", "sha256", "bytes", "logical_bytes", "descriptor_bytes",
                       "zone_names_bytes", "filtered_bytes", "cgns_nodes", "point_list_ids"});
  (void)source.string("path");
  in.mesh_info_bytes = source.integer("bytes", 1);
  (void)source.integer("logical_bytes", 0);
  (void)source.integer("descriptor_bytes", 0);
  (void)source.integer("zone_names_bytes", 0);
  (void)source.integer("filtered_bytes", 0);
  (void)source.integer("cgns_nodes", 0);
  (void)source.integer("point_list_ids", 0);
  (void)source.tag("format", {"cgns", "gmsh", "fluent", "polymesh"});
  const std::string sha = sha256(source, "sha256");
  if (sha != in.mesh_sha256) {
    throw InputError(
        Diagnostic{.code = "E0203",
                   .message = "mesh_info describes another mesh than mesh.sha256",
                   .path = source.at("sha256"),
                   .data_json = Json{{"expected", in.mesh_sha256}, {"found", sha}}.dump()});
  }

  const Object counts(
      root.get("counts"), root.at("counts"),
      {"cells", "faces", "nodes", "source_nodes", "face_nodes", "cells_by_type", "stencil_entries",
       "vertex_stencil_entries", "band_entries", "boundary_fit_entries", "periodic_ghosts"});
  const std::int64_t cells = counts.integer("cells", 1), faces = counts.integer("faces", 1);
  (void)counts.integer("source_nodes", counts.integer("nodes", 1));
  (void)counts.integer("face_nodes", 3 * faces);
  (void)count(counts, "stencil_entries");
  (void)count(counts, "vertex_stencil_entries");
  (void)count(counts, "band_entries");
  (void)count(counts, "boundary_fit_entries");
  (void)count(counts, "periodic_ghosts");
  const Object by_type(counts.get("cells_by_type"), counts.at("cells_by_type"),
                       {"tetra", "pyramid", "prism", "hexa", "polyhedron"});
  std::int64_t typed = 0;
  for (const char* t : {"tetra", "pyramid", "prism", "hexa", "polyhedron"})
    typed += count(by_type, t);
  if (typed != cells) invalid(counts.at("cells_by_type"), "does not sum to counts.cells");

  const Object bounds(root.get("bounds"), root.at("bounds"), {"min", "max"});
  const auto low = bounds.vector3("min"), high = bounds.vector3("max");
  for (std::size_t k = 0; k < 3; ++k) {
    if (low[k] > high[k]) invalid(bounds.at("min"), "exceeds bounds.max");
  }

  std::set<std::string> boundary_names, periodic_zones;
  std::map<std::string, std::int64_t> zone_faces;
  std::map<std::string, std::size_t> zone_order;
  std::int64_t boundary_faces = 0;
  const Json& boundary = list(root, "boundary_zones");
  for (std::size_t i = 0; i < boundary.size(); ++i) {
    const Object z(boundary[i], indexed(root.at("boundary_zones"), i),
                   {"name", "type", "native_type", "faces"});
    const std::string n = name(z);
    if (!boundary_names.insert(n).second) invalid(z.at("name"), "is not unique");
    const std::string type = null_or_tag(
        z, "type",
        {"wall", "thermal-wall", "slip", "symmetry", "farfield", "inlet", "outlet", "periodic"});
    if (!z.is_null("native_type")) (void)z.string("native_type");
    zone_faces[n] = count(z, "faces");
    boundary_faces += zone_faces[n];
    if (type == "periodic") periodic_zones.insert(n);
    zone_order[n] = i;
  }
  // mesh-info.md, Zone groups: distinct names in order, each of
  // [A-Za-z0-9_.][A-Za-z0-9_.-]*, its boundary zones distinct and in MeshInfo order, from the
  // sidecar the input names (mesh.groups_sha256); none without one. The SDK checks the same.
  const Json& groups = list(root, "groups");
  std::string previous;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const Object group(groups[i], indexed(root.at("groups"), i), {"name", "zones"});
    const std::string group_name = name(group);
    const auto allowed = [](char c, bool first) {
      return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.' ||
             (!first && c == '-');
    };
    bool valid = allowed(group_name.front(), true);
    for (std::size_t k = 1; k < group_name.size(); ++k)
      valid = valid && allowed(group_name[k], false);
    if (!valid) invalid(group.at("name"), "is not a zone group name");
    if (i > 0 && group_name <= previous) invalid(group.at("name"), "breaks the groups' name order");
    previous = group_name;
    const Json& members = list(group, "zones");
    if (members.empty()) invalid(group.at("zones"), "expected at least one zone");
    std::size_t last = 0;
    for (std::size_t k = 0; k < members.size(); ++k) {
      const std::string at = indexed(group.at("zones"), k);
      if (!members[k].is_string()) invalid(at, "expected a zone name");
      const auto found = zone_order.find(members[k].get<std::string>());
      if (found == zone_order.end()) invalid(at, "is not a boundary zone");
      if (k > 0 && found->second <= last) invalid(at, "breaks the zones' MeshInfo order");
      last = found->second;
    }
  }
  std::optional<std::string> groups_sha;
  if (!root.is_null("groups_source")) {
    const Object groups_source(root.get("groups_source"), root.at("groups_source"),
                               {"path", "sha256", "provenance"});
    (void)groups_source.string("path");
    groups_sha = sha256(groups_source, "sha256");
    const Json& provenance = groups_source.get("provenance");
    if (!provenance.is_object()) invalid(groups_source.at("provenance"), "expected an object");
    for (const auto& [key, value] : provenance.items()) {
      if (!value.is_string())
        invalid(groups_source.at("provenance") + "." + key, "expected a string");
    }
  } else if (!groups.empty()) {
    invalid(root.at("groups"), "has groups without groups_source");
  }
  if (groups_sha != in.mesh_groups_sha256)
    invalid(root.at("groups_source"), "is not the zone-groups file mesh.groups_sha256 names");
  if (boundary_faces > faces)
    invalid(root.at("boundary_zones"), "hold more faces than counts.faces");

  std::set<std::string> cell_names;
  std::int64_t zoned = 0;
  const Json& cell_zones = list(root, "cell_zones");
  if (cell_zones.empty()) invalid(root.at("cell_zones"), "expected at least one cell zone");
  for (std::size_t i = 0; i < cell_zones.size(); ++i) {
    const Object z(cell_zones[i], indexed(root.at("cell_zones"), i),
                   {"name", "type", "native_type", "cells"});
    if (!cell_names.insert(name(z)).second) invalid(z.at("name"), "is not unique");
    (void)null_or_tag(z, "type", {"fluid", "solid"});
    if (!z.is_null("native_type")) (void)z.string("native_type");
    zoned += z.integer("cells", 1);
  }
  if (zoned != cells) invalid(root.at("cell_zones"), "do not sum to counts.cells");

  // Two names of `pool`, as a list of two strings.
  const auto pair_of = [](const Object& o, const std::set<std::string>& pool) {
    const Json& v = o.get("zones");
    if (!v.is_array() || v.size() != 2 || !v[0].is_string() || !v[1].is_string() ||
        !pool.contains(v[0].get<std::string>()) || !pool.contains(v[1].get<std::string>()) ||
        v[0] == v[1])
      invalid(o.at("zones"), "expected two distinct zone names");
    return std::pair{v[0].get<std::string>(), v[1].get<std::string>()};
  };
  std::set<std::pair<std::string, std::string>> shared;
  std::int64_t interface_faces = 0;
  const Json& interfaces = list(root, "interfaces");
  for (std::size_t i = 0; i < interfaces.size(); ++i) {
    const Object f(interfaces[i], indexed(root.at("interfaces"), i), {"zones", "faces"});
    const auto zones = pair_of(f, cell_names);
    if (!(zones.first < zones.second)) invalid(f.at("zones"), "expected the names sorted");
    if (!shared.insert(zones).second) invalid(f.at("zones"), "lists a pair twice");
    interface_faces += f.integer("faces", 1);
  }
  if (interface_faces > faces - boundary_faces)
    invalid(root.at("interfaces"), "hold more faces than the interior faces");

  std::multiset<std::string> paired;
  const Json& periodic = list(root, "periodic");
  for (std::size_t i = 0; i < periodic.size(); ++i) {
    const Object pr(periodic[i], indexed(root.at("periodic"), i), {"zones", "transform", "faces"});
    const auto [a, b] = pair_of(pr, periodic_zones);
    const Object t(pr.get("transform"), pr.at("transform"), {"kind", "vector"});
    (void)t.tag("kind", {"translation"});
    (void)t.vector3("vector");
    const std::int64_t n = pr.integer("faces", 1);
    if (zone_faces[a] != n || zone_faces[b] != n)
      invalid(pr.at("faces"), "differs from its zones' face counts");
    paired.insert(a);
    paired.insert(b);
  }
  if (!std::ranges::equal(paired, periodic_zones))
    invalid(root.at("periodic"), "must hold each periodic zone in exactly one pair");

  const Object quality(root.get("quality"), root.at("quality"),
                       {"max_non_orthogonality_deg", "max_faces_per_cell", "lines", "multigrid"});
  const double angle = quality.number("max_non_orthogonality_deg");
  if (angle < 0.0 || angle > 180.0)
    invalid(quality.at("max_non_orthogonality_deg"), "expected 0 to 180");
  (void)quality.integer("max_faces_per_cell", 4);
  const Object lines(quality.get("lines"), quality.at("lines"),
                     {"order", "scale", "cells", "long_cells"});
  in.mesh_info_order = lines.tag("order", {"rcm", "input"});
  in.mesh_info_scale = lines.positive("scale");
  const std::int64_t in_lines = count(lines, "cells"), long_lines = count(lines, "long_cells");
  if (!(long_lines <= in_lines && in_lines <= cells))
    invalid(lines.at("cells"), "must hold long_cells <= cells <= counts.cells");
  const Object multigrid(quality.get("multigrid"), quality.at("multigrid"),
                         {"cells_by_level", "edges_by_level"});
  const Json& level_cells = list(multigrid, "cells_by_level");
  const Json& level_edges = list(multigrid, "edges_by_level");
  if (level_cells.empty() || level_edges.size() != level_cells.size())
    invalid(multigrid.at("edges_by_level"), "expected one value per level, at least one level");
  for (std::size_t l = 0; l < level_cells.size(); ++l) {
    if (!level_cells[l].is_number_integer() || level_cells[l].get<std::int64_t>() < 1 ||
        !level_edges[l].is_number_integer() || level_edges[l].get<std::int64_t>() < 0)
      invalid(indexed(multigrid.at("cells_by_level"), l), "expected counts, cells >= 1");
  }
  if (level_cells[0].get<std::int64_t>() != cells)
    invalid(multigrid.at("cells_by_level"), "must start with counts.cells");
  if (level_edges[0].get<std::int64_t>() != faces - boundary_faces)
    invalid(multigrid.at("edges_by_level"), "must start with the interior faces");

  if (info.contains("partition")) {
    const Object part(
        root.get("partition"), root.at("partition"),
        {"ranks", "groups", "ghost_stencil", "cells_by_rank", "ghosts_by_rank", "faces_by_rank",
         "interior_faces_by_rank", "stencil_entries_by_rank", "vertex_stencil_entries_by_rank",
         "band_entries_by_rank", "boundary_fit_entries_by_rank", "nodes_by_rank",
         "face_nodes_by_rank", "sent_cells_by_rank"});
    const std::int64_t ranks = part.integer("ranks", 1);
    (void)part.tag("groups", {"none", "lines"});
    (void)part.tag("ghost_stencil", {"face2", "vertex"});
    for (const char* key :
         {"cells_by_rank", "ghosts_by_rank", "faces_by_rank", "interior_faces_by_rank",
          "stencil_entries_by_rank", "vertex_stencil_entries_by_rank", "band_entries_by_rank",
          "boundary_fit_entries_by_rank", "nodes_by_rank", "face_nodes_by_rank",
          "sent_cells_by_rank"}) {
      const Json& v = list(part, key);
      if (static_cast<std::int64_t>(v.size()) != ranks)
        invalid(part.at(key), "expected one value per rank");
      std::int64_t sum = 0;
      for (const Json& n : v) {
        if (!n.is_number_integer() || n.get<std::int64_t>() < 0)
          invalid(part.at(key), "expected integers >= 0");
        sum += n.get<std::int64_t>();
      }
      if (std::string_view(key) == "cells_by_rank" && sum != cells)
        invalid(part.at(key), "does not sum to counts.cells");
    }
  }
  in.mesh_info = info.dump(2);
}

// A pointer, not a reference: GCC 13 flags references bound to the result of
// a call taking a temporary (-Wdangling-reference).
const Json* array_at(const Object& o, const std::string& key) {
  const Json& v = o.get(key);
  if (!v.is_array()) invalid(o.at(key), "expected an array");
  return &v;
}

Turbulence parse_turbulence(const Object& physics, const std::string& kind) {
  const Object t(physics.get("turbulence"), physics.at("turbulence"),
                 {"model", "base", "wall_model", "nu_tilde_ratio", "intensity", "viscosity_ratio"});
  Turbulence out;
  out.model = t.tag("model", {"inviscid", "laminar", "sa", "sst", "ddes", "wmles"});
  const bool ddes = out.model == "ddes";
  out.base = t.maybe<std::string>("base", ddes,
                                  [&](const std::string& k) { return t.tag(k, {"sa", "sst"}); });
  out.wall_model =
      t.maybe<std::string>("wall_model", out.model == "wmles",
                           [&](const std::string& k) { return t.tag(k, {"equilibrium"}); });
  const bool sa = out.model == "sa" || (ddes && out.base == "sa");
  const bool sst = out.model == "sst" || (ddes && out.base == "sst");
  out.nu_tilde_ratio = t.positive_if("nu_tilde_ratio", sa);
  out.intensity = t.positive_if("intensity", sst);
  out.viscosity_ratio = t.positive_if("viscosity_ratio", sst);
  (void)kind;
  return out;
}

Medium parse_medium(const Json& value, const std::string& path) {
  const Json kind = value.is_object() && value.contains("kind") ? value.at("kind") : Json();
  Medium m;
  if (kind == "ideal-gas") {
    const Object o(value, path, {"kind", "name", "gas_constant", "gamma", "prandtl", "viscosity"});
    m.kind = "ideal-gas";
    m.name = o.string("name");
    m.gas_constant = o.positive("gas_constant");
    m.gamma = o.number("gamma");
    if (!(*m.gamma > 1.0)) invalid(o.at("gamma"), "must be greater than 1");
    m.prandtl = o.positive("prandtl");
  } else if (kind == "constant-density-fluid") {
    const Object o(value, path, {"kind", "name", "density", "viscosity"});
    m.kind = "constant-density-fluid";
    m.name = o.string("name");
    m.density = o.positive("density");
    m.viscosity = Viscosity{
        .law = "constant", .mu_ref = {}, .t_ref = {}, .s = {}, .mu = o.positive("viscosity")};
    return m;
  } else if (kind == "solid") {
    const Object o(value, path, {"kind", "name", "density", "specific_heat", "conductivity"});
    m.kind = "solid";
    m.name = o.string("name");
    m.density = o.positive("density");
    m.specific_heat = o.positive("specific_heat");
    m.conductivity = o.positive("conductivity");
    return m;
  } else {
    invalid(path + ".kind", "expected ideal-gas, constant-density-fluid or solid");
  }
  const std::string vpath = path + ".viscosity";
  const Json& v = value.at("viscosity");
  const Json law = v.is_object() && v.contains("law") ? v.at("law") : Json();
  Viscosity visc;
  if (law == "sutherland") {
    const Object o(v, vpath, {"law", "mu_ref", "t_ref", "s"});
    visc = Viscosity{.law = "sutherland",
                     .mu_ref = o.positive("mu_ref"),
                     .t_ref = o.positive("t_ref"),
                     .s = o.positive("s"),
                     .mu = std::nullopt};
  } else if (law == "constant") {
    const Object o(v, vpath, {"law", "mu"});
    visc = Viscosity{.law = "constant", .mu_ref = {}, .t_ref = {}, .s = {}, .mu = o.positive("mu")};
  } else {
    invalid(vpath + ".law", "expected sutherland or constant");
  }
  m.viscosity = visc;
  return m;
}

InitialState parse_initial(const Json& value, const std::string& path, const SolverInput& in,
                           const Medium& medium) {
  const Json kind = value.is_object() && value.contains("kind") ? value.at("kind") : Json();
  InitialState s;
  if (kind == "solid") {
    const Object o(value, path, {"kind", "temperature"});
    if (medium.kind != "solid") invalid(o.at("kind"), "a fluid zone needs a fluid initial state");
    s.kind = "solid";
    s.temperature = o.positive("temperature");
    return s;
  }
  if (kind != "fluid") invalid(path + ".kind", "expected fluid or solid");
  const Object o(
      value, path,
      {"kind", "density", "pressure", "temperature", "velocity", "nu_tilde", "k", "omega"});
  if (medium.kind == "solid") invalid(o.at("kind"), "a solid zone needs a solid initial state");
  const bool sa = in.turbulence.nu_tilde_ratio.has_value();
  const bool sst = in.turbulence.intensity.has_value();
  s.kind = "fluid";
  s.density = o.positive("density");
  s.pressure = o.number("pressure");
  s.temperature = o.positive_if("temperature", in.physics_kind == "compressible");
  s.velocity = o.vector3("velocity");
  s.nu_tilde = o.positive_if("nu_tilde", sa);
  s.k = o.positive_if("k", sst);
  s.omega = o.positive_if("omega", sst);
  return s;
}

BoundaryInput parse_boundary(const Json& value, const std::string& path) {
  const Object o(value, path,
                 {"zone", "rule", "kind", "temperature", "total_pressure", "total_temperature",
                  "velocity", "static_pressure", "states", "point_vortex"});
  BoundaryInput b;
  b.zone = o.string("zone");
  // The case rule the SDK resolved the zone with: checked, of no use to the run.
  if (!o.is_null("rule")) {
    const Json& rule = o.get("rule");
    if (!rule.is_number_integer() || rule.get<std::int64_t>() < 0)
      invalid(path + ".rule", "expected an integer >= 0 or null");
  }
  b.kind = o.tag("kind", {"wall", "slip", "symmetry", "farfield", "inlet", "velocity-inlet",
                          "outlet", "manufactured", "oracle-state"});
  if (b.kind == "oracle-state") {
    const Object s(o.get("states"), path + ".states", {"path", "sha256"});
    BoundaryStates states;
    states.path = s.string("path");
    if (states.path.find('\0') != std::string::npos) {
      invalid(path + ".states.path", "contains a NUL character");
    }
    if (!std::filesystem::path(states.path).is_absolute()) {
      invalid(path + ".states.path", "must be an absolute path");
    }
    states.sha256 = sha256(s, "sha256");
    b.states = std::move(states);
  } else if (!o.is_null("states")) {
    invalid(path + ".states", "only an oracle-state boundary takes states");
  }
  if (!o.is_null("point_vortex")) {
    if (b.kind != "farfield")
      invalid(path + ".point_vortex", "only a farfield takes a point vortex");
    const Object v(o.get("point_vortex"), path + ".point_vortex", {"zones", "center"});
    const Json& names = v.get("zones");
    if (!names.is_array() || names.empty()) invalid(v.at("zones"), "expected zone names");
    PointVortex vortex;
    for (std::size_t j = 0; j < names.size(); ++j) {
      if (!names[j].is_string()) invalid(indexed(v.at("zones"), j), "expected a zone name");
      vortex.zones.push_back(names[j].get<std::string>());
    }
    vortex.center = v.vector3("center");
    b.point_vortex = std::move(vortex);
  }
  b.temperature = b.kind == "wall" ? (o.is_null("temperature")
                                          ? std::nullopt
                                          : std::optional<double>(o.positive("temperature")))
                                   : o.positive_if("temperature", false);
  b.total_pressure = o.positive_if("total_pressure", b.kind == "inlet");
  b.total_temperature = o.positive_if("total_temperature", b.kind == "inlet");
  b.static_pressure = o.positive_if("static_pressure", b.kind == "outlet");
  b.velocity = o.positive_if("velocity", b.kind == "velocity-inlet");
  return b;
}

}  // namespace

std::string Diagnostic::json() const {
  static const Json registry = Json::parse(autan_diagnostics_json).at("codes");
  if (!registry.contains(code))
    throw std::logic_error("diagnostic code " + code + " is not in the registry");
  const Json& entry = registry.at(code);
  const Json d{{"code", code},
               {"severity", entry.at("severity")},
               {"message", message},
               {"location", path.empty() ? Json() : Json{{"path", path}}},
               {"fix", entry.at("fix")},
               {"doc", entry.at("doc")},
               {"data", Json::parse(data_json.empty() ? "{}" : data_json)}};
  return d.dump();
}

namespace {

// results.md: run-YYYYMMDDTHHMMSSZ-xxxxxxxx, x lowercase hexadecimal.
bool is_run_id(std::string_view id) {
  constexpr std::string_view shape = "run-DDDDDDDDTDDDDDDZ-hhhhhhhh";
  if (id.size() != shape.size()) return false;
  for (std::size_t i = 0; i < id.size(); ++i) {
    const char c = id[i];
    const bool digit = c >= '0' && c <= '9';
    const bool ok = shape[i] == 'D'   ? digit
                    : shape[i] == 'h' ? digit || (c >= 'a' && c <= 'f')
                                      : c == shape[i];
    if (!ok) return false;
  }
  return true;
}

// time.checkpoint: {seconds} in [60, 86400] or {iterations} >= 1 (case-schema.md#time).
void parse_checkpoint(const Object& time, TimeInput& out) {
  const Json& c = time.get("checkpoint");
  const std::string path = time.at("checkpoint");
  if (c.is_object() && c.contains("seconds")) {
    const Object o(c, path, {"seconds"});
    const double s = o.number("seconds");
    if (s < 60.0 || s > 86400.0) invalid(o.at("seconds"), "must be in [60, 86400]");
    out.checkpoint_seconds = s;
  } else {
    const Object o(c, path, {"iterations"});
    out.checkpoint_iterations = o.integer("iterations", 1);
  }
}

// Needs the rest of the input: the residual reference is keyed by its
// equations and the mode's rules depend on its time.
Start parse_start(const Json& value, const SolverInput& in) {
  const Object o(value, "start",
                 {"run", "run_id", "identity", "mode", "iterations", "time", "solution_sha256",
                  "file", "residual_reference", "woken"});
  Start start;
  start.run = o.string("run");
  if (!std::filesystem::path(start.run).is_absolute())
    invalid("start.run", "must be an absolute path");
  start.run_id = o.string("run_id");
  if (!is_run_id(start.run_id)) invalid("start.run_id", "expected run-YYYYMMDDTHHMMSSZ-xxxxxxxx");
  start.identity = sha256(o, "identity");
  start.mode = o.tag("mode", {"continue", "warm"});
  start.iterations = o.integer("iterations", 1);
  start.solution_sha256 = sha256(o, "solution_sha256");
  start.file = o.tag("file", {"solution", "checkpoint"});
  const bool steady = in.time.kind == "steady";
  const bool continued = start.mode == "continue";
  // A continued run has the previous run's kind of time; a warm one may not.
  start.time = continued ? o.number_if("time", !steady) : o.optional_number("time");
  if (continued && start.time && *start.time < 0.0) invalid("start.time", "must not be negative");
  if (continued && (steady ? in.time.max_iterations : in.time.steps) <= start.iterations) {
    invalid(o.at("iterations"), "the time budget must exceed the iterations already done");
  }
  if (o.is_null("residual_reference")) {
    if (continued && steady) invalid(o.at("residual_reference"), "must not be null here");
    if (!o.is_null("woken")) invalid(o.at("woken"), "must be null here");
    return start;
  }
  if (!(continued && steady)) invalid(o.at("residual_reference"), "must be null here");
  const auto equations = residual_names(in);
  const Object r(o.get("residual_reference"), o.at("residual_reference"), equations);
  std::vector<double> reference;
  for (const auto& name : equations) {
    const double v = r.number(name);
    if (v < 0.0) invalid(r.at(name), "must not be negative");
    reference.push_back(v);
  }
  start.residual_reference = std::move(reference);
  if (o.is_null("woken")) invalid(o.at("woken"), "must not be null here");
  const Json& woken = *array_at(o, "woken");
  std::vector<std::string> names;
  for (std::size_t i = 0; i < woken.size(); ++i) {
    const std::string path = indexed(o.at("woken"), i);
    if (!woken[i].is_string()) invalid(path, "expected an equation name");
    names.push_back(woken[i].get<std::string>());
    if (std::ranges::find(equations, names.back()) == equations.end())
      invalid(path, "no equation '" + names.back() + "'");
    if (i > 0 && !(names[i - 1] < names[i])) invalid(path, "names must be sorted and distinct");
  }
  start.woken = std::move(names);
  return start;
}

}  // namespace

SolverInput parse_solver_input(const std::string& text) {
  Json doc;
  try {
    doc = Json::parse(text);
  } catch (const Json::exception& e) {
    invalid("", std::string("not JSON: ") + e.what());
  }
  const Object root(
      doc, "",
      {"schema", "case", "mesh", "mesh_info", "verification", "start", "initial_state", "periodic",
       "precision", "reproducibility", "devices", "memory", "physics", "freestream", "boundaries",
       "cell_zones", "time", "monitors"});
  if (root.string("schema") != "autan.solver-input/1")
    invalid("schema", "expected autan.solver-input/1");
  SolverInput in;
  {
    const Object c(root.get("case"), "case", {"name", "sha256", "toml"});
    in.case_name = c.string("name");
    in.case_sha256 = sha256(c, "sha256");
    in.case_toml = c.string("toml");
    const Object m(root.get("mesh"), "mesh", {"path", "sha256", "scale", "groups_sha256"});
    in.mesh_path = m.string("path");
    if (!std::filesystem::path(in.mesh_path).is_absolute())
      invalid("mesh.path", "must be an absolute path");
    in.mesh_sha256 = sha256(m, "sha256");
    in.mesh_scale = m.positive("scale");
    if (!m.is_null("groups_sha256")) in.mesh_groups_sha256 = sha256(m, "groups_sha256");
  }
  read_mesh_info(root.get("mesh_info"), in);
  if (!root.is_null("verification")) {
    const Object v(root.get("verification"), "verification", {"path", "sha256"});
    Verification verification;
    verification.path = v.string("path");
    if (!std::filesystem::path(verification.path).is_absolute()) {
      invalid("verification.path", "must be an absolute path");
    }
    verification.sha256 = sha256(v, "sha256");
    in.verification = std::move(verification);
  }
  in.precision = root.tag("precision", {"fp32+acc64", "fp64"});
  in.reproducibility = root.tag("reproducibility", {"none", "run", "ranks"});
  {
    const Object d(root.get("devices"), "devices", {"kind", "count"});
    in.device_kind = d.tag("kind", {"cpu", "cuda", "hip"});
    in.device_count = d.integer("count", 1);
  }
  {
    const Object m(root.get("memory"), "memory", {"bytes_host", "bytes_device"});
    in.memory_bytes_host = m.integer("bytes_host", 0);
    in.memory_bytes_device = m.integer("bytes_device", 0);
  }
  {
    const Object p(root.get("physics"), "physics", {"kind", "turbulence", "flux"});
    in.physics_kind = p.tag("kind", {"compressible", "incompressible"});
    in.turbulence = parse_turbulence(p, in.physics_kind);
    // case-schema.md#physics: the convective flux of compressible flow, null otherwise.
    if (in.physics_kind == "compressible") {
      in.flux = p.tag("flux", {"roe", "hllc", "ausm+up"});
    } else if (!p.is_null("flux")) {
      invalid(p.at("flux"), "expected null: only compressible physics names a flux");
    }
  }
  const bool compressible = in.physics_kind == "compressible";
  const bool sa = in.turbulence.nu_tilde_ratio.has_value();
  const bool sst = in.turbulence.intensity.has_value();
  {
    const Object f(root.get("freestream"), "freestream",
                   {"density", "pressure", "temperature", "velocity", "speed", "viscosity", "mach",
                    "re", "ref_length", "aoa", "sideslip", "nu_tilde", "k", "omega"});
    Freestream& fs = in.freestream;
    fs.density = f.positive("density");
    fs.pressure = f.number("pressure");
    fs.temperature = f.positive_if("temperature", compressible);
    fs.velocity = f.vector3("velocity");
    fs.speed = f.positive("speed");
    fs.viscosity = f.positive("viscosity");
    fs.mach = f.is_null("mach") ? std::nullopt : std::optional<double>(f.positive("mach"));
    fs.re = f.is_null("re") ? std::nullopt : std::optional<double>(f.positive("re"));
    fs.ref_length =
        f.is_null("ref_length") ? std::nullopt : std::optional<double>(f.positive("ref_length"));
    fs.aoa = f.number("aoa");
    fs.sideslip = f.number("sideslip");
    fs.nu_tilde = f.positive_if("nu_tilde", sa);
    fs.k = f.positive_if("k", sst);
    fs.omega = f.positive_if("omega", sst);
  }
  const Json& boundaries = *array_at(root, "boundaries");
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    in.boundaries.push_back(parse_boundary(boundaries[i], indexed("boundaries", i)));
  }
  const Json& zones = *array_at(root, "cell_zones");
  if (zones.empty()) invalid("cell_zones", "at least one cell zone is required");
  for (std::size_t i = 0; i < zones.size(); ++i) {
    const std::string path = indexed("cell_zones", i);
    const Object z(zones[i], path, {"zone", "medium", "initial"});
    CellZoneInput zone;
    zone.zone = z.string("zone");
    zone.medium = parse_medium(z.get("medium"), z.at("medium"));
    zone.initial = parse_initial(z.get("initial"), z.at("initial"), in, zone.medium);
    in.cell_zones.push_back(std::move(zone));
  }
  {
    const Json& t = root.get("time");
    const Json kind = t.is_object() && t.contains("kind") ? t.at("kind") : Json();
    if (kind == "steady") {
      const Object o(t, "time",
                     {"kind", "max_iterations", "residual_orders", "linear_solver",
                      "coupled_turbulence", "matrix_free", "checkpoint"});
      in.time.kind = "steady";
      in.time.max_iterations = o.integer("max_iterations", 1);
      in.time.residual_orders = o.positive("residual_orders");
      in.time.linear_solver = o.tag("linear_solver", {"multigrid", "gauss-seidel", "lines"});
      in.time.coupled_turbulence = o.boolean("coupled_turbulence");
      in.time.matrix_free = o.boolean("matrix_free");
      parse_checkpoint(o, in.time);
    } else if (kind == "unsteady") {
      const Object o(t, "time", {"kind", "steps", "dt", "cfl", "average_from", "checkpoint"});
      in.time.kind = "unsteady";
      in.time.steps = o.integer("steps", 1);
      in.time.dt = o.is_null("dt") ? std::nullopt : std::optional<double>(o.positive("dt"));
      in.time.cfl = o.positive_if("cfl", !in.time.dt.has_value());
      if (!o.is_null("average_from")) {
        in.time.average_from = o.integer("average_from", 0);
        if (*in.time.average_from >= in.time.steps)
          invalid("time.average_from", "must be below steps");
      }
      parse_checkpoint(o, in.time);
    } else {
      invalid("time.kind", "expected steady or unsteady");
    }
  }
  // With verification every boundary is manufactured; without, none is.
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    if ((in.boundaries[i].kind == "manufactured") != in.verification.has_value()) {
      invalid(indexed("boundaries", i) + ".kind",
              in.verification ? "a verification case has only manufactured boundaries"
                              : "manufactured needs verification");
    }
  }
  if (in.verification && in.time.kind != "steady")
    invalid("time.kind", "verification cases are steady");
  // Per-face states carry no turbulence variables: compressible inviscid or laminar flow only.
  for (std::size_t i = 0; i < in.boundaries.size(); ++i) {
    if (in.boundaries[i].kind == "oracle-state" &&
        (in.physics_kind != "compressible" ||
         (in.turbulence.model != "inviscid" && in.turbulence.model != "laminar"))) {
      invalid(indexed("boundaries", i) + ".kind",
              "oracle-state needs compressible inviscid or laminar flow");
    }
  }
  if ((in.turbulence.model == "ddes" || in.turbulence.model == "wmles") &&
      in.time.kind != "unsteady") {
    invalid("time.kind", in.turbulence.model + " needs an unsteady time");
  }
  if (!root.is_null("start")) in.start = parse_start(root.get("start"), in);
  if (!root.is_null("initial_state")) {
    if (in.start)
      invalid("initial_state", "a run starts from start or from initial_state, not both");
    // The file carries no turbulence variables (initial-state.md).
    if (in.physics_kind != "compressible" ||
        (in.turbulence.model != "inviscid" && in.turbulence.model != "laminar")) {
      invalid("initial_state", "needs compressible inviscid or laminar flow");
    }
    const Object s(root.get("initial_state"), "initial_state", {"path", "sha256"});
    InitialStateFile file;
    file.path = s.string("path");
    if (file.path.find('\0') != std::string::npos) {
      invalid("initial_state.path", "contains a NUL character");
    }
    if (!std::filesystem::path(file.path).is_absolute()) {
      invalid("initial_state.path", "must be an absolute path");
    }
    file.sha256 = sha256(s, "sha256");
    in.initial_state = std::move(file);
  }
  const Json& periodic = *array_at(root, "periodic");
  for (std::size_t i = 0; i < periodic.size(); ++i) {
    const Object o(periodic[i], indexed("periodic", i), {"zones", "transform"});
    const Json& names = o.get("zones");
    if (!names.is_array() || names.size() != 2 || !names[0].is_string() || !names[1].is_string() ||
        names[0] == names[1]) {
      invalid(o.at("zones"), "expected two distinct zone names");
    }
    const Object t(o.get("transform"), o.at("transform"), {"kind", "vector"});
    (void)t.tag("kind", {"translation"});
    in.periodic.push_back({.zones = {names[0].get<std::string>(), names[1].get<std::string>()},
                           .translation = t.vector3("vector")});
  }
  const Json& monitors = *array_at(root, "monitors");
  for (std::size_t i = 0; i < monitors.size(); ++i) {
    const std::string path = indexed("monitors", i);
    const Object m(monitors[i], path,
                   {"kind", "name", "every", "zones", "ref_area", "ref_length", "moment_center"});
    (void)m.tag("kind", {"forces"});
    ForcesMonitor fm;
    fm.name = m.string("name");
    for (const auto& other : in.monitors) {
      if (other.name == fm.name)
        invalid(m.at("name"), "monitor name '" + fm.name + "' is used twice");
    }
    fm.every = m.integer("every", 1);
    const Json& names = *array_at(m, "zones");
    for (std::size_t j = 0; j < names.size(); ++j) {
      if (!names[j].is_string()) invalid(indexed(m.at("zones"), j), "expected a zone name");
      fm.zones.push_back(names[j].get<std::string>());
    }
    fm.ref_area =
        m.is_null("ref_area") ? std::nullopt : std::optional<double>(m.positive("ref_area"));
    fm.ref_length =
        m.is_null("ref_length") ? std::nullopt : std::optional<double>(m.positive("ref_length"));
    fm.moment_center = m.vector3("moment_center");
    in.monitors.push_back(std::move(fm));
  }
  return in;
}

std::vector<std::string> residual_names(const SolverInput& in) {
  std::vector<std::string> names =
      in.physics_kind == "compressible"
          ? std::vector<std::string>{"density", "momentum-x", "momentum-y", "momentum-z", "energy"}
          : std::vector<std::string>{"pressure", "momentum-x", "momentum-y", "momentum-z"};
  if (in.turbulence.nu_tilde_ratio) names.emplace_back("nu-tilde");
  if (in.turbulence.intensity) {
    names.emplace_back("k");
    names.emplace_back("omega");
  }
  return names;
}

std::vector<std::string> field_names(const SolverInput& in) {
  std::vector<std::string> names{"Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"};
  if (in.physics_kind == "compressible") names.emplace_back("Temperature");
  if (in.turbulence.nu_tilde_ratio) names.emplace_back("TurbulentSANuTilde");
  if (in.turbulence.intensity) {
    names.emplace_back("TurbulentEnergyKinetic");
    names.emplace_back("TurbulentDissipationRate");
  }
  return names;
}

}  // namespace autan::runtime
