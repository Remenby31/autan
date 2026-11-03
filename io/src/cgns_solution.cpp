// Reading back the cell fields of a solution.cgns (results.md) to restart.

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cgnslib.h>

#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "cgns_util.hpp"
#include "self_contained.hpp"
#include "text_limit.hpp"

namespace autan::io {
namespace {

void ok(int status, const std::string& what) {
  if (status != CG_OK)
    throw Error(ErrorCode::io_malformed, "CGNS: cannot " + what + ": " + cg_get_error());
}

[[noreturn]] void unsupported(const std::filesystem::path& file, const std::string& what) {
  throw Error(ErrorCode::io_unsupported, file.string() + ": " + what);
}

}  // namespace

std::int64_t solution_cells(const std::filesystem::path& file) {
  reject_outside_data(file);
  const CgioTree tree(file);
  const auto base = tree.child(tree.root(), "CGNSBase_t");
  const auto zone = base ? tree.child(*base, "Zone_t") : std::nullopt;
  if (!zone) throw Error(ErrorCode::io_malformed, file.string() + ": no zone");
  std::array<std::int64_t, 2> size{};  // [nodes, cells, ...]
  tree.read_integers(*zone, size);
  return size[1];
}

void check_start_solution(const std::filesystem::path& file, const StartLimits& limits) {
  reject_outside_data(file);
  const CgioTree tree(file);
  const auto refuse = [&](const std::string& what) {
    throw Error(ErrorCode::io_malformed,
                file.string() + ": " + what + ", which results.md's writer does not write");
  };
  std::int64_t translations = 0, families = 0;
  // Every descriptor of the tree, where the writer puts them and no larger than it writes them.
  const auto walk = [&](const auto& self, double node, const std::string& parent) -> void {
    for (const double child : tree.children(node)) {
      const std::string label = tree.label(child);
      if (label == "Descriptor_t") {
        const std::string name = tree.name(child);
        const std::int64_t bytes = tree.logical_bytes(child);
        const bool in_base = parent == "CGNSBase_t";
        if (in_base && name == "RunMetadata" && bytes <= (std::int64_t{64} << 10)) continue;
        if (in_base && name == "ZoneNames" && bytes <= (limits.zones + 1) * 25600) continue;
        if (parent == "GridConnectivity_t" && name == "PeriodicTranslation" && bytes <= 4096 &&
            ++translations <= 2 * limits.periodic_pairs) {
          continue;
        }
        refuse("a descriptor '" + name + "' of " + std::to_string(bytes) + " bytes");
      }
      if (label == "Family_t" && ++families > limits.zones)
        refuse(std::to_string(families) + " families for " + std::to_string(limits.zones) +
               " zones");
      self(self, child, label);
    }
  };
  walk(walk, tree.root(), "");
  if (const std::int64_t n = total(count_nodes(tree)); n > max_cgns_nodes) {
    refuse(std::to_string(n) + " nodes (at most " + std::to_string(max_cgns_nodes) + ")");
  }
  if (largest_filtered_bytes(file) > 0) refuse("a filtered (compressed) dataset");
}

StoredSolution read_solution(const std::filesystem::path& file, bool whole_state) {
  reject_outside_data(file);  // E0134, before the library reads anything
  const CgnsFile in(file, CG_MODE_READ);
  const int fn = in.handle();
  StoredSolution out;
  int count = 0;
  ok(cg_nbases(fn, &count), "count bases");
  if (count != 1) unsupported(file, std::to_string(count) + " bases (one is supported)");
  ok(cg_nzones(fn, 1, &count), "count zones");
  if (count != 1) unsupported(file, std::to_string(count) + " zones (one is supported)");
  // cg_zone_read writes 3 sizes for an unstructured zone, up to 9 otherwise.
  CGNS_ENUMT(ZoneType_t) zone_type{};
  ok(cg_zone_type(fn, 1, 1, &zone_type), "read the zone type");
  if (zone_type != CGNS_ENUMV(Unstructured)) unsupported(file, "a structured zone");
  char name[33];
  cgsize_t size[3] = {0, 0, 0};
  ok(cg_zone_read(fn, 1, 1, name, size), "read the zone");
  if (size[1] < 1 || size[1] > std::numeric_limits<index_t>::max()) {
    unsupported(file, std::to_string(size[1]) + " cells");
  }
  out.cells = static_cast<index_t>(size[1]);
  // results.md, RunMetadata: run.json's fields with two free texts (the case name, the mesh
  // path) of at most 4096 bytes, each escaped up to 6 bytes per byte by ensure_ascii, and fixed
  // structure: at most 64 KiB, sized from its node before it is read. No other descriptor is read.
  const CgioTree tree(fn);
  const auto base = tree.child(tree.root(), "CGNSBase_t");
  if (const auto node = base ? tree.child(*base, "Descriptor_t", "RunMetadata") : std::nullopt) {
    check_text(file.string(), "RunMetadata descriptor",
               static_cast<std::size_t>(tree.logical_bytes(*node)), std::size_t{64} << 10);
    out.run_metadata = tree.text(*node);
  }
  int solutions = 0;
  ok(cg_nsols(fn, 1, 1, &solutions), "count solutions");
  for (int s = 1; s <= solutions; ++s) {
    CGNS_ENUMT(GridLocation_t) location{};
    ok(cg_sol_info(fn, 1, 1, s, name, &location), "read a solution");
    const std::string solution = name;
    std::vector<Field>* target = solution == "FlowSolution"       ? &out.fields
                                 : !whole_state                   ? nullptr
                                 : solution == "FlowSolutionMean" ? &out.mean_fields
                                 : solution == "RestartState"     ? &out.restart_fields
                                                                  : nullptr;
    if (target == nullptr) continue;
    if (location != CGNS_ENUMV(CellCenter)) {
      throw Error(ErrorCode::io_malformed,
                  file.string() + ": " + solution + " is not cell-centred");
    }
    int fields = 0;
    ok(cg_nfields(fn, 1, 1, s, &fields), "count fields");
    for (int f = 1; f <= fields; ++f) {
      CGNS_ENUMT(DataType_t) type{};
      ok(cg_field_info(fn, 1, 1, s, f, &type, name), "read a field");
      Field field{.name = name, .values = std::vector<double>(static_cast<std::size_t>(out.cells))};
      const cgsize_t first = 1;
      const cgsize_t last = out.cells;
      ok(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &last,
                       field.values.data()),
         "read field " + field.name);
      target->push_back(std::move(field));
    }
  }
  return out;
}

}  // namespace autan::io
