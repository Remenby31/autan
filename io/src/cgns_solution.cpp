// Reading back the cell fields of a solution.cgns (results.md) to restart.

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <cgnslib.h>

#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "cgns_util.hpp"
#include "self_contained.hpp"

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

StoredSolution read_solution(const std::filesystem::path& file) {
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
  ok(cg_goto(fn, 1, "end"), "go to the base");
  int descriptors = 0;
  ok(cg_ndescriptors(&descriptors), "count descriptors");
  for (int d = 1; d <= descriptors; ++d) {
    char* text = nullptr;
    ok(cg_descriptor_read(d, name, &text), "read a descriptor");
    if (std::string_view(name) == "RunMetadata") out.run_metadata = text;
    cg_free(text);
  }
  int solutions = 0;
  ok(cg_nsols(fn, 1, 1, &solutions), "count solutions");
  for (int s = 1; s <= solutions; ++s) {
    CGNS_ENUMT(GridLocation_t) location{};
    ok(cg_sol_info(fn, 1, 1, s, name, &location), "read a solution");
    const std::string solution = name;
    std::vector<Field>* target = solution == "FlowSolution"       ? &out.fields
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
