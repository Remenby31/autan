// Reading back the cell fields of a solution.cgns (results.md) to restart.

#include <string>
#include <string_view>
#include <vector>

#include <cgnslib.h>

#include "autan/core/error.hpp"
#include "autan/io/cgns.hpp"
#include "cgns_util.hpp"

namespace autan::io {
namespace {

void ok(int status, const std::string& what) {
  if (status != CG_OK)
    throw Error(ErrorCode::io_malformed, "CGNS: cannot " + what + ": " + cg_get_error());
}

}  // namespace

StoredSolution read_solution(const std::filesystem::path& file, index_t cells) {
  CgnsFile in(file, CG_MODE_READ);
  const int fn = in.handle();
  StoredSolution out;
  char name[33];
  cgsize_t size[3] = {0, 0, 0};
  ok(cg_zone_read(fn, 1, 1, name, size), "read the zone");
  if (size[1] != cells) {
    throw Error(ErrorCode::io_malformed, file.string() + ": the solution has " +
                                             std::to_string(size[1]) + " cells, the mesh " +
                                             std::to_string(cells));
  }
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
      Field field{.name = name, .values = std::vector<double>(static_cast<std::size_t>(cells))};
      const cgsize_t first = 1;
      const cgsize_t last = cells;
      ok(cg_field_read(fn, 1, 1, s, name, CGNS_ENUMV(RealDouble), &first, &last,
                       field.values.data()),
         "read field " + field.name);
      target->push_back(std::move(field));
    }
  }
  return out;
}

}  // namespace autan::io
