// autan-mesh-prisms: merges the boundary-layer tetrahedra of a mesh into prisms (#337) and
// writes the result as a polyMesh, with a JSON report of what it merged and where it stopped.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "autan/core/error.hpp"
#include "autan/core/prisms.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/io/write_polymesh.hpp"

namespace {

constexpr const char* usage =
    "usage: autan-mesh-prisms [--planarity DEG] [--pillar DEG] [--aspect A] MESH OUT\n"
    "\n"
    "Reads MESH, merges in its boundary layer each set of three tetrahedra that\n"
    "fills a triangular prism into one cell (its sides kept as their two\n"
    "triangles, so that the cell is exactly their union), advancing in layers\n"
    "from its wall zones, and writes the result to the polyMesh directory OUT,\n"
    "with the"
    " same nodes, boundary faces and zones. A prism's side quadrilateral must be\n"
    "planar within --planarity degrees (default 5), its pillars within --pillar\n"
    "degrees of its wall-side cap's normal (default 30) and its height at most A\n"
    "times its cap's shortest edge (default 1). Prints a JSON report. Exits 1 with\n"
    "a coded error (E01xx) if the mesh cannot be read or written, 2 on a usage\n"
    "error.\n";

std::optional<double> positive(std::string_view text) {
  const std::string copy(text);
  char* end = nullptr;
  const double v = std::strtod(copy.c_str(), &end);
  if (copy.empty() || end != copy.c_str() + copy.size() || !std::isfinite(v) || v <= 0.0)
    return std::nullopt;
  return v;
}

}  // namespace

int main(int argc, char* const* argv) {
  autan::PrismOptions options;
  const char* in = nullptr;
  const char* out = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const std::string_view next = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--planarity" && positive(next) && *positive(next) < 90.0) {
      options.planarity_deg = *positive(next);
      ++i;
    } else if (arg == "--pillar" && positive(next) && *positive(next) < 90.0) {
      options.pillar_deg = *positive(next);
      ++i;
    } else if (arg == "--aspect" && positive(next)) {
      options.aspect = *positive(next);
      ++i;
    } else if (arg == "-h" || arg == "--help") {
      std::fputs(usage, stdout);
      return 0;
    } else if (!arg.starts_with('-') && in == nullptr) {
      in = argv[i];
    } else if (!arg.starts_with('-') && out == nullptr) {
      out = argv[i];
    } else {
      std::fputs(usage, stderr);
      return 2;
    }
  }
  if (in == nullptr || out == nullptr) {
    std::fputs(usage, stderr);
    return 2;
  }
  try {
    const autan::Mesh mesh = autan::io::read_mesh(in);
    const auto merged = autan::merge_layer_prisms(mesh, options);
    autan::io::write_polymesh(out, merged.mesh);
    const auto& r = merged.report;
    const nlohmann::ordered_json report{{"cells_in", mesh.n_cells()},
                                        {"cells_out", merged.mesh.n_cells()},
                                        {"faces_in", mesh.n_faces()},
                                        {"faces_out", merged.mesh.n_faces()},
                                        {"prisms", r.prisms},
                                        {"tetrahedra", r.tetrahedra},
                                        {"other_cells", r.other_cells},
                                        {"layers", r.layers},
                                        {"options",
                                         {{"planarity_deg", options.planarity_deg},
                                          {"pillar_deg", options.pillar_deg},
                                          {"aspect", options.aspect}}},
                                        {"stopped",
                                         {{"no_candidate", r.no_candidate},
                                          {"ambiguous", r.ambiguous},
                                          {"not_planar", r.not_planar},
                                          {"not_aligned", r.not_aligned},
                                          {"not_layer", r.not_layer},
                                          {"invalid", r.invalid},
                                          {"across_zones", r.across_zones},
                                          {"not_tetrahedron", r.not_tetrahedron}}}};
    std::printf("%s\n", report.dump(2).c_str());
  } catch (const autan::Error& e) {
    std::fprintf(stderr, "autan-mesh-prisms: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "autan-mesh-prisms: internal error: %s\n", e.what());
    return 1;
  }
  return 0;
}
