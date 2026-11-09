// autan-mesh-info: reads a mesh and prints its counts, zones and quality, or
// with --json the MeshInfo document of docs/contracts/mesh-info.md, or with
// --surface OUT writes its boundary surface (docs/contracts/mesh.md).

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/quality.hpp"
#include "autan/io/mesh_counts.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"

namespace {

constexpr const char* usage =
    "usage: autan-mesh-info [--json] [--scale S] [--cell-order rcm|input]\n"
    "                       [--multigrid-order rcm|input]\n"
    "                       [--ranks N [--groups lines]] MESH\n"
    "       autan-mesh-info --surface OUT MESH\n"
    "\n"
    "Reads MESH (Gmsh .msh 4.1, Fluent .msh, CGNS, or an OpenFOAM polyMesh or\n"
    "case directory) and prints its counts, zones and quality. With --json,\n"
    "prints the MeshInfo document (schema autan.mesh-info/1) instead: what a\n"
    "run computes on it with S metres per mesh unit (default 1), its smoother\n"
    "lines in the cell order of --cell-order (default rcm, the run's), the\n"
    "multigrid hierarchy in the cell order of --multigrid-order (default\n"
    "--cell-order's; input for a multigrid run on a partition: several ranks,\n"
    "or reproducibility \"ranks\"),\n"
    "and the partition of a run on one rank, or with --ranks N on N ranks,\n"
    "keeping the smoother's lines whole with --groups lines (N >= 2). With\n"
    "--surface OUT, writes its boundary faces by zone to OUT (schema\n"
    "autan.surface/1) and prints nothing.\n"
    "Exits 1 with a coded error (E01xx) if the mesh cannot be read or cannot\n"
    "be partitioned, 2 on a usage error.\n";

double max_of(const std::vector<double>& v) {
  return v.empty() ? 0.0 : *std::ranges::max_element(v);
}

void print_text(const autan::Mesh& mesh, const std::string& hash) {
  const autan::Geometry g = autan::compute_geometry(mesh);
  const autan::MeshQuality q = autan::compute_quality(mesh, g);
  const auto closure = autan::cell_closure_error(mesh, g);
  std::printf("mesh      %s (%s)\nsha256    %s\n", mesh.source().path.c_str(),
              mesh.source().format.c_str(), hash.c_str());
  std::printf("cells     %d\nfaces     %d (%d interior)\nnodes     %d\nvolume    %.17g\n",
              mesh.n_cells(), mesh.n_faces(), mesh.n_interior_faces(), mesh.n_points(),
              std::accumulate(g.cell_volume.begin(), g.cell_volume.end(), 0.0));
  if (const auto& snap = mesh.periodic_snap(); snap.nodes > 0) {
    std::printf(
        "W0150     %lld periodic nodes moved, at most %.3g (pair mismatch %.3g of the "
        "tolerance, volume change %.3g)\n",
        static_cast<long long>(snap.nodes), snap.max_distance, snap.max_relative,
        snap.max_volume_change);
  }
  std::printf("\nboundary zones\n");
  const auto offsets = mesh.boundary_zone_offsets();
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    const auto& zone = mesh.face_zones()[z];
    if (!zone.boundary) continue;
    std::printf("  %-24s %-20s %d faces\n", zone.name.c_str(),
                zone.bc_type.empty() ? "-" : zone.bc_type.c_str(), offsets[z + 1] - offsets[z]);
  }
  std::printf("\ncell zones\n");
  for (std::size_t z = 0; z < mesh.cell_zones().size(); ++z) {
    const auto& zone = mesh.cell_zones()[z];
    const auto cells = std::count(mesh.cell_zone().begin(), mesh.cell_zone().end(),
                                  static_cast<autan::index_t>(z));
    std::printf("  %-24s %-20s %td cells\n", zone.name.c_str(),
                zone.type.empty() ? "-" : zone.type.c_str(), cells);
  }
  std::printf("\nquality\n");
  const double min_ratio = q.volume_ratio.empty() ? 1.0 : *std::ranges::min_element(q.volume_ratio);
  std::printf("  max non-orthogonality  %.3f deg\n", max_of(q.non_orthogonality));
  std::printf("  max skewness           %.3g\n", max_of(q.skewness));
  std::printf("  max aspect ratio       %.3g\n", max_of(q.aspect_ratio));
  std::printf("  min volume ratio       %.3g\n", min_ratio);
  std::printf("  max closure error      %.3g\n", max_of(closure));
}

std::optional<double> positive(std::string_view text) {
  const std::string copy(text);
  char* end = nullptr;
  const double v = std::strtod(copy.c_str(), &end);
  if (copy.empty() || end != copy.c_str() + copy.size() || !std::isfinite(v) || v <= 0.0)
    return std::nullopt;
  return v;
}

std::optional<std::int32_t> rank_count(std::string_view text) {
  std::int32_t n = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
  if (error != std::errc{} || end != text.data() + text.size() || n < 1) return std::nullopt;
  return n;
}

}  // namespace

int main(int argc, char* const* argv) {
  bool json = false;
  const char* surface = nullptr;
  const char* path = nullptr;
  std::optional<autan::io::CellOrder> order;
  std::optional<autan::io::CellOrder> multigrid_order;
  std::optional<double> scale;
  std::optional<autan::io::RanksRequest> ranks;
  bool groups = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const std::string_view next = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--json" && !json) {
      json = true;
    } else if (arg == "--surface" && surface == nullptr && i + 1 < argc) {
      surface = argv[++i];
    } else if (arg == "--scale" && !scale && positive(next)) {
      scale = positive(next);
      ++i;
    } else if (arg == "--cell-order" && !order && (next == "rcm" || next == "input")) {
      order = next == "rcm" ? autan::io::CellOrder::rcm : autan::io::CellOrder::input;
      ++i;
    } else if (arg == "--multigrid-order" && !multigrid_order &&
               (next == "rcm" || next == "input")) {
      multigrid_order = next == "rcm" ? autan::io::CellOrder::rcm : autan::io::CellOrder::input;
      ++i;
    } else if (const auto count = arg == "--ranks" && !ranks ? rank_count(next) : std::nullopt;
               count) {
      ranks = autan::io::RanksRequest{.ranks = *count, .groups = autan::io::RankGroups::none};
      ++i;
    } else if (arg == "--groups" && !groups && next == "lines") {
      groups = true;
      ++i;
    } else if (arg == "-h" || arg == "--help") {
      std::fputs(usage, stdout);
      return 0;
    } else if (path == nullptr && !arg.starts_with('-')) {
      path = argv[i];
    } else {
      std::fputs(usage, stderr);
      return 2;
    }
  }
  const bool counted = scale || order || multigrid_order || ranks || groups;
  if (path == nullptr || (surface != nullptr && (json || counted)) || (counted && !json) ||
      (groups && (!ranks || ranks->ranks < 2))) {
    std::fputs(usage, stderr);
    return 2;
  }
  if (groups) ranks->groups = autan::io::RankGroups::lines;
  try {
    const autan::Mesh mesh = autan::io::read_mesh(path);
    const std::string hash = autan::io::mesh_hash(path);
    if (surface != nullptr) {
      const std::string bytes = autan::io::surface_bytes(mesh, hash);
      std::ofstream out(surface, std::ios::binary);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      out.close();
      if (out.fail()) {
        throw autan::Error(autan::ErrorCode::io_cannot_open,
                           "cannot write '" + std::string(surface) + "'");
      }
    } else if (json) {
      const auto zone_groups = autan::io::read_zone_groups(path, mesh, hash);
      const auto run =
          autan::io::run_counts(mesh, scale.value_or(1.0),
                                order.value_or(autan::io::CellOrder::rcm), ranks, multigrid_order);
      std::printf("%s\n",
                  autan::io::mesh_info_json(mesh, hash, autan::io::mesh_bytes(path),
                                            autan::io::mesh_logical_bytes(path), run, zone_groups)
                      .c_str());
    } else {
      print_text(mesh, hash);
    }
  } catch (const autan::Error& e) {
    std::fprintf(stderr, "autan-mesh-info: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "autan-mesh-info: internal error: %s\n", e.what());
    return 1;
  }
  return 0;
}
