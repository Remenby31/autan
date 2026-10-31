// autan-partition-report: partitions a structured hex box of n^3 cells and
// prints the time of each step and the partition quality as one JSON object
// per part count. Usage: autan-partition-report n method parts...
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/partition/partition.hpp"

namespace {

using autan::index_t;

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

autan::MeshInput hex_box(index_t n) {
  autan::MeshInput in;
  const auto np = n + 1;
  auto point = [np](index_t i, index_t j, index_t k) { return i + np * (j + np * k); };
  auto cell = [n](index_t i, index_t j, index_t k) { return i + n * (j + n * k); };
  for (index_t k = 0; k <= n; ++k)
    for (index_t j = 0; j <= n; ++j)
      for (index_t i = 0; i <= n; ++i) {
        in.x.push_back(static_cast<double>(i) / n);
        in.y.push_back(static_cast<double>(j) / n);
        in.z.push_back(static_cast<double>(k) / n);
      }
  in.face_node_offsets.push_back(0);
  // Each face is listed with its normal along +axis, from the cell at the
  // lower coordinate to the one at the higher; boundary faces at the lower
  // end are reversed so that they point out of their only cell.
  auto add = [&](std::initializer_list<index_t> nodes, index_t lower, index_t upper) {
    if (lower == autan::no_cell) {
      std::vector<index_t> reversed(nodes);
      in.face_nodes.push_back(reversed[0]);
      for (std::size_t r = reversed.size() - 1; r > 0; --r) in.face_nodes.push_back(reversed[r]);
      in.face_cell0.push_back(upper);
      in.face_cell1.push_back(autan::no_cell);
      in.face_zone.push_back(1);
    } else {
      in.face_nodes.insert(in.face_nodes.end(), nodes);
      in.face_cell0.push_back(lower);
      in.face_cell1.push_back(upper);
      in.face_zone.push_back(upper == autan::no_cell ? 1 : 0);
    }
    in.face_node_offsets.push_back(static_cast<autan::offset_t>(in.face_nodes.size()));
  };
  auto side = [n](index_t a, auto at) { return a < n ? at(a) : autan::no_cell; };
  for (index_t k = 0; k < n; ++k)
    for (index_t j = 0; j < n; ++j)
      for (index_t i = 0; i <= n; ++i)
        add({point(i, j, k), point(i, j + 1, k), point(i, j + 1, k + 1), point(i, j, k + 1)},
            i > 0 ? cell(i - 1, j, k) : autan::no_cell,
            side(i, [&](index_t a) { return cell(a, j, k); }));
  for (index_t k = 0; k < n; ++k)
    for (index_t j = 0; j <= n; ++j)
      for (index_t i = 0; i < n; ++i)
        add({point(i, j, k), point(i, j, k + 1), point(i + 1, j, k + 1), point(i + 1, j, k)},
            j > 0 ? cell(i, j - 1, k) : autan::no_cell,
            side(j, [&](index_t a) { return cell(i, a, k); }));
  for (index_t k = 0; k <= n; ++k)
    for (index_t j = 0; j < n; ++j)
      for (index_t i = 0; i < n; ++i)
        add({point(i, j, k), point(i + 1, j, k), point(i + 1, j + 1, k), point(i, j + 1, k)},
            k > 0 ? cell(i, j, k - 1) : autan::no_cell,
            side(k, [&](index_t a) { return cell(i, j, a); }));
  in.cell_zone.assign(static_cast<std::size_t>(n) * n * n, 0);
  in.face_zones.resize(2);
  in.face_zones[0].name = "interior";
  in.face_zones[0].boundary = false;
  in.face_zones[1].name = "walls";
  in.cell_zones.resize(1);
  in.cell_zones[0].name = "fluid";
  in.source = autan::MeshSource{.format = "memory", .path = ""};
  return in;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s n metis|scotch parts...\n", argv[0]);
    return 2;
  }
  const auto n = static_cast<index_t>(std::atoi(argv[1]));
  const std::string method_name = argv[2];
  autan::PartitionOptions options;
  options.method =
      method_name == "scotch" ? autan::PartitionMethod::scotch : autan::PartitionMethod::metis;

  auto start = std::chrono::steady_clock::now();
  const autan::Mesh mesh = autan::assemble(hex_box(n));
  const double t_mesh = seconds_since(start);
  start = std::chrono::steady_clock::now();
  const autan::Geometry geometry = autan::compute_geometry(mesh);
  const double t_geometry = seconds_since(start);

  for (int a = 3; a < argc; ++a) {
    const auto n_parts = static_cast<std::int32_t>(std::atoi(argv[a]));
    start = std::chrono::steady_clock::now();
    const auto parts = autan::partition_cells(mesh, n_parts, options);
    const double t_partition = seconds_since(start);
    const auto q = autan::partition_quality(mesh, parts, n_parts);
    start = std::chrono::steady_clock::now();
    const auto local = autan::extract_partition(mesh, geometry, parts, n_parts, 0);
    const double t_extract = seconds_since(start);
    std::printf(
        "{\"cells\": %d, \"method\": \"%s\", \"parts\": %d, \"cut_faces\": %lld, "
        "\"cut_fraction\": %.4f, \"imbalance\": %.4f, \"ghost_ratio\": %.4f, "
        "\"max_ghost_ratio\": %.4f, \"max_neighbours\": %d, \"assemble_s\": %.2f, "
        "\"geometry_s\": %.2f, \"partition_s\": %.2f, \"extract_rank0_s\": %.2f, "
        "\"rank0_cells\": %d}\n",
        mesh.n_cells(), method_name.c_str(), n_parts, static_cast<long long>(q.cut_faces),
        static_cast<double>(q.cut_faces) / mesh.n_interior_faces(), q.imbalance, q.ghost_ratio,
        q.max_ghost_ratio, q.max_neighbours, t_mesh, t_geometry, t_partition, t_extract,
        local.mesh.n_cells());
    std::fflush(stdout);
  }
  return 0;
}
