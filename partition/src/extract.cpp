#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/partition/partition.hpp"

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

void check_request(const Mesh& global, const Geometry& geometry,
                   std::span<const std::int32_t> parts, std::int32_t n_parts, std::int32_t rank) {
  if (global.n_owned_cells() != global.n_cells()) {
    throw Error(ErrorCode::partition_input_has_ghosts, "the mesh to split already has ghost cells");
  }
  if (!global.periodic_pairs().empty()) {
    throw Error(ErrorCode::partition_invalid_request, "periodic meshes are not partitioned yet");
  }
  if (parts.size() != at(global.n_cells()) || n_parts < 1 || rank < 0 || rank >= n_parts) {
    throw Error(ErrorCode::partition_invalid_request,
                "parts has " + std::to_string(parts.size()) + " entries for " +
                    std::to_string(global.n_cells()) + " cells; rank " + std::to_string(rank) +
                    " of " + std::to_string(n_parts));
  }
  if (geometry.cell_volume.size() != at(global.n_cells()) ||
      geometry.face_area.size() != at(global.n_faces())) {
    throw Error(ErrorCode::partition_invalid_request,
                "the geometry does not belong to the mesh to split");
  }
  for (std::size_t c = 0; c < parts.size(); ++c) {
    if (parts[c] < 0 || parts[c] >= n_parts) {
      throw Error(ErrorCode::partition_invalid_request,
                  "cell " + std::to_string(c) + " is in part " + std::to_string(parts[c]));
    }
  }
}

}  // namespace

LocalPartition extract_partition(const Mesh& global, const Geometry& global_geometry,
                                 std::span<const std::int32_t> parts, std::int32_t n_parts,
                                 std::int32_t rank) {
  check_request(global, global_geometry, parts, n_parts, rank);
  const auto owner = global.owner();
  const auto neighbour = global.neighbour();
  const auto n_interior = at(global.n_interior_faces());
  const auto gid = global.cell_global_id();

  // Owned cells keep their relative global order; ghosts follow, by owning
  // part then by global order, so each neighbour's ghosts are contiguous.
  std::vector<index_t> local(at(global.n_cells()), no_cell);
  std::vector<index_t> owned;
  for (std::size_t c = 0; c < parts.size(); ++c) {
    if (parts[c] == rank) owned.push_back(static_cast<index_t>(c));
  }
  if (owned.empty()) {
    throw Error(ErrorCode::partition_invalid_request,
                "rank " + std::to_string(rank) + " owns no cell");
  }
  for (std::size_t i = 0; i < owned.size(); ++i) local[at(owned[i])] = static_cast<index_t>(i);

  std::vector<index_t> faces;
  std::vector<std::pair<std::int32_t, index_t>> ghosts;  // (part, global cell)
  std::vector<std::pair<std::int32_t, index_t>> sends;   // (part, owned global cell)
  for (std::size_t f = 0; f < at(global.n_faces()); ++f) {
    const index_t o = owner[f];
    const index_t n = f < n_interior ? neighbour[f] : no_cell;
    const bool owns_o = parts[at(o)] == rank;
    const bool owns_n = n != no_cell && parts[at(n)] == rank;
    if (!owns_o && !owns_n) continue;
    faces.push_back(static_cast<index_t>(f));
    if (n == no_cell || owns_o == owns_n) continue;
    const index_t inside = owns_o ? o : n;
    const index_t outside = owns_o ? n : o;
    ghosts.emplace_back(parts[at(outside)], outside);
    sends.emplace_back(parts[at(outside)], inside);
  }
  std::sort(ghosts.begin(), ghosts.end());
  ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
  std::sort(sends.begin(), sends.end());
  sends.erase(std::unique(sends.begin(), sends.end()), sends.end());
  const auto n_owned = static_cast<index_t>(owned.size());
  for (std::size_t g = 0; g < ghosts.size(); ++g) {
    local[at(ghosts[g].second)] = n_owned + static_cast<index_t>(g);
  }
  if (owned.size() + ghosts.size() > at(std::numeric_limits<index_t>::max())) {
    throw Error(ErrorCode::mesh_too_large, "the partition exceeds the index_t range");
  }

  MeshInput in;
  in.face_zones.assign(global.face_zones().begin(), global.face_zones().end());
  in.cell_zones.assign(global.cell_zones().begin(), global.cell_zones().end());
  in.source = global.source();
  in.n_ghost_cells = static_cast<index_t>(ghosts.size());
  const auto n_local_cells = owned.size() + ghosts.size();
  in.cell_zone.resize(n_local_cells);
  in.cell_global_id.resize(n_local_cells);
  auto put_cell = [&](index_t global_cell) {
    const auto l = at(local[at(global_cell)]);
    in.cell_zone[l] = global.cell_zone()[at(global_cell)];
    in.cell_global_id[l] = gid[at(global_cell)];
  };
  for (const index_t c : owned) put_cell(c);
  for (const auto& [p, c] : ghosts) put_cell(c);

  // Points in ascending global order, so the same global mesh yields the
  // same local arrays whatever the other parts are.
  std::vector<index_t> point_local(at(global.n_points()), -1);
  for (const index_t f : faces) {
    for (const index_t p : global.face_nodes(f)) point_local[at(p)] = 0;
  }
  index_t n_points = 0;
  for (std::size_t p = 0; p < point_local.size(); ++p) {
    if (point_local[p] < 0) continue;
    point_local[p] = n_points++;
    in.x.push_back(global.x()[p]);
    in.y.push_back(global.y()[p]);
    in.z.push_back(global.z()[p]);
  }

  in.face_node_offsets.reserve(faces.size() + 1);
  in.face_node_offsets.push_back(0);
  for (const index_t f : faces) {
    for (const index_t p : global.face_nodes(f)) in.face_nodes.push_back(point_local[at(p)]);
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(local[at(owner[at(f)])]);
    in.face_cell1.push_back(at(f) < n_interior ? local[at(neighbour[at(f)])] : no_cell);
    in.face_zone.push_back(global.face_zone()[at(f)]);
  }

  AssemblyMap map;
  LocalPartition out{
      .mesh = assemble(std::move(in), &map), .geometry = {}, .face_global_id = {}, .halo = {}};
  const Mesh& mesh = out.mesh;

  const auto n_faces = at(mesh.n_faces());
  const Geometry& gg = global_geometry;
  Geometry& g = out.geometry;
  g.face_area_x.resize(n_faces);
  g.face_area_y.resize(n_faces);
  g.face_area_z.resize(n_faces);
  g.face_area.resize(n_faces);
  g.face_centroid_x.resize(n_faces);
  g.face_centroid_y.resize(n_faces);
  g.face_centroid_z.resize(n_faces);
  out.face_global_id.resize(n_faces);
  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto source = at(faces[at(map.face_source[f])]);
    const double sign = map.face_reversed[f] != 0 ? -1.0 : 1.0;
    out.face_global_id[f] = static_cast<std::int64_t>(source);
    g.face_area_x[f] = sign * gg.face_area_x[source];
    g.face_area_y[f] = sign * gg.face_area_y[source];
    g.face_area_z[f] = sign * gg.face_area_z[source];
    g.face_area[f] = gg.face_area[source];
    g.face_centroid_x[f] = gg.face_centroid_x[source];
    g.face_centroid_y[f] = gg.face_centroid_y[source];
    g.face_centroid_z[f] = gg.face_centroid_z[source];
  }
  const auto n_cells = at(mesh.n_cells());
  constexpr double unset = std::numeric_limits<double>::quiet_NaN();
  g.cell_volume.assign(n_cells, unset);
  g.cell_centroid_x.assign(n_cells, unset);
  g.cell_centroid_y.assign(n_cells, unset);
  g.cell_centroid_z.assign(n_cells, unset);
  for (std::size_t c = 0; c < owned.size(); ++c) {
    const auto source = at(owned[c]);
    g.cell_volume[c] = gg.cell_volume[source];
    g.cell_centroid_x[c] = gg.cell_centroid_x[source];
    g.cell_centroid_y[c] = gg.cell_centroid_y[source];
    g.cell_centroid_z[c] = gg.cell_centroid_z[source];
  }

  HaloPattern& h = out.halo;
  h.rank = rank;
  h.n_ranks = n_parts;
  for (std::size_t g_i = 0; g_i < ghosts.size(); ++g_i) {
    if (h.neighbours.empty() || h.neighbours.back() != ghosts[g_i].first) {
      if (!h.neighbours.empty()) h.recv_offsets.push_back(static_cast<index_t>(g_i));
      h.neighbours.push_back(ghosts[g_i].first);
    }
  }
  if (!h.neighbours.empty()) h.recv_offsets.push_back(static_cast<index_t>(ghosts.size()));
  // A cell adjacent to part q is a ghost of q exactly when q is adjacent to
  // it, so send lists and the neighbours' ghost lists hold the same cells in
  // the same (global) order.
  for (std::size_t s = 0; s < sends.size(); ++s) {
    h.send_cells.push_back(local[at(sends[s].second)]);
    if (s + 1 == sends.size() || sends[s + 1].first != sends[s].first) {
      h.send_offsets.push_back(static_cast<index_t>(s + 1));
    }
  }

  std::vector<std::pair<index_t, index_t>> slots(h.send_cells.size());
  for (std::size_t s = 0; s < slots.size(); ++s) {
    slots[s] = {h.send_cells[s], static_cast<index_t>(s)};
  }
  std::sort(slots.begin(), slots.end());
  for (std::size_t s = 0; s < slots.size(); ++s) {
    if (s == 0 || slots[s].first != slots[s - 1].first) {
      if (s > 0) h.reverse_offsets.push_back(static_cast<index_t>(s));
      h.reverse_cells.push_back(slots[s].first);
    }
    h.reverse_slots.push_back(slots[s].second);
  }
  if (!slots.empty()) h.reverse_offsets.push_back(static_cast<index_t>(slots.size()));

  const auto local_neighbour = mesh.neighbour();
  for (std::size_t f = 0; f < local_neighbour.size(); ++f) {
    if (local_neighbour[f] >= n_owned) h.interface_faces.push_back(static_cast<index_t>(f));
  }
  return out;
}

}  // namespace autan
