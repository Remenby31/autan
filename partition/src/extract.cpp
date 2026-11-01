#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/partition/partition.hpp"

namespace autan {
namespace {

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

// A copy of a global cell or node in the periodic covering of the mesh: its
// position is the original one plus sum_p shift[p] * translation of pair p.
constexpr std::size_t max_pairs = 3;
using Shift = std::array<std::int32_t, max_pairs>;
struct Key {
  index_t entity;
  Shift shift;
  auto operator<=>(const Key&) const = default;
};

void check_request(const Mesh& global, const Geometry& geometry,
                   std::span<const std::int32_t> parts, std::int32_t n_parts, std::int32_t rank,
                   int layers) {
  if (global.n_owned_cells() != global.n_cells()) {
    throw Error(ErrorCode::partition_input_has_ghosts, "the mesh to split already has ghost cells");
  }
  if (global.periodic_pairs().size() > max_pairs) {
    throw Error(ErrorCode::partition_invalid_request,
                "more than " + std::to_string(max_pairs) + " periodic pairs");
  }
  if (layers != 1 && layers != 2) {
    throw Error(ErrorCode::partition_invalid_request, "a halo of 1 or 2 layers");
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

// The periodic structure of the global mesh: for each face of a periodic
// zone, its pair, its side and its partner face; for each node of a zone a,
// per pair, the coinciding node of zone b.
class Covering {
 public:
  explicit Covering(const Mesh& mesh) : mesh_(mesh) {
    const auto n_faces = at(mesh.n_faces());
    partner_.assign(n_faces, -1);
    pair_.assign(n_faces, -1);
    side_.assign(n_faces, 0);
    const auto off = mesh.boundary_zone_offsets();
    const auto x = mesh.x(), y = mesh.y(), z = mesh.z();
    for (std::size_t p = 0; p < mesh.periodic_pairs().size(); ++p) {
      const auto& pair = mesh.periodic_pairs()[p];
      const auto partners = mesh.periodic_partners(p);
      std::vector<index_t>& image = node_image_.emplace_back(at(mesh.n_points()), -1);
      for (std::size_t i = 0; i < partners.size(); ++i) {
        const auto fa = at(off[at(pair.zone_a)]) + i;
        const auto fb = at(partners[i]);
        partner_[fa] = static_cast<index_t>(fb);
        partner_[fb] = static_cast<index_t>(fa);
        pair_[fa] = pair_[fb] = static_cast<std::int8_t>(p);
        side_[fa] = 1;
        side_[fb] = -1;
        // The core matched these faces node for node; recover the pairs.
        for (const index_t na : mesh.face_nodes(static_cast<index_t>(fa))) {
          const auto a = at(na);
          double best = std::numeric_limits<double>::infinity();
          for (const index_t nb : mesh.face_nodes(static_cast<index_t>(fb))) {
            const auto b = at(nb);
            const double d =
                std::hypot(x[b] - x[a] - pair.translation[0], y[b] - y[a] - pair.translation[1],
                           z[b] - z[a] - pair.translation[2]);
            if (d < best) {
              best = d;
              image[a] = nb;
            }
          }
        }
      }
    }
  }

  // Translation of a shift.
  [[nodiscard]] std::array<double, 3> translation(const Shift& k) const {
    std::array<double, 3> t{};
    for (std::size_t p = 0; p < mesh_.periodic_pairs().size(); ++p) {
      for (std::size_t d = 0; d < 3; ++d) t[d] += k[p] * mesh_.periodic_pairs()[p].translation[d];
    }
    return t;
  }

  // A node copy named by a node of zone b wherever it coincides with one:
  // the two sides of a pair share their nodes in the covering.
  [[nodiscard]] Key canonical_node(index_t node, Shift k) const {
    for (bool moved = true; moved;) {
      moved = false;
      for (std::size_t p = 0; p < node_image_.size(); ++p) {
        const index_t image = node_image_[p][at(node)];
        if (image >= 0) {
          node = image;
          --k[p];
          moved = true;
        }
      }
    }
    return {node, k};
  }

  // One face of a cell copy seen from it: the face that represents it (the
  // zone-a face of a periodic pair), the shift of that face's owner copy,
  // and the cell copy on the other side (entity no_cell on a boundary).
  struct Side {
    index_t face;
    Shift home;
    Key other;
  };
  [[nodiscard]] Side across(index_t face, const Key& from) const {
    const auto f = at(face);
    if (face < mesh_.n_interior_faces()) {
      const index_t o = mesh_.owner()[f];
      const index_t n = mesh_.neighbour()[f];
      return {face, from.shift, {o == from.entity ? n : o, from.shift}};
    }
    if (pair_[f] < 0) return {face, from.shift, {no_cell, from.shift}};
    const auto p = at(pair_[f]);
    const auto partner = at(partner_[f]);
    Shift k = from.shift;
    if (side_[f] > 0) {
      --k[p];
      return {face, from.shift, {mesh_.owner()[partner], k}};
    }
    ++k[p];
    return {static_cast<index_t>(partner), k, {mesh_.owner()[partner], k}};
  }

  // The cell copy on the far side of a representing face from its home.
  [[nodiscard]] Key far_side(index_t face, const Shift& home) const {
    const auto f = at(face);
    if (face < mesh_.n_interior_faces()) return {mesh_.neighbour()[f], home};
    if (pair_[f] < 0) return {no_cell, home};
    Shift k = home;
    --k[at(pair_[f])];
    return {mesh_.owner()[at(partner_[f])], k};
  }

  [[nodiscard]] bool periodic(index_t face) const { return pair_[at(face)] >= 0; }

 private:
  const Mesh& mesh_;
  std::vector<index_t> partner_;
  std::vector<std::int8_t> pair_;
  std::vector<std::int8_t> side_;  // +1 zone a, -1 zone b
  std::vector<std::vector<index_t>> node_image_;
};

// The ghost copies part `q` holds: cells of the covering within `layers`
// faces of its owned cells, ordered by owning part, then global cell, then
// shift. Ghosts in part q itself are periodic images of its own cells.
std::vector<Key> ghosts_of(const Mesh& global, const Covering& cover,
                           std::span<const std::int32_t> parts, std::int32_t q, int layers) {
  std::set<Key> seen;
  std::vector<Key> frontier;
  for (std::size_t c = 0; c < parts.size(); ++c) {
    if (parts[c] == q) {
      seen.insert({static_cast<index_t>(c), Shift{}});
      frontier.push_back({static_cast<index_t>(c), Shift{}});
    }
  }
  std::vector<Key> ghosts;
  for (int layer = 0; layer < layers; ++layer) {
    std::vector<Key> next;
    for (const Key& u : frontier) {
      for (const index_t f : global.cell_faces(u.entity)) {
        const Key v = cover.across(f, u).other;
        if (v.entity == no_cell || !seen.insert(v).second) continue;
        next.push_back(v);
        ghosts.push_back(v);
      }
    }
    frontier = std::move(next);
  }
  std::ranges::sort(ghosts, [&](const Key& a, const Key& b) {
    const auto pa = parts[at(a.entity)], pb = parts[at(b.entity)];
    return pa != pb ? pa < pb : a < b;
  });
  return ghosts;
}

}  // namespace

LocalPartition extract_partition(const Mesh& global, const Geometry& global_geometry,
                                 std::span<const std::int32_t> parts, std::int32_t n_parts,
                                 std::int32_t rank, int layers) {
  check_request(global, global_geometry, parts, n_parts, rank, layers);
  const Covering cover(global);

  // Owned cells keep their relative global order; ghosts follow, by owning
  // part, global order and shift, so each neighbour's ghosts are contiguous.
  std::map<Key, index_t> local;
  std::vector<index_t> owned;
  for (std::size_t c = 0; c < parts.size(); ++c) {
    if (parts[c] == rank) owned.push_back(static_cast<index_t>(c));
  }
  if (owned.empty()) {
    throw Error(ErrorCode::partition_invalid_request,
                "rank " + std::to_string(rank) + " owns no cell");
  }
  for (std::size_t i = 0; i < owned.size(); ++i)
    local[{owned[i], Shift{}}] = static_cast<index_t>(i);
  const std::vector<Key> ghosts = ghosts_of(global, cover, parts, rank, layers);
  const auto n_owned = static_cast<index_t>(owned.size());
  if (owned.size() + ghosts.size() > at(std::numeric_limits<index_t>::max())) {
    throw Error(ErrorCode::mesh_too_large, "the partition exceeds the index_t range");
  }
  for (std::size_t g = 0; g < ghosts.size(); ++g)
    local[ghosts[g]] = n_owned + static_cast<index_t>(g);

  // Faces: every face of the owned cells and, with two layers, of the first
  // ghost layer (whose neighbours are all local), each physical face once.
  std::set<Key> first_layer;
  if (layers == 2) {
    for (const index_t c : owned) {
      for (const index_t f : global.cell_faces(c)) {
        const Key v = cover.across(f, {c, Shift{}}).other;
        if (v.entity != no_cell && local.at(v) >= n_owned) first_layer.insert(v);
      }
    }
  }
  std::set<Key> faces;  // (representing global face, shift of its home)
  const auto add_faces = [&](const Key& u) {
    for (const index_t f : global.cell_faces(u.entity)) {
      const auto side = cover.across(f, u);
      faces.insert({side.face, side.home});
    }
  };
  for (const index_t c : owned) add_faces({c, Shift{}});
  for (const Key& g : first_layer) add_faces(g);

  MeshInput in;
  in.face_zones.assign(global.face_zones().begin(), global.face_zones().end());
  for (const auto& pair : global.periodic_pairs()) {
    in.face_zones[at(pair.zone_a)].boundary = false;
    in.face_zones[at(pair.zone_b)].boundary = false;
  }
  in.cell_zones.assign(global.cell_zones().begin(), global.cell_zones().end());
  in.source = global.source();
  in.n_ghost_cells = static_cast<index_t>(ghosts.size());
  const auto n_local_cells = owned.size() + ghosts.size();
  in.cell_zone.resize(n_local_cells);
  in.cell_global_id.resize(n_local_cells);
  in.ghost_translation.resize(ghosts.size());
  const auto gid = global.cell_global_id();
  for (std::size_t i = 0; i < owned.size(); ++i) {
    in.cell_zone[i] = global.cell_zone()[at(owned[i])];
    in.cell_global_id[i] = gid[at(owned[i])];
  }
  for (std::size_t g = 0; g < ghosts.size(); ++g) {
    in.cell_zone[owned.size() + g] = global.cell_zone()[at(ghosts[g].entity)];
    in.cell_global_id[owned.size() + g] = gid[at(ghosts[g].entity)];
    in.ghost_translation[g] = cover.translation(ghosts[g].shift);
  }

  // Node copies in ascending (global node, shift) order, so the same global
  // mesh yields the same local arrays whatever the other parts are.
  std::map<Key, index_t> nodes;
  for (const Key& face : faces) {
    for (const index_t p : global.face_nodes(face.entity))
      nodes.emplace(cover.canonical_node(p, face.shift), 0);
  }
  index_t n_points = 0;
  for (auto& [node, index] : nodes) {
    index = n_points++;
    const auto t = cover.translation(node.shift);
    in.x.push_back(global.x()[at(node.entity)] + t[0]);
    in.y.push_back(global.y()[at(node.entity)] + t[1]);
    in.z.push_back(global.z()[at(node.entity)] + t[2]);
  }

  std::vector<Key> face_list(faces.begin(), faces.end());
  in.face_node_offsets.reserve(face_list.size() + 1);
  in.face_node_offsets.push_back(0);
  for (const Key& face : face_list) {
    for (const index_t p : global.face_nodes(face.entity))
      in.face_nodes.push_back(nodes.at(cover.canonical_node(p, face.shift)));
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(local.at({global.owner()[at(face.entity)], face.shift}));
    const Key other = cover.far_side(face.entity, face.shift);
    in.face_cell1.push_back(other.entity == no_cell ? no_cell : local.at(other));
    in.face_zone.push_back(global.face_zone()[at(face.entity)]);
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
    const Key& face = face_list[at(map.face_source[f])];
    const auto source = at(face.entity);
    const double sign = map.face_reversed[f] != 0 ? -1.0 : 1.0;
    const auto t = cover.translation(face.shift);
    out.face_global_id[f] = static_cast<std::int64_t>(source);
    g.face_area_x[f] = sign * gg.face_area_x[source];
    g.face_area_y[f] = sign * gg.face_area_y[source];
    g.face_area_z[f] = sign * gg.face_area_z[source];
    g.face_area[f] = gg.face_area[source];
    g.face_centroid_x[f] = gg.face_centroid_x[source] + t[0];
    g.face_centroid_y[f] = gg.face_centroid_y[source] + t[1];
    g.face_centroid_z[f] = gg.face_centroid_z[source] + t[2];
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

  // Ghosts received from part q: our ghosts owned by q. Cells sent to q: q's
  // ghosts that we own, in q's order. The relation is symmetric (shift the
  // path), so both lists cover the same neighbours; part `rank` itself is
  // one when its cells have periodic images.
  HaloPattern& h = out.halo;
  h.rank = rank;
  h.n_ranks = n_parts;
  std::map<std::int32_t, std::vector<index_t>> sends;
  for (std::int32_t q = 0; q < n_parts; ++q) {
    for (const Key& k : ghosts_of(global, cover, parts, q, layers)) {
      if (parts[at(k.entity)] == rank) sends[q].push_back(local.at({k.entity, Shift{}}));
    }
  }
  for (std::size_t i = 0; i < ghosts.size(); ++i) {
    const std::int32_t q = parts[at(ghosts[i].entity)];
    if (h.neighbours.empty() || h.neighbours.back() != q) {
      if (!h.neighbours.empty()) h.recv_offsets.push_back(static_cast<index_t>(i));
      h.neighbours.push_back(q);
    }
  }
  if (!h.neighbours.empty()) h.recv_offsets.push_back(static_cast<index_t>(ghosts.size()));
  for (const std::int32_t q : h.neighbours) {
    const auto& cells = sends[q];
    h.send_cells.insert(h.send_cells.end(), cells.begin(), cells.end());
    h.send_offsets.push_back(static_cast<index_t>(h.send_cells.size()));
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
  for (index_t f = 0; f < mesh.n_owned_interior_faces(); ++f) {
    if (local_neighbour[at(f)] >= n_owned) h.interface_faces.push_back(f);
  }
  return out;
}

}  // namespace autan
