#include "cfd/core/mesh.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>

#include "cfd/core/error.hpp"

namespace cfd {
namespace {

std::size_t to_size(index_t value) { return static_cast<std::size_t>(value); }
std::size_t to_size(offset_t value) { return static_cast<std::size_t>(value); }

[[noreturn]] void fail(ErrorCode code, const std::string& message) { throw Error(code, message); }

void check_sizes(const MeshInput& in) {
  const std::size_t n_points = in.x.size();
  if (in.y.size() != n_points || in.z.size() != n_points) {
    fail(ErrorCode::mesh_inconsistent_sizes, "x, y and z have different sizes");
  }
  const std::size_t n_faces = in.face_cell0.size();
  if (in.face_cell1.size() != n_faces || in.face_zone.size() != n_faces ||
      in.face_node_offsets.size() != n_faces + 1) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "face_cell0, face_cell1, face_zone and face_node_offsets disagree on the face count");
  }
  if (in.cell_zone.empty() || n_faces == 0 || n_points == 0) {
    fail(ErrorCode::mesh_empty, "mesh has no cells, faces or points");
  }
  constexpr auto max_index = static_cast<std::size_t>(std::numeric_limits<index_t>::max());
  if (n_points > max_index || n_faces > max_index || in.cell_zone.size() > max_index ||
      in.face_zones.size() > max_index || in.cell_zones.size() > max_index) {
    fail(ErrorCode::mesh_too_large, "entity count exceeds the 32-bit index range");
  }
  if (in.face_node_offsets.front() != 0 ||
      to_size(in.face_node_offsets.back()) != in.face_nodes.size()) {
    fail(ErrorCode::mesh_inconsistent_sizes,
         "face_node_offsets must start at 0 and end at face_nodes.size()");
  }
  for (std::size_t f = 0; f < n_faces; ++f) {
    if (in.face_node_offsets[f + 1] < in.face_node_offsets[f]) {
      fail(ErrorCode::mesh_inconsistent_sizes,
           "face_node_offsets decreases at face " + std::to_string(f));
    }
  }
}

template <typename Zone>
void check_unique_names(const std::vector<Zone>& zones, const char* what) {
  std::unordered_set<std::string> seen;
  for (const Zone& zone : zones) {
    if (!seen.insert(zone.name).second) {
      fail(ErrorCode::mesh_duplicate_zone_name,
           std::string(what) + " name '" + zone.name + "' is used twice");
    }
  }
}

void check_entities(const MeshInput& in) {
  const auto n_points = static_cast<index_t>(in.x.size());
  const auto n_cells = static_cast<index_t>(in.cell_zone.size());
  const auto n_face_zones = static_cast<index_t>(in.face_zones.size());
  const auto n_cell_zones = static_cast<index_t>(in.cell_zones.size());
  const std::size_t n_faces = in.face_cell0.size();

  std::vector<index_t> sorted;
  for (std::size_t f = 0; f < n_faces; ++f) {
    const std::string where = " (face " + std::to_string(f) + ")";
    const auto begin = in.face_nodes.begin() + in.face_node_offsets[f];
    const auto end = in.face_nodes.begin() + in.face_node_offsets[f + 1];
    if (end - begin < 3) {
      fail(ErrorCode::mesh_face_too_few_nodes, "face has fewer than 3 nodes" + where);
    }
    sorted.assign(begin, end);
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= n_points) {
      fail(ErrorCode::mesh_index_out_of_range, "node index out of range" + where);
    }
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      fail(ErrorCode::mesh_face_repeated_node, "face lists a node twice" + where);
    }
    const index_t c0 = in.face_cell0[f];
    const index_t c1 = in.face_cell1[f];
    if (c0 < 0 || c0 >= n_cells || (c1 != no_cell && (c1 < 0 || c1 >= n_cells))) {
      fail(ErrorCode::mesh_index_out_of_range, "cell index out of range" + where);
    }
    if (c0 == c1) {
      fail(ErrorCode::mesh_face_same_cell_both_sides,
           "face has the same cell on both sides" + where);
    }
    if (in.face_zone[f] < 0 || in.face_zone[f] >= n_face_zones) {
      fail(ErrorCode::mesh_index_out_of_range, "face zone index out of range" + where);
    }
  }
  for (std::size_t c = 0; c < in.cell_zone.size(); ++c) {
    if (in.cell_zone[c] < 0 || in.cell_zone[c] >= n_cell_zones) {
      fail(ErrorCode::mesh_index_out_of_range,
           "cell zone index out of range (cell " + std::to_string(c) + ")");
    }
  }

  check_unique_names(in.face_zones, "face zone");
  check_unique_names(in.cell_zones, "cell zone");

  std::vector<unsigned char> zone_sides(in.face_zones.size(), 0);
  for (std::size_t f = 0; f < n_faces; ++f) {
    zone_sides[to_size(in.face_zone[f])] |=
        static_cast<unsigned char>(in.face_cell1[f] == no_cell ? 2 : 1);
  }
  for (std::size_t z = 0; z < zone_sides.size(); ++z) {
    if (zone_sides[z] == 3) {
      fail(ErrorCode::mesh_zone_mixes_interior_and_boundary,
           "face zone '" + in.face_zones[z].name + "' holds both interior and boundary faces");
    }
  }
}

// Every edge of a closed, consistently oriented cell surface is traversed
// exactly twice, once in each direction, when each face is walked with its
// normal pointing out of the cell.
void check_cells_closed(const Mesh& mesh) {
  struct Edge {
    std::uint64_t key;
    bool forward;
  };
  std::vector<Edge> edges;
  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    edges.clear();
    for (const index_t f : mesh.cell_faces(c)) {
      const auto nodes = mesh.face_nodes(f);
      const bool outward = mesh.owner()[to_size(f)] == c;
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        index_t a = nodes[i];
        index_t b = nodes[(i + 1) % nodes.size()];
        if (!outward) std::swap(a, b);
        const auto lo = static_cast<std::uint64_t>(std::min(a, b));
        const auto hi = static_cast<std::uint64_t>(std::max(a, b));
        edges.push_back({(lo << 32) | hi, a < b});
      }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& l, const Edge& r) {
      return l.key != r.key ? l.key < r.key : l.forward < r.forward;
    });
    bool closed = edges.size() % 2 == 0;
    for (std::size_t i = 0; closed && i < edges.size(); i += 2) {
      closed = edges[i].key == edges[i + 1].key && !edges[i].forward && edges[i + 1].forward &&
               (i + 2 == edges.size() || edges[i + 2].key != edges[i].key);
    }
    if (!closed) {
      fail(ErrorCode::mesh_cell_not_closed,
           "cell " + std::to_string(c) +
               " is not a closed, consistently oriented surface: an edge is not shared by exactly "
               "two of its faces with opposite directions");
    }
  }
}

}  // namespace

std::span<const index_t> Mesh::face_nodes(index_t face) const noexcept {
  const auto f = to_size(face);
  return std::span<const index_t>(face_nodes_)
      .subspan(to_size(face_node_offsets_[f]),
               to_size(face_node_offsets_[f + 1] - face_node_offsets_[f]));
}

std::span<const index_t> Mesh::cell_faces(index_t cell) const noexcept {
  const auto c = to_size(cell);
  return std::span<const index_t>(cell_faces_)
      .subspan(to_size(cell_face_offsets_[c]),
               to_size(cell_face_offsets_[c + 1] - cell_face_offsets_[c]));
}

MeshView Mesh::view() const noexcept {
  return MeshView{
      .n_points = n_points(),
      .n_cells = n_cells(),
      .n_faces = n_faces(),
      .n_interior_faces = n_interior_faces(),
      .n_face_zones = static_cast<index_t>(face_zones_.size()),
      .n_cell_zones = static_cast<index_t>(cell_zones_.size()),
      .x = x_.data(),
      .y = y_.data(),
      .z = z_.data(),
      .face_node_offsets = face_node_offsets_.data(),
      .face_nodes = face_nodes_.data(),
      .owner = owner_.data(),
      .neighbour = neighbour_.data(),
      .face_zone = face_zone_.data(),
      .cell_zone = cell_zone_.data(),
      .boundary_zone_offsets = boundary_zone_offsets_.data(),
      .cell_face_offsets = cell_face_offsets_.data(),
      .cell_faces = cell_faces_.data(),
  };
}

Mesh assemble(MeshInput in) {
  check_sizes(in);
  check_entities(in);

  const std::size_t n_faces = in.face_cell0.size();
  const std::size_t n_cells = in.cell_zone.size();

  // Orient interior faces from the lower to the higher cell index.
  for (std::size_t f = 0; f < n_faces; ++f) {
    if (in.face_cell1[f] != no_cell && in.face_cell1[f] < in.face_cell0[f]) {
      std::swap(in.face_cell0[f], in.face_cell1[f]);
      std::reverse(in.face_nodes.begin() + in.face_node_offsets[f] + 1,
                   in.face_nodes.begin() + in.face_node_offsets[f + 1]);
    }
  }

  std::vector<index_t> order(n_faces);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&in](index_t a, index_t b) {
    const auto ia = to_size(a);
    const auto ib = to_size(b);
    const bool boundary_a = in.face_cell1[ia] == no_cell;
    const bool boundary_b = in.face_cell1[ib] == no_cell;
    if (boundary_a != boundary_b) return boundary_b;
    if (boundary_a) {
      if (in.face_zone[ia] != in.face_zone[ib]) return in.face_zone[ia] < in.face_zone[ib];
      return in.face_cell0[ia] < in.face_cell0[ib];
    }
    if (in.face_cell0[ia] != in.face_cell0[ib]) return in.face_cell0[ia] < in.face_cell0[ib];
    return in.face_cell1[ia] < in.face_cell1[ib];
  });

  std::vector<index_t> point_map(in.x.size(), -1);
  for (const index_t node : in.face_nodes) point_map[to_size(node)] = 0;
  Mesh mesh;
  index_t n_used = 0;
  for (std::size_t p = 0; p < point_map.size(); ++p) {
    if (point_map[p] == 0) {
      point_map[p] = n_used++;
      mesh.x_.push_back(in.x[p]);
      mesh.y_.push_back(in.y[p]);
      mesh.z_.push_back(in.z[p]);
    }
  }

  mesh.face_node_offsets_.reserve(n_faces + 1);
  mesh.face_node_offsets_.push_back(0);
  mesh.face_nodes_.reserve(in.face_nodes.size());
  mesh.owner_.reserve(n_faces);
  mesh.face_zone_.reserve(n_faces);
  for (const index_t face : order) {
    const auto f = to_size(face);
    for (offset_t k = in.face_node_offsets[f]; k < in.face_node_offsets[f + 1]; ++k) {
      mesh.face_nodes_.push_back(point_map[to_size(in.face_nodes[to_size(k)])]);
    }
    mesh.face_node_offsets_.push_back(static_cast<offset_t>(mesh.face_nodes_.size()));
    mesh.owner_.push_back(in.face_cell0[f]);
    mesh.face_zone_.push_back(in.face_zone[f]);
    if (in.face_cell1[f] != no_cell) mesh.neighbour_.push_back(in.face_cell1[f]);
  }

  const std::size_t n_zones = in.face_zones.size();
  mesh.boundary_zone_offsets_.assign(n_zones + 1, 0);
  for (std::size_t f = mesh.neighbour_.size(); f < n_faces; ++f) {
    ++mesh.boundary_zone_offsets_[to_size(mesh.face_zone_[f]) + 1];
  }
  mesh.boundary_zone_offsets_[0] = static_cast<index_t>(mesh.neighbour_.size());
  std::partial_sum(mesh.boundary_zone_offsets_.begin(), mesh.boundary_zone_offsets_.end(),
                   mesh.boundary_zone_offsets_.begin());

  mesh.cell_face_offsets_.assign(n_cells + 1, 0);
  for (std::size_t f = 0; f < n_faces; ++f) {
    ++mesh.cell_face_offsets_[to_size(mesh.owner_[f]) + 1];
    if (f < mesh.neighbour_.size()) ++mesh.cell_face_offsets_[to_size(mesh.neighbour_[f]) + 1];
  }
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (mesh.cell_face_offsets_[c + 1] < 4) {
      fail(ErrorCode::mesh_cell_too_few_faces,
           "cell " + std::to_string(c) + " has fewer than 4 faces");
    }
  }
  std::partial_sum(mesh.cell_face_offsets_.begin(), mesh.cell_face_offsets_.end(),
                   mesh.cell_face_offsets_.begin());
  mesh.cell_faces_.resize(to_size(mesh.cell_face_offsets_.back()));
  std::vector<offset_t> fill(mesh.cell_face_offsets_.begin(), mesh.cell_face_offsets_.end() - 1);
  for (std::size_t f = 0; f < n_faces; ++f) {
    const auto face = static_cast<index_t>(f);
    mesh.cell_faces_[to_size(fill[to_size(mesh.owner_[f])]++)] = face;
    if (f < mesh.neighbour_.size()) {
      mesh.cell_faces_[to_size(fill[to_size(mesh.neighbour_[f])]++)] = face;
    }
  }

  mesh.cell_zone_ = std::move(in.cell_zone);
  mesh.face_zones_ = std::move(in.face_zones);
  mesh.cell_zones_ = std::move(in.cell_zones);
  mesh.source_ = std::move(in.source);

  check_cells_closed(mesh);
  return mesh;
}

}  // namespace cfd
