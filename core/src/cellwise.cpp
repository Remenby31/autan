#include "cfd/core/cellwise.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>

#include "cfd/core/error.hpp"

namespace cfd {
namespace {

std::size_t to_size(std::int64_t value) { return static_cast<std::size_t>(value); }

[[noreturn]] void fail(ErrorCode code, const std::string& message) { throw Error(code, message); }

struct ShapeFaces {
  std::size_t n_nodes;
  std::size_t n_faces;
  std::array<std::array<int, 5>, 6> faces;  // node count, then local nodes
};

// Outward faces of the Gmsh reference elements: the right-hand-rule normal of
// each node cycle points out of the element.
constexpr ShapeFaces shape_faces(CellShape shape) {
  switch (shape) {
    case CellShape::tetra:
      return {4, 4, {{{3, 0, 2, 1}, {3, 0, 1, 3}, {3, 0, 3, 2}, {3, 1, 2, 3}}}};
    case CellShape::pyramid:
      return {5, 5, {{{4, 0, 3, 2, 1}, {3, 0, 1, 4}, {3, 1, 2, 4}, {3, 2, 3, 4}, {3, 3, 0, 4}}}};
    case CellShape::prism:
      return {
          6, 5, {{{3, 0, 2, 1}, {3, 3, 4, 5}, {4, 0, 1, 4, 3}, {4, 1, 2, 5, 4}, {4, 0, 3, 5, 2}}}};
    case CellShape::hexa:
      return {8,
              6,
              {{{4, 0, 3, 2, 1},
                {4, 4, 5, 6, 7},
                {4, 0, 1, 5, 4},
                {4, 1, 2, 6, 5},
                {4, 2, 3, 7, 6},
                {4, 3, 0, 4, 7}}}};
  }
  return {};
}

// Faces stored as a CSR plus a sorted copy of each node list, which is the
// identity of the face independent of its starting node and orientation.
struct FaceKeys {
  std::span<const offset_t> offsets;
  std::vector<index_t> sorted;

  FaceKeys(std::span<const offset_t> face_offsets, std::span<const index_t> nodes)
      : offsets(face_offsets), sorted(nodes.begin(), nodes.end()) {
    for (std::size_t f = 0; f + 1 < offsets.size(); ++f) {
      std::sort(sorted.begin() + offsets[f], sorted.begin() + offsets[f + 1]);
    }
  }
  [[nodiscard]] std::size_t size() const { return offsets.size() - 1; }
  [[nodiscard]] std::span<const index_t> key(std::size_t f) const {
    return std::span<const index_t>(sorted).subspan(to_size(offsets[f]),
                                                    to_size(offsets[f + 1] - offsets[f]));
  }
};

int compare_keys(std::span<const index_t> a, std::span<const index_t> b) {
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

// True when `b` is the cycle `a` walked in the opposite direction.
bool is_reversed_cycle(std::span<const index_t> a, std::span<const index_t> b) {
  const std::size_t n = a.size();
  const auto start = std::find(b.begin(), b.end(), a[0]);
  if (n != b.size() || start == b.end()) return false;
  const auto j = static_cast<std::size_t>(start - b.begin());
  for (std::size_t k = 0; k < n; ++k) {
    if (b[(j + n - k) % n] != a[k]) return false;
  }
  return true;
}

index_t append_zone(std::vector<FaceZone>& zones, const char* name, const char* detail) {
  zones.push_back(FaceZone{.name = name,
                           .bc_type = "",
                           .kind = BcKind::unspecified,
                           .provenance = Provenance{.source_id = -1, .detail = detail}});
  return static_cast<index_t>(zones.size() - 1);
}

}  // namespace

void CellwiseMesh::add_cell(CellShape shape, std::span<const index_t> nodes, index_t zone) {
  const ShapeFaces sf = shape_faces(shape);
  if (nodes.size() != sf.n_nodes) {
    fail(ErrorCode::mesh_inconsistent_sizes, "cell node count does not match its shape");
  }
  for (std::size_t f = 0; f < sf.n_faces; ++f) {
    const auto& local = sf.faces[f];
    for (int k = 1; k <= local[0]; ++k)
      face_nodes.push_back(nodes[static_cast<std::size_t>(local[static_cast<std::size_t>(k)])]);
    face_node_offsets.push_back(static_cast<offset_t>(face_nodes.size()));
  }
  cell_face_offsets.push_back(static_cast<offset_t>(face_node_offsets.size() - 1));
  cell_zone.push_back(zone);
}

void CellwiseMesh::add_polyhedron(std::span<const offset_t> local_face_offsets,
                                  std::span<const index_t> local_face_nodes, index_t zone) {
  if (local_face_offsets.empty() || local_face_offsets.front() != 0 ||
      to_size(local_face_offsets.back()) != local_face_nodes.size()) {
    fail(ErrorCode::mesh_inconsistent_sizes, "polyhedron face offsets do not match its nodes");
  }
  const auto base = static_cast<offset_t>(face_nodes.size());
  face_nodes.insert(face_nodes.end(), local_face_nodes.begin(), local_face_nodes.end());
  for (std::size_t f = 1; f < local_face_offsets.size(); ++f) {
    face_node_offsets.push_back(base + local_face_offsets[f]);
  }
  cell_face_offsets.push_back(static_cast<offset_t>(face_node_offsets.size() - 1));
  cell_zone.push_back(zone);
}

void CellwiseMesh::add_labelled_face(std::span<const index_t> nodes, index_t zone) {
  labelled_face_nodes.insert(labelled_face_nodes.end(), nodes.begin(), nodes.end());
  labelled_face_node_offsets.push_back(static_cast<offset_t>(labelled_face_nodes.size()));
  labelled_face_zone.push_back(zone);
}

MeshInput to_face_based(CellwiseMesh mesh) {
  if (mesh.cell_face_offsets.size() != mesh.cell_zone.size() + 1 ||
      to_size(mesh.cell_face_offsets.back()) + 1 != mesh.face_node_offsets.size() ||
      to_size(mesh.face_node_offsets.back()) != mesh.face_nodes.size() ||
      mesh.labelled_face_node_offsets.size() != mesh.labelled_face_zone.size() + 1) {
    fail(ErrorCode::mesh_inconsistent_sizes, "cellwise mesh arrays disagree on their sizes");
  }

  const FaceKeys cell_faces(mesh.face_node_offsets, mesh.face_nodes);
  const std::size_t n_cell_faces = cell_faces.size();
  std::vector<index_t> face_cell(n_cell_faces);
  for (std::size_t c = 0; c + 1 < mesh.cell_face_offsets.size(); ++c) {
    for (offset_t f = mesh.cell_face_offsets[c]; f < mesh.cell_face_offsets[c + 1]; ++f) {
      face_cell[to_size(f)] = static_cast<index_t>(c);
    }
  }

  std::vector<std::size_t> order(n_cell_faces);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return compare_keys(cell_faces.key(a), cell_faces.key(b)) < 0;
  });

  // One entry per distinct face: the first cell face of the group (the
  // orientation kept) and the second one if the face is shared.
  struct Group {
    std::size_t first;
    std::size_t second;
  };
  constexpr std::size_t none = static_cast<std::size_t>(-1);
  std::vector<Group> groups;
  for (std::size_t i = 0; i < n_cell_faces;) {
    std::size_t j = i + 1;
    while (j < n_cell_faces &&
           compare_keys(cell_faces.key(order[i]), cell_faces.key(order[j])) == 0) {
      ++j;
    }
    const std::size_t a = std::min(order[i], j - i > 1 ? order[i + 1] : order[i]);
    const std::size_t b = std::max(order[i], j - i > 1 ? order[i + 1] : order[i]);
    if (j - i > 2) {
      fail(ErrorCode::mesh_face_shared_by_more_than_two_cells,
           "a face is shared by " + std::to_string(j - i) + " cell faces (cell " +
               std::to_string(face_cell[a]) + ")");
    }
    if (j - i == 2) {
      if (face_cell[a] == face_cell[b]) {
        fail(ErrorCode::mesh_face_same_cell_both_sides,
             "cell " + std::to_string(face_cell[a]) + " lists the same face twice");
      }
      const auto nodes_a =
          std::span<const index_t>(mesh.face_nodes)
              .subspan(to_size(mesh.face_node_offsets[a]),
                       to_size(mesh.face_node_offsets[a + 1] - mesh.face_node_offsets[a]));
      const auto nodes_b =
          std::span<const index_t>(mesh.face_nodes)
              .subspan(to_size(mesh.face_node_offsets[b]),
                       to_size(mesh.face_node_offsets[b + 1] - mesh.face_node_offsets[b]));
      if (!is_reversed_cycle(nodes_a, nodes_b)) {
        fail(ErrorCode::mesh_shared_face_orientation,
             "cells " + std::to_string(face_cell[a]) + " and " + std::to_string(face_cell[b]) +
                 " list their common face with the same orientation (inverted cell?)");
      }
      groups.push_back({a, b});
    } else {
      groups.push_back({a, none});
    }
    i = j;
  }
  std::sort(groups.begin(), groups.end(),
            [](const Group& l, const Group& r) { return l.first < r.first; });

  std::vector<index_t> group_zone(groups.size(), -1);
  if (!mesh.labelled_face_zone.empty()) {
    std::vector<std::size_t> by_key(groups.size());
    std::iota(by_key.begin(), by_key.end(), std::size_t{0});
    std::sort(by_key.begin(), by_key.end(), [&](std::size_t a, std::size_t b) {
      return compare_keys(cell_faces.key(groups[a].first), cell_faces.key(groups[b].first)) < 0;
    });
    const FaceKeys labelled(mesh.labelled_face_node_offsets, mesh.labelled_face_nodes);
    for (std::size_t l = 0; l < labelled.size(); ++l) {
      const auto key = labelled.key(l);
      const auto it = std::lower_bound(
          by_key.begin(), by_key.end(), key, [&](std::size_t g, std::span<const index_t> k) {
            return compare_keys(cell_faces.key(groups[g].first), k) < 0;
          });
      if (it == by_key.end() || compare_keys(cell_faces.key(groups[*it].first), key) != 0) {
        fail(ErrorCode::mesh_labelled_face_not_found,
             "labelled face " + std::to_string(l) + " matches no cell face");
      }
      if (group_zone[*it] != -1) {
        fail(ErrorCode::mesh_face_labelled_twice,
             "labelled face " + std::to_string(l) + " labels a face that is already labelled");
      }
      group_zone[*it] = mesh.labelled_face_zone[l];
    }
  }

  MeshInput out;
  out.face_node_offsets.reserve(groups.size() + 1);
  out.face_node_offsets.push_back(0);
  index_t interior_zone = -1;
  index_t unassigned_zone = -1;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    const std::size_t f = groups[g].first;
    const bool interior = groups[g].second != none;
    index_t zone = group_zone[g];
    if (zone == -1 && interior) {
      if (interior_zone == -1) {
        interior_zone =
            append_zone(mesh.face_zones, "interior", "synthesised: unlabelled interior faces");
      }
      zone = interior_zone;
    } else if (zone == -1) {
      if (unassigned_zone == -1) {
        unassigned_zone = append_zone(mesh.face_zones, "unassigned",
                                      "synthesised: boundary faces without a label in the source");
      }
      zone = unassigned_zone;
    }
    out.face_nodes.insert(out.face_nodes.end(), mesh.face_nodes.begin() + mesh.face_node_offsets[f],
                          mesh.face_nodes.begin() + mesh.face_node_offsets[f + 1]);
    out.face_node_offsets.push_back(static_cast<offset_t>(out.face_nodes.size()));
    out.face_cell0.push_back(face_cell[f]);
    out.face_cell1.push_back(interior ? face_cell[groups[g].second] : no_cell);
    out.face_zone.push_back(zone);
  }

  out.x = std::move(mesh.x);
  out.y = std::move(mesh.y);
  out.z = std::move(mesh.z);
  out.cell_zone = std::move(mesh.cell_zone);
  out.face_zones = std::move(mesh.face_zones);
  out.cell_zones = std::move(mesh.cell_zones);
  out.source = std::move(mesh.source);
  return out;
}

}  // namespace cfd
