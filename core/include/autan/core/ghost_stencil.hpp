#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace autan {

/// The ghost cells and faces a partition holds around its owned cells (extract_partition), so
/// that an owned cell's stencil of that kind is the same on the part as on the whole mesh. Each
/// kind holds the ones before it.
/// - `face`: the face neighbours.
/// - `face2`: the cells within two face steps, with every face of the first layer
///   (kernels::StencilKind::face2).
/// - `vertex`: face2's, and every cell and face around every node of an owned cell, with the
///   boundary faces of those cells (kernels::StencilKind::vertex, and any per-cell union of the
///   two). Nodes are those of the periodic covering: a ring runs across periodic faces.
/// - `vertex2`, `vertex3`, `vertex4`: face2's, and the cells within 2, 3 or 4 vertex rings of an
///   owned cell, with every face around the nodes of the cells within one ring fewer and the
///   boundary faces of the outermost ring's cells: the vertex stencil of every cell within one
///   ring fewer is whole (the quadratic gradient's two rings, the viscous fits' rings).
enum class GhostStencil : std::uint8_t { face, face2, vertex, vertex2, vertex3, vertex4 };

/// The vertex rings of `ghosts`: 0 for face and face2, 1 for vertex, 2 to 4 for vertex2 to
/// vertex4.
constexpr int vertex_rings(GhostStencil ghosts) {
  return ghosts < GhostStencil::vertex
             ? 0
             : static_cast<int>(ghosts) - static_cast<int>(GhostStencil::vertex) + 1;
}

/// The name of each kind in MeshInfo and on the command line, in enum order.
inline constexpr std::array<std::string_view, 6> ghost_stencil_names{
    "face", "face2", "vertex", "vertex2", "vertex3", "vertex4"};

constexpr std::string_view name(GhostStencil ghosts) {
  return ghost_stencil_names[static_cast<std::size_t>(ghosts)];
}

/// The kind named `text`, or none.
constexpr std::optional<GhostStencil> ghost_stencil_named(std::string_view text) {
  for (std::size_t k = 0; k < ghost_stencil_names.size(); ++k) {
    if (ghost_stencil_names[k] == text) return static_cast<GhostStencil>(k);
  }
  return std::nullopt;
}

}  // namespace autan
