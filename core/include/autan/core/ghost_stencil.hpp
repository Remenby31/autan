#pragma once

#include <cstdint>

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
enum class GhostStencil : std::uint8_t { face, face2, vertex };

}  // namespace autan
