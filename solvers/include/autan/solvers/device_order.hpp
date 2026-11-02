#pragma once

#include <span>
#include <vector>

#include "autan/core/mesh.hpp"
#include "autan/solvers/agglomeration.hpp"

namespace autan::solvers::gpu {

/// Device numbering of a hierarchy (docs/design/multigrid-gpu.md#transfers): per level, the
/// host index of each device node. Level 0 keeps the mesh's numbering; a coarse node takes the
/// rank of its least child in the finer level's device numbering. Deterministic.
std::vector<std::vector<index_t>> device_orders(std::span<const Level> levels);

}  // namespace autan::solvers::gpu
