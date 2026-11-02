#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/mesh.hpp"
#include "autan/solvers/agglomeration.hpp"

namespace autan::solvers::gpu {

/// Where a level's off-diagonal blocks live in its device operator: for each edge e of the
/// level (host numbering), the position of block A(owner, neighbour), held by the owner's
/// incidence of e, and of block A(neighbour, owner), held by the neighbour's.
struct EdgeSlots {
  std::vector<offset_t> owner, neighbour;
};

/// Gather lists of the Galerkin coarse operator A_c = Pᵀ A P from one level to the next
/// (docs/design/multigrid-gpu.md#galerkin-coarse-operator), in the host multigrid's summation
/// order (`Multigrid::set_operator`). Each coarse block is the sum, from +0.0, of its list's
/// blocks in list order:
/// - diagonal of coarse device node J, entries `diag[diag_offsets[J] .. diag_offsets[J + 1])`:
///   the diagonal blocks of its children in increasing host fine index (an entry d >= 0 is the
///   fine device node d), then, for each fine edge inside the agglomerate in increasing edge
///   index, its owner's then its neighbour's off-diagonal block (an entry d < 0 is the fine
///   slot -(d + 1));
/// - off-diagonal block at coarse slot s, entries `off[off_offsets[s] .. off_offsets[s + 1])`:
///   for each fine edge e mapped to the coarse edge, in increasing e, the fine slot of the
///   block coupling the same pair of coarse nodes in the same direction. Slots no coarse edge
///   uses have empty lists.
struct GalerkinLists {
  std::vector<offset_t> diag_offsets;
  std::vector<std::int64_t> diag;
  std::vector<offset_t> off_offsets;
  std::vector<offset_t> off;
};

/// Builds the lists from `fine` to `coarse` (consecutive levels of `agglomerate`), with each
/// level's device numbering (`Transfers::order`: host index of each device node) and slots;
/// `n_coarse_slots` is the size of the coarse level's off-diagonal array, padding included.
/// Throws std::invalid_argument when the inputs do not describe consecutive levels.
GalerkinLists galerkin_lists(const Level& fine, const Level& coarse,
                             std::span<const index_t> fine_order,
                             std::span<const index_t> coarse_order, const EdgeSlots& fine_slots,
                             const EdgeSlots& coarse_slots, offset_t n_coarse_slots);

}  // namespace autan::solvers::gpu
