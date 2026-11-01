#pragma once

// Face loop race strategies and colourings, shared by the GPU backends (CUDA/HIP and Metal).

#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// Face loop race strategies (docs/research/kernel-generator.md section 6).
enum class Strategy : std::uint8_t {
  atomic,            ///< one thread per face, atomic increments
  warp_aggregated,   ///< owner-sorted faces: one atomic per owner segment of a warp
  coloured,          ///< one launch per face colour, plain increments
  store_gather,      ///< face fluxes stored, then summed per cell in incidence order
  gather_recompute,  ///< each cell evaluates the fluxes of its own faces
};

inline const char* name(Strategy s) {
  switch (s) {
    case Strategy::atomic:
      return "atomic";
    case Strategy::warp_aggregated:
      return "warp-aggregated";
    case Strategy::coloured:
      return "coloured";
    case Strategy::store_gather:
      return "store-gather";
    case Strategy::gather_recompute:
      return "gather-recompute";
  }
  return "?";
}

/// The strategy named `s` (as `name` spells it); throws E1103 for an unknown name.
inline Strategy strategy_of(std::string_view s) {
  for (auto k : {Strategy::atomic, Strategy::warp_aggregated, Strategy::coloured,
                 Strategy::store_gather, Strategy::gather_recompute}) {
    if (s == name(k)) {
      return k;
    }
  }
  throw std::invalid_argument("E1103: unknown race strategy");
}

/// True for strategies whose sums run in an order fixed by the mesh numbering.
inline bool deterministic(Strategy s) {
  return s == Strategy::coloured || s == Strategy::store_gather || s == Strategy::gather_recompute;
}

/// Faces of one range grouped in colours: no two faces of a colour touch the same cell.
/// `faces` is a device array; `offsets` (host) delimits each colour in it.
struct Colouring {
  const index_t* faces = nullptr;
  std::vector<index_t> offsets;
};

}  // namespace autan::kernels
