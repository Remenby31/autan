#pragma once

#include <vector>

#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// Faces [first, last) grouped in colours so that no two faces of a colour touch the same cell.
/// Greedy in face order (first colour not used by a face of either cell), so the result is a
/// function of the mesh numbering only. `faces` lists the faces colour by colour, each colour in
/// increasing face order; colour k is faces[offsets[k], offsets[k + 1]).
struct HostColouring {
  std::vector<index_t> faces;
  std::vector<index_t> offsets;
};

HostColouring colour_faces(const Topology& topo, index_t first, index_t last);

}  // namespace autan::kernels
