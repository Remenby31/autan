#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// Element counts of a Topology, the multipliers of the generated <loop>_bytes functions
/// (autan_codegen.traffic): each is the number of elements a loop's range reaches.
struct Counts {
  double cells = 0;                   ///< owned and halo cells (cells across faces and entries)
  double owned_cells = 0;             ///< cell loops and reductions
  double owned_interior_faces = 0;    ///< face loops
  double boundary_faces = 0;          ///< boundary loops writing face data
  double owned_boundary_faces = 0;    ///< boundary loops incrementing cells
  double boundary_cells = 0;          ///< distinct owners of the owned boundary faces
  double faces = 0;                   ///< distinct faces of the owned cells (incidence reductions)
  double incidences = 0;              ///< incidences of the owned cells, padding excluded
  double stencil = 0;                 ///< stencil entries of the owned cells, padding excluded
  double stencil_boundary_faces = 0;  ///< distinct boundary faces among those entries
  bool stencil_sliced = false;        ///< Topology::stencil_stride > 1
  bool incidence_sliced = false;      ///< Topology::incidence_stride > 1
  bool incidence_cell = false;        ///< Topology::incidence_cell is set
};

/// Bytes of one element of a Data<P> member (a pointer to the stored type).
template <class Ptr>
constexpr double element_size() {
  return static_cast<double>(sizeof(std::remove_cvref_t<std::remove_pointer_t<Ptr>>));
}

/// The counts of a host Topology (its arrays in host memory).
inline Counts counts(const Topology& t) {
  Counts n;
  const auto owned = static_cast<std::size_t>(t.n_owned_cells);
  const auto ni = static_cast<std::size_t>(t.n_interior_faces);
  n.cells = static_cast<double>(t.n_cells);
  n.owned_cells = static_cast<double>(owned);
  n.owned_interior_faces = static_cast<double>(t.n_owned_interior_faces);
  n.boundary_faces = static_cast<double>(t.n_faces - t.n_interior_faces);
  n.owned_boundary_faces = static_cast<double>(t.n_owned_faces - t.n_interior_faces);
  std::vector<char> seen(static_cast<std::size_t>(t.n_cells), 0);
  for (auto f = ni; f < static_cast<std::size_t>(t.n_owned_faces); ++f) {
    seen[static_cast<std::size_t>(t.owner[f])] = 1;
  }
  n.boundary_cells = static_cast<double>(std::count(seen.begin(), seen.end(), 1));
  n.incidence_sliced = t.incidence_stride > 1;
  n.incidence_cell = t.incidence_cell != nullptr;
  std::vector<char> face(static_cast<std::size_t>(t.n_faces), 0);
  for (std::size_t c = 0; c < owned; ++c) {
    const offset_t first = t.cell_face_offsets[c];
    const offset_t end = t.incidence_stride == 1
                             ? t.cell_face_offsets[c + 1]
                             : first + offset_t(t.incidence_length[c]) * t.incidence_stride;
    for (offset_t k = first; k < end; k += t.incidence_stride) {
      face[static_cast<std::size_t>(t.cell_faces[k])] = 1;
      n.incidences += 1;
    }
  }
  n.faces = static_cast<double>(std::count(face.begin(), face.end(), 1));
  if (t.stencil_offsets != nullptr) {
    n.stencil_sliced = t.stencil_stride > 1;
    std::vector<char> bnd(static_cast<std::size_t>(t.n_faces), 0);
    for (std::size_t c = 0; c < owned; ++c) {
      const offset_t first = t.stencil_offsets[c];
      const offset_t end = t.stencil_stride == 1
                               ? t.stencil_offsets[c + 1]
                               : first + offset_t(t.stencil_length[c]) * t.stencil_stride;
      for (offset_t k = first; k < end; k += t.stencil_stride) {
        if (t.stencil[k] < 0) bnd[static_cast<std::size_t>(-t.stencil[k] - 1)] = 1;
        n.stencil += 1;
      }
    }
    n.stencil_boundary_faces = static_cast<double>(std::count(bnd.begin(), bnd.end(), 1));
  }
  return n;
}

}  // namespace autan::kernels
