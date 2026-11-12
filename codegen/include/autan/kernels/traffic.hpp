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
  double owned_cells = 0;              ///< cell loops, reductions, their outputs
  double face_owner_cells = 0;         ///< distinct owners of the owned interior faces
  double face_neighbour_cells = 0;     ///< distinct neighbours of the owned interior faces
  double face_cells = 0;               ///< distinct cells of the owned interior faces
  double owned_interior_faces = 0;     ///< face loops
  double boundary_faces = 0;           ///< boundary loops writing face data
  double owned_boundary_faces = 0;     ///< boundary loops incrementing cells
  double boundary_cells = 0;           ///< distinct owners of the owned boundary faces
  double all_boundary_cells = 0;       ///< distinct owners of every boundary face
  double faces = 0;                    ///< distinct faces of the owned cells
  double incidences = 0;               ///< incidences of the owned cells, padding excluded
  double incidence_cells = 0;          ///< distinct cells across those incidences
  double incidence_touched_cells = 0;  ///< the owned cells and those across
  double stencil = 0;                  ///< stencil entries of the owned cells, padding excluded
  double stencil_rows = 0;             ///< owned cells with at least one entry
  double stencil_cells = 0;            ///< distinct cells among the entries
  double stencil_touched_cells = 0;    ///< the rows' cells and the entries' cells
  double stencil_owned_cells = 0;      ///< the owned cells and the entries' cells
  double stencil_boundary_faces = 0;   ///< distinct boundary faces among the entries
  bool stencil_sliced = false;         ///< Topology::stencil_stride > 1
  bool incidence_sliced = false;       ///< Topology::incidence_stride > 1
  bool incidence_cell = false;         ///< Topology::incidence_cell is set
};

/// Bytes of one element of a Data<P> member (a pointer to the stored type).
template <class Ptr>
constexpr double element_size() {
  return static_cast<double>(sizeof(std::remove_cvref_t<std::remove_pointer_t<Ptr>>));
}

/// The counts of a host Topology (its arrays in host memory).
inline Counts counts(const Topology& t) {
  Counts n;
  const auto nc = static_cast<std::size_t>(t.n_cells);
  const auto owned = static_cast<std::size_t>(t.n_owned_cells);
  const auto ni = static_cast<std::size_t>(t.n_interior_faces);
  const auto nf = static_cast<std::size_t>(t.n_faces);
  const auto count = [](const std::vector<char>& v) {
    return static_cast<double>(std::count(v.begin(), v.end(), char{1}));
  };
  const auto any = [](const std::vector<char>& a, const std::vector<char>& b) {
    std::size_t m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m += (a[i] != 0 || b[i] != 0) ? 1 : 0;
    return static_cast<double>(m);
  };
  n.owned_cells = static_cast<double>(owned);
  n.owned_interior_faces = static_cast<double>(t.n_owned_interior_faces);
  n.boundary_faces = static_cast<double>(t.n_faces - t.n_interior_faces);
  n.owned_boundary_faces = static_cast<double>(t.n_owned_faces - t.n_interior_faces);
  std::vector<char> left(nc, 0), right(nc, 0), bnd_owned(nc, 0), bnd_all(nc, 0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(t.n_owned_interior_faces); ++f) {
    left[static_cast<std::size_t>(t.owner[f])] = 1;
    right[static_cast<std::size_t>(t.neighbour[f])] = 1;
  }
  for (auto f = ni; f < nf; ++f) {
    bnd_all[static_cast<std::size_t>(t.owner[f])] = 1;
    if (f < static_cast<std::size_t>(t.n_owned_faces))
      bnd_owned[static_cast<std::size_t>(t.owner[f])] = 1;
  }
  n.face_owner_cells = count(left);
  n.face_neighbour_cells = count(right);
  n.face_cells = any(left, right);
  n.boundary_cells = count(bnd_owned);
  n.all_boundary_cells = count(bnd_all);
  n.incidence_sliced = t.incidence_stride > 1;
  n.incidence_cell = t.incidence_cell != nullptr;
  std::vector<char> face(nf, 0), across(nc, 0), self(nc, 0);
  for (std::size_t c = 0; c < owned; ++c) {
    self[c] = 1;
    const offset_t first = t.cell_face_offsets[c];
    const offset_t end = t.incidence_stride == 1
                             ? t.cell_face_offsets[c + 1]
                             : first + offset_t(t.incidence_length[c]) * t.incidence_stride;
    for (offset_t k = first; k < end; k += t.incidence_stride) {
      const auto f = static_cast<std::size_t>(t.cell_faces[k]);
      face[f] = 1;
      n.incidences += 1;
      if (t.incidence_cell != nullptr) {
        if (t.incidence_cell[k] >= 0) across[static_cast<std::size_t>(t.incidence_cell[k])] = 1;
      } else if (f < ni) {
        const auto o = static_cast<std::size_t>(t.owner[f]);
        across[o == c ? static_cast<std::size_t>(t.neighbour[f]) : o] = 1;
      }
    }
  }
  n.faces = count(face);
  n.incidence_cells = count(across);
  n.incidence_touched_cells = any(self, across);
  if (t.stencil_offsets != nullptr) {
    n.stencil_sliced = t.stencil_stride > 1;
    std::vector<char> bnd(nf, 0), rows(nc, 0), cells(nc, 0);
    for (std::size_t c = 0; c < owned; ++c) {
      const offset_t first = t.stencil_offsets[c];
      const offset_t end = t.stencil_stride == 1
                               ? t.stencil_offsets[c + 1]
                               : first + offset_t(t.stencil_length[c]) * t.stencil_stride;
      for (offset_t k = first; k < end; k += t.stencil_stride) {
        rows[c] = 1;
        const auto e = t.stencil[k];
        if (e < 0) {
          bnd[static_cast<std::size_t>(-e - 1)] = 1;
        } else {
          cells[static_cast<std::size_t>(e)] = 1;
        }
        n.stencil += 1;
      }
    }
    n.stencil_rows = count(rows);
    n.stencil_cells = count(cells);
    n.stencil_touched_cells = any(rows, cells);
    n.stencil_owned_cells = any(self, cells);
    n.stencil_boundary_faces = count(bnd);
  }
  return n;
}

}  // namespace autan::kernels
