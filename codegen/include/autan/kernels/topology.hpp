#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// Rows of a CSR (n + 1 offsets) in slices of `slice` consecutive rows: row r of lane l of slice
/// s starts at start(s) + l, its entry j at that start + j * slice, start(s) the sum of slice *
/// (longest row) over the slices before. `starts` has n + 1 entries (the last the total size,
/// padding included); `position[k]` is where CSR entry k went. Throws E1101 for slice < 1.
struct SlicedRows {
  std::vector<offset_t> starts;
  std::vector<index_t> length;
  std::vector<offset_t> position;
};
SlicedRows slice_rows(std::span<const offset_t> offsets, int slice);

/// The cell -> face incidences of `csr` (a Topology in CSR layout, with incidence_twin) in the
/// sliced layout of slice_rows (Topology::incidence_stride): cell_face_offsets holds n_cells + 1
/// starts, `cell` the cell across each incidence's face (-1 on boundary faces and padding),
/// twins sliced positions. Each cell keeps its faces in the same order, so a reduction gives the
/// same bits. Padding holds face 0.
struct SlicedIncidences {
  std::vector<offset_t> cell_face_offsets;
  std::vector<index_t> length;
  std::vector<offset_t> position;
  std::vector<index_t> cell_faces;
  std::vector<index_t> cell;
  std::vector<offset_t> twin;
  offset_t stride = 1;
};
SlicedIncidences sliced_incidences(const Topology& csr, int slice);

/// Per-incidence data (`comps` x n_incidences, component-major) moved to the sliced positions of
/// `s` (`comps` x its total size), padding 0.
std::vector<double> sliced_values(const SlicedIncidences& s, const std::vector<double>& csr,
                                  int comps);

/// The Topology of a graph without geometry, such as a coarse multigrid level: nodes as owned
/// cells, edge e (lo[e] < hi[e]) as interior face e with owner lo[e] and neighbour hi[e], no
/// boundary face; each node's incidences are its edges in increasing order, in CSR (slice 0) or
/// sliced layout. topology() views this object's arrays. Throws E1101 for an edge that is not a
/// pair lo < hi < n_nodes.
struct GraphTopology {
  index_t n_nodes = 0;
  std::vector<index_t> owner, neighbour, cell_faces, length, cell;
  std::vector<offset_t> cell_face_offsets;
  offset_t stride = 1;
  [[nodiscard]] Topology topology() const;
};
GraphTopology graph_topology(index_t n_nodes, std::span<const index_t> lo,
                             std::span<const index_t> hi, int slice);

/// The owned rows sorted by colour, in their order within a colour, then the ghost rows in
/// their order: `new_to_old[p]` is the row placed at position p (autan::renumber_cells' order),
/// and colour k's rows are positions offsets[k] .. offsets[k + 1]. A block system renumbered so
/// relaxes each colour over a contiguous range of rows. Throws E1101 for a colour outside
/// [0, n_colours).
struct ColourOrder {
  std::vector<index_t> new_to_old;
  std::vector<index_t> offsets;
};
ColourOrder colour_order(std::span<const std::int32_t> colours, std::int32_t n_colours,
                         index_t n_owned);

}  // namespace autan::kernels
