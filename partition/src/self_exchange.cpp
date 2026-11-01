#include <array>
#include <cstddef>
#include <span>

#include "autan/core/error.hpp"
#include "autan/partition/partition.hpp"

namespace autan {
namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

}  // namespace

template <class T>
void forward_self(const HaloPattern& halo, index_t n_owned_cells, std::span<T* const> components) {
  for (std::size_t i = 0; i < halo.neighbours.size(); ++i) {
    if (halo.neighbours[i] != halo.rank) continue;
    // The ghosts of block i are filled from send_cells of block i, in order.
    const index_t count = halo.recv_offsets[i + 1] - halo.recv_offsets[i];
    for (index_t k = 0; k < count; ++k) {
      const auto ghost = at(n_owned_cells + halo.recv_offsets[i] + k);
      const auto owner = at(halo.send_cells[at(halo.send_offsets[i] + k)]);
      for (T* const c : components) c[ghost] = c[owner];
    }
  }
}

template void forward_self<float>(const HaloPattern&, index_t, std::span<float* const>);
template void forward_self<double>(const HaloPattern&, index_t, std::span<double* const>);

void fill_ghost_geometry(LocalPartition& part) {
  if (part.halo.n_ranks != 1) {
    throw Error(ErrorCode::partition_invalid_request,
                "fill_ghost_geometry fills one-part partitions; several exchange their geometry");
  }
  Geometry& g = part.geometry;
  const std::array<double*, 4> fields{g.cell_volume.data(), g.cell_centroid_x.data(),
                                      g.cell_centroid_y.data(), g.cell_centroid_z.data()};
  const index_t n_owned = part.mesh.n_owned_cells();
  forward_self(part.halo, n_owned, std::span<double* const>(fields));
  const auto t = part.mesh.ghost_translation();
  for (std::size_t k = 0; k < t.size(); ++k) {
    g.cell_centroid_x[at(n_owned) + k] += t[k][0];
    g.cell_centroid_y[at(n_owned) + k] += t[k][1];
    g.cell_centroid_z[at(n_owned) + k] += t[k][2];
  }
}

}  // namespace autan
