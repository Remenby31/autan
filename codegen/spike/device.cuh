#pragma once

// Device copies of a spike mesh and its geometry, shared by the GPU driver and the benchmark.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "autan/kernels/colouring.hpp"
#include "autan/kernels/gpu.hpp"
#include "spike.hpp"

namespace autan::spike {

using autan::kernels::index_t;
using autan::kernels::offset_t;
namespace gpu = autan::kernels::gpu;

template <class T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    gpu::check(cudaMalloc(&p_, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    gpu::check(cudaMemset(p_, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
  }
  explicit DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) { upload(v.data()); }
  DeviceArray(const T* host, std::size_t n) : DeviceArray(n) { upload(host); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(p_); }
  void upload(const T* host) {
    gpu::check(cudaMemcpy(p_, host, n_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  std::vector<T> download() const {
    std::vector<T> v(n_);
    gpu::check(cudaMemcpy(v.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
  }
  T* get() const { return p_; }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

/// Topology, colourings and scratch of a spike mesh on the device.
struct DeviceMesh {
  static constexpr int kScratchWidth = 10;          // widest spike face loop: the Roe adjoint
  static constexpr int kIncidenceScratchWidth = 1;  // spike reductions gather one scalar
  DeviceArray<index_t> owner, neighbour, cell_faces;
  DeviceArray<offset_t> offsets;
  autan::kernels::HostColouring int_colours, bnd_colours;
  DeviceArray<index_t> int_faces, bnd_faces;
  DeviceArray<double> scratch;
  DeviceArray<offset_t> twin;
  DeviceArray<double> incidence_scratch;
  autan::kernels::Topology topo;
  gpu::Schedule schedule;

  DeviceMesh(const SpikeMesh& m, int block)
      : owner(m.topo.owner, static_cast<std::size_t>(m.topo.n_faces)),
        neighbour(m.topo.neighbour, static_cast<std::size_t>(m.topo.n_interior_faces)),
        cell_faces(m.topo.cell_faces, static_cast<std::size_t>(m.topo.n_incidences)),
        offsets(m.topo.cell_face_offsets, static_cast<std::size_t>(m.topo.n_cells) + 1),
        int_colours(autan::kernels::colour_faces(m.topo, 0, m.topo.n_interior_faces)),
        bnd_colours(autan::kernels::colour_faces(m.topo, m.topo.n_interior_faces, m.topo.n_faces)),
        int_faces(int_colours.faces),
        bnd_faces(bnd_colours.faces),
        scratch(kScratchWidth * static_cast<std::size_t>(m.topo.n_faces)),
        twin(m.topo.incidence_twin, static_cast<std::size_t>(m.topo.n_incidences)),
        incidence_scratch(kIncidenceScratchWidth * static_cast<std::size_t>(m.topo.n_incidences)),
        topo(m.topo) {
    topo.owner = owner.get();
    topo.neighbour = neighbour.get();
    topo.cell_faces = cell_faces.get();
    topo.cell_face_offsets = offsets.get();
    schedule.block = block;
    schedule.interior = {int_faces.get(), int_colours.offsets};
    schedule.boundary = {bnd_faces.get(), bnd_colours.offsets};
    schedule.face_scratch = scratch.get();
    schedule.face_scratch_width = kScratchWidth;
    schedule.incidence_scratch = incidence_scratch.get();
    schedule.incidence_scratch_width = kIncidenceScratchWidth;
    topo.incidence_twin = twin.get();
  }
};

template <class P>
struct DeviceGeom {
  using G = typename P::geom;
  DeviceArray<G> face_area, face_centroid, face_moment, face_normal, face_area_magnitude,
      face_r_left, face_r_right, cell_centroid, cell_volume, lsq_weight;
  explicit DeviceGeom(const GeomCopy<P>& g)
      : face_area(g.face_area),
        face_centroid(g.face_centroid),
        face_moment(g.face_moment),
        face_normal(g.face_normal),
        face_area_magnitude(g.face_area_magnitude),
        face_r_left(g.face_r_left),
        face_r_right(g.face_r_right),
        cell_centroid(g.cell_centroid),
        cell_volume(g.cell_volume),
        lsq_weight(g.lsq_weight) {}
};

inline gpu::Strategy strategy_of(std::string_view s) {
  for (auto k : {gpu::Strategy::atomic, gpu::Strategy::warp_aggregated, gpu::Strategy::coloured,
                 gpu::Strategy::store_gather, gpu::Strategy::gather_recompute}) {
    if (s == gpu::name(k)) {
      return k;
    }
  }
  throw std::invalid_argument("E1103: unknown race strategy");
}

}  // namespace autan::spike
