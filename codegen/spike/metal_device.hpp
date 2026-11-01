#pragma once

// Metal copies of a spike mesh and its geometry, shared by the Metal driver and benchmark.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "autan/kernels/colouring.hpp"
#include "autan/kernels/metal.hpp"
#include "spike.hpp"

namespace autan::spike::metal {

using autan::kernels::index_t;
using autan::kernels::offset_t;
namespace mtl = autan::kernels::metal;

/// A typed shared buffer of n elements.
template <class T>
class Array {
 public:
  Array(mtl::Context& ctx, std::size_t n) : buf_(ctx.allocate(n * sizeof(T))), n_(n) {}
  Array(mtl::Context& ctx, const std::vector<T>& v) : Array(ctx, v.size()) {
    if (!v.empty()) {
      std::memcpy(buf_.host(), v.data(), v.size() * sizeof(T));
    }
  }
  Array(mtl::Context& ctx, const T* p, std::size_t n) : Array(ctx, n) {
    if (n != 0) {
      std::memcpy(buf_.host(), p, n * sizeof(T));
    }
  }
  T* get() const { return buf_.device<T>(); }
  T* host() const { return buf_.data<T>(); }
  std::vector<T> download() const { return std::vector<T>(host(), host() + n_); }
  void zero() { std::memset(buf_.host(), 0, n_ * sizeof(T)); }

 private:
  mtl::Buffer buf_;
  std::size_t n_;
};

/// Topology, colourings and scratch of a spike mesh in Metal buffers.
template <class P>
struct DeviceMesh {
  static constexpr int kScratchWidth = 10;          // widest spike face loop: the Roe adjoint
  static constexpr int kIncidenceScratchWidth = 1;  // spike reductions gather one scalar
  Array<index_t> owner, neighbour, cell_faces;
  Array<offset_t> offsets;
  autan::kernels::HostColouring int_colours, bnd_colours;
  Array<index_t> int_faces, bnd_faces;
  Array<typename P::accum> scratch;
  Array<offset_t> twin;
  Array<offset_t> stencil_offsets;
  Array<index_t> stencil;
  Array<offset_t> stencil_twin;
  Array<std::uint8_t> reversed;
  Array<typename P::accum> incidence_scratch;
  autan::kernels::Topology topo;
  mtl::Schedule schedule;

  DeviceMesh(mtl::Context& ctx, const SpikeMesh& m, int block)
      : owner(ctx, m.topo.owner, static_cast<std::size_t>(m.topo.n_faces)),
        neighbour(ctx, m.topo.neighbour, static_cast<std::size_t>(m.topo.n_interior_faces)),
        cell_faces(ctx, m.topo.cell_faces, static_cast<std::size_t>(m.topo.n_incidences)),
        offsets(ctx, m.topo.cell_face_offsets, static_cast<std::size_t>(m.topo.n_cells) + 1),
        int_colours(autan::kernels::colour_faces(m.topo, 0, m.topo.n_owned_interior_faces)),
        bnd_colours(
            autan::kernels::colour_faces(m.topo, m.topo.n_interior_faces, m.topo.n_owned_faces)),
        int_faces(ctx, int_colours.faces),
        bnd_faces(ctx, bnd_colours.faces),
        scratch(ctx, kScratchWidth * static_cast<std::size_t>(m.topo.n_faces)),
        twin(ctx, m.topo.incidence_twin, static_cast<std::size_t>(m.topo.n_incidences)),
        stencil_offsets(
            ctx, m.topo.stencil_offsets,
            m.topo.stencil_offsets != nullptr ? static_cast<std::size_t>(m.topo.n_cells) + 1 : 0),
        stencil(ctx, m.topo.stencil, static_cast<std::size_t>(m.topo.n_stencil)),
        stencil_twin(ctx, m.topo.stencil_twin, static_cast<std::size_t>(m.topo.n_stencil)),
        reversed(ctx, m.topo.face_reversed,
                 m.topo.face_reversed != nullptr ? static_cast<std::size_t>(m.topo.n_faces) : 0),
        incidence_scratch(
            ctx, kIncidenceScratchWidth *
                     static_cast<std::size_t>(std::max(m.topo.n_incidences, m.topo.n_stencil))),
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
    topo.face_reversed = m.topo.face_reversed != nullptr ? reversed.get() : nullptr;
    if (m.topo.stencil != nullptr) {
      topo.stencil_offsets = stencil_offsets.get();
      topo.stencil = stencil.get();
      topo.stencil_twin = stencil_twin.get();
    }
  }
};

/// One geometry array on the device in both stored types (spike.hpp, GeomArray).
template <class P>
struct DeviceGeomArray {
  Array<typename P::geom> geom;
  Array<typename P::geom_value> value;  // empty when GeomArray<P>::same
  DeviceGeomArray(mtl::Context& ctx, const GeomArray<P>& a)
      : geom(ctx, a.geom), value(ctx, a.value) {}
  template <class T>
  [[nodiscard]] const T* as() const {
    if constexpr (std::is_same_v<T, typename P::geom>) {
      return geom.get();
    } else {
      return value.get();
    }
  }
};

template <class P>
struct DeviceGeom {
  DeviceGeomArray<P> face_area, face_centroid, face_moment, face_normal, face_area_magnitude,
      face_r_left, face_r_right, cell_centroid, cell_volume, lsq_weight, incidence_side,
      stencil_lsq_weight;
  DeviceGeom(mtl::Context& ctx, const GeomCopy<P>& g)
      : face_area(ctx, g.face_area),
        face_centroid(ctx, g.face_centroid),
        face_moment(ctx, g.face_moment),
        face_normal(ctx, g.face_normal),
        face_area_magnitude(ctx, g.face_area_magnitude),
        face_r_left(ctx, g.face_r_left),
        face_r_right(ctx, g.face_r_right),
        cell_centroid(ctx, g.cell_centroid),
        cell_volume(ctx, g.cell_volume),
        lsq_weight(ctx, g.lsq_weight),
        incidence_side(ctx, g.incidence_side),
        stencil_lsq_weight(ctx, g.stencil_lsq_weight) {}
};

/// Calls f with the Metal policy named `policy`; fp64 is refused (Metal has no FP64).
template <class F>
void with_policy(std::string_view policy, F&& f) {
  if (policy == "fp32+acc64") {
    f(mtl::fp32_acc64{});
  } else if (policy == "fp32") {
    f(mtl::fp32{});
  } else if (policy == "fp64") {
    throw std::invalid_argument("E1013: Metal has no FP64: policy fp64 is not available");
  } else {
    throw std::invalid_argument("E1102: unknown precision policy");
  }
}

}  // namespace autan::spike::metal
