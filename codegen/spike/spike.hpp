#pragma once

// Pieces shared by the host and GPU spike drivers.
#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"

namespace autan::spike {

struct SpikeMesh {
  autan::Mesh mesh;
  autan::Geometry geometry;
  autan::kernels::PackedGeometry packed;
  std::vector<autan::kernels::offset_t> twins;
  autan::kernels::CellStencil stencil;
  autan::kernels::Topology topo{};
  std::vector<std::int64_t> face_global_id;  ///< partitions: face of the global mesh
};

inline int fail(char* err, std::size_t len, std::string_view code, std::string_view what) {
  if (err != nullptr && len > 0) {
    std::string s(what);
    if (!(s.size() > 6 && s[0] == 'E' && s[5] == ':')) {
      s = std::string(code) + ": " + s;
    }
    std::strncpy(err, s.c_str(), len - 1);
    err[len - 1] = '\0';
  }
  return 1;
}

/// Runs `body`, turning exceptions into a return code and an "E####: message" string.
template <class F>
int guarded(char* err, std::size_t len, F&& body) {
  try {
    body();
    return 0;
  } catch (const autan::Error& e) {
    return fail(err, len, autan::error_code_string(e.code()), e.what());
  } catch (const std::exception& e) {
    return fail(err, len, "E1100", e.what());
  }
}

/// Geometry arrays converted to the policy's Geom storage type.
template <class P>
struct GeomCopy {
  using G = typename P::geom;
  std::vector<G> face_area, face_centroid, face_moment, face_normal, face_area_magnitude,
      face_r_left, face_r_right, cell_centroid, cell_volume, lsq_weight, stencil_lsq_weight;
  explicit GeomCopy(const autan::kernels::PackedGeometry& g)
      : face_area(autan::kernels::converted<G>(g.face_area)),
        face_centroid(autan::kernels::converted<G>(g.face_centroid)),
        face_moment(autan::kernels::converted<G>(g.face_moment)),
        face_normal(autan::kernels::converted<G>(g.face_normal)),
        face_area_magnitude(autan::kernels::converted<G>(g.face_area_magnitude)),
        face_r_left(autan::kernels::converted<G>(g.face_r_left)),
        face_r_right(autan::kernels::converted<G>(g.face_r_right)),
        cell_centroid(autan::kernels::converted<G>(g.cell_centroid)),
        cell_volume(autan::kernels::converted<G>(g.cell_volume)),
        lsq_weight(autan::kernels::converted<G>(g.lsq_weight)),
        stencil_lsq_weight(autan::kernels::converted<G>(g.stencil_lsq_weight)) {}
};

/// Calls f with the policy object named by `policy`; throws for an unknown name.
template <class F>
void with_policy(std::string_view policy, F&& f) {
  if (policy == "fp64") {
    f(autan::kernels::fp64{});
  } else if (policy == "fp32+acc64") {
    f(autan::kernels::fp32_acc64{});
  } else if (policy == "fp32") {
    f(autan::kernels::fp32{});
  } else {
    throw std::invalid_argument("E1102: unknown precision policy");
  }
}

}  // namespace autan::spike
