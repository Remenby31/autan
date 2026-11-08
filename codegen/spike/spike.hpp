#pragma once

// Pieces shared by the host and GPU spike drivers.
#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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
  autan::kernels::PackedGeometry packed{};
  std::vector<autan::kernels::offset_t> twins{};
  autan::kernels::CellStencil stencil{};
  autan::kernels::Topology topo{};
  std::vector<std::int64_t> face_global_id{};  ///< partitions: face of the global mesh
  std::vector<std::uint8_t> face_reversed{};   ///< partitions: LocalPartition::face_reversed
  /// Set by autan_spike_set_global_order: keys and the incidence order they give.
  std::vector<std::int64_t> cell_key{}, face_key{};
  autan::kernels::GlobalOrder order{};
  bool global = false;
  /// Set only by autan_spike_test_only_drop_incidence (ADR 0003, control 3): the incidence list
  /// without the dropped entry, which topo then points to.
  std::vector<autan::kernels::offset_t> test_only_offsets{};
  std::vector<autan::kernels::index_t> test_only_cell_faces{};
  [[nodiscard]] autan::kernels::GlobalKeys keys() const {
    return {cell_key, face_key, face_reversed, mesh.ghost_translation()};
  }
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

/// One geometry array in both types a dataset may be stored in: the policy's Geom type, and its
/// geom_value type (Geom datasets no body reads through rel(); kernel-generator.md, Precision).
template <class P>
struct GeomArray {
  // `value` stays empty when the two types coincide (fp64, fp32): `geom` serves both.
  static constexpr bool same = std::is_same_v<typename P::geom, typename P::geom_value>;
  std::vector<typename P::geom> geom;
  std::vector<typename P::geom_value> value;
  explicit GeomArray(const std::vector<double>& v)
      : geom(autan::kernels::converted<typename P::geom>(v)),
        value(same ? std::vector<typename P::geom_value>{}
                   : autan::kernels::converted<typename P::geom_value>(v)) {}
  template <class T>
  [[nodiscard]] const T* as() const {
    if constexpr (std::is_same_v<T, typename P::geom>) {
      return geom.data();
    } else {
      return value.data();
    }
  }
};

/// Geometry arrays for the generated Data<P> structs; bind() picks the stored type.
template <class P>
struct GeomCopy {
  GeomArray<P> face_area, face_centroid, face_moment, face_normal, face_area_magnitude, face_r_left,
      face_r_right, cell_centroid, cell_volume, lsq_weight, incidence_side, stencil_lsq_weight;
  explicit GeomCopy(const autan::kernels::PackedGeometry& g)
      : face_area(g.face_area),
        face_centroid(g.face_centroid),
        face_moment(g.face_moment),
        face_normal(g.face_normal),
        face_area_magnitude(g.face_area_magnitude),
        face_r_left(g.face_r_left),
        face_r_right(g.face_r_right),
        cell_centroid(g.cell_centroid),
        cell_volume(g.cell_volume),
        lsq_weight(g.lsq_weight),
        incidence_side(g.incidence_side),
        stencil_lsq_weight(g.stencil_lsq_weight) {}
};

/// Points a Data<P> geometry member at the array of its stored type.
template <class T, class A>
void bind(const T*& member, const A& array) {
  member = array.template as<T>();
}

/// TEST ONLY (ADR 0003, control 11): fp32_acc64 with its accumulation narrowed to binary32
/// (accum and residual float). It exists only to show that check 3 and check 4 catch a wrong
/// precision. It lives in the spike, is dispatched only by autan_spike_gradients under the name
/// "fp32+acc32-test-only", and must never appear in runtime.hpp or a product target
/// (test_spike_isolation). The kernels it runs are the generator's, instantiated with it.
struct Fp32Acc32TestOnly {
  using state = float;
  using grad = float;
  using residual = float;
  using geom = double;
  using geom_value = float;
  using param = double;
  using jac = float;
  using compute = float;
  using accum = float;
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
