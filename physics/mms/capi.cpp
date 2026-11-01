// C ABI over autan::physics::Euler for the manufactured-solution study (euler_mms.py): reads a
// mesh through the core readers and evaluates the residual on the host in FP64 arrays.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan_build_info.hpp"

namespace {

using autan::physics::Euler;

struct MeshHandle {
  autan::Mesh mesh;
  autan::Geometry geometry;
};

struct EulerHandle {
  std::variant<std::unique_ptr<Euler<autan::kernels::fp64>>,
               std::unique_ptr<Euler<autan::kernels::fp32_acc64>>>
      euler;
};

int fail(char* err, std::size_t len, std::string_view what) {
  if (err != nullptr && len > 0) {
    const std::string s(what);
    std::strncpy(err, s.c_str(), len - 1);
    err[len - 1] = '\0';
  }
  return 1;
}

template <class F>
int guarded(char* err, std::size_t len, F&& body) {
  try {
    body();
    return 0;
  } catch (const autan::Error& e) {
    return fail(err, len, std::string(autan::error_code_string(e.code())) + ": " + e.what());
  } catch (const std::exception& e) {
    return fail(err, len, std::string("E1100: ") + e.what());
  }
}

/// Converts FP64 arrays to the policy's state type, runs f, converts the result back.
template <class E, class F>
void in_policy(E& e, const double* a, const double* b, double* out, F&& f) {
  using S = typename std::remove_reference_t<decltype(*e)>::State;
  using R = typename std::remove_reference_t<decltype(*e)>::Res;
  const auto n = 5 * static_cast<std::size_t>(e->n_cells());
  std::vector<S> sa(a, a + n);
  std::vector<S> sb(b != nullptr ? b : a, (b != nullptr ? b : a) + n);
  std::vector<R> r(n, R(0));
  f(*e, std::span<const S>(sa), std::span<const S>(sb), std::span<R>(r));
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<double>(r[i]);
  }
}

}  // namespace

extern "C" {

/// Git commit the library was built from.
const char* autan_mms_commit() { return AUTAN_BUILD_COMMIT; }

int autan_mms_mesh_read(const char* path, void** out, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    auto m = std::make_unique<MeshHandle>(MeshHandle{autan::io::read_mesh(path), {}});
    m->geometry = autan::compute_geometry(m->mesh);
    *out = m.release();
  });
}

void autan_mms_mesh_free(void* mesh) { delete static_cast<MeshHandle*>(mesh); }

/// counts = {n_cells, n_faces, n_interior_faces}
void autan_mms_mesh_counts(const void* mesh, std::int64_t* counts) {
  const auto& m = static_cast<const MeshHandle*>(mesh)->mesh;
  counts[0] = m.n_cells();
  counts[1] = m.n_faces();
  counts[2] = m.n_interior_faces();
}

/// Owner and neighbour (n_interior_faces) of every face; cell centroids and volumes; face
/// centroids and unit normals (n_faces x 3, row-major).
void autan_mms_mesh_arrays(const void* mesh, std::int32_t* owner, std::int32_t* neighbour,
                           double* cell_centroid, double* cell_volume, double* face_centroid,
                           double* face_normal) {
  const auto& h = *static_cast<const MeshHandle*>(mesh);
  const auto& g = h.geometry;
  const auto v = h.mesh.view();
  const auto nf = static_cast<std::size_t>(h.mesh.n_faces());
  const auto ni = static_cast<std::size_t>(h.mesh.n_interior_faces());
  const auto nc = static_cast<std::size_t>(h.mesh.n_cells());
  std::memcpy(owner, v.owner, sizeof(std::int32_t) * nf);
  std::memcpy(neighbour, v.neighbour, sizeof(std::int32_t) * ni);
  for (std::size_t c = 0; c < nc; ++c) {
    cell_centroid[3 * c] = g.cell_centroid_x[c];
    cell_centroid[3 * c + 1] = g.cell_centroid_y[c];
    cell_centroid[3 * c + 2] = g.cell_centroid_z[c];
    cell_volume[c] = g.cell_volume[c];
  }
  for (std::size_t f = 0; f < nf; ++f) {
    face_centroid[3 * f] = g.face_centroid_x[f];
    face_centroid[3 * f + 1] = g.face_centroid_y[f];
    face_centroid[3 * f + 2] = g.face_centroid_z[f];
    face_normal[3 * f] = g.face_area_x[f] / g.face_area[f];
    face_normal[3 * f + 1] = g.face_area_y[f] / g.face_area[f];
    face_normal[3 * f + 2] = g.face_area_z[f] / g.face_area[f];
  }
}

/// scheme = {flux (0 Roe, 1 HLLC, 2 AUSM+-up), second_order, limiter, gradient stencil (0 face2,
/// 1 vertex)}; real = {gamma,
/// gas_constant, venkat_k, length, mach_ref, scale[5]}; policy "fp64" or "fp32+acc64".
int autan_mms_euler_create(const void* mesh, const char* policy, const std::int32_t* scheme,
                           const double* real, void** out, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    const auto& m = *static_cast<const MeshHandle*>(mesh);
    const autan::physics::Gas gas{real[0], real[1]};
    autan::physics::Scheme s;
    s.flux = static_cast<autan::physics::Flux>(scheme[0]);
    s.second_order = scheme[1] != 0;
    s.limiter = scheme[2] != 0;
    s.gradient = static_cast<autan::physics::GradientStencil>(scheme[3]);
    s.venkat_k = real[2];
    s.length = real[3];
    s.mach_ref = real[4];
    for (std::size_t c = 0; c < 5; ++c) {
      s.scale[c] = real[5 + c];
    }
    auto h = std::make_unique<EulerHandle>();
    const std::string_view p(policy);
    if (p == "fp64") {
      h->euler = std::make_unique<Euler<autan::kernels::fp64>>(m.mesh, m.geometry, gas, s);
    } else if (p == "fp32+acc64") {
      h->euler = std::make_unique<Euler<autan::kernels::fp32_acc64>>(m.mesh, m.geometry, gas, s);
    } else {
      throw std::invalid_argument("E1102: unknown precision policy");
    }
    *out = h.release();
  });
}

void autan_mms_euler_free(void* euler) { delete static_cast<EulerHandle*>(euler); }

/// kind: one code per boundary face (autan::physics::BoundaryKind); data: 5 x n_boundary_faces.
int autan_mms_euler_set_boundary(void* euler, const std::uint8_t* kind, const double* data,
                                 std::int64_t n_boundary, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    const auto nb = static_cast<std::size_t>(n_boundary);
    std::vector<autan::physics::BoundaryKind> k(nb);
    for (std::size_t b = 0; b < nb; ++b) {
      k[b] = static_cast<autan::physics::BoundaryKind>(kind[b]);
    }
    std::visit([&](auto& e) { e->set_boundary(k, std::span<const double>(data, 5 * nb)); },
               static_cast<EulerHandle*>(euler)->euler);
  });
}

int autan_mms_euler_set_source(void* euler, const double* source, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    std::visit(
        [&](auto& e) {
          e->set_source(
              std::span<const double>(source, 5 * static_cast<std::size_t>(e->n_cells())));
        },
        static_cast<EulerHandle*>(euler)->euler);
  });
}

/// res = R(u); arrays 5 x n_cells in FP64, converted to and from the policy's types.
int autan_mms_euler_residual(void* euler, const double* u, double* res, char* err,
                             std::size_t err_len) {
  return guarded(err, err_len, [&] {
    std::visit(
        [&](auto& e) {
          in_policy(e, u, nullptr, res,
                    [](auto& eu, auto su, auto, auto r) { eu.residual(su, r); });
        },
        static_cast<EulerHandle*>(euler)->euler);
  });
}

/// out = dR1/dU du, the tangent of the first-order residual.
int autan_mms_euler_first_order_tangent(void* euler, const double* u, const double* du, double* out,
                                        char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    std::visit(
        [&](auto& e) {
          in_policy(e, u, du, out, [](auto& eu, auto su, auto sdu, auto r) {
            eu.first_order_tangent(su, sdu, r);
          });
        },
        static_cast<EulerHandle*>(euler)->euler);
  });
}

}  // extern "C"
