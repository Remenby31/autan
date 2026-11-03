// C ABI over autan::physics::Euler for the manufactured-solution study (euler_mms.py): reads a
// mesh through the core readers and evaluates the residual on the host in FP64 arrays.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
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

/// Converts the conserved variables `a` to the stored state and `b` (a direction) to the state
/// type, runs f, converts the result back to FP64.
template <class E, class F>
void in_policy(E& e, const double* a, const double* b, double* out, F&& f) {
  using S = typename std::remove_reference_t<decltype(*e)>::State;
  using R = typename std::remove_reference_t<decltype(*e)>::Res;
  const auto n = 5 * static_cast<std::size_t>(e->n_cells());
  std::vector<S> sa(n);
  e->state(std::span<const double>(a, n), sa);
  std::vector<S> sb(b != nullptr ? b : a, (b != nullptr ? b : a) + n);
  std::vector<R> r(n, R(0));
  f(*e, std::span<const S>(sa), std::span<const S>(sb), std::span<R>(r));
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = static_cast<double>(r[i]);
  }
}

using Block = std::array<double, 25>;

// c = a b for 5x5 row-major blocks.
void multiply(const double* a, const double* b, double* c) {
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      double s = 0.0;
      for (int k = 0; k < 5; ++k) s += a[5 * i + k] * b[5 * k + j];
      c[5 * i + j] = s;
    }
  }
}

// Inverse of a 5x5 block by Gauss-Jordan elimination with partial pivoting; false on a zero pivot.
bool invert(const double* a, double* inverse) {
  std::array<std::array<double, 10>, 5> m{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) m[i][j] = a[5 * i + j];
    m[i][5 + i] = 1.0;
  }
  for (int p = 0; p < 5; ++p) {
    int best = p;
    for (int r = p + 1; r < 5; ++r) {
      if (std::abs(m[r][p]) > std::abs(m[best][p])) best = r;
    }
    if (!(std::abs(m[best][p]) > 0.0) || !std::isfinite(m[best][p])) return false;
    std::swap(m[p], m[best]);
    const double d = m[p][p];
    for (double& v : m[p]) v /= d;
    for (int r = 0; r < 5; ++r) {
      if (r == p) continue;
      const double f = m[r][p];
      for (int j = 0; j < 10; ++j) m[r][j] -= f * m[p][j];
    }
  }
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) inverse[5 * i + j] = m[i][5 + j];
  }
  return true;
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
/// gas_constant, venkat_k, length, mach_ref, scale[5], reference[5]} (reference: the conserved
/// state the cells store differences from); policy "fp64" or "fp32+acc64".
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
      s.reference[c] = real[10 + c];
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

/// Block ILU(0) of a matrix of 5x5 blocks in block CSR (n block rows, indptr of n + 1 offsets,
/// sorted column indices, row-major blocks), in place: the strictly lower blocks become L (unit
/// block diagonal), the others U, and `inverse` receives the inverse of each pivot block. No block
/// outside the matrix's own pattern is created. Fails on a block row without a diagonal block or
/// with a singular pivot block.
int autan_mms_block_ilu0(std::int64_t n, const std::int64_t* indptr, const std::int32_t* indices,
                         double* blocks, double* inverse, char* err, std::size_t err_len) {
  return guarded(err, err_len, [&] {
    const auto rows = static_cast<std::size_t>(n);
    std::vector<std::int64_t> where(rows, -1);
    std::vector<std::int64_t> diagonal(rows, -1);
    Block product{};
    // Saad (2003), Algorithm 10.4 (ILU(0), IKJ variant) with blocks for scalars.
    for (std::size_t i = 0; i < rows; ++i) {
      for (auto p = indptr[i]; p < indptr[i + 1]; ++p) {
        const auto j = static_cast<std::size_t>(indices[p]);
        where[j] = p;
        if (j == i) diagonal[i] = p;
      }
      if (diagonal[i] < 0) {
        throw std::domain_error("block row " + std::to_string(i) + " has no diagonal block");
      }
      for (auto p = indptr[i]; p < diagonal[i]; ++p) {
        const auto k = static_cast<std::size_t>(indices[p]);
        multiply(blocks + 25 * p, inverse + 25 * k, product.data());
        std::copy(product.begin(), product.end(), blocks + 25 * p);
        for (auto q = diagonal[k] + 1; q < indptr[k + 1]; ++q) {
          const auto at = where[static_cast<std::size_t>(indices[q])];
          if (at < 0) continue;
          multiply(blocks + 25 * p, blocks + 25 * q, product.data());
          for (std::size_t e = 0; e < 25; ++e)
            blocks[25 * at + static_cast<std::int64_t>(e)] -= product[e];
        }
      }
      if (!invert(blocks + 25 * diagonal[i], inverse + 25 * i)) {
        throw std::domain_error("singular pivot block in block row " + std::to_string(i));
      }
      for (auto p = indptr[i]; p < indptr[i + 1]; ++p)
        where[static_cast<std::size_t>(indices[p])] = -1;
    }
  });
}

/// x = (LU)^-1 b for the factors autan_mms_block_ilu0 left in `blocks` and `inverse`; b and x hold
/// 5 values per block row and may not alias.
void autan_mms_block_ilu0_solve(std::int64_t n, const std::int64_t* indptr,
                                const std::int32_t* indices, const double* blocks,
                                const double* inverse, const double* b, double* x) {
  std::array<double, 5> s{};
  for (std::int64_t i = 0; i < n; ++i) {
    std::copy(b + 5 * i, b + 5 * i + 5, s.begin());
    for (auto p = indptr[i]; p < indptr[i + 1] && indices[p] < i; ++p) {
      const double* l = blocks + 25 * p;
      const double* y = x + 5 * static_cast<std::int64_t>(indices[p]);
      for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) s[static_cast<std::size_t>(r)] -= l[5 * r + c] * y[c];
      }
    }
    std::copy(s.begin(), s.end(), x + 5 * i);
  }
  for (std::int64_t i = n - 1; i >= 0; --i) {
    std::copy(x + 5 * i, x + 5 * i + 5, s.begin());
    for (auto p = indptr[i + 1] - 1; p >= indptr[i] && indices[p] > i; --p) {
      const double* u = blocks + 25 * p;
      const double* y = x + 5 * static_cast<std::int64_t>(indices[p]);
      for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) s[static_cast<std::size_t>(r)] -= u[5 * r + c] * y[c];
      }
    }
    const double* d = inverse + 25 * i;
    for (int r = 0; r < 5; ++r) {
      double v = 0.0;
      for (int c = 0; c < 5; ++c) v += d[5 * r + c] * s[static_cast<std::size_t>(c)];
      x[5 * i + r] = v;
    }
  }
}

}  // extern "C"
