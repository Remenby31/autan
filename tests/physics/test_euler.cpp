// The Euler residual through autan::physics::Euler on the test meshes of tests/data/mesh.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <unistd.h>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/geometry.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/point_vortex.hpp"
#include "euler_cpu.hpp"
#include "fixtures.hpp"
#include "navier_stokes_cpu.hpp"
#include "stretched_slab.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::Scheme;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};

// Tetrahedra, hexahedra, polyhedra and hybrid cells; every mesh fills the unit cube with
// boundary zones xmin, xmax, ymin, ymax, zmin, zmax.
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

constexpr Gas air{1.4, 287.0};

struct Prim {
  double rho;
  std::array<double, 3> v;
  double p;
};

double sound(const Prim& w) { return std::sqrt(air.gamma * w.p / w.rho); }

std::array<double, 5> conserved(const Prim& w) {
  const double q2 = w.v[0] * w.v[0] + w.v[1] * w.v[1] + w.v[2] * w.v[2];
  return {w.rho, w.rho * w.v[0], w.rho * w.v[1], w.rho * w.v[2],
          w.p / (air.gamma - 1.0) + 0.5 * w.rho * q2};
}

struct Case {
  autan::Mesh mesh;
  autan::Geometry geometry;
  explicit Case(const char* file)
      : mesh(autan::io::read_mesh(mesh_dir / file)), geometry(autan::compute_geometry(mesh)) {}

  [[nodiscard]] std::string zone(std::size_t boundary_face) const {
    const auto f = static_cast<std::size_t>(mesh.n_interior_faces()) + boundary_face;
    return mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
  }

  /// Sum of face areas around each cell.
  [[nodiscard]] std::vector<double> cell_surface() const {
    std::vector<double> s(static_cast<std::size_t>(mesh.n_cells()), 0.0);
    for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
      s[static_cast<std::size_t>(mesh.owner()[f])] += geometry.face_area[f];
      if (f < static_cast<std::size_t>(mesh.n_interior_faces())) {
        s[static_cast<std::size_t>(mesh.neighbour()[f])] += geometry.face_area[f];
      }
    }
    return s;
  }
};

template <class P>
constexpr double eps() {
  return std::numeric_limits<typename P::compute>::epsilon();
}

/// Largest |R_k| of each cell relative to the flux bound of the uniform state times the cell's
/// surface: the residual of a steady state is round-off of the face fluxes it sums.
template <class P>
double relative_residual(Euler<P>& e, const Case& c, std::span<const typename P::state> u,
                         const Prim& w) {
  const auto n = static_cast<std::size_t>(e.n_cells());
  std::vector<typename P::residual> r(5 * n);
  e.residual(u, r);
  const double speed = std::hypot(w.v[0], w.v[1], w.v[2]) + sound(w);
  const auto uc = conserved(w);
  const std::array<double, 5> flux = {w.rho * speed, w.rho * speed * speed + w.p,
                                      w.rho * speed * speed + w.p, w.rho * speed * speed + w.p,
                                      (uc[4] + w.p) * speed};
  const auto surface = c.cell_surface();
  double worst = 0.0;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < n; ++i) {
      const double v = std::abs(static_cast<double>(r[k * n + i])) / (flux[k] * surface[i]);
      if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
      worst = std::max(worst, v);
    }
  }
  return worst;
}

std::vector<double> uniform(const Prim& w, std::size_t n) {
  const auto uc = conserved(w);
  std::vector<double> u(5 * n);
  for (std::size_t k = 0; k < 5; ++k) {
    std::fill_n(u.begin() + static_cast<std::ptrdiff_t>(k * n), n, uc[k]);
  }
  return u;
}

template <class P>
std::vector<typename P::state> stored(const Euler<P>& e, const std::vector<double>& u) {
  std::vector<typename P::state> s(u.size());
  e.state(u, s);
  return s;
}

Scheme second_order(Flux flux, double mach) {
  Scheme s;
  s.flux = flux;
  s.mach_ref = mach;
  s.scale = {1.0, 100.0, 100.0, 100.0, 1e5};
  // A reference state apart from every test state, so that the stored differences are not zero.
  s.reference = conserved({1.1, {10.0, 5.0, 0.0}, 0.95e5});
  return s;
}

}  // namespace

TEMPLATE_TEST_CASE("a uniform flow that satisfies every boundary condition is steady", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  INFO(file << " flux " << static_cast<int>(flux));
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  auto put = [&](std::size_t b, BoundaryKind k, std::array<double, 5> v) {
    kind[b] = k;
    for (std::size_t j = 0; j < 5; ++j) {
      data[j * nb + b] = v[j];
    }
  };

  SECTION("inlet at total conditions, outlet at static pressure, slip walls along the flow") {
    const Prim w{1.2, {0.5 * std::sqrt(1.4 * 1e5 / 1.2), 0.0, 0.0}, 1e5};
    const double t = w.p / (w.rho * air.gas_constant);
    const double m2 = (w.v[0] / sound(w)) * (w.v[0] / sound(w));
    const double t0 = t * (1.0 + 0.2 * m2);
    const double p0 = w.p * std::pow(t0 / t, 3.5);
    for (std::size_t b = 0; b < nb; ++b) {
      const auto z = c.zone(b);
      if (z == "xmin") {
        put(b, BoundaryKind::inlet, {p0, t0, 0.0, 0.0, 0.0});
      } else if (z == "xmax") {
        put(b, BoundaryKind::outlet, {0.0, 0.0, 0.0, 0.0, w.p});
      } else {
        put(b, BoundaryKind::slip, {});
      }
    }
    Euler<P> e(c.mesh, c.geometry, air, second_order(flux, 0.5));
    e.set_boundary(kind, data);
    const auto u = stored(e, uniform(w, nc));
    CHECK(relative_residual(e, c, std::span<const typename P::state>(u), w) <= 64 * eps<P>());
  }

  SECTION("farfield on every face, subsonic and supersonic, oblique flow") {
    const double mach = GENERATE(0.5, 2.0);
    const double a = std::sqrt(1.4 * 1e5 / 1.2);
    const Prim w{1.2, {mach * a * 0.8, mach * a * 0.36, mach * a * 0.48}, 1e5};
    for (std::size_t b = 0; b < nb; ++b) {
      put(b, BoundaryKind::farfield, {w.rho, w.v[0], w.v[1], w.v[2], w.p});
    }
    Euler<P> e(c.mesh, c.geometry, air, second_order(flux, mach));
    e.set_boundary(kind, data);
    const auto u = stored(e, uniform(w, nc));
    CHECK(relative_residual(e, c, std::span<const typename P::state>(u), w) <= 64 * eps<P>());
  }
}

TEMPLATE_TEST_CASE("gas at rest under uniform pressure is steady for any density field", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // Each flux resolves a stationary contact exactly (Toro 2009, sections 10.4 and 11.2).
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  INFO(file << " flux " << static_cast<int>(flux));
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  const std::vector<double> data(5 * nb, 0.0);
  Euler<P> e(c.mesh, c.geometry, air, second_order(flux, 0.1));
  e.set_boundary(kind, data);
  const Prim w{1.0, {0.0, 0.0, 0.0}, 1e5};
  auto absolute = uniform(w, nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const auto x = c.geometry.cell_centroid_x[i];
    const auto y = c.geometry.cell_centroid_y[i];
    const auto z = c.geometry.cell_centroid_z[i];
    absolute[i] =
        1.0 + 0.5 * std::sin(7.0 * x + 3.0 * y) * std::cos(5.0 * z) + (x > 0.5 ? 9.0 : 0.0);
  }
  const auto u = stored(e, absolute);
  // A 10:1 density jump: the linear reconstruction would make the density negative at some
  // faces, where the scheme falls back to the cell value.
  const Prim scale{11.0, {0.0, 0.0, 0.0}, 1e5};
  CHECK(relative_residual(e, c, std::span<const typename P::state>(u), scale) <= 64 * eps<P>());
}

TEMPLATE_TEST_CASE("slip walls keep a gauge pressure finer than FP32 resolves p_ref", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // One skew tetrahedron with four slip faces: no interior neighbour determines its gradient, so
  // the boundary values do. At rest under uniform pressure the gradient must vanish and the
  // residual reduce to the closure of the gauge pressure, even when p - p_ref = 1e-3 is below the
  // FP32 spacing of p_ref = 0.95e5 (7.8e-3).
  using P = TestType;
  // One file per process: the two precisions run as separate tests, possibly at once (ctest -j).
  const fs::path file =
      fs::temp_directory_path() / ("autan-one-skew-tet-" + std::to_string(::getpid()) + ".msh");
  {
    std::ofstream out(file);
    out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        << "$PhysicalNames\n2\n2 1 \"walls\"\n3 101 \"fluid\"\n$EndPhysicalNames\n"
        << "$Entities\n0 0 1 1\n1 0 0 0 1 1 1.3 1 1 0\n1 0 0 0 1 1 1.3 1 101 1 1\n"
        << "$EndEntities\n"
        << "$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n"
        << "0 0 0\n1 0 0\n0.3 1 0\n0.2 0.4 1.3\n$EndNodes\n"
        << "$Elements\n2 5 1 5\n2 1 2 4\n1 1 3 2\n2 1 2 4\n3 2 3 4\n4 1 4 3\n"
        << "3 1 4 1\n5 1 2 3 4\n$EndElements\n";
  }
  const Case c(file.c_str());
  fs::remove(file);
  REQUIRE(c.mesh.n_cells() == 1);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  Euler<P> e(c.mesh, c.geometry, air, second_order(Flux::roe, 0.1));
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::slip), std::vector<double>(5 * nb));
  const double gauge = 1e-3;
  const auto u = stored(e, uniform(Prim{1.0, {0.0, 0.0, 0.0}, 0.95e5 + gauge}, 1));
  std::vector<typename P::residual> r(5);
  e.residual(std::span<const typename P::state>(u), r);
  const double surface = c.cell_surface()[0];
  for (const auto v : r) CHECK(std::abs(static_cast<double>(v)) <= 64 * eps<P>() * gauge * surface);
}

TEMPLATE_TEST_CASE("the planar loops give the full loops' residual where every face is planar",
                   "[euler]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // Where every face moment is an exact zero (kernels::PackedGeometry::all_planar), the
  // warped-face terms are signed zeros for a finite state, and x + (+-0) == x: the loops without
  // them give the same values (== also equates -0 and +0). Tetrahedra and the axis-aligned
  // hexahedra are planar; a mesh with a face not proven planar refuses the planar loops.
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  CAPTURE(file, static_cast<int>(flux), limiter);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto ni = static_cast<std::size_t>(c.mesh.n_interior_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  auto flow = [](double x, double y, double z) {
    return Prim{1.2 + 0.1 * std::sin(2.0 * x + y),
                {80.0 + 15.0 * std::cos(x + 2.0 * z), -20.0 + 10.0 * std::sin(y + z),
                 5.0 * std::cos(3.0 * x)},
                1.0e5 + 4.0e3 * std::sin(x + y + z)};
  };
  Scheme s = second_order(flux, 0.3);
  s.limiter = limiter;
  s.venkat_k = 0.1;
  Euler<P> e(c.mesh, c.geometry, air, s);
  const std::string name(file);
  if (name.starts_with("tet") || name.starts_with("hex")) REQUIRE(e.all_planar());
  if (!e.all_planar()) {
    CHECK_THROWS_AS(e.set_planar(true), std::invalid_argument);
    return;
  }
  // Farfield on x and z, slip walls on y: the wall's branch and the flux's both run.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto w = flow(c.geometry.face_centroid_x[f], c.geometry.face_centroid_y[f],
                        c.geometry.face_centroid_z[f]);
    kind[b] = c.zone(b)[0] == 'y' ? BoundaryKind::slip : BoundaryKind::farfield;
    const std::array<double, 5> v{w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = v[k];
  }
  e.set_boundary(kind, data);
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const auto uc = conserved(flow(c.geometry.cell_centroid_x[i], c.geometry.cell_centroid_y[i],
                                   c.geometry.cell_centroid_z[i]));
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = uc[k];
  }
  const auto stored_u = stored(e, u);
  std::vector<typename P::residual> full(5 * nc), planar(5 * nc);
  e.residual(stored_u, full);
  e.set_planar(true);
  REQUIRE(e.planar());
  e.residual(stored_u, planar);
  std::size_t differ = 0;
  for (std::size_t i = 0; i < full.size(); ++i) {
    REQUIRE(std::isfinite(static_cast<double>(full[i])));
    if (!(planar[i] == full[i])) ++differ;
  }
  CHECK(differ == 0);
}

TEST_CASE("the first-order tangent matches central differences of the first-order residual",
          "[euler]") {
  using P = autan::kernels::fp64;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  INFO(file << " flux " << static_cast<int>(flux));
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const double a = std::sqrt(1.4 * 1e5 / 1.2);
  const Prim w{1.2, {0.6 * a, 0.2 * a, -0.1 * a}, 1e5};
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> v{w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t j = 0; j < 5; ++j) {
      data[j * nb + b] = v[j];
    }
  }
  // M_co = 1 holds AUSM+-up's low-Mach scaling fixed: fa(M_o) has a kink where the mean
  // normal Mach number crosses M_co, which central differences straddle.
  Scheme s = second_order(flux, 1.0);
  s.second_order = false;
  Euler<P> e(c.mesh, c.geometry, air, s);
  e.set_boundary(kind, data);
  auto absolute = uniform(w, nc);
  std::vector<double> du(5 * nc);
  const auto uc = conserved(w);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < nc; ++i) {
      const double x = c.geometry.cell_centroid_x[i];
      const double y = c.geometry.cell_centroid_y[i];
      absolute[k * nc + i] *=
          1.0 + 0.1 * std::sin(3.0 * x + static_cast<double>(k)) * std::cos(2.0 * y);
      du[k * nc + i] = uc[k] * std::cos(5.0 * x + 2.0 * y + static_cast<double>(k));
    }
  }
  const auto u = stored(e, absolute);
  std::vector<double> jv(5 * nc), rp(5 * nc), rm(5 * nc);
  e.first_order_tangent(u, du, jv);
  const double h = 1e-6;
  std::vector<double> up(u), um(u);
  for (std::size_t i = 0; i < u.size(); ++i) {
    up[i] += h * du[i];
    um[i] -= h * du[i];
  }
  e.residual(up, rp);
  e.residual(um, rm);
  double diff = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    const double fd = (rp[i] - rm[i]) / (2.0 * h);
    REQUIRE(std::isfinite(fd));
    REQUIRE(std::isfinite(jv[i]));
    diff = std::max(diff, std::abs(fd - jv[i]));
    norm = std::max(norm, std::abs(jv[i]));
  }
  // Central differences: truncation O(h^2) and round-off O(eps / h) relative.
  CHECK(diff <= 1e-6 * norm);
  // The tangent's arrays are allocated by its first call; residuals between calls leave them as
  // they were: a second call gives the first one's bits.
  std::vector<double> again(5 * nc);
  e.first_order_tangent(u, du, again);
  CHECK(std::memcmp(again.data(), jv.data(), jv.size() * sizeof(double)) == 0);
}

TEST_CASE("the stencil twins built on demand are those of the cell stencil", "[euler]") {
  // Euler keeps no twins and no copy of the weights (no forward loop reads the twins, and the
  // weights live in the loops' data): their memory is released, not only emptied. The adjoint
  // asks for the twins: they must be kernels::cell_stencil's, bit for bit, and the topology the
  // generated reverse reductions read must point at them, after every call.
  using P = autan::kernels::fp32_acc64;
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const Case c(file);
  Euler<P> e(c.mesh, c.geometry, air, second_order(Flux::roe, 0.5));
  CHECK(e.stencil().twin.capacity() == 0);
  CHECK(e.stencil().lsq_weight.capacity() == 0);
  CHECK(e.data().topo.stencil_twin == nullptr);
  const auto reference =
      autan::kernels::cell_stencil(c.mesh, autan::kernels::pack_geometry(c.mesh, c.geometry),
                                   autan::kernels::StencilKind::face2);
  for (int call = 0; call < 2; ++call) {
    const auto twin = e.stencil_twin();
    REQUIRE(twin.size() == reference.twin.size());
    CHECK(std::equal(twin.begin(), twin.end(), reference.twin.begin()));
    CHECK(e.data().topo.stencil_twin == twin.data());
  }
}

TEMPLATE_TEST_CASE("a linear velocity along slip walls has the exact mass divergence", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // rho = 1, v = (a x, 0, 0): tangent to the y and z sides (slip), exact state on the x sides.
  // The linear reconstruction is exact, so the mass residual of each cell is a V_c exactly.
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const auto ni = static_cast<std::size_t>(c.mesh.n_interior_faces());
  const double a = 50.0, p = 1e5;
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto z = c.zone(b);
    const double x = c.geometry.face_centroid_x[ni + b];
    kind[b] = (z == "xmin" || z == "xmax") ? BoundaryKind::state : BoundaryKind::slip;
    const std::array<double, 5> w = {1.0, a * x, 0.0, 0.0, p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
  }
  Scheme s = second_order(Flux::roe, 0.3);
  s.limiter = false;
  Euler<P> e(c.mesh, c.geometry, air, s);
  e.set_boundary(kind, data);
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const auto uc = conserved({1.0, {a * c.geometry.cell_centroid_x[i], 0.0, 0.0}, p});
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = uc[k];
  }
  const auto st = stored(e, u);
  std::vector<typename P::residual> r(5 * nc);
  e.residual(st, r);
  const auto surface = c.cell_surface();
  for (std::size_t i = 0; i < nc; ++i) {
    const double v = c.geometry.cell_volume[i];
    // Round-off of the face mass fluxes, each at most a |x| <= a times the face area.
    CHECK(std::abs(static_cast<double>(r[i]) - a * v) <= 64 * eps<P>() * a * surface[i]);
  }
}

TEST_CASE("the local time step bounds the faster side of every face", "[euler]") {
  // Opposing flows in neighbouring cells: sum over the faces of max over both sides of
  // (|v . n| + c) A, computed here from the mesh; slip walls mirror the cell.
  using P = autan::kernels::fp64;
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const auto ni = static_cast<std::size_t>(c.mesh.n_interior_faces());
  Euler<P> e(c.mesh, c.geometry, air, second_order(Flux::roe, 0.5));
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::slip), std::vector<double>(5 * nb));
  std::vector<Prim> w(nc);
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const double s = i % 2 == 0 ? 1.0 : -1.0;
    w[i] = {1.0 + 0.1 * static_cast<double>(i % 3), {1000.0 * s, 300.0 * s, -200.0 * s}, 1e5};
    const auto uc = conserved(w[i]);
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = uc[k];
  }
  const auto st = stored(e, u);
  std::vector<P::residual> r(5 * nc);
  e.residual(st, r);
  std::vector<P::state> lambda(nc);
  e.spectral_radius(lambda);
  auto speed = [&](std::size_t i, std::size_t f) {
    const auto& g = c.geometry;
    const double n[3] = {g.face_area_x[f] / g.face_area[f], g.face_area_y[f] / g.face_area[f],
                         g.face_area_z[f] / g.face_area[f]};
    const double vn = w[i].v[0] * n[0] + w[i].v[1] * n[1] + w[i].v[2] * n[2];
    return (std::abs(vn) + sound(w[i])) * g.face_area[f];
  };
  std::vector<double> expected(nc, 0.0);
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(c.mesh.owner()[f]);
    const auto n = static_cast<std::size_t>(c.mesh.neighbour()[f]);
    const double m = std::max(speed(o, f), speed(n, f));
    expected[o] += m;
    expected[n] += m;
  }
  for (std::size_t f = ni; f < ni + nb; ++f) {
    expected[static_cast<std::size_t>(c.mesh.owner()[f])] +=
        speed(static_cast<std::size_t>(c.mesh.owner()[f]), f);
  }
  for (std::size_t i = 0; i < nc; ++i) {
    CHECK(std::abs(lambda[i] - expected[i]) <= 1e-12 * expected[i]);
  }
}

TEMPLATE_TEST_CASE("a linear flow through a thin tetrahedron has the exact mass divergence",
                   "[euler]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // One right tetrahedron with edges 1, h, h (aspect ratio 1/h up to 1e4), every face at the
  // exact state of rho = 1, v = (a x, 0, 0): its four faces determine the linear reconstruction,
  // which is then exact, so the mass residual is a V. Thin cells keep a full gradient.
  using P = TestType;
  const double h = GENERATE(5e-2, 5e-4, 1e-4);
  INFO("h " << h);
  std::random_device random;
  const fs::path file =
      fs::temp_directory_path() / ("autan-thin-tet-" + std::to_string(random()) + ".msh");
  {
    std::ofstream out(file);
    out.precision(17);
    out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        << "$PhysicalNames\n2\n2 1 \"walls\"\n3 101 \"fluid\"\n$EndPhysicalNames\n"
        << "$Entities\n0 0 1 1\n1 0 0 0 1 " << h << " " << h << " 1 1 0\n1 0 0 0 1 " << h << " "
        << h << " 1 101 1 1\n$EndEntities\n"
        << "$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n"
        << "0 0 0\n1 0 0\n0 " << h << " 0\n0 0 " << h << "\n$EndNodes\n"
        << "$Elements\n2 5 1 5\n2 1 2 4\n1 1 3 2\n2 1 2 4\n3 2 3 4\n4 1 4 3\n"
        << "3 1 4 1\n5 1 2 3 4\n$EndElements\n";
  }
  const Case c(file.c_str());
  REQUIRE(c.mesh.n_cells() == 1);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const double a = 50.0, p = 1e5;
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> w = {1.0, a * c.geometry.face_centroid_x[b], 0.0, 0.0, p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
  }
  Scheme s = second_order(Flux::roe, 0.3);
  s.limiter = false;
  s.gradient_fit = autan::physics::GradientFit::quadratic;
  Euler<P> e(c.mesh, c.geometry, air, s);
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::state), data);
  const auto u = stored(e, uniform(Prim{1.0, {a * c.geometry.cell_centroid_x[0], 0.0, 0.0}, p}, 1));
  std::vector<typename P::residual> r(5);
  e.residual(u, r);
  const double v = c.geometry.cell_volume[0];
  CHECK(std::abs(static_cast<double>(r[0]) - a * v) <= 64 * eps<P>() * a * c.cell_surface()[0]);
  // Its only neighbours are its four faces: too few for the quadratic fit, so it keeps the linear
  // one, and says so.
  const auto linear = e.linear_cells();
  CHECK(linear.count == 1);
  CHECK(linear.first == 0);
}

TEMPLATE_TEST_CASE("a one-cell-thick slab keeps a flow without spanwise velocity planar", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // n x n perturbed hexahedra in x-z, one layer along y as thick as 1e-3 to 10 in-plane cells:
  // slip (symmetry) on the two y planes, the flow's state on the sides. Some top nodes sit a few
  // ulps off the top plane, as a mesh writer's rounding leaves them, so the cell centroids differ
  // across the layer by round-off only: no stencil of cells determines a y gradient, and the
  // planar flow below must leave the y-momentum residual at round-off.
  using P = TestType;
  const int n = 8;
  const double thickness = GENERATE(1e-3, 1.0, 10.0) / n;
  INFO("thickness " << thickness);
  std::random_device random;
  const fs::path file =
      fs::temp_directory_path() / ("autan-one-layer-slab-" + std::to_string(random()) + ".msh");
  {
    const int layer = (n + 1) * (n + 1);
    auto node = [&](int i, int k, int j) { return 1 + j * layer + k * (n + 1) + i; };
    std::ofstream out(file);
    out.precision(17);
    out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$PhysicalNames\n3\n2 1 \"planes\"\n"
        << "2 2 \"sides\"\n3 101 \"fluid\"\n$EndPhysicalNames\n$Entities\n0 0 2 1\n"
        << "1 0 0 0 1 " << 1.1 * thickness << " 1 1 1 0\n2 0 0 0 1 " << 1.1 * thickness
        << " 1 1 2 0\n1 0 0 0 1 " << 1.1 * thickness << " 1 1 101 2 1 2\n"
        << "$EndEntities\n$Nodes\n1 " << 2 * layer << " 1 " << 2 * layer << "\n3 1 0 " << 2 * layer
        << "\n";
    for (int t = 1; t <= 2 * layer; ++t) out << t << "\n";
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k <= n; ++k) {
        for (int i = 0; i <= n; ++i) {
          const bool inner = i > 0 && i < n && k > 0 && k < n;
          const double dx = inner ? 0.3 / n * std::sin(1.7 * i + 2.9 * k) : 0.0;
          const double dz = inner ? 0.3 / n * std::cos(2.3 * i + 1.1 * k) : 0.0;
          const double top =
              (i + 2 * k) % 3 == 0
                  ? thickness + 8 * std::numeric_limits<double>::epsilon() * thickness
                  : thickness;
          const double y = j == 0 ? 0.0 : top;
          out << double(i) / n + dx << " " << y << " " << double(k) / n + dz << "\n";
        }
      }
    }
    out << "$EndNodes\n";
    std::vector<std::array<int, 4>> planes, sides;
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < n; ++i) {
        planes.push_back(
            {node(i, k, 0), node(i + 1, k, 0), node(i + 1, k + 1, 0), node(i, k + 1, 0)});
        planes.push_back(
            {node(i, k, 1), node(i, k + 1, 1), node(i + 1, k + 1, 1), node(i + 1, k, 1)});
      }
    }
    for (int t = 0; t < n; ++t) {
      sides.push_back({node(t, 0, 0), node(t, 0, 1), node(t + 1, 0, 1), node(t + 1, 0, 0)});
      sides.push_back({node(t, n, 0), node(t + 1, n, 0), node(t + 1, n, 1), node(t, n, 1)});
      sides.push_back({node(0, t, 0), node(0, t + 1, 0), node(0, t + 1, 1), node(0, t, 1)});
      sides.push_back({node(n, t, 0), node(n, t, 1), node(n, t + 1, 1), node(n, t + 1, 0)});
    }
    const auto total = planes.size() + sides.size() + static_cast<std::size_t>(n * n);
    out << "$Elements\n3 " << total << " 1 " << total << "\n";
    std::size_t tag = 1;
    for (const auto& [entity, quads] : {std::pair{1, &planes}, std::pair{2, &sides}}) {
      out << "2 " << entity << " 3 " << quads->size() << "\n";
      for (const auto& q : *quads) {
        out << tag++ << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
      }
    }
    out << "3 1 5 " << n * n << "\n";
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < n; ++i) {
        // Bottom quad counter-clockwise seen from +y, where the top layer is.
        const std::array<int, 4> q{node(i, k, 0), node(i, k + 1, 0), node(i + 1, k + 1, 0),
                                   node(i + 1, k, 0)};
        out << tag++;
        for (const int v : q) out << " " << v;
        for (const int v : q) out << " " << v + layer;
        out << "\n";
      }
    }
    out << "$EndElements\n";
  }
  const Case c(file.c_str());
  REQUIRE(c.mesh.n_cells() == n * n);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const auto ni = static_cast<std::size_t>(c.mesh.n_interior_faces());
  const double p0 = 1e5, speed = 100.0;
  const auto flow = [&](double x, double z) {
    return Prim{1.0 + 0.2 * std::sin(3.0 * x + 2.0 * z),
                {speed * std::cos(2.0 * z), 0.0, speed * std::sin(2.0 * x)},
                p0 * (1.0 + 0.1 * std::cos(3.0 * x - z))};
  };
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    if (c.zone(b) != "sides") {
      kind[b] = BoundaryKind::slip;
      continue;
    }
    kind[b] = BoundaryKind::state;
    const auto w = flow(c.geometry.face_centroid_x[ni + b], c.geometry.face_centroid_z[ni + b]);
    const std::array<double, 5> v{w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  Euler<P> e(c.mesh, c.geometry, air, second_order(Flux::roe, 0.3));
  e.set_boundary(kind, data);
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const auto uc = conserved(flow(c.geometry.cell_centroid_x[i], c.geometry.cell_centroid_z[i]));
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = uc[k];
  }
  const auto st = stored(e, u);
  std::vector<typename P::residual> r(5 * nc);
  e.residual(std::span<const typename P::state>(st), r);
  const auto surface = c.cell_surface();
  for (std::size_t i = 0; i < nc; ++i) {
    INFO("cell " << i);
    // The y planes carry p A each way: their round-off bounds the y-momentum residual.
    CHECK(std::abs(static_cast<double>(r[2 * nc + i])) <= 64 * eps<P>() * 1.2 * p0 * surface[i]);
  }
}

// In every cell the reconstruction's gradient is that of a quadratic fit to the values at cell and
// face centroids (docs/design/euler.md, Quadratic gradient): with a density that is a quadratic
// field, the solver's density gradient is the field's exact gradient at each cell centroid, where
// the linear least-squares fit is off by O(h) next to the boundary.
TEMPLATE_TEST_CASE("the quadratic gradient of a quadratic field is exact in every cell", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  auto phi = [](const std::array<double, 3>& x) {
    return 10.0 + 2.0 * x[0] - 3.0 * x[1] + x[2] + 0.5 * x[0] * x[0] + x[0] * x[1] -
           2.0 * x[1] * x[2] + 1.5 * x[2] * x[2] - 0.75 * x[1] * x[1] + 0.25 * x[0] * x[2];
  };
  auto exact = [](const std::array<double, 3>& x) {
    return std::array<double, 3>{2.0 + x[0] + x[1] + 0.25 * x[2],
                                 -3.0 + x[0] - 2.0 * x[2] - 1.5 * x[1],
                                 1.0 - 2.0 * x[1] + 3.0 * x[2] + 0.25 * x[0]};
  };
  const std::array<double, 3> v{80.0, -10.0, 5.0};
  constexpr double p = 1e5;
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const std::array<double, 5> w{
        phi({g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]}), v[0], v[1], v[2],
        p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
  }
  Scheme scheme = second_order(Flux::roe, 0.3);
  scheme.limiter = false;
  scheme.gradient_fit = autan::physics::GradientFit::quadratic;
  Euler<P> e(mesh, g, air, scheme);
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::state), data);
  std::vector<double> value(n), u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    value[c] = phi({g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]});
    const auto uc = conserved({value[c], v, p});
    for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = uc[k];
  }
  std::vector<typename P::residual> r(5 * n);
  e.residual(stored(e, u), r);
  REQUIRE(e.linear_cells().count == 0);
  const auto grad = e.arrays().grad;
  // The round-off the fit carries: sum_k |w_k| (|phi_k| + |phi_c|), from its weights.
  const auto band = e.quadratic_gradient();
  const auto nst = band.entries.size();
  double worst = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    REQUIRE(band.fitted[c] == 1.0);
    double magnitude = 0.0;
    for (auto k = static_cast<std::size_t>(band.offsets[c]);
         k < static_cast<std::size_t>(band.offsets[c + 1]); ++k) {
      const auto en = band.entries[k];
      const auto f = static_cast<std::size_t>(-en - 1);
      const double pk =
          en >= 0 ? value[static_cast<std::size_t>(en)]
                  : phi({g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]});
      for (std::size_t j = 0; j < 3; ++j) {
        magnitude += std::abs(band.weights[j * nst + k]) * (std::abs(pk) + std::abs(value[c]));
      }
    }
    const auto expected = exact({g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]});
    for (std::size_t j = 0; j < 3; ++j) {
      const double error = std::abs(static_cast<double>(grad[j * n + c]) - expected[j]) / magnitude;
      if (!std::isnan(worst) && !(error <= worst)) worst = error;  // NaN sticks
    }
  }
  // Round-off of the stored state, of the weights (QR in scaled coordinates, then stored in the
  // policy's geometry type) and of the sum, relative to its terms.
  CHECK(worst <= 1024 * eps<P>());
}

// Under the quadratic gradient the linear one stays available (viscous terms that keep it), the
// same bits as the linear fit's own.
TEMPLATE_TEST_CASE("the linear gradient is the same under either fit", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> w{1.2, 80.0, -10.0, 5.0, 1e5};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = w[j];
  }
  Scheme s = second_order(Flux::roe, 0.3);
  Euler<P> face2(c.mesh, c.geometry, air, s);
  s.gradient_fit = autan::physics::GradientFit::quadratic;
  Euler<P> quadratic(c.mesh, c.geometry, air, s);
  const std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  face2.set_boundary(kind, data);
  quadratic.set_boundary(kind, data);
  auto u = uniform(Prim{1.2, {80.0, -10.0, 5.0}, 1e5}, nc);
  for (std::size_t i = 0; i < u.size(); ++i)
    u[i] *= 1.0 + 0.01 * std::sin(0.7 * static_cast<double>(i));
  const auto st = stored(face2, u);
  std::vector<typename P::residual> r(5 * nc);
  face2.residual(std::span<const typename P::state>(st), r);
  quadratic.residual(std::span<const typename P::state>(st), r);
  std::vector<typename P::grad> a(15 * nc), b(15 * nc);
  face2.linear_gradient(a);
  quadratic.linear_gradient(b);
  CHECK(a == b);
  CHECK(quadratic.linear_cells().count == 0);
  CHECK(face2.quadratic_gradient().entries.empty());
}

TEST_CASE("the point-vortex farfield state is the freestream plus the Prandtl-Glauert vortex",
          "[euler]") {
  // Velocity: the gradient of the compressible vortex potential phi = -Gamma/(2 pi)
  // atan2(beta eta, xi), (xi, eta) the freestream and lift directions of the x-z plane
  // (Thomas and Salas 1986), by central differences on a circle of radius R; total enthalpy and
  // entropy: those of the freestream.
  const double gamma = 1.4, aoa = 0.3, circulation = 2.5, radius = 7.0;
  const double mach = GENERATE(1e-3, 0.5, 0.8);
  CAPTURE(mach);
  const double rho = 1.2, p = 1.0e5, c = std::sqrt(gamma * p / rho), speed = mach * c;
  const std::array<double, 5> w{rho, speed * std::cos(aoa), 0.0, speed * std::sin(aoa), p};
  const double beta = std::sqrt(1.0 - mach * mach);
  auto phi = [&](double x, double z) {
    const double xi = std::cos(aoa) * x + std::sin(aoa) * z;
    const double eta = -std::sin(aoa) * x + std::cos(aoa) * z;
    return -circulation / (2.0 * std::numbers::pi) * std::atan2(beta * eta, xi);
  };
  const double scale = circulation / (2.0 * std::numbers::pi * radius * beta);
  for (int k = 0; k < 16; ++k) {
    // Off the branch cut of atan2 (eta = 0, xi < 0).
    const double theta = aoa + 2.0 * std::numbers::pi * (k + 0.3) / 16.0;
    const double x = radius * std::cos(theta), z = radius * std::sin(theta);
    CAPTURE(theta);
    const double h = 1e-5 * radius;
    const double du = (phi(x + h, z) - phi(x - h, z)) / (2.0 * h);
    const double dw = (phi(x, z + h) - phi(x, z - h)) / (2.0 * h);
    const auto b = autan::physics::point_vortex_state(w, gamma, aoa, circulation, x, z);
    CHECK(std::abs(b[1] - w[1] - du) <= 1e-8 * scale);
    CHECK(b[2] == 0.0);
    CHECK(std::abs(b[3] - w[3] - dw) <= 1e-8 * scale);
    const double eps = std::numeric_limits<double>::epsilon();
    auto enthalpy = [&](const std::array<double, 5>& s) {
      return gamma / (gamma - 1.0) * s[4] / s[0] + 0.5 * (s[1] * s[1] + s[2] * s[2] + s[3] * s[3]);
    };
    CHECK(std::abs(enthalpy(b) - enthalpy(w)) <= 64 * eps * enthalpy(w));
    CHECK(std::abs(b[4] / std::pow(b[0], gamma) - p / std::pow(rho, gamma)) <=
          64 * eps * p / std::pow(rho, gamma));
  }
}

TEMPLATE_TEST_CASE("the limited residual is that of the generated loops run one after another",
                   "[euler]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // The reference runs the loops in the scheme's order on a second instance's arrays.
  namespace ek = autan::kernels::euler;
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  INFO(file);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const Prim w{1.2, {100.0, 30.0, 10.0}, 1e5};
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> v = {w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  // Jumps across x = 0.5 in every primitive, which the limiter must clip there, and a source.
  std::vector<double> u(5 * nc), source(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const bool left = c.geometry.cell_centroid_x[i] < 0.5;
    const double f = left ? 2.0 : 1.0;
    const std::array<double, 3> v = {left ? 60.0 : w.v[0], left ? -20.0 : w.v[1],
                                     left ? 40.0 : w.v[2]};
    const auto q = conserved({f * w.rho, v, f * w.p});
    for (std::size_t k = 0; k < 5; ++k) {
      u[k * nc + i] = q[k];
      source[k * nc + i] = 1e-3 * q[k];
    }
  }
  Scheme s = second_order(Flux::roe, 0.3);
  s.venkat_k = 0.5;
  Euler<P> e(c.mesh, c.geometry, air, s);
  e.set_boundary(kind, data);
  e.set_source(source);
  const auto input = stored(e, u);
  auto state = input;
  std::vector<typename P::residual> r(5 * nc, typename P::residual(7));
  e.residual(state, r);
  REQUIRE(state == input);
  std::array<std::size_t, 5> limited{};
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < nc; ++i) {
      limited[k] += static_cast<double>(e.data().phi[k * nc + i]) < 1.0 ? 1U : 0U;
    }
    INFO("component " << k);
    REQUIRE(limited[k] > 0);
  }

  Euler<P> f(c.mesh, c.geometry, air, s);
  f.set_boundary(kind, data);
  f.set_source(source);
  auto d = f.data();
  auto reference_state = input;
  std::vector<typename P::residual> ref(5 * nc, typename P::residual(7));
  d.u = reference_state.data();
  d.res = ref.data();
  ek::cpu::primitives(d);
  ek::cpu::boundary_states(d);
  ek::cpu::gradients(d);
  ek::cpu::limiter(d);
  ek::cpu::sources(d);
  ek::cpu::roe_faces(d);
  ek::cpu::roe_boundaries(d);
  CHECK(std::memcmp(r.data(), ref.data(), r.size() * sizeof(r[0])) == 0);
}

TEMPLATE_TEST_CASE("the face part of the residual and the face fluxes do not depend on the source",
                   "[euler]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // The conservation check's face sums (docs/design/euler.md, Conservation check): on one state,
  // a zero and a constant non-zero source in every equation give the same bits, through the
  // residual's loops and the steady step's fused Jacobian loops alike, and every face's flux is
  // written.
  using P = TestType;
  using R = typename P::residual;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool jacobian = GENERATE(false, true);
  INFO(file << " flux " << static_cast<int>(flux) << " jacobian " << jacobian);
  const Case c(file);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const auto nf = static_cast<std::size_t>(c.mesh.n_faces());
  const Prim w{1.2, {100.0, 30.0, 10.0}, 1e5};
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> v = {w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  std::vector<double> u(5 * nc), zero(5 * nc, 0.0), source(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const double f = c.geometry.cell_centroid_x[i] < 0.5 ? 2.0 : 1.0;
    const auto q = conserved({f * w.rho, w.v, f * w.p});
    for (std::size_t k = 0; k < 5; ++k) {
      u[k * nc + i] = q[k];
      source[k * nc + i] = 0.25 * static_cast<double>(k + 1);
    }
  }
  Euler<P> e(c.mesh, c.geometry, air, second_order(flux, 0.3));
  e.set_boundary(kind, data);
  e.set_planar(e.all_planar());
  const auto s = stored(e, u);
  std::vector<R> faces0(5 * nc), fluxes0(5 * nf, R(0)), faces1(5 * nc), fluxes1(5 * nf, R(7));
  e.face_parts(s, faces0, fluxes0, jacobian);
  e.set_source(source);
  e.face_parts(s, faces1, fluxes1, jacobian);
  CHECK(std::memcmp(faces0.data(), faces1.data(), faces0.size() * sizeof(R)) == 0);
  CHECK(std::memcmp(fluxes0.data(), fluxes1.data(), fluxes0.size() * sizeof(R)) == 0);
  std::vector<R> with(5 * nc), without(5 * nc);
  e.residual(s, with);
  e.set_source(zero);
  e.residual(s, without);
  CHECK(with != without);
}

TEMPLATE_TEST_CASE("a cell's face part sums the fluxes its faces are written with", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // Under global keys every cell evaluates its own faces (the gather strategies), so an interior
  // face is evaluated once from each cell. Each cell's face part must be, bit for bit, the sum of
  // the write-only loops' fluxes in the cell's global face order, interior faces first, signed by
  // the cell's side, as the gather adds them. A sum can hide compensating differences: this
  // checks the cells' sums, not each evaluation.
  using P = TestType;
  using R = typename P::residual;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool jacobian = GENERATE(false, true);
  INFO(file << " flux " << static_cast<int>(flux) << " jacobian " << jacobian);
  const Case c(file);
  const auto ni = static_cast<std::size_t>(c.mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  const auto nf = ni + nb;
  std::vector<std::int64_t> face_id(nf);
  std::iota(face_id.begin(), face_id.end(), std::int64_t{0});
  const std::vector<std::uint8_t> not_reversed(nf, 0);
  const autan::kernels::GlobalKeys keys{
      .cell = c.mesh.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};
  const Prim w{1.2, {100.0, 30.0, 10.0}, 1e5};
  std::vector<BoundaryKind> kind(nb, BoundaryKind::farfield);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const std::array<double, 5> v = {w.rho, w.v[0], w.v[1], w.v[2], w.p};
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const double x = c.geometry.cell_centroid_x[i], y = c.geometry.cell_centroid_y[i];
    const auto q = conserved(
        {w.rho * (1.0 + 0.3 * x), {w.v[0], w.v[1] * (1.0 - y), w.v[2]}, w.p * (1.0 + 0.2 * x * y)});
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = q[k];
  }
  Euler<P> e(c.mesh, c.geometry, air, second_order(flux, 0.3), &keys);
  e.set_boundary(kind, data);
  e.set_planar(e.all_planar());
  const auto s = stored(e, u);
  std::vector<R> faces(5 * nc), fluxes(5 * nf);
  e.face_parts(s, faces, fluxes, jacobian);
  const auto order = autan::kernels::global_order(c.mesh, keys);
  const auto offsets = c.mesh.cell_face_offsets();
  const auto owner = c.mesh.owner();
  std::vector<R> expected(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    for (std::size_t k = 0; k < 5; ++k) {
      R interior = 0, boundary = 0;
      for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
        const auto f = static_cast<std::size_t>(order.cell_faces[static_cast<std::size_t>(j)]);
        const R value = fluxes[k * nf + f];
        if (f < ni) {
          interior += static_cast<std::size_t>(owner[f]) == i ? value : -value;
        } else {
          boundary += value;
        }
      }
      expected[k * nc + i] = (R(0) + interior) + boundary;
    }
  }
  CHECK(std::memcmp(expected.data(), faces.data(), faces.size() * sizeof(R)) == 0);
}

namespace {

// A stencil entry by global keys: a cell (0, global id, translation) or a boundary face
// (1, global face id, translation of its cell), with its three LSQ weights.
using StencilKey = std::tuple<int, std::int64_t, std::array<double, 3>>;
using Row = std::vector<std::pair<StencilKey, std::array<double, 3>>>;

// The gradient stencil of every owned cell of `part`, by global cell id, as Euler builds it
// under the global keys of reproducibility "ranks", every boundary face a slip wall. Every
// entry to a ghost must have its twin: the adjoint's partial sums for the reverse exchange.
template <class P>
std::map<std::int64_t, Row> owned_rows(const autan::LocalPartition& part,
                                       autan::physics::GradientStencil gradient) {
  const auto& m = part.mesh;
  const autan::kernels::GlobalKeys keys{.cell = m.cell_global_id(),
                                        .face = part.face_global_id,
                                        .reversed = part.face_reversed,
                                        .shift = m.ghost_translation()};
  Scheme s = second_order(Flux::roe, 0.5);
  s.gradient = gradient;
  Euler<P> e(m, part.geometry, air, s, &keys, part.ghosts);
  const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::slip),
                 std::vector<double>(5 * nb, 0.0));
  const auto& st = e.stencil();
  const auto twin = e.stencil_twin();
  const auto* w = e.data().stencil_lsq_weight;
  const auto ns = st.entries.size();
  const auto n_owned = m.n_owned_cells();
  const auto shift = [&](autan::index_t c) {
    return c < n_owned ? std::array<double, 3>{}
                       : m.ghost_translation()[static_cast<std::size_t>(c - n_owned)];
  };
  std::map<std::int64_t, Row> rows;
  for (autan::index_t c = 0; c < n_owned; ++c) {
    Row& row = rows[m.cell_global_id()[static_cast<std::size_t>(c)]];
    const auto uc = static_cast<std::size_t>(c);
    for (auto k = st.offsets[uc]; k < st.offsets[uc + 1]; ++k) {
      const auto uk = static_cast<std::size_t>(k);
      const autan::index_t e_k = st.entries[uk];
      const auto f = static_cast<std::size_t>(-1 - e_k);
      const StencilKey key =
          e_k >= 0 ? StencilKey{0, m.cell_global_id()[static_cast<std::size_t>(e_k)], shift(e_k)}
                   : StencilKey{1, part.face_global_id[f], shift(m.owner()[f])};
      row.push_back({key,
                     {static_cast<double>(w[uk]), static_cast<double>(w[ns + uk]),
                      static_cast<double>(w[2 * ns + uk])}});
      if (e_k >= n_owned) {
        REQUIRE(twin[uk] >= 0);
        CHECK(twin[static_cast<std::size_t>(twin[uk])] == k);
      }
    }
  }
  return rows;
}

// Cells of `m` whose vertex stencil holds an entry (cell or boundary face) its face2 stencil does
// not.
std::size_t cells_beyond_face2(const autan::Mesh& m, const autan::Geometry& g) {
  const auto packed = autan::kernels::pack_geometry(m, g);
  const auto v = autan::kernels::cell_stencil(m, packed, autan::kernels::StencilKind::vertex);
  const auto f = autan::kernels::cell_stencil(m, packed, autan::kernels::StencilKind::face2);
  std::size_t cells = 0;
  for (std::size_t c = 0; c + 1 < v.offsets.size(); ++c) {
    const std::set<autan::index_t> face2(f.entries.begin() + f.offsets[c],
                                         f.entries.begin() + f.offsets[c + 1]);
    bool beyond = false;
    for (auto k = v.offsets[c]; k < v.offsets[c + 1]; ++k) {
      beyond = beyond || !face2.contains(v.entries[static_cast<std::size_t>(k)]);
    }
    cells += beyond ? 1 : 0;
  }
  return cells;
}

// Blocks of the unit cube that meet at corners, numbered modulo n.
std::vector<std::int32_t> blocks(const autan::Geometry& g, std::int32_t n) {
  std::vector<std::int32_t> parts(g.cell_volume.size());
  for (std::size_t c = 0; c < parts.size(); ++c) {
    const auto at = [](double x, int k) { return static_cast<int>(std::floor(x * k)); };
    const int b =
        at(g.cell_centroid_x[c], 3) + 2 * at(g.cell_centroid_y[c], 2) + at(g.cell_centroid_z[c], 2);
    parts[c] = std::clamp(b, 0, 99) % n;
  }
  return parts;
}

autan::Mesh periodic_hex_box(int n) {
  autan::CellwiseMesh cells = autan::test::box(n, autan::test::BoxFill::hexa, 0.2);
  for (int k = 0; k < 2; ++k) {
    cells.face_zones[static_cast<std::size_t>(2 * k)].kind = autan::BcKind::periodic;
    cells.face_zones[static_cast<std::size_t>(2 * k + 1)].kind = autan::BcKind::periodic;
    std::array<double, 3> t{};
    t[static_cast<std::size_t>(k)] = 1.0;
    cells.periodic.push_back({.zone_a = 2 * k, .zone_b = 2 * k + 1, .translation = t});
  }
  return autan::assemble(autan::to_face_based(std::move(cells)));
}

}  // namespace

TEMPLATE_TEST_CASE(
    "on a part holding its ghost stencil, every owned cell's gradient stencil "
    "and weights are those of one rank",
    "[euler][partition]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // A partition whose ghosts hold the scheme's stencil (GhostStencil) gives each owned cell the
  // entries the whole mesh gives it, in the same global order, so its LSQ weights, and the
  // gradient they make, have the same bits on any number of ranks. Blocks of the cube meet at
  // corners, where a vertex neighbour lies three face steps from a part. The periodic boxes, two
  // and three cells wide, give a cell two images of one boundary face. A face2 partition would
  // truncate the vertex stencil (on every mesh but the polyhedral dual, below): Euler refuses it
  // (E0148).
  using P = TestType;
  using autan::physics::GradientStencil;
  const int which = GENERATE(range(0, static_cast<int>(meshes.size()) + 2));
  const auto gradient = GENERATE(GradientStencil::face2, GradientStencil::vertex);
  const autan::Mesh global =
      which < static_cast<int>(meshes.size())
          ? autan::io::read_mesh(mesh_dir / meshes[static_cast<std::size_t>(which)])
          : periodic_hex_box(which - static_cast<int>(meshes.size()) + 2);
  INFO("mesh " << which << " gradient " << static_cast<int>(gradient));
  const autan::Geometry gg = autan::compute_geometry(global);
  const auto ghosts = gradient == GradientStencil::vertex ? autan::GhostStencil::vertex
                                                          : autan::GhostStencil::face2;
  const auto part_of = [&](const std::vector<std::int32_t>& parts, std::int32_t n, std::int32_t r,
                           autan::GhostStencil g) {
    auto part = autan::extract_partition(global, gg, parts, n, r, g);
    // Ghost geometry as the forward exchange gives it: the owner's, an image's centroid moved by
    // its translation.
    auto& lg = part.geometry;
    const auto& m = part.mesh;
    for (autan::index_t c = m.n_owned_cells(); c < m.n_cells(); ++c) {
      const auto u = static_cast<std::size_t>(c);
      const auto o = static_cast<std::size_t>(m.cell_global_id()[u]);
      const auto& t = m.ghost_translation()[static_cast<std::size_t>(c - m.n_owned_cells())];
      lg.cell_volume[u] = gg.cell_volume[o];
      lg.cell_centroid_x[u] = gg.cell_centroid_x[o] + t[0];
      lg.cell_centroid_y[u] = gg.cell_centroid_y[o] + t[1];
      lg.cell_centroid_z[u] = gg.cell_centroid_z[o] + t[2];
    }
    return part;
  };
  const auto reference = owned_rows<P>(
      part_of(std::vector<std::int32_t>(gg.cell_volume.size(), 0), 1, 0, ghosts), gradient);
  REQUIRE(std::ssize(reference) == global.n_cells());
  for (const std::int32_t n : {2, 3, 4}) {
    const auto parts = blocks(gg, n);
    for (std::int32_t r = 0; r < n; ++r) {
      for (const auto& [id, row] : owned_rows<P>(part_of(parts, n, r, ghosts), gradient)) {
        CHECK(row == reference.at(id));
      }
    }
  }
  if (gradient == GradientStencil::vertex) {
    // The gap the ghost stencil closes: on face2 parts, some owned cell's vertex stencil, built
    // directly, misses entries, and Euler refuses such a part.
    const auto parts = blocks(gg, 4);
    bool truncated = false;
    for (std::int32_t r = 0; r < 4; ++r) {
      const auto part = part_of(parts, 4, r, autan::GhostStencil::face2);
      const auto& m = part.mesh;
      const auto st = autan::kernels::cell_stencil(
          m, autan::kernels::pack_geometry(m, part.geometry), autan::kernels::StencilKind::vertex);
      for (autan::index_t c = 0; c < m.n_owned_cells(); ++c) {
        const auto uc = static_cast<std::size_t>(c);
        const auto size = static_cast<std::size_t>(st.offsets[uc + 1] - st.offsets[uc]);
        truncated = truncated || size < reference.at(m.cell_global_id()[uc]).size();
      }
      if (m.n_cells() > m.n_owned_cells()) {
        try {
          Scheme s = second_order(Flux::roe, 0.5);
          s.gradient = GradientStencil::vertex;
          Euler<P> e(m, part.geometry, air, s, nullptr, part.ghosts);
          FAIL("expected E0148");
        } catch (const autan::Error& e) {
          CHECK(e.code() == autan::ErrorCode::partition_stencil_not_local);
        }
      }
    }
    // The polyhedral dual mesh cannot show the power: every cell's vertex stencil lies inside its
    // face2 stencil, so no partition truncates it. That measured fact is asserted, so that a
    // change of the mesh or of the stencils brings the power check back for it. The hanging-node
    // polyhedron goes beyond face2 but fills half its box, leaving the declared grid partitions
    // empty: it is not used (#606, #630).
    if (which < static_cast<int>(meshes.size()) &&
        std::string(meshes[static_cast<std::size_t>(which)]) == "poly/polyMesh") {
      CHECK(cells_beyond_face2(global, gg) == 0);
    } else {
      CHECK(truncated);
    }
  }
}

namespace {

// The viscous boundary fit of the owned cells (and, with `layer1`, of their ghost face neighbours)
// of a Navier-Stokes physics built on `part` under its global keys, every boundary face an
// isothermal wall, or with `ymin_only` zone ymin alone and the others slip walls: per cell by
// global id, whether it carries a fit, and its stencil entries by global key (StencilKey) with
// their 19 weights.
using FitRow = std::pair<bool, std::vector<std::pair<StencilKey, std::array<double, 19>>>>;

template <class P>
std::map<std::int64_t, FitRow> fit_rows(const autan::LocalPartition& part, bool layer1,
                                        bool ymin_only) {
  const auto& m = part.mesh;
  const autan::kernels::GlobalKeys keys{.cell = m.cell_global_id(),
                                        .face = part.face_global_id,
                                        .reversed = part.face_reversed,
                                        .shift = m.ghost_translation()};
  autan::physics::NavierStokes<P> ns(m, part.geometry, air, second_order(Flux::roe, 0.5),
                                     autan::physics::Transport{}, &keys, part.ghosts);
  const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
  const auto ni = static_cast<std::size_t>(m.n_interior_faces());
  std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto zone = static_cast<std::size_t>(m.face_zone()[ni + b]);
    if (!ymin_only || m.face_zones()[zone].name == "ymin") {
      kind[b] = BoundaryKind::wall;
      data[b] = 300.0;
    }
  }
  ns.set_boundary(kind, data);
  const auto& d = ns.data();
  const auto n_owned = m.n_owned_cells();
  const auto shift = [&](autan::index_t c) {
    return c < n_owned ? std::array<double, 3>{}
                       : m.ghost_translation()[static_cast<std::size_t>(c - n_owned)];
  };
  std::vector<char> listed(static_cast<std::size_t>(m.n_cells()), 0);
  std::fill(listed.begin(), listed.begin() + static_cast<std::ptrdiff_t>(n_owned), 1);
  if (layer1) {
    for (autan::index_t f = 0; f < m.n_interior_faces(); ++f) {
      const auto o = m.owner()[static_cast<std::size_t>(f)];
      const auto n = m.neighbour()[static_cast<std::size_t>(f)];
      if (o < n_owned) listed[static_cast<std::size_t>(n)] = 1;
      if (n < n_owned) listed[static_cast<std::size_t>(o)] = 1;
    }
  }
  const auto ns_entries = static_cast<std::size_t>(d.topo.n_stencil);
  const std::array q{d.q0, d.q1, d.q2, d.q3, d.q4, d.q5, d.q6};
  std::map<std::int64_t, FitRow> rows;
  for (std::size_t c = 0; c < listed.size(); ++c) {
    if (listed[c] == 0) continue;
    FitRow& row = rows[m.cell_global_id()[c]];
    row.first = static_cast<double>(d.quad[c]) > 0.5;
    for (auto k = d.topo.stencil_offsets[c]; k < d.topo.stencil_offsets[c + 1]; ++k) {
      const auto uk = static_cast<std::size_t>(k);
      const autan::index_t e = d.topo.stencil[uk];
      const auto f = static_cast<std::size_t>(-1 - e);
      const StencilKey key =
          e >= 0 ? StencilKey{0, m.cell_global_id()[static_cast<std::size_t>(e)], shift(e)}
                 : StencilKey{1, part.face_global_id[f], shift(m.owner()[f])};
      std::array<double, 19> w{};
      for (std::size_t i = 0; i < 19; ++i) {
        w[i] = static_cast<double>(q[i / 3][(i % 3) * ns_entries + uk]);
      }
      row.second.emplace_back(key, w);
    }
  }
  return rows;
}

}  // namespace

TEMPLATE_TEST_CASE(
    "the owned cells of a part holding the Navier-Stokes ghost stencil and their face neighbours "
    "have the boundary fits of one rank",
    "[navier-stokes][partition]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // The viscous faces of an owned cell read the boundary fits of both their cells, built from up
  // to three vertex rings of each (docs/design/ns-sa.md). A part holding four rings
  // (navier_stokes_ghost_stencil under reproducibility "ranks") gives the owned cells and their
  // ghost face neighbours the
  // fit of the whole mesh: the same flag, entries in the same global order, and weights of the
  // same bits. Every boundary face is a wall on the four meshes. On a stretched one-cell-thick
  // slab, zone ymin alone is, as on a flat plate: the cells at its ends, where two rings hold
  // three positions along it, take a fit over three rings (docs/design/ns-sa.md). A part holding
  // two rings is refused (E0148), with ghosts or not.
  using P = TestType;
  const int which = GENERATE(range(0, static_cast<int>(meshes.size()) + 1));
  const bool ymin_only = which == static_cast<int>(meshes.size());
  INFO((ymin_only ? "stretched slab, wall ymin" : meshes[static_cast<std::size_t>(which)]));
  const autan::Mesh global =
      ymin_only ? autan::test::stretched_slab()
                : autan::io::read_mesh(mesh_dir / meshes[static_cast<std::size_t>(which)]);
  const autan::Geometry gg = autan::compute_geometry(global);
  const auto stencil = autan::physics::navier_stokes_ghost_stencil(true);
  REQUIRE(stencil == autan::GhostStencil::vertex4);
  REQUIRE(autan::physics::navier_stokes_ghost_stencil(false) == autan::GhostStencil::vertex3);
  const auto part_of = [&](const std::vector<std::int32_t>& parts, std::int32_t n, std::int32_t r,
                           autan::GhostStencil g) {
    auto part = autan::extract_partition(global, gg, parts, n, r, g);
    auto& lg = part.geometry;
    const auto& m = part.mesh;
    for (autan::index_t c = m.n_owned_cells(); c < m.n_cells(); ++c) {
      const auto u = static_cast<std::size_t>(c);
      const auto o = static_cast<std::size_t>(m.cell_global_id()[u]);
      const auto& t = m.ghost_translation()[static_cast<std::size_t>(c - m.n_owned_cells())];
      lg.cell_volume[u] = gg.cell_volume[o];
      lg.cell_centroid_x[u] = gg.cell_centroid_x[o] + t[0];
      lg.cell_centroid_y[u] = gg.cell_centroid_y[o] + t[1];
      lg.cell_centroid_z[u] = gg.cell_centroid_z[o] + t[2];
    }
    return part;
  };
  const auto reference =
      fit_rows<P>(part_of(std::vector<std::int32_t>(gg.cell_volume.size(), 0), 1, 0, stencil),
                  false, ymin_only);
  REQUIRE(std::ssize(reference) == global.n_cells());
  // Cells whose fit reaches a third vertex ring, whose stencil the parts must hold whole too.
  const auto vertex = autan::kernels::cell_stencil(
      global, autan::kernels::pack_geometry(global, gg), autan::kernels::StencilKind::vertex);
  std::size_t three_rings = 0;
  for (const auto& [id, row] : reference) {
    std::set<std::int64_t> near{id};
    for (int ring = 0; ring < 2; ++ring) {
      const std::vector<std::int64_t> reached(near.begin(), near.end());
      for (const auto c : reached) {
        const auto uc = static_cast<std::size_t>(c);
        for (auto k = vertex.offsets[uc]; k < vertex.offsets[uc + 1]; ++k) {
          const auto e = vertex.entries[static_cast<std::size_t>(k)];
          if (e >= 0) near.insert(e);
        }
      }
    }
    bool beyond = false;
    for (const auto& entry : row.second) {
      beyond =
          beyond || (std::get<0>(entry.first) == 0 && !near.contains(std::get<1>(entry.first)));
    }
    three_rings += beyond ? 1 : 0;
  }
  INFO("cells fitted over three rings: " << three_rings);
  if (ymin_only) CHECK(three_rings > 0);
  std::size_t compared = 0;
  for (const std::int32_t n : {2, 3, 4}) {
    const auto parts = blocks(gg, n);
    for (std::int32_t r = 0; r < n; ++r) {
      for (const auto& [id, row] : fit_rows<P>(part_of(parts, n, r, stencil), true, ymin_only)) {
        INFO("cell " << id << " on part " << r << " of " << n);
        CHECK(row == reference.at(id));
        compared += row.first ? 1 : 0;
      }
    }
  }
  CHECK(compared > 0);
  const auto parts = blocks(gg, 4);
  for (std::int32_t r = 0; r < 4; ++r) {
    const auto part = part_of(parts, 4, r, autan::GhostStencil::vertex2);
    try {
      autan::physics::NavierStokes<P> ns(part.mesh, part.geometry, air,
                                         second_order(Flux::roe, 0.5), autan::physics::Transport{},
                                         nullptr, part.ghosts);
      FAIL("expected E0148");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::partition_stencil_not_local);
    }
  }
}

TEST_CASE("a part without ghosts refuses a stencil its ghost stencil does not hold", "[euler]") {
  // Two hexahedra sharing one corner node: on two face2 parts neither cell has a ghost, yet each
  // one's vertex stencil on the whole mesh holds the other. A part's declared ghost stencil, not
  // its ghost count, decides: the vertex gradient is E0148 there, face2 runs.
  autan::test::ReferenceCell first = autan::test::affine_cell(
      autan::CellShape::hexa, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0});
  autan::CellwiseMesh cells = std::move(first.mesh);
  std::vector<autan::index_t> nodes{6};  // the first cell's corner (1, 1, 1)
  for (const auto& r : std::vector<std::array<double, 3>>{
           {2, 1, 1}, {2, 2, 1}, {1, 2, 1}, {1, 1, 2}, {2, 1, 2}, {2, 2, 2}, {1, 2, 2}}) {
    cells.x.push_back(r[0]);
    cells.y.push_back(r[1]);
    cells.z.push_back(r[2]);
    nodes.push_back(static_cast<autan::index_t>(cells.x.size() - 1));
  }
  cells.add_cell(autan::CellShape::hexa, nodes, 0);
  const autan::Mesh global = autan::assemble(autan::to_face_based(std::move(cells)));
  const auto part = autan::extract_partition(global, autan::compute_geometry(global),
                                             std::vector<std::int32_t>{0, 1}, 2, 0);
  REQUIRE(part.mesh.n_cells() == part.mesh.n_owned_cells());
  Scheme s = second_order(Flux::roe, 0.5);
  CHECK_NOTHROW(
      Euler<autan::kernels::fp64>(part.mesh, part.geometry, air, s, nullptr, part.ghosts));
  s.gradient = autan::physics::GradientStencil::vertex;
  try {
    Euler<autan::kernels::fp64> e(part.mesh, part.geometry, air, s, nullptr, part.ghosts);
    FAIL("expected E0148");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::partition_stencil_not_local);
  }
  // The Navier-Stokes boundary fits reach the other cell through the node too.
  try {
    autan::physics::NavierStokes<autan::kernels::fp64> ns(
        part.mesh, part.geometry, air, second_order(Flux::roe, 0.5), autan::physics::Transport{},
        nullptr, part.ghosts);
    FAIL("expected E0148");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::partition_stencil_not_local);
  }
}
