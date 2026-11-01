// The Euler residual through autan::physics::Euler on the test meshes of tests/data/mesh.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"

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

template <class P>
std::vector<typename P::state> uniform(const Prim& w, std::size_t n) {
  const auto uc = conserved(w);
  std::vector<typename P::state> u(5 * n);
  for (std::size_t k = 0; k < 5; ++k) {
    std::fill_n(u.begin() + static_cast<std::ptrdiff_t>(k * n), n, typename P::state(uc[k]));
  }
  return u;
}

Scheme second_order(Flux flux, double mach) {
  Scheme s;
  s.flux = flux;
  s.mach_ref = mach;
  s.scale = {1.0, 100.0, 100.0, 100.0, 1e5};
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
    const auto u = uniform<P>(w, nc);
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
    const auto u = uniform<P>(w, nc);
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
  auto u = uniform<P>(w, nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const auto x = c.geometry.cell_centroid_x[i];
    const auto y = c.geometry.cell_centroid_y[i];
    const auto z = c.geometry.cell_centroid_z[i];
    u[i] = typename P::state(1.0 + 0.5 * std::sin(7.0 * x + 3.0 * y) * std::cos(5.0 * z) +
                             (x > 0.5 ? 9.0 : 0.0));
  }
  // A 10:1 density jump: the linear reconstruction would make the density negative at some
  // faces, where the scheme falls back to the cell value.
  const Prim scale{11.0, {0.0, 0.0, 0.0}, 1e5};
  CHECK(relative_residual(e, c, std::span<const typename P::state>(u), scale) <= 64 * eps<P>());
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
  auto u = uniform<P>(w, nc);
  std::vector<double> du(5 * nc);
  const auto uc = conserved(w);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t i = 0; i < nc; ++i) {
      const double x = c.geometry.cell_centroid_x[i];
      const double y = c.geometry.cell_centroid_y[i];
      u[k * nc + i] *= 1.0 + 0.1 * std::sin(3.0 * x + static_cast<double>(k)) * std::cos(2.0 * y);
      du[k * nc + i] = uc[k] * std::cos(5.0 * x + 2.0 * y + static_cast<double>(k));
    }
  }
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
  const std::vector<typename P::state> st(u.begin(), u.end());
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
  const std::vector<P::state> st(u.begin(), u.end());
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

TEMPLATE_TEST_CASE("a one-cell-thick slab keeps a flow without spanwise velocity planar", "[euler]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // n x n perturbed hexahedra in x-z, one layer 0.1 thick along y, slip on every side. The cell
  // centroids' y agree only to round-off, so no stencil determines a y gradient: the planar flow
  // below must leave the y-momentum residual at round-off (issue #161).
  using P = TestType;
  const int n = 8;
  const fs::path file = fs::temp_directory_path() / "autan-one-layer-slab.msh";
  {
    auto node = [&](int i, int k, int j) { return 1 + j * (n + 1) * (n + 1) + k * (n + 1) + i; };
    std::ofstream out(file);
    out.precision(17);
    const int nodes = 2 * (n + 1) * (n + 1);
    out << "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        << "$PhysicalNames\n2\n2 1 \"walls\"\n3 101 \"fluid\"\n$EndPhysicalNames\n"
        << "$Entities\n0 0 1 1\n1 0 0 0 1 0.1 1 1 1 0\n1 0 0 0 1 0.1 1 1 101 1 1\n$EndEntities\n"
        << "$Nodes\n1 " << nodes << " 1 " << nodes << "\n3 1 0 " << nodes << "\n";
    for (int t = 1; t <= nodes; ++t) out << t << "\n";
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k <= n; ++k) {
        for (int i = 0; i <= n; ++i) {
          const bool inner = i > 0 && i < n && k > 0 && k < n;
          const double dx = inner ? 0.3 / n * std::sin(1.7 * i + 2.9 * k) : 0.0;
          const double dz = inner ? 0.3 / n * std::cos(2.3 * i + 1.1 * k) : 0.0;
          out << double(i) / n + dx << " " << 0.1 * j << " " << double(k) / n + dz << "\n";
        }
      }
    }
    out << "$EndNodes\n";
    std::vector<std::array<int, 4>> quads;
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < n; ++i) {
        quads.push_back({node(i, k, 0), node(i, k + 1, 0), node(i + 1, k + 1, 0), node(i + 1, k, 0)});
        quads.push_back({node(i, k, 1), node(i + 1, k, 1), node(i + 1, k + 1, 1), node(i, k + 1, 1)});
      }
    }
    for (int t = 0; t < n; ++t) {
      quads.push_back({node(t, 0, 0), node(t + 1, 0, 0), node(t + 1, 0, 1), node(t, 0, 1)});
      quads.push_back({node(t, n, 0), node(t, n, 1), node(t + 1, n, 1), node(t + 1, n, 0)});
      quads.push_back({node(0, t, 0), node(0, t, 1), node(0, t + 1, 1), node(0, t + 1, 0)});
      quads.push_back({node(n, t, 0), node(n, t + 1, 0), node(n, t + 1, 1), node(n, t, 1)});
    }
    const auto nq = quads.size();
    const auto nh = static_cast<std::size_t>(n * n);
    out << "$Elements\n2 " << nq + nh << " 1 " << nq + nh << "\n2 1 3 " << nq << "\n";
    std::size_t tag = 1;
    for (const auto& q : quads) out << tag++ << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    out << "3 1 5 " << nh << "\n";
    for (int k = 0; k < n; ++k) {
      for (int i = 0; i < n; ++i) {
        out << tag++ << " " << node(i, k, 0) << " " << node(i + 1, k, 0) << " "
            << node(i + 1, k + 1, 0) << " " << node(i, k + 1, 0) << " " << node(i, k, 1) << " "
            << node(i + 1, k, 1) << " " << node(i + 1, k + 1, 1) << " " << node(i, k + 1, 1) << "\n";
      }
    }
    out << "$EndElements\n";
  }
  const Case c(file.c_str());
  REQUIRE(c.mesh.n_cells() == n * n);
  const auto nb = static_cast<std::size_t>(c.mesh.n_boundary_faces());
  const auto nc = static_cast<std::size_t>(c.mesh.n_cells());
  Euler<P> e(c.mesh, c.geometry, air, second_order(Flux::roe, 0.3));
  e.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::slip), std::vector<double>(5 * nb));
  const double p0 = 1e5, speed = 100.0;
  std::vector<double> u(5 * nc);
  for (std::size_t i = 0; i < nc; ++i) {
    const double x = c.geometry.cell_centroid_x[i], z = c.geometry.cell_centroid_z[i];
    const auto uc = conserved({1.0 + 0.2 * std::sin(3.0 * x + 2.0 * z),
                               {speed * std::cos(2.0 * z), 0.0, speed * std::sin(2.0 * x)},
                               p0 * (1.0 + 0.1 * std::cos(3.0 * x - z))});
    for (std::size_t k = 0; k < 5; ++k) u[k * nc + i] = uc[k];
  }
  const std::vector<typename P::state> st(u.begin(), u.end());
  std::vector<typename P::residual> r(5 * nc);
  e.residual(st, r);
  const auto surface = c.cell_surface();
  for (std::size_t i = 0; i < nc; ++i) {
    // The y faces carry p A each way: their round-off bounds the y-momentum residual.
    CHECK(std::abs(static_cast<double>(r[2 * nc + i])) <= 64 * eps<P>() * 1.2 * p0 * surface[i]);
  }
}
