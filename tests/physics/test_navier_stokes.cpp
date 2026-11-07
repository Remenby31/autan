// The viscous fluxes of autan::physics::NavierStokes and the boundary fluxes of
// autan::physics::SpalartAllmaras, on the test meshes of tests/data/mesh and generated slabs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::offset_t;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Gas;
using autan::physics::NavierStokes;
using autan::physics::Scheme;
using autan::physics::SpalartAllmaras;
using autan::physics::Transport;
using Catch::Matchers::WithinRel;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

constexpr Gas air{1.4, 287.0};
constexpr double mu = 0.5;
constexpr double prandtl = 0.72;
constexpr double rho = 1.2;
// Velocity y a, zero on the plane y = 0; pressure linear in x and z, so that with a uniform density
// the temperature is linear and independent of y (no heat flux through y = 0).
constexpr std::array<double, 3> a{30.0, 6.0, -10.0};

std::array<double, 5> primitive(double x, double y, double z) {
  return {rho, a[0] * y, a[1] * y, a[2] * y, 1.0e5 + 2000.0 * x + 500.0 * z};
}

struct Slab {
  double first = 1.0e-5;  // spacing in y at y = 0
  int nz = 1;             // layers in z
  double angle = 0.0;     // turned about z (radians)
  double shear = 0.0;     // x shifted by shear y
  double warp = 0.0;      // the nodes of y = 0 at y = warp (2 z - 1) x: the wall's faces warped
};

// Hexahedra over [0,1]^3, 8 along x, 12 along y with the spacing growing geometrically from
// `first` at y = 0 (1e-5: aspect ratio 10^4 at the wall, as in the wall-resolved 2D grids of the
// NASA TMR), `nz` along z; zones xmin ... ymax and `sides` for z = 0 and z = 1.
autan::Mesh stretched_slab(const Slab& slab = {}) {
  constexpr int nx = 8;
  constexpr int ny = 12;
  const int nz = slab.nz;
  double lo = 1.0, hi = 4.0;
  for (int it = 0; it < 200; ++it) {
    const double ratio = 0.5 * (lo + hi);
    (slab.first * (std::pow(ratio, ny) - 1.0) / (ratio - 1.0) > 1.0 ? hi : lo) = ratio;
  }
  autan::MeshInput in;
  double y = 0.0, dy = slab.first;
  for (int j = 0; j <= ny; ++j) {
    for (int k = 0; k <= nz; ++k) {
      const double zk = static_cast<double>(k) / nz;
      for (int i = 0; i <= nx; ++i) {
        const double xi = static_cast<double>(i) / nx;
        const double yj = j == ny ? 1.0 : j == 0 ? slab.warp * (2 * zk - 1) * xi : y;
        const double xs = xi + slab.shear * yj;
        in.x.push_back(std::cos(slab.angle) * xs - std::sin(slab.angle) * yj);
        in.y.push_back(std::sin(slab.angle) * xs + std::cos(slab.angle) * yj);
        in.z.push_back(zk);
      }
    }
    y += dy;
    dy *= lo;
  }
  auto p = [&](int i, int j, int k) {
    return static_cast<index_t>(i + (nx + 1) * (k + (nz + 1) * j));
  };
  auto cell = [&](int i, int j, int k) { return static_cast<index_t>(i + nx * (k + nz * j)); };
  in.face_node_offsets.push_back(0);
  auto face = [&](std::array<index_t, 4> nodes, bool flip, index_t c0, index_t c1, index_t zone) {
    if (flip) std::reverse(nodes.begin(), nodes.end());
    in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
    in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
    in.face_cell0.push_back(c0);
    in.face_cell1.push_back(c1);
    in.face_zone.push_back(zone);
  };
  constexpr index_t interior = 5;
  for (int j = 0; j < ny; ++j) {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i <= nx; ++i) {
        const std::array<index_t, 4> q{p(i, j, k), p(i, j + 1, k), p(i, j + 1, k + 1),
                                       p(i, j, k + 1)};
        if (i == 0)
          face(q, true, cell(0, j, k), autan::no_cell, 0);
        else if (i == nx)
          face(q, false, cell(nx - 1, j, k), autan::no_cell, 1);
        else
          face(q, false, cell(i - 1, j, k), cell(i, j, k), interior);
      }
    }
  }
  for (int j = 0; j <= ny; ++j) {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        const std::array<index_t, 4> q{p(i, j, k), p(i, j, k + 1), p(i + 1, j, k + 1),
                                       p(i + 1, j, k)};
        if (j == 0)
          face(q, true, cell(i, 0, k), autan::no_cell, 2);
        else if (j == ny)
          face(q, false, cell(i, ny - 1, k), autan::no_cell, 3);
        else
          face(q, false, cell(i, j - 1, k), cell(i, j, k), interior);
      }
    }
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      for (int k = 0; k <= nz; ++k) {
        const std::array<index_t, 4> q{p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k),
                                       p(i, j + 1, k)};
        if (k == 0)
          face(q, true, cell(i, j, 0), autan::no_cell, 4);
        else if (k == nz)
          face(q, false, cell(i, j, nz - 1), autan::no_cell, 4);
        else
          face(q, false, cell(i, j, k - 1), cell(i, j, k), interior);
      }
    }
  }
  in.cell_zone.assign(static_cast<std::size_t>(nx * ny * nz), 0);
  for (const char* name : {"xmin", "xmax", "ymin", "ymax", "sides"}) {
    in.face_zones.push_back(autan::FaceZone{
        .name = name, .bc_type = "", .kind = autan::BcKind::unspecified, .provenance = {}});
  }
  in.face_zones.push_back(autan::FaceZone{.name = "interior",
                                          .bc_type = "",
                                          .kind = autan::BcKind::unspecified,
                                          .boundary = false,
                                          .provenance = {}});
  in.cell_zones = {autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}}};
  in.source = autan::MeshSource{.format = "memory", .path = ""};
  return autan::assemble(std::move(in));
}

template <class P>
constexpr double eps() {
  return std::numeric_limits<typename P::compute>::epsilon();
}

// The larger error, NaN when either is: std::max(worst, NaN) would keep worst.
double worse(double worst, double error) {
  return std::isnan(worst) || error <= worst ? worst : error;
}

}  // namespace

TEST_CASE("y+ of a wall face from the wall state by hand", "[navier-stokes]") {
  // results.md, WallSolution YPlus: y1 sqrt(|tau_w| rho_w) / mu(T_w), rho_w = p_w / (R T_w). For
  // y1 = 1e-5 m, |tau_w| = 2 Pa, p_w = 101325 Pa, R = 287 J/(kg K) and air's Sutherland law
  // (1.716e-5 Pa s, 273.15 K, 110.4 K): at T_w = 300 K mu = 1.84592e-5 Pa s, rho_w = 1.17683 kg/m^3
  // and y+ = 0.831112; at 250 K, y+ = 1.05099; with a constant mu of 1.8e-5 Pa s at 300 K,
  // 0.852313.
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const Transport sutherland_air{};
  CHECK_THAT(autan::physics::viscosity(sutherland_air, 300.0),
             WithinRel(1.8459162511975804e-05, 64 * eps));
  CHECK_THAT(autan::physics::wall_y_plus(1e-5, 2.0, 101325.0, 300.0, 287.0, sutherland_air),
             WithinRel(0.8311123484984475, 64 * eps));
  CHECK_THAT(autan::physics::wall_y_plus(1e-5, 2.0, 101325.0, 250.0, 287.0, sutherland_air),
             WithinRel(1.0509925978906687, 64 * eps));
  Transport constant = sutherland_air;
  constant.sutherland = false;
  constant.mu_ref = 1.8e-5;
  CHECK_THAT(autan::physics::wall_y_plus(1e-5, 2.0, 101325.0, 300.0, 287.0, constant),
             WithinRel(0.8523132170357062, 64 * eps));
  // A slip face carries no shear: y+ 0.
  CHECK(autan::physics::wall_y_plus(1e-5, 0.0, 101325.0, 300.0, 287.0, sutherland_air) == 0.0);
}

// With velocity and temperature linear, the least-squares gradients and the corrected face
// gradients are exact, the stress tau and the heat flux are uniform, and the face values are exact
// at the centroids of planar faces. The viscous residual -sum_f F_v . S of each cell is then
// exactly -V (0, 0, 0, 0, tau : grad u): the stress and the heat flux are divergence-free and
// div(u . tau) = tau : grad u (viscous dissipation). The same holds when y = 0 is a no-slip wall,
// isothermal or adiabatic, since the wall state is the exact one.
TEMPLATE_TEST_CASE("viscous residual of linear velocity and temperature is the dissipation",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  const auto wall = GENERATE(BoundaryKind::state, BoundaryKind::wall);
  const auto isothermal = GENERATE(true, false);
  if (wall == BoundaryKind::state && !isothermal) return;
  CAPTURE(file, wall == BoundaryKind::wall, isothermal);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());

  Scheme scheme;
  scheme.limiter = false;
  const auto w0 = primitive(0.5, 0.5, 0.5);
  scheme.reference = {
      rho, rho * w0[1], rho * w0[2], rho * w0[3],
      w0[4] / (air.gamma - 1.0) + 0.5 * rho * (w0[1] * w0[1] + w0[2] * w0[2] + w0[3] * w0[3])};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  Euler<P> euler(mesh, g, air, scheme);

  std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto w = primitive(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    if (wall == BoundaryKind::wall && zone == "ymin") {
      kind[b] = BoundaryKind::wall;
      data[b] = isothermal ? w[4] / (rho * air.gas_constant) : 0.0;
    } else {
      for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
    }
  }
  ns.set_boundary(kind, data);
  euler.set_boundary(kind, data);

  std::vector<double> u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto w = primitive(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    u[c] = rho;
    for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = rho * w[1 + k];
    u[4 * n + c] = w[4] / (air.gamma - 1.0) + 0.5 * rho * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
  }
  std::vector<typename P::state> s(5 * n);
  ns.state(u, s);
  std::vector<typename P::residual> r_ns(5 * n), r_euler(5 * n);
  ns.residual(s, r_ns);
  euler.residual(s, r_euler);

  // grad u: row i the gradient of u_i, only d/dy is non-zero.
  const double div = a[1];
  double dissipation = 0.0;  // tau : grad u
  double tau_max = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double gij = j == 1 ? a[i] : 0.0;
      const double gji = i == 1 ? a[j] : 0.0;
      const double tij = mu * (gij + gji - (i == j ? 2.0 / 3.0 * div : 0.0));
      dissipation += tij * gij;
      tau_max = std::max(tau_max, std::abs(tij));
    }
  }
  std::vector<double> surface(n, 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
    if (f < ni) surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
  }
  const double cp = air.gamma * air.gas_constant / (air.gamma - 1.0);
  const double heat = cp * mu / prandtl * std::hypot(2000.0, 500.0) / (rho * air.gas_constant);
  const double speed = std::hypot(a[0], a[1], a[2]);
  double worst = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    // Round-off of the face fluxes summed around the cell: stress, work and heat flux bounds.
    const double scale = (tau_max * (1.0 + speed) + heat) * surface[c];
    for (std::size_t k = 0; k < 5; ++k) {
      const double viscous =
          static_cast<double>(r_ns[k * n + c]) - static_cast<double>(r_euler[k * n + c]);
      const double expected = k == 4 ? -g.cell_volume[c] * dissipation : 0.0;
      worst = worse(worst, std::abs(viscous - expected) / scale);
    }
  }
  // The least-squares gradients of the stored states carry the round-off of each cell value
  // amplified by the stencil's conditioning (measured: 1.1e3 eps on the tetrahedra in fp64).
  CHECK(worst <= 4096 * eps<P>());
}

// A cubic velocity vanishing on the wall y = 0, at uniform density and pressure: the stress on the
// wall is mu (du/dy, 0, dw/dy) at each face centroid, and the cubic fit of the boundary cells gives
// it exactly, where a linear fit would be first order and a quadratic one second. Also on a
// one-layer slab stretched 10^4:1 at the wall, with slip sides (the fields are those of the plane
// z = 1/2) and an outlet at x = 1, which gives neither T nor the velocity at the wall's end; and on
// that slab turned 30 degrees about z, whose wall lies along no axis (#448): the fields and the
// stress are those of the slab's own frame, turned with it.
TEMPLATE_TEST_CASE("wall shear stress of a cubic velocity is exact", "[navier-stokes]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const std::string file = GENERATE(Catch::Generators::values<std::string>(
      {meshes[0], meshes[1], meshes[2], meshes[3], "stretched slab", "turned slab"}));
  CAPTURE(file);
  const bool slab = file == "stretched slab" || file == "turned slab";
  const double angle = file == "turned slab" ? std::acos(-1.0) / 6.0 : 0.0;
  const double ca = std::cos(angle), sa = std::sin(angle);
  const auto mesh = slab ? stretched_slab({.angle = angle}) : autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  // In the slab's frame (x, y) = R^T (x, y), the components turned back by R.
  auto velocity = [&](double gx, double gy, double z) {
    const double x = ca * gx + sa * gy, y = -sa * gx + ca * gy;
    const double u =
        y * (30.0 + 8.0 * x - 5.0 * z + 12.0 * y + 4.0 * x * x + 3.0 * y * y - 2.0 * x * z);
    const double v = 4.0 * y * y + y * y * y;
    return std::array<double, 3>{ca * u - sa * v, sa * u + ca * v,
                                 y * (-10.0 + 6.0 * x + 2.0 * z * z)};
  };
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  std::vector<double> data(5 * nb);
  std::vector<std::size_t> wall;
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto v = velocity(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    if (zone == "ymin") {
      kind[b] = BoundaryKind::wall;
      data[b] = p / (rho * air.gas_constant);
      wall.push_back(b);
    } else if (zone == "sides") {
      kind[b] = BoundaryKind::slip;
    } else if (zone == "xmax" && slab) {
      kind[b] = BoundaryKind::outlet;
      data[4 * nb + b] = p;
    } else {
      data[b] = rho;
      for (std::size_t k = 0; k < 3; ++k) data[(1 + k) * nb + b] = v[k];
      data[4 * nb + b] = p;
    }
  }
  ns.set_boundary(kind, data);
  std::vector<double> u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto v = velocity(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    u[c] = rho;
    for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = rho * v[k];
    u[4 * n + c] = p / (air.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
  std::vector<typename P::state> s(5 * n);
  std::vector<typename P::residual> r(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  std::vector<double> traction(3 * nb);
  ns.viscous_traction(traction);
  REQUIRE(!wall.empty());
  double worst = 0.0;
  for (const auto b : wall) {
    const auto f = ni + b;
    const double x = ca * g.face_centroid_x[f] + sa * g.face_centroid_y[f];
    const double z = g.face_centroid_z[f];
    const double tx = mu * (30.0 + 8.0 * x - 5.0 * z + 4.0 * x * x - 2.0 * x * z);
    const std::array<double, 3> exact{ca * tx, sa * tx, mu * (-10.0 + 6.0 * x + 2.0 * z * z)};
    for (std::size_t k = 0; k < 3; ++k) {
      worst = worse(worst, std::abs(traction[k * nb + b] - exact[k]) / (mu * 45.0));
    }
  }
  // Round-off of the fit, amplified by its conditioning like the linear gradients above. The turned
  // slab's nodes are rounded at eps_64 |x| (|x| < 1.5) against the wall spacing 1e-5, which
  // perturbs the offsets across the wall by eps_64 |x| / 1e-5 relative, whatever the fit.
  const double geometry =
      angle != 0.0 ? std::numeric_limits<double>::epsilon() * 1.5 / 1.0e-5 : 0.0;
  CHECK(worst <= 4096 * (eps<P>() + geometry));
}

// An adiabatic wall at rest takes no energy from the flow: its velocity is zero over the whole of
// each face, so neither conduction nor the work of the stress crosses it, warped faces included
// (here the wall y = 0.02 (2 z - 1) x). In a slab closed by such walls (y = 0, x = 0 and x = 1) and
// slip faces, which pass no energy either, the energy residuals of the cells sum to zero: each
// interior face adds its flux to one cell and subtracts it from the other.
TEMPLATE_TEST_CASE("an adiabatic wall at rest exchanges no energy", "[navier-stokes]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto mesh = stretched_slab({.first = 0.05, .nz = 4, .warp = 0.02});
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  auto velocity = [](double x, double y, double z) {
    return std::array<double, 3>{y * (30.0 + 8.0 * x), 4.0 * y * y,
                                 y * (-10.0 + 6.0 * x + 2.0 * z)};
  };
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = static_cast<std::size_t>(mesh.n_interior_faces()) + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    if (zone == "ymin" || zone == "xmin" || zone == "xmax") kind[b] = BoundaryKind::wall;
  }
  ns.set_boundary(kind, std::vector<double>(5 * nb, 0.0));
  std::vector<double> u(5 * n);
  double speed = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    const auto v = velocity(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    u[c] = rho;
    for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = rho * v[k];
    u[4 * n + c] = p / (air.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    speed = std::max(speed, std::hypot(v[0], v[1], v[2]));
  }
  std::vector<typename P::state> s(5 * n);
  std::vector<typename P::residual> r(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  double total = 0.0;
  for (std::size_t c = 0; c < n; ++c) total += static_cast<double>(r[4 * n + c]);
  // The face fluxes cancel in pairs as computed, so only the accumulation in the residual's
  // precision rounds; the enthalpy flux bounds each face's energy flux.
  double scale = 0.0;
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    scale += 2.0 * air.gamma / (air.gamma - 1.0) * p * speed * g.face_area[f];
  }
  CHECK(std::abs(total) <= 64 * std::numeric_limits<typename P::residual>::epsilon() * scale);
}

// A slip or outlet face takes nt from the interior and passes none by diffusion (zero normal
// derivative). On the slab sheared by x += y / 2, nt is a linear function of x - y / 2, the same
// at the centroid of each y face as at its cell's, so the interior value is exact there, but
// dnt/dy is not zero. The flow at rest and the wall distance (1e10 m) leave production and
// destruction negligible, and the residuals sum to the budget of the equation: -1/sigma int rho
// (nu + nt f_n) grad nt . dS over the faces that give nt, on each of which nt is constant, and the
// cb2 term -int rho cb2 |grad nt|^2 / sigma dV. Also for negative nt, whose diffusion coefficient
// takes f_n = (cn1 + chi^3) / (cn1 - chi^3), cn1 = 16 (Allmaras et al. 2012, eq. 21).
TEMPLATE_TEST_CASE("slip and outlet faces pass no nt by diffusion", "[spalart-allmaras]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto neumann = GENERATE(BoundaryKind::slip, BoundaryKind::outlet);
  const bool negative = GENERATE(false, true);
  CAPTURE(neumann == BoundaryKind::slip, negative);
  constexpr double shear = 0.5;
  const auto mesh = stretched_slab({.first = 0.05, .shear = shear});
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  constexpr double mu_air = 1.8e-5;
  constexpr double sigma = 2.0 / 3.0;
  constexpr double cb2 = 0.622;
  const double nu = mu_air / rho;
  const double base = negative ? -0.5 : 50.0, slope = negative ? -0.1 : 5.0;
  const std::array<double, 3> grad{slope * nu, -slope * shear * nu, 0.0};
  auto nt = [&](double x, double y) { return nu * (base + slope * (x - shear * y)); };
  auto fn = [&](double v) {
    const double chi3 = (v / nu) * (v / nu) * (v / nu);
    return v < 0.0 ? (16.0 + chi3) / (16.0 - chi3) : 1.0;
  };
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu_air;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  std::vector<double> data(5 * nb, 0.0), nt_b(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    nt_b[b] = nt(g.face_centroid_x[f], g.face_centroid_y[f]);
    data[4 * nb + b] = p;
    if (zone == "ymax") {
      kind[b] = neumann;
    } else if (zone == "sides") {
      kind[b] = BoundaryKind::slip;
    } else {
      data[b] = rho;
    }
  }
  ns.set_boundary(kind, data);
  std::vector<double> u(5 * n, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    u[c] = rho;
    u[4 * n + c] = p / (air.gamma - 1.0);
  }
  std::vector<typename P::state> s(5 * n);
  std::vector<typename P::residual> r(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  SpalartAllmaras<P> sa(mesh, g, ns, t, /*ft2=*/false);
  sa.set_boundary(nt_b);
  sa.set_wall_distance(std::vector<double>(n, 1.0e10));
  std::vector<typename P::state> rn(n);
  for (std::size_t c = 0; c < n; ++c) {
    rn[c] = static_cast<typename P::state>(rho * nt(g.cell_centroid_x[c], g.cell_centroid_y[c]));
  }
  std::vector<typename P::residual> res(n);
  sa.residual(rn, res);

  const double g2 = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
  double total = 0.0, expected = 0.0, scale = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    total += static_cast<double>(res[c]);
    expected -= g.cell_volume[c] * rho * cb2 * g2 / sigma;
  }
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    const double nt_f = nt(g.face_centroid_x[f], g.face_centroid_y[f]);
    const double coefficient = rho * (nu + nt_f * fn(nt_f)) / sigma;
    scale += coefficient * std::sqrt(g2) * g.face_area[f];
    if (f >= ni && kind[f - ni] == BoundaryKind::state) {
      expected -= coefficient * (grad[0] * g.face_area_x[f] + grad[1] * g.face_area_y[f] +
                                 grad[2] * g.face_area_z[f]);
    }
  }
  CAPTURE(total, expected, scale);
  CHECK(std::abs(total - expected) <= 4096 * eps<P>() * scale);
}

// A slip face passes no nt by convection: the flow does not cross it. With nt uniform, a flow and
// its reverse give the same source (|curl u| does not see the sign) and opposite convective
// fluxes, so half the difference of their residuals sums to the nt convected through the
// boundary, zero in a box closed by slip faces; here for u = 10 (x - 1/2, y - 1/2, z - 1/2).
TEMPLATE_TEST_CASE("slip faces pass no nt by convection", "[spalart-allmaras]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  constexpr double mu_air = 1.8e-5;
  constexpr double alpha = 10.0;
  const double nt = 50.0 * mu_air / rho;
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu_air;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  ns.set_boundary(std::vector<BoundaryKind>(nb, BoundaryKind::slip),
                  std::vector<double>(5 * nb, 0.0));
  SpalartAllmaras<P> sa(mesh, g, ns, t);
  const std::vector<typename P::state> rn(n, static_cast<typename P::state>(rho * nt));
  std::vector<typename P::state> s(5 * n);
  std::vector<typename P::residual> r(5 * n), res(n);
  double half = 0.0, speed = 0.0;
  for (const double sign : {1.0, -1.0}) {
    std::vector<double> u(5 * n);
    for (std::size_t c = 0; c < n; ++c) {
      const std::array<double, 3> v{sign * alpha * (g.cell_centroid_x[c] - 0.5),
                                    sign * alpha * (g.cell_centroid_y[c] - 0.5),
                                    sign * alpha * (g.cell_centroid_z[c] - 0.5)};
      u[c] = rho;
      for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = rho * v[k];
      u[4 * n + c] = p / (air.gamma - 1.0) + 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      speed = std::max(speed, std::hypot(v[0], v[1], v[2]));
    }
    ns.state(u, s);
    ns.residual(s, r);
    sa.residual(rn, res);
    for (std::size_t c = 0; c < n; ++c) half += 0.5 * sign * static_cast<double>(res[c]);
  }
  double scale = 0.0;
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    scale += rho * speed * nt * g.face_area[f];
  }
  CAPTURE(half, scale);
  CHECK(std::abs(half) <= 4096 * eps<P>() * scale);
}

// The local time step's viscous spectral radius takes, per face, the larger of the momentum
// diffusion 4/3 mu A^2 / (rho V) and the heat diffusion gamma mu / Pr A^2 / (rho V): at Pr = 100
// the momentum's, which a radius scaled by 1 / Pr alone would lose. NavierStokes adds 4 times it
// to the convective radius of Euler (Blazek 2015, eq. 6.20).
TEMPLATE_TEST_CASE("the viscous spectral radius bounds momentum diffusion at any Prandtl number",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = 100.0;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  Euler<P> euler(mesh, g, air, scheme);
  const std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  const std::vector<double> data(5 * nb, 0.0);
  ns.set_boundary(kind, data);
  euler.set_boundary(kind, data);
  std::vector<double> u(5 * n, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    u[c] = rho;
    u[4 * n + c] = p / (air.gamma - 1.0);
  }
  std::vector<typename P::state> s(5 * n), lambda_ns(n), lambda_euler(n);
  std::vector<typename P::residual> r(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  ns.spectral_radius(lambda_ns);
  euler.residual(s, r);
  euler.spectral_radius(lambda_euler);
  std::vector<double> viscous(n, 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    const double a2 = g.face_area[f] * g.face_area[f] * 4.0 / 3.0 * mu / rho;
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    viscous[o] += a2 / g.cell_volume[o];
    if (f < ni) {
      const auto nb_cell = static_cast<std::size_t>(mesh.neighbour()[f]);
      viscous[nb_cell] += a2 / g.cell_volume[nb_cell];
    }
  }
  double worst = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    const double total = static_cast<double>(lambda_ns[c]);
    const double expected = static_cast<double>(lambda_euler[c]) + 4.0 * viscous[c];
    worst = worse(worst, std::abs(total - expected) / expected);
  }
  CHECK(worst <= 64 * eps<P>());
}

// An inlet gives SA the ratio nt / nu of the case, nu the laminar viscosity of the inlet's own
// state at each evaluation (case-schema.md#turbulence), not a fixed nt. A uniform flow enters the
// slab through xmin at its own total pressure and temperature, so that the inlet's state is the
// cell's, and leaves through the outlet xmax at its static pressure, with nt = 3 nu uniform:
// nothing varies, and every cell's residual is zero to round-off, the wall distance (1e10 m)
// leaving production and destruction negligible. The same SA model then sees the flow at a second
// temperature, where Sutherland's law gives another nu: no fixed inlet value serves both.
TEMPLATE_TEST_CASE("an inlet gives nt the case's ratio to its own viscosity", "[spalart-allmaras]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto mesh = stretched_slab({.first = 0.05});
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  constexpr double speed = 50.0;
  constexpr double ratio = 3.0;
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {1.2, 1.2 * speed, 0.0, 0.0, p / (air.gamma - 1.0) + 0.6 * speed * speed};
  Transport t;  // Sutherland's law for air
  t.prandtl = prandtl;
  auto sutherland = [&](double temperature) {
    return t.mu_ref * std::pow(temperature / t.t_ref, 1.5) * (t.t_ref + t.s) / (temperature + t.s);
  };
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  std::vector<double> nt_b(nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[ni + b])].name;
    if (zone == "xmin") {
      kind[b] = BoundaryKind::inlet;
      nt_b[b] = ratio;
    } else if (zone == "xmax") {
      kind[b] = BoundaryKind::outlet;
    }
  }
  std::vector<double> surface(n, 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
    if (f < ni) surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
  }
  std::unique_ptr<SpalartAllmaras<P>> sa;
  for (const double temperature : {300.0, 400.0}) {
    CAPTURE(temperature);
    const double density = p / (air.gas_constant * temperature);
    const double cp = air.gamma * air.gas_constant / (air.gamma - 1.0);
    const double t0 = temperature + 0.5 * speed * speed / cp;
    const double p0 = p * std::pow(t0 / temperature, air.gamma / (air.gamma - 1.0));
    std::vector<double> data(5 * nb, 0.0);
    for (std::size_t b = 0; b < nb; ++b) {
      if (kind[b] == BoundaryKind::inlet) {
        data[b] = p0;
        data[nb + b] = t0;
      } else if (kind[b] == BoundaryKind::outlet) {
        data[4 * nb + b] = p;
      }
    }
    ns.set_boundary(kind, data);
    std::vector<double> u(5 * n, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
      u[c] = density;
      u[n + c] = density * speed;
      u[4 * n + c] = p / (air.gamma - 1.0) + 0.5 * density * speed * speed;
    }
    std::vector<typename P::state> s(5 * n);
    std::vector<typename P::residual> r(5 * n);
    ns.state(u, s);
    ns.residual(s, r);
    if (!sa) {
      sa = std::make_unique<SpalartAllmaras<P>>(mesh, g, ns, t);
      sa->set_boundary(nt_b);
      sa->set_wall_distance(std::vector<double>(n, 1.0e10));
    }
    const double nt = ratio * sutherland(temperature) / density;
    const std::vector<typename P::state> rn(n, static_cast<typename P::state>(density * nt));
    std::vector<typename P::residual> res(n);
    sa->residual(rn, res);
    double worst = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      worst =
          worse(worst, std::abs(static_cast<double>(res[c])) / (density * speed * nt * surface[c]));
    }
    CHECK(worst <= 4096 * eps<P>());
  }
}

// SA-neg (Allmaras et al. 2012; TMR SA page) for nt < 0: production cb1 (1 - ct3) Omega nt,
// destruction -cw1 (nt / d)^2, and no eddy viscosity. A uniform negative nt in a flow at rest
// closed by slip faces leaves each cell only its destruction, -V rho cw1 (nt / d)^2 with
// cw1 = cb1 / kappa^2 + (1 + cb2) / sigma; and the flow's next residual is the laminar one.
TEMPLATE_TEST_CASE("negative nt is destroyed and gives no eddy viscosity", "[spalart-allmaras]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  constexpr double p = 1.0e5;
  constexpr double mu_air = 1.8e-5;
  constexpr double kappa = 0.41;
  constexpr double cw1 = 0.1355 / (kappa * kappa) + (1.0 + 0.622) / (2.0 / 3.0);
  const double nt = -0.5 * mu_air / rho;
  Scheme scheme;
  scheme.limiter = false;
  scheme.reference = {rho, 0.0, 0.0, 0.0, p / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu_air;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  const std::vector<BoundaryKind> kind(nb, BoundaryKind::slip);
  const std::vector<double> data(5 * nb, 0.0);
  ns.set_boundary(kind, data);
  std::vector<double> u(5 * n, 0.0), d(n);
  for (std::size_t c = 0; c < n; ++c) {
    u[c] = rho;
    u[4 * n + c] = p / (air.gamma - 1.0);
    d[c] = 0.05 + g.cell_centroid_y[c];
  }
  std::vector<typename P::state> s(5 * n);
  std::vector<typename P::residual> r(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  SpalartAllmaras<P> sa(mesh, g, ns, t);
  sa.set_wall_distance(d);
  const std::vector<typename P::state> rn(n, static_cast<typename P::state>(rho * nt));
  std::vector<typename P::residual> res(n);
  sa.residual(rn, res);
  double worst = 0.0;
  for (std::size_t c = 0; c < n; ++c) {
    const double expected = -g.cell_volume[c] * rho * cw1 * (nt / d[c]) * (nt / d[c]);
    worst = worse(worst, std::abs(static_cast<double>(res[c]) - expected) / std::abs(expected));
  }
  CHECK(worst <= 64 * eps<P>());

  // The flow's residual of a sheared state, after SA wrote the eddy viscosity of rn, against a
  // flow that never had one.
  for (std::size_t c = 0; c < n; ++c) {
    const auto w = primitive(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = rho * w[1 + k];
    u[4 * n + c] = w[4] / (air.gamma - 1.0) + 0.5 * rho * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
  }
  NavierStokes<P> laminar(mesh, g, air, scheme, t);
  laminar.set_boundary(kind, data);
  std::vector<typename P::residual> r_laminar(5 * n);
  ns.state(u, s);
  ns.residual(s, r);
  laminar.residual(s, r_laminar);
  CHECK(r == r_laminar);
}
