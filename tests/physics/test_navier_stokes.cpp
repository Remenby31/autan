// The viscous fluxes of autan::physics::NavierStokes on the test meshes of tests/data/mesh.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

// One layer of hexahedra over [0,1]^2 x [0,1], the spacing in y growing geometrically from 1e-5 at
// y = 0 (aspect ratio 10^4 at the wall, as in the wall-resolved 2D grids of the NASA TMR), zones
// xmin ... ymax and `sides` for z = 0 and z = 1; turned by `angle` (radians) about z.
autan::Mesh stretched_slab(double angle = 0.0) {
  constexpr int nx = 8;
  constexpr int ny = 12;
  constexpr double first = 1.0e-5;
  double lo = 1.0, hi = 4.0;
  for (int it = 0; it < 200; ++it) {
    const double ratio = 0.5 * (lo + hi);
    (first * (std::pow(ratio, ny) - 1.0) / (ratio - 1.0) > 1.0 ? hi : lo) = ratio;
  }
  autan::MeshInput in;
  double y = 0.0, dy = first;
  for (int j = 0; j <= ny; ++j) {
    for (int k = 0; k <= 1; ++k) {
      for (int i = 0; i <= nx; ++i) {
        const double xi = static_cast<double>(i) / nx, yj = j == ny ? 1.0 : y;
        in.x.push_back(std::cos(angle) * xi - std::sin(angle) * yj);
        in.y.push_back(std::sin(angle) * xi + std::cos(angle) * yj);
        in.z.push_back(k);
      }
    }
    y += dy;
    dy *= lo;
  }
  auto p = [](int i, int j, int k) { return static_cast<index_t>(i + (nx + 1) * (k + 2 * j)); };
  auto cell = [](int i, int j) { return static_cast<index_t>(i + nx * j); };
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
    for (int i = 0; i <= nx; ++i) {
      const std::array<index_t, 4> q{p(i, j, 0), p(i, j + 1, 0), p(i, j + 1, 1), p(i, j, 1)};
      if (i == 0)
        face(q, true, cell(0, j), autan::no_cell, 0);
      else if (i == nx)
        face(q, false, cell(nx - 1, j), autan::no_cell, 1);
      else
        face(q, false, cell(i - 1, j), cell(i, j), interior);
    }
  }
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const std::array<index_t, 4> q{p(i, j, 0), p(i, j, 1), p(i + 1, j, 1), p(i + 1, j, 0)};
      if (j == 0)
        face(q, true, cell(i, 0), autan::no_cell, 2);
      else if (j == ny)
        face(q, false, cell(i, ny - 1), autan::no_cell, 3);
      else
        face(q, false, cell(i, j - 1), cell(i, j), interior);
    }
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      for (int k = 0; k <= 1; ++k) {
        face({p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k), p(i, j + 1, k)}, k == 0, cell(i, j),
             autan::no_cell, 4);
      }
    }
  }
  in.cell_zone.assign(static_cast<std::size_t>(nx * ny), 0);
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

}  // namespace

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
      worst = std::max(worst, std::abs(viscous - expected) / scale);
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
  const auto mesh = slab ? stretched_slab(angle) : autan::io::read_mesh(mesh_dir / file);
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
      worst = std::max(worst, std::abs(traction[k * nb + b] - exact[k]) / (mu * 45.0));
    }
  }
  // Round-off of the fit, amplified by its conditioning like the linear gradients above. The turned
  // slab's nodes are rounded at eps_64 |x| (|x| < 1.5) against the wall spacing 1e-5, which
  // perturbs the offsets across the wall by eps_64 |x| / 1e-5 relative, whatever the fit.
  const double geometry =
      angle != 0.0 ? std::numeric_limits<double>::epsilon() * 1.5 / 1.0e-5 : 0.0;
  CHECK(worst <= 4096 * (eps<P>() + geometry));
}

// The matrix-free product's residual (Euler<P>::freeze's contract, with the viscous terms): at the
// frozen state it is the residual bit for bit, and evaluating it elsewhere leaves the residual's
// own cache, which the wall traction reads, untouched.
TEMPLATE_TEST_CASE("the frozen residual is the residual at its state and keeps the cache",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  Scheme scheme;  // limited: freeze holds the limiter
  scheme.reference = {rho, 0.0, 0.0, 0.0, 1e5 / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  std::vector<double> data(5 * nb);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto w = primitive(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    if (zone == "ymin") {
      kind[b] = BoundaryKind::wall;
      data[b] = 0.0;
    } else {
      for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
    }
  }
  ns.set_boundary(kind, data);
  // `thin`: a cell whose conserved state is scaled by 1e-3.
  auto conserved = [&](double amplitude,
                       std::size_t thin = std::numeric_limits<std::size_t>::max()) {
    std::vector<double> u(5 * n);
    for (std::size_t c = 0; c < n; ++c) {
      const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
      auto w = primitive(x, y, z);
      w[1] += amplitude * std::sin(7.0 * x + 3.0 * z);
      w[4] += amplitude * 300.0 * std::cos(5.0 * y);
      u[c] = w[0];
      for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = w[0] * w[1 + k];
      u[4 * n + c] =
          w[4] / (air.gamma - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
      if (c == thin) {
        for (std::size_t k = 0; k < 5; ++k) u[k * n + c] *= 1e-3;
      }
    }
    std::vector<typename P::state> s(5 * n);
    ns.state(u, s);
    return s;
  };
  const auto s1 = conserved(1.0), s2 = conserved(2.0);
  std::vector<typename P::residual> r(5 * n), rf(5 * n), other(5 * n);
  ns.residual(s1, r);
  std::vector<double> traction(3 * nb), after(3 * nb);
  ns.viscous_traction(traction);
  ns.freeze(s1);
  ns.residual_frozen(s1, rf);
  CHECK(r == rf);
  CHECK(ns.frozen_switches() == 0);
  ns.residual_frozen(s2, other);
  CHECK(ns.frozen_switches() == 0);
  ns.viscous_traction(after);
  CHECK(traction == after);
  // One cell's density down to 1e-3 of its value: its neighbours' reconstructions towards it
  // reach a non-positive density, so face sides switch to the cell value.
  // The convective reconstruction's switches alone, from Euler<P> at the same states (slip faces
  // and walls alike out of its gradients): the viscous one switches too.
  Euler<P> euler(mesh, g, air, scheme);
  auto slip = kind;
  for (auto& k : slip) {
    if (k == BoundaryKind::wall) k = BoundaryKind::slip;
  }
  euler.set_boundary(slip, data);
  euler.freeze(s1);
  std::vector<typename P::residual> convective(5 * n);
  euler.residual_frozen(conserved(1.0, n / 2), convective);
  ns.residual_frozen(conserved(1.0, n / 2), other);
  CAPTURE(euler.frozen_switches(), ns.frozen_switches());
  CHECK(euler.frozen_switches() > 0);
  CHECK(ns.frozen_switches() > euler.frozen_switches());
  ns.residual_frozen(s1, rf);
  CHECK(ns.frozen_switches() == 0);
  ns.release_frozen();
  CHECK_THROWS_AS(ns.residual_frozen(s1, rf), std::logic_error);
  CHECK_THROWS_AS(static_cast<void>(ns.frozen_switches()), std::logic_error);
}

TEMPLATE_TEST_CASE(
    "the coupled frozen residual is the residual at its state and smooth across branches",
    "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  using P = TestType;
  using S = typename P::state;
  using R = typename P::residual;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  Scheme scheme;
  scheme.reference = {rho, 0.0, 0.0, 0.0, 1e5 / (air.gamma - 1.0)};
  Transport t;
  t.sutherland = false;
  t.mu_ref = mu;
  t.prandtl = prandtl;
  NavierStokes<P> ns(mesh, g, air, scheme, t);
  SpalartAllmaras<P> sa(mesh, g, ns, t);
  // nt from -2.5 nu to 3.5 nu: chi through 0 (sign of nt, f_n), fv2 < 0 at chi of a few with
  // the wall distance y (the S_tilde limit, S_tilde <= 0, r at its cap). Along s(tau), nt grows
  // by tau nu, and every fifth cell with y > 0.3 starts at nt = 0.5 e nu or -1.5 e nu, so that
  // the path changes its sign there between the points of the second differences below.
  constexpr double nu = mu / rho;
  double e = 1e-3;
  auto nt = [](double x, double y, double z, double tau) {
    return nu * (0.5 + 3.0 * std::sin(4.0 * x + 5.0 * y - 3.0 * z) + tau);
  };
  std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
  std::vector<double> data(5 * nb), nt_b(nb), wall(n);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const double x = g.face_centroid_x[f], y = g.face_centroid_y[f], z = g.face_centroid_z[f];
    const auto w = primitive(x, y, z);
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    if (zone == "ymin") {
      kind[b] = BoundaryKind::wall;
    } else {
      for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
    }
    nt_b[b] = nt(x, y, z, 0.0);
  }
  ns.set_boundary(kind, data);
  sa.set_boundary(nt_b);
  for (std::size_t c = 0; c < n; ++c) wall[c] = g.cell_centroid_y[c];
  sa.set_wall_distance(wall);
  // The states along s(tau): the velocity and pressure perturbations of the flow test above, nt's.
  auto flow = [&](double tau) {
    std::vector<double> u(5 * n);
    for (std::size_t c = 0; c < n; ++c) {
      const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
      auto w = primitive(x, y, z);
      w[1] += (1.0 + tau) * std::sin(7.0 * x + 3.0 * z);
      w[4] += (1.0 + tau) * 300.0 * std::cos(5.0 * y);
      u[c] = w[0];
      for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = w[0] * w[1 + k];
      u[4 * n + c] =
          w[4] / (air.gamma - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
    }
    std::vector<S> s(5 * n);
    ns.state(u, s);
    return s;
  };
  auto seeded = [&](std::size_t c) { return c % 5 == 0 && g.cell_centroid_y[c] > 0.3; };
  auto turbulence = [&](double tau) {
    std::vector<S> s(n);
    for (std::size_t c = 0; c < n; ++c) {
      const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
      const double v = seeded(c) ? nu * ((c % 10 == 0 ? 0.5 : -1.5) * e + tau) : nt(x, y, z, tau);
      s[c] = S(rho * v);
    }
    return s;
  };
  auto held = [&](double tau) {
    const auto s = flow(tau), v = turbulence(tau);
    std::vector<R> r(6 * n);
    ns.residual_frozen(s, std::span(r).first(5 * n), [&] { sa.eddy_viscosity_frozen(v); });
    sa.residual_frozen(v, std::span(r).last(n));
    return r;
  };
  auto unheld = [&](double tau) {
    const auto s = flow(tau), v = turbulence(tau);
    std::vector<R> r(6 * n);
    ns.residual(s, std::span(r).first(5 * n));
    sa.eddy_viscosity(v);
    ns.residual(s, std::span(r).first(5 * n));
    sa.residual(v, std::span(r).last(n));
    return r;
  };
  const auto r0 = unheld(0.0);
  ns.freeze(flow(0.0));
  sa.freeze(turbulence(0.0));
  CHECK(held(0.0) == r0);

  std::vector<R> cached(5 * n);
  ns.residual(flow(0.0), cached);
  CHECK(std::equal(cached.begin(), cached.end(), r0.begin()));

  // Second differences of nt's residual, D(e) = R(e) - 2 R(0) + R(-e): q(e) = |D(2e) - 4 D(e)|
  // falls as e^4 where R is smooth along s(tau), as e^2 where a branch changes sign of nt (a jump
  // of the second derivative). In FP32 these second differences are round-off. The flow's held
  // residual is Euler<P>'s and its own test's.
  if constexpr (std::is_same_v<P, autan::kernels::fp64>) {
    // Over every cell, or over those where nt changes sign.
    auto q = [&](auto&& r, bool crossing) {
      std::array<std::vector<R>, 5> at{r(-2 * e), r(-e), r(0.0), r(e), r(2 * e)};
      double out = 0.0;
      for (std::size_t c = 0; c < n; ++c) {
        if (crossing && !seeded(c)) continue;
        auto d = [&](std::size_t lo, std::size_t hi) {
          return at[hi][5 * n + c] - 2.0 * at[2][5 * n + c] + at[lo][5 * n + c];
        };
        out = std::max(out, std::abs(d(0, 4) - 4.0 * d(1, 3)));
      }
      return out;
    };
    std::array<double, 2> qh{}, qu{};
    for (std::size_t i = 0; i < 2; ++i) {
      e = i == 0 ? 1e-3 : 5e-4;
      ns.residual(flow(0.0), cached);
      ns.freeze(flow(0.0));
      sa.freeze(turbulence(0.0));
      qh[i] = q(held, false);
      qu[i] = q(unheld, true);
    }
    const double held_order = std::log2(qh[0] / qh[1]);
    const double unheld_order = std::log2(qu[0] / qu[1]);
    CAPTURE(held_order, unheld_order);
    CHECK(held_order > 3.5);
    CHECK(unheld_order < 3.0);
  }
  sa.release_frozen();
  std::vector<R> rn(n);
  CHECK_THROWS_AS(sa.residual_frozen(turbulence(0.0), rn), std::logic_error);
}
