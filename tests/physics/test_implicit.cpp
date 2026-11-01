// The approximate Jacobian of the Euler residual and its block Gauss-Seidel solver.
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/euler.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BlockMatrix;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

std::array<double, 5> conserved(double rho, std::array<double, 3> v, double p) {
  const double q2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  return {rho, rho * v[0], rho * v[1], rho * v[2], p / 0.4 + 0.5 * rho * q2};
}

}  // namespace

TEST_CASE("the assembled Jacobian of the first-order residual matches its central differences",
          "[implicit]") {
  // Without reconstruction the fused face Jacobians are exact: A v = dR/dU v.
  using P = autan::kernels::fp64;
  const char* file = GENERATE(from_range(meshes));
  const Flux flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  INFO(file << " flux " << static_cast<int>(flux));
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto geometry = autan::compute_geometry(mesh);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  const double a = std::sqrt(1.4 * 1e5 / 1.2);
  const std::array<double, 5> w = {1.2, 0.6 * a, 0.2 * a, -0.1 * a, 1e5};
  // Every condition, so that every boundary Jacobian is exercised.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb);
  const std::array<BoundaryKind, 5> kinds = {BoundaryKind::farfield, BoundaryKind::slip,
                                             BoundaryKind::outlet, BoundaryKind::state,
                                             BoundaryKind::inlet};
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  for (std::size_t b = 0; b < nb; ++b) {
    kind[b] = kinds[b % kinds.size()];
    // An inlet where the flow enters: where it leaves, the inlet's normal velocity is clamped
    // at zero and the fluxes switch upwind side there, a kink central differences straddle.
    const auto f = ni + b;
    const double vn = w[1] * geometry.face_area_x[f] + w[2] * geometry.face_area_y[f] +
                      w[3] * geometry.face_area_z[f];
    if (kind[b] == BoundaryKind::inlet && vn >= 0.0) kind[b] = BoundaryKind::state;
    // Inlet data are total pressure and temperature, above the static ones of w.
    const std::array<double, 5> v =
        kind[b] == BoundaryKind::inlet ? std::array<double, 5>{1.3e5, 320.0, 0.0, 0.0, 0.0} : w;
    for (std::size_t j = 0; j < 5; ++j) data[j * nb + b] = v[j];
  }
  autan::physics::Scheme s;
  s.flux = flux;
  s.second_order = false;
  s.mach_ref = 1.0;  // AUSM+-up's scaling held fixed, away from its kink
  s.reference = conserved(1.1, {10.0, 5.0, 0.0}, 0.95e5);
  Euler<P> e(mesh, geometry, {1.4, 287.0}, s);
  e.set_boundary(kind, data);
  std::vector<double> u(5 * nc), v(5 * nc);
  const auto uc = conserved(w[0], {w[1], w[2], w[3]}, w[4]);
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < nc; ++c) {
      const double x = geometry.cell_centroid_x[c], y = geometry.cell_centroid_y[c];
      u[k * nc + c] =
          uc[k] * (1.0 + 0.1 * std::sin(3.0 * x + static_cast<double>(k)) * std::cos(2.0 * y));
      v[k * nc + c] = uc[k] * std::cos(5.0 * x + 2.0 * y + static_cast<double>(k));
    }
  }
  std::vector<double> state(5 * nc), res(5 * nc), plain(5 * nc), av(5 * nc), rp(5 * nc), rm(5 * nc);
  e.state(u, state);
  e.residual(state, plain);
  e.residual_and_jacobian(state, res);
  // The fused loops compute the same residual, operation for operation.
  CHECK(res == plain);
  BlockMatrix m(mesh);
  e.assemble(std::vector<double>(nc, 0.0), m);
  m.multiply(v, av);
  const double h = 1e-6;
  std::vector<double> up(u), um(u);
  for (std::size_t i = 0; i < u.size(); ++i) {
    up[i] += h * v[i];
    um[i] -= h * v[i];
  }
  e.state(up, state);
  e.residual(state, rp);
  e.state(um, state);
  e.residual(state, rm);
  double diff = 0.0, norm = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(std::isfinite(av[i]));
    REQUIRE(std::isfinite(rp[i] - rm[i]));
    diff = std::max(diff, std::abs((rp[i] - rm[i]) / (2.0 * h) - av[i]));
    norm = std::max(norm, std::abs(av[i]));
  }
  // Central differences: O(h^2) truncation and O(eps / h) round-off, relative.
  CHECK(diff <= 1e-6 * norm);

  // The same blocks in fp32+acc64: FP32 Jacobians, so A v agrees with FP64's to FP32 rounding.
  using Q = autan::kernels::fp32_acc64;
  Euler<Q> e32(mesh, geometry, {1.4, 287.0}, s);
  e32.set_boundary(kind, data);
  std::vector<float> state32(5 * nc);
  std::vector<double> res32(5 * nc), plain32(5 * nc), av32(5 * nc);
  e32.state(u, state32);
  e32.residual(state32, plain32);
  e32.residual_and_jacobian(state32, res32);
  CHECK(res32 == plain32);
  BlockMatrix m32(mesh);
  e32.assemble(std::vector<double>(nc, 0.0), m32);
  m32.multiply(v, av32);
  double diff32 = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    REQUIRE(std::isfinite(av32[i]));
    diff32 = std::max(diff32, std::abs(av32[i] - av[i]));
  }
  CHECK(diff32 <= 1024 * std::numeric_limits<float>::epsilon() * norm);
}

TEST_CASE("multicolour block Gauss-Seidel solves a block diagonally dominant system",
          "[implicit]") {
  const char* file = GENERATE(from_range(meshes));
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto nc = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  BlockMatrix m(mesh);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> r(-1.0, 1.0);
  for (std::size_t f = 0; f < ni; ++f) {
    for (std::size_t q = 0; q < 25; ++q) {
      m.upper(f)[q] = r(rng);
      m.lower(f)[q] = r(rng);
    }
  }
  // Each block row sums at most 5 * (faces of the cell) in absolute value off the diagonal.
  std::vector<double> faces(nc, 0.0);
  for (std::size_t f = 0; f < ni; ++f) {
    faces[static_cast<std::size_t>(mesh.owner()[f])] += 1.0;
    faces[static_cast<std::size_t>(mesh.neighbour()[f])] += 1.0;
  }
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t q = 0; q < 25; ++q) m.diagonal(c)[q] = 0.1 * r(rng);
    for (std::size_t i = 0; i < 5; ++i) m.diagonal(c)[6 * i] = 10.0 * faces[c] + 10.0;
  }
  std::vector<double> x_exact(5 * nc), b(5 * nc), x(5 * nc, 0.0), ax(5 * nc);
  for (auto& v : x_exact) v = r(rng);
  m.multiply(x_exact, b);
  m.factor();
  m.gauss_seidel(b, x, 20);
  m.multiply(x, ax);
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    REQUIRE(std::isfinite(x[i]));
    err = std::max(err, std::abs(x[i] - x_exact[i]));
    ref = std::max(ref, std::abs(x_exact[i]));
  }
  CHECK(err <= 1e-12 * ref);
}
