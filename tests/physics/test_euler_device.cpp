// The Euler residual on a CUDA device (DeviceStepper) against the host's (Euler<P>), cell by
// cell, on the test meshes of the unit cube.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
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
#include "autan/physics/euler_device.hpp"
#include "autan/runtime/ranks.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::Scheme;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};
constexpr Gas air{1.4, 287.0};

// One rank, no periodic pairs: no ghost cells to fill.
class NoGhosts final : public autan::runtime::GhostExchange {
 public:
  void forward(std::span<float* const> /*c*/, autan::MemorySpace /*s*/, void* /*stream*/) override {
  }
  void forward(std::span<double* const> /*c*/, autan::MemorySpace /*s*/,
               void* /*stream*/) override {}
};

// A smooth subsonic flow over the unit cube: rho, u, v, w, p at (x, y, z).
std::array<double, 5> flow(double x, double y, double z) {
  return {1.2 + 0.1 * std::sin(2.0 * x + y), 80.0 + 15.0 * std::cos(x + 2.0 * z),
          -20.0 + 10.0 * std::sin(y + z), 5.0 * std::cos(3.0 * x),
          1.0e5 + 4.0e3 * std::sin(x + y + z)};
}

std::array<double, 5> conserved(const std::array<double, 5>& w) {
  const double q2 = w[1] * w[1] + w[2] * w[2] + w[3] * w[3];
  return {w[0], w[0] * w[1], w[0] * w[2], w[0] * w[3], w[4] / (air.gamma - 1.0) + 0.5 * w[0] * q2};
}

}  // namespace

TEMPLATE_TEST_CASE("one device residual equals the host's to round-off, cell by cell",
                   "[euler][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // |R_device - R_host| <= 500 eps f_k A_i: eps of the storage precision, f_k the face flux
  // magnitude of the reference state (rho q, rho q^2, rho H q with q = |v| + c) and A_i the
  // surface of cell i (docs/contracts/case-schema.md, Time: Phi_k taken per cell). Both sum the
  // same face fluxes, in different orders and with or without fused multiply-adds.
  using P = TestType;
  const char* file = GENERATE(from_range(meshes));
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  CAPTURE(file, static_cast<int>(flux), limiter);
  const auto mesh = autan::io::read_mesh(mesh_dir / file);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  const auto ref = flow(0.5, 0.5, 0.5);
  Scheme s;
  s.flux = flux;
  s.limiter = limiter;
  const double c_ref = std::sqrt(air.gamma * ref[4] / ref[0]);
  s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / c_ref;
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  s.reference = conserved(ref);
  Euler<P> host(mesh, g, air, s);
  // Every condition: farfield on x, slip walls on y, the flow's state on z.
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    std::array<double, 5> w = ref;
    if (zone[0] == 'x') {
      kind[b] = BoundaryKind::farfield;
    } else if (zone[0] == 'y') {
      kind[b] = BoundaryKind::slip;
    } else {
      kind[b] = BoundaryKind::state;
      w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    }
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  host.set_boundary(kind, data);
  std::vector<double> source(5 * n), u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto w =
        conserved(flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]));
    for (std::size_t k = 0; k < 5; ++k) {
      u[k * n + c] = w[k];
      source[k * n + c] = 1e-3 * w[k];
    }
  }
  host.set_source(source);
  host.set_second_order(true);
  std::vector<typename P::state> stored(5 * n);
  host.state(u, stored);
  std::vector<typename P::residual> r_host(5 * n);
  host.residual(stored, r_host);

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device =
      autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, true, rank, ghosts, nullptr);
  std::vector<double> r_device(5 * n);
  device->residual(u, r_device);

  const double q = std::hypot(ref[1], ref[2], ref[3]) + c_ref;
  const double h = c_ref * c_ref / (air.gamma - 1.0) + 0.5 * std::pow(q - c_ref, 2);
  const std::array<double, 5> f_k{ref[0] * q, ref[0] * q * q, ref[0] * q * q, ref[0] * q * q,
                                  ref[0] * h * q};
  std::vector<double> surface(n, 0.0);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
    if (f < ni) surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
  }
  const double eps = std::numeric_limits<typename P::state>::epsilon();
  double worst = 0.0;
  for (std::size_t k = 0; k < 5; ++k) {
    for (std::size_t c = 0; c < n; ++c) {
      const double d = std::abs(r_device[k * n + c] - static_cast<double>(r_host[k * n + c]));
      worst = std::max(worst, d / (eps * f_k[k] * surface[c]));
    }
  }
  INFO("worst difference " << worst << " eps f_k A_i");
  CHECK(worst <= 500.0);
}
