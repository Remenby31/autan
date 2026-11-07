// The Navier-Stokes residual on a CUDA device (make_device_stepper of a NavierStokes<P>) against
// the host's, cell by cell (#586).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/navier_stokes_device.hpp"
#include "autan/runtime/ranks.hpp"
#include "fixtures.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BoundaryKind;
using autan::physics::Euler;
using autan::physics::Flux;
using autan::physics::Gas;
using autan::physics::GradientFit;
using autan::physics::NavierStokes;
using autan::physics::Scheme;
using autan::physics::Transport;
using autan::test::BoxFill;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
constexpr Gas air{1.4, 287.0};

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

// The test meshes as read, and boxes whose interior nodes move by 20 % of the spacing.
using Source = std::variant<const char*, BoxFill>;
autan::Mesh mesh_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s))
    return autan::io::read_mesh(mesh_dir / *file);
  return autan::assemble(autan::to_face_based(autan::test::box(4, std::get<BoxFill>(s), 0.2)));
}

std::string name_of(const Source& s) {
  if (const auto* file = std::get_if<const char*>(&s)) return *file;
  return "perturbed " + autan::test::to_string(std::get<BoxFill>(s));
}

}  // namespace

TEMPLATE_TEST_CASE("the device's Navier-Stokes residual has the host's bits",
                   "[navier-stokes][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #586, criterion 1: under the mesh's own numbering as global keys the host gathers each cell's
  // faces in the device's order, and every operation of these conditions (state, slip, outlet,
  // walls) is correctly rounded on both sides, so every component has the same bits.
  using P = TestType;
  const Source source =
      GENERATE(Source{"tet-gmsh/tet-gmsh.ascii.msh"}, Source{"hex/hex.ascii.msh"},
               Source{"poly/polyMesh"}, Source{"hybrid/hybrid.ascii.msh"}, Source{BoxFill::tetra},
               Source{BoxFill::hexa}, Source{BoxFill::prism}, Source{BoxFill::hybrid});
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  const auto fit = GENERATE(GradientFit::linear, GradientFit::quadratic);
  const double t_wall = GENERATE(0.0, 300.0);  // adiabatic, isothermal
  CAPTURE(name_of(source), static_cast<int>(flux), limiter, static_cast<int>(fit), t_wall);
  const auto mesh = mesh_of(source);
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
  std::vector<std::int64_t> face_id(ni + nb);
  std::iota(face_id.begin(), face_id.end(), std::int64_t{0});
  const std::vector<std::uint8_t> not_reversed(ni + nb, 0);
  const autan::kernels::GlobalKeys keys{
      .cell = mesh.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};

  const auto ref = flow(0.5, 0.5, 0.5);
  Scheme s;
  s.flux = flux;
  s.limiter = limiter;
  s.gradient_fit = fit;
  s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / std::sqrt(air.gamma * ref[4] / ref[0]);
  s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
  s.reference = conserved(ref);
  s.venkat_k = 0.1;
  // A viscosity at which the viscous fluxes are of the order of the convective ones.
  Transport t;
  t.mu_ref = 0.5;
  std::vector<BoundaryKind> kind(nb);
  std::vector<double> data(5 * nb, 0.0);
  for (std::size_t b = 0; b < nb; ++b) {
    const auto f = ni + b;
    const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
    auto w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    if (zone == "xmax") {
      kind[b] = BoundaryKind::outlet;
    } else if (zone == "ymin") {
      kind[b] = BoundaryKind::wall;
      w = {t_wall, 0.0, 0.0, 0.0, 0.0};
    } else if (zone == "ymax") {
      kind[b] = BoundaryKind::slip;
    } else {
      kind[b] = BoundaryKind::state;
    }
    for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
  }
  std::vector<double> u(5 * n);
  for (std::size_t c = 0; c < n; ++c) {
    const auto w =
        conserved(flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]));
    for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = w[k];
  }

  NavierStokes<P> host(mesh, g, air, s, t, &keys);
  host.set_boundary(kind, data);
  host.set_second_order(true);
  std::vector<typename P::state> stored(5 * n);
  host.state(u, stored);
  std::vector<typename P::residual> r_host(5 * n);
  host.residual(stored, r_host);

  // The viscous fluxes change nearly every momentum and energy residual: a device that left them
  // out, or the boundary fit, could not pass.
  Euler<P> inviscid(mesh, g, air, s, &keys);
  inviscid.set_boundary(kind, data);
  inviscid.set_second_order(true);
  std::vector<typename P::residual> r_euler(5 * n);
  inviscid.residual(stored, r_euler);
  std::size_t viscous = 0;
  for (std::size_t i = n; i < 5 * n; ++i) viscous += r_host[i] != r_euler[i] ? 1 : 0;
  REQUIRE(viscous >= 4 * n * 9 / 10);

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, true, rank, ghosts,
                                                       nullptr, &keys);
  std::vector<double> r_device(5 * n);
  device->residual(u, r_device);
  // Distance in units in the last place of a double, from the ordered bit patterns.
  auto ordered = [](std::int64_t v) {
    return v < 0 ? std::numeric_limits<std::int64_t>::min() - v : v;
  };
  std::size_t differ = 0;
  std::uint64_t ulps = 0;
  for (std::size_t i = 0; i < r_device.size(); ++i) {
    const auto a = std::bit_cast<std::int64_t>(static_cast<double>(r_host[i]));
    const auto b = std::bit_cast<std::int64_t>(r_device[i]);
    if (a == b) continue;
    ++differ;
    const auto d = static_cast<std::uint64_t>(ordered(a)) - static_cast<std::uint64_t>(ordered(b));
    ulps = std::max(ulps, std::min(d, std::uint64_t{0} - d));
  }
  INFO(differ << " of " << r_device.size() << " components differ, by at most " << ulps
              << " ulp of the double they are read as");
  CHECK(differ == 0);
}
