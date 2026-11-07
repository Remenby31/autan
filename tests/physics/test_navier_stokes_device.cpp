// The Navier-Stokes and Spalart-Allmaras residuals on a CUDA device against the host's, cell by
// cell (#586).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/physics/euler.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/navier_stokes_device.hpp"
#include "autan/physics/spalart_allmaras.hpp"
#include "autan/physics/spalart_allmaras_device.hpp"
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
using autan::physics::SpalartAllmaras;
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

// The mesh's own numbering as global keys, the scheme and the conditions of the tests below
// (outlet on xmax, a no-slip wall on ymin, slip on ymax, the flow's state elsewhere: no exp or log
// in any boundary state), and the conserved variables of the flow at the cell centroids.
struct Problem {
  Problem(const Source& source, Flux flux, bool limiter, GradientFit fit, double t_wall)
      : mesh(mesh_of(source)),
        g(autan::compute_geometry(mesh)),
        n(static_cast<std::size_t>(mesh.n_cells())),
        ni(static_cast<std::size_t>(mesh.n_interior_faces())),
        nb(static_cast<std::size_t>(mesh.n_boundary_faces())),
        face_id(ni + nb),
        not_reversed(ni + nb, 0),
        kind(nb),
        data(5 * nb, 0.0),
        u(5 * n) {
    std::iota(face_id.begin(), face_id.end(), std::int64_t{0});
    keys = {.cell = mesh.cell_global_id(), .face = face_id, .reversed = not_reversed, .shift = {}};
    const auto ref = flow(0.5, 0.5, 0.5);
    s.flux = flux;
    s.limiter = limiter;
    s.gradient_fit = fit;
    s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / std::sqrt(air.gamma * ref[4] / ref[0]);
    s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
    s.reference = conserved(ref);
    s.venkat_k = 0.1;
    // A viscosity at which the viscous fluxes are of the order of the convective ones.
    t.mu_ref = 0.5;
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
    for (std::size_t c = 0; c < n; ++c) {
      const auto w =
          conserved(flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]));
      for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = w[k];
    }
  }
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;

  autan::Mesh mesh;
  autan::Geometry g;
  std::size_t n, ni, nb;
  std::vector<std::int64_t> face_id;
  std::vector<std::uint8_t> not_reversed;
  autan::kernels::GlobalKeys keys;
  Scheme s;
  Transport t;
  std::vector<BoundaryKind> kind;
  std::vector<double> data, u;
};

template <class R>
bool finite(const R& r) {
  return std::ranges::all_of(r, [](auto x) { return std::isfinite(static_cast<double>(x)); });
}

// How many components differ in their bits, and by at most how many units in the last place of
// a double, from the ordered bit patterns.
template <class R>
std::pair<std::size_t, std::uint64_t> bit_difference(const R& host,
                                                     std::span<const double> device) {
  auto ordered = [](std::int64_t v) {
    return v < 0 ? std::numeric_limits<std::int64_t>::min() - v : v;
  };
  std::size_t differ = 0;
  std::uint64_t ulps = 0;
  for (std::size_t i = 0; i < device.size(); ++i) {
    const auto a = std::bit_cast<std::int64_t>(static_cast<double>(host[i]));
    const auto b = std::bit_cast<std::int64_t>(device[i]);
    if (a == b) continue;
    ++differ;
    const auto d = static_cast<std::uint64_t>(ordered(a)) - static_cast<std::uint64_t>(ordered(b));
    ulps = std::max(ulps, std::min(d, std::uint64_t{0} - d));
  }
  return {differ, ulps};
}

const std::array<Source, 8> sources = {Source{"tet-gmsh/tet-gmsh.ascii.msh"},
                                       Source{"hex/hex.ascii.msh"},
                                       Source{"poly/polyMesh"},
                                       Source{"hybrid/hybrid.ascii.msh"},
                                       Source{BoxFill::tetra},
                                       Source{BoxFill::hexa},
                                       Source{BoxFill::prism},
                                       Source{BoxFill::hybrid}};

}  // namespace

TEMPLATE_TEST_CASE("the device's Navier-Stokes residual has the host's bits",
                   "[navier-stokes][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #586, criterion 1: under the mesh's own numbering as global keys the host gathers each cell's
  // faces in the device's order, and every operation of these conditions (state, slip, outlet,
  // walls) is correctly rounded on both sides, so every component has the same bits.
  using P = TestType;
  const Source source = GENERATE(from_range(sources));
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const bool limiter = GENERATE(false, true);
  const auto fit = GENERATE(GradientFit::linear, GradientFit::quadratic);
  const double t_wall = GENERATE(0.0, 300.0);  // adiabatic, isothermal
  CAPTURE(name_of(source), static_cast<int>(flux), limiter, static_cast<int>(fit), t_wall);
  const Problem pb(source, flux, limiter, fit, t_wall);
  const auto n = pb.n;

  NavierStokes<P> host(pb.mesh, pb.g, air, pb.s, pb.t, &pb.keys);
  host.set_boundary(pb.kind, pb.data);
  host.set_second_order(true);
  std::vector<typename P::state> stored(5 * n);
  host.state(pb.u, stored);
  // The fraction of momentum and energy residuals two evaluations give different values.
  auto changed = [&](const auto& x, const auto& y) {
    std::size_t m = 0;
    for (std::size_t i = n; i < 5 * n; ++i) m += x[i] != y[i] ? 1 : 0;
    return static_cast<double>(m) / static_cast<double>(4 * n);
  };
  std::vector<typename P::residual> r_laminar(5 * n), r_host(5 * n);
  host.residual(stored, r_laminar);
  // An eddy viscosity of the order of the laminar one, different in every cell and boundary
  // face, so that a device reading it wrongly or not at all could not pass.
  std::vector<double> mu_t(n), mu_t_b(pb.nb);
  for (std::size_t c = 0; c < n; ++c) {
    mu_t[c] = 0.3 * (1.0 + std::sin(5.0 * pb.g.cell_centroid_x[c] + 3.0 * pb.g.cell_centroid_y[c]) *
                               std::cos(4.0 * pb.g.cell_centroid_z[c]));
  }
  for (std::size_t b = 0; b < pb.nb; ++b) {
    const auto f = pb.ni + b;
    mu_t_b[b] =
        0.2 * (1.0 + std::cos(3.0 * pb.g.face_centroid_x[f] - 2.0 * pb.g.face_centroid_z[f]));
  }
  host.set_eddy_viscosity(mu_t, mu_t_b);
  host.residual(stored, r_host);
  REQUIRE(finite(r_laminar));
  REQUIRE(finite(r_host));
  REQUIRE(changed(r_host, r_laminar) >= 0.9);

  // The viscous fluxes change nearly every momentum and energy residual: a device that left them
  // out, or the boundary fit, could not pass.
  Euler<P> inviscid(pb.mesh, pb.g, air, pb.s, &pb.keys);
  inviscid.set_boundary(pb.kind, pb.data);
  inviscid.set_second_order(true);
  std::vector<typename P::residual> r_euler(5 * n);
  inviscid.residual(stored, r_euler);
  REQUIRE(finite(r_euler));
  REQUIRE(changed(r_laminar, r_euler) >= 0.9);

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(pb.mesh, pb.g, host, air.gamma, true, rank,
                                                       ghosts, nullptr, &pb.keys);
  std::vector<double> r_device(5 * n);
  device->residual(pb.u, r_device);
  REQUIRE(finite(r_device));
  const auto [differ, ulps] = bit_difference(r_host, r_device);
  INFO(differ << " of " << r_device.size() << " components differ, by at most " << ulps
              << " ulp of the double they are read as");
  CHECK(differ == 0);
}

TEMPLATE_TEST_CASE("the device's Spalart-Allmaras residual and eddy viscosity match the host's",
                   "[spalart-allmaras][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #586. With nt < 0 in every cell SA-neg's source takes its negative branch, which evaluates no
  // exp or log: the residual has the host's bits (criterion 1). With nt > 0, fw and ft2 call exp
  // and log, whose last bits differ between glibc and libdevice: |R_dev - R_host| <= 64 eps_C
  // s_ref V_i per cell (criterion 2a), s_ref = rho_max (cb1 Omega_max nt_max + 2 cw1 (nt_max /
  // d_min)^2) from the analytic fields below. Either way the eddy viscosity (fv1: no exp or log)
  // reaches the flow's next residual with the host's bits.
  using P = TestType;
  const Source source = GENERATE(from_range(sources));
  const double sign = GENERATE(-1.0, 1.0);
  CAPTURE(name_of(source), sign);
  const Problem pb(source, Flux::roe, true, GradientFit::quadratic, 300.0);
  const auto n = pb.n;
  const auto& g = pb.g;

  NavierStokes<P> host(pb.mesh, pb.g, air, pb.s, pb.t, &pb.keys);
  host.set_boundary(pb.kind, pb.data);
  host.set_second_order(true);
  std::vector<typename P::state> stored(5 * n);
  host.state(pb.u, stored);
  // nt of the order of the laminar nu = mu / rho ~ 0.4, so that the eddy viscosity matters; the
  // wall distance of the ymin wall plus 0.05.
  constexpr double nt_max = 1.5;
  auto nt_at = [&](double x, double y, double z) {
    return sign * (1.0 + 0.5 * std::sin(2.0 * x + 3.0 * y + z));
  };
  std::vector<double> rnu(n), dist(n), nt_b(pb.nb, 0.0);
  double d_min = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < n; ++c) {
    rnu[c] = pb.u[c] * nt_at(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    dist[c] = g.cell_centroid_y[c] + 0.05;
    d_min = std::min(d_min, dist[c]);
  }
  for (std::size_t b = 0; b < pb.nb; ++b) {
    const auto f = pb.ni + b;
    nt_b[b] = nt_at(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
  }
  auto model = [&](bool ft2, double far) {
    auto m = std::make_unique<SpalartAllmaras<P>>(pb.mesh, pb.g, host, pb.t, ft2);
    m->set_boundary(nt_b);
    std::vector<double> d(dist);
    for (auto& v : d) v += far;
    m->set_wall_distance(d);
    return m;
  };
  std::vector<typename P::state> rs(n);
  for (std::size_t c = 0; c < n; ++c) rs[c] = typename P::state(rnu[c]);
  std::vector<typename P::residual> r_flow(5 * n), r_sa(n), r_flow_t(5 * n);
  host.residual(stored, r_flow);
  auto sa = model(true, 0.0);
  sa->residual(rs, r_sa);
  REQUIRE(finite(r_sa));
  // The device copies the host's eddy viscosity now, before the model sets it: zero.
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(pb.mesh, pb.g, host, air.gamma, true, rank,
                                                       ghosts, nullptr, &pb.keys);
  auto sa_device = autan::physics::make_device_spalart_allmaras<P>(*sa, *device);

  // Omega_max: |curl v| of `flow` is at most sqrt(10^2 + 45^2) < 46.1; the discrete vorticity of
  // the gradients the model reads is checked against twice that.
  constexpr double omega_max = 2.0 * 46.1;
  {
    const auto a = host.arrays();
    double omega = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      auto gr = [&](std::size_t k, std::size_t j) {
        return static_cast<double>(a.flow.grad[(3 * k + j) * n + c]);
      };
      omega = std::max(omega,
                       std::hypot(gr(3, 1) - gr(2, 2), gr(1, 2) - gr(3, 0), gr(2, 0) - gr(1, 1)));
    }
    REQUIRE(omega <= omega_max);
  }
  constexpr double rho_max = 1.3, cb1 = 0.1355,
                   cw1 = cb1 / (0.41 * 0.41) + (1.0 + 0.622) / (2.0 / 3.0);
  const double s_ref =
      rho_max * (cb1 * omega_max * nt_max + 2.0 * cw1 * std::pow(nt_max / d_min, 2));
  const double eps = std::numeric_limits<typename P::compute>::epsilon();
  // The largest |R - R_host| of a cell, in units of 64 eps_C s_ref V_i.
  auto distance = [&](const auto& r) {
    double worst = 0.0;
    for (std::size_t c = 0; c < n; ++c) {
      const double d = std::abs(static_cast<double>(r[c]) - static_cast<double>(r_sa[c]));
      worst = std::max(worst, d / (64.0 * eps * s_ref * g.cell_volume[c]));
    }
    return worst;
  };
  if (sign > 0.0) {
    // The tolerance is tight: leaving out ft2, or the near-wall destruction (the wall moved far
    // away), moves the residual by more than ten times it.
    std::vector<typename P::residual> r_other(n);
    model(false, 0.0)->residual(rs, r_other);
    CHECK(distance(r_other) > 10.0);
    model(true, 1.0e6)->residual(rs, r_other);
    CHECK(distance(r_other) > 10.0);
  }
  sa->eddy_viscosity(rs);
  host.residual(stored, r_flow_t);
  REQUIRE(finite(r_flow_t));
  if (sign > 0.0) {
    std::size_t moved = 0;
    for (std::size_t i = n; i < 5 * n; ++i) moved += r_flow_t[i] != r_flow[i] ? 1 : 0;
    REQUIRE(moved >= 4 * n * 9 / 10);
  }

  std::vector<double> d_flow(5 * n), d_sa(n), d_flow_t(5 * n);
  device->residual(pb.u, d_flow);
  sa_device->residual(rnu, d_sa);
  sa_device->eddy_viscosity(rnu);
  device->residual(pb.u, d_flow_t);
  REQUIRE(finite(d_sa));
  {
    const auto [differ, ulps] = bit_difference(r_sa, d_sa);
    INFO("SA residual: " << differ << " of " << n << " differ, by at most " << ulps << " ulp; "
                         << distance(d_sa) << " times the tolerance");
    if (sign < 0.0) {
      CHECK(differ == 0);
    } else {
      CHECK(distance(d_sa) <= 1.0);
    }
  }
  const auto [differ, ulps] = bit_difference(r_flow_t, d_flow_t);
  INFO("flow residual with the eddy viscosity: " << differ << " of " << 5 * n
                                                 << " differ, by at most " << ulps << " ulp");
  CHECK(differ == 0);
}
