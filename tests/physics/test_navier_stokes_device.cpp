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
#include <stdexcept>
#include <string>
#include <type_traits>
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

  // Farfield on xmin (the flow's state as the freestream) and a subsonic inlet on zmin (total
  // pressure and temperature of the flow's state there): the conditions whose boundary states
  // call exp and log. ymin becomes slip and zmax an outlet: a wall or a given state beside them
  // can leave the viscous boundary fit undetermined, and those conditions are criterion 1's.
  void farfield_and_inlet(bool zmax_outlet = true) {
    const double cp = air.gamma * air.gas_constant / (air.gamma - 1.0);
    for (std::size_t b = 0; b < nb; ++b) {
      const auto f = ni + b;
      const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
      const auto w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
      std::array<double, 5> d = w;
      if (zone == "ymin") {
        kind[b] = BoundaryKind::slip;
        continue;
      }
      if (zone == "zmax" && zmax_outlet) {
        kind[b] = BoundaryKind::outlet;
        continue;
      }
      if (zone == "xmin") {
        kind[b] = BoundaryKind::farfield;
      } else if (zone == "zmin") {
        kind[b] = BoundaryKind::inlet;
        const double t_static = w[4] / (w[0] * air.gas_constant);
        const double t0 = t_static + (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]) / (2.0 * cp);
        d = {w[4] * std::pow(t0 / t_static, air.gamma / (air.gamma - 1.0)), t0, 0.0, 0.0, 0.0};
      } else {
        continue;
      }
      for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = d[k];
    }
  }

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
  std::vector<double> rnu(n), dist(n), nt_b(pb.nb, 0.0), nt_source(n);
  double d_min = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < n; ++c) {
    rnu[c] = pb.u[c] * nt_at(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    dist[c] = g.cell_centroid_y[c] + 0.05;
    nt_source[c] = 0.1 * std::cos(4.0 * g.cell_centroid_z[c] - g.cell_centroid_x[c]);
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
    m->set_source(nt_source);
    return m;
  };
  std::vector<typename P::state> rs(n);
  for (std::size_t c = 0; c < n; ++c) rs[c] = typename P::state(rnu[c]);
  std::vector<typename P::residual> r_flow(5 * n), r_sa(n), r_flow_t(5 * n);
  host.residual(stored, r_flow);
  auto sa = model(true, 0.0);
  sa->residual(rs, r_sa);
  REQUIRE(finite(r_sa));
  // An eddy viscosity the model must overwrite everywhere, zeros included: the device copies it
  // now, before the model sets it.
  std::vector<double> stale(n, 0.25), stale_b(pb.nb, 0.15);
  host.set_eddy_viscosity(stale, stale_b);
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(pb.mesh, pb.g, host, air.gamma, true, rank,
                                                       ghosts, nullptr, &pb.keys);
  auto sa_device = autan::physics::make_device_spalart_allmaras<P>(*sa, *device);
  {
    // Only the stepper of the flow the model is coupled to.
    NavierStokes<P> other(pb.mesh, pb.g, air, pb.s, pb.t, &pb.keys);
    other.set_boundary(pb.kind, pb.data);
    auto other_device = autan::physics::make_device_stepper<P>(
        pb.mesh, pb.g, other, air.gamma, true, rank, ghosts, nullptr, &pb.keys);
    CHECK_THROWS_AS(autan::physics::make_device_spalart_allmaras<P>(*sa, *other_device),
                    std::invalid_argument);
  }

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
    // The tolerance is tight: leaving out ft2, or the wall's terms (destruction and the wall term
    // of S_tilde, the wall moved far away), moves the residual by more than ten times it.
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
  } else {
    // nt < 0 gives no eddy viscosity: the stale one is gone and the flow is laminar again.
    REQUIRE(bit_difference(r_flow, std::vector<double>(r_flow_t.begin(), r_flow_t.end())).first ==
            0);
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

TEMPLATE_TEST_CASE("the device's residual with farfield and inlet faces stays within #586's bound",
                   "[navier-stokes][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #586, criterion 2(b), as revised on 2026-11-08 before any device comparison: farfield_state
  // and inlet_state call exp and log, whose last bits differ between glibc and libdevice. Each
  // boundary state then differs by at most 32 (1 + L) eps_C |w_b|, L the largest |log| of the
  // densities and temperature ratios involved. It reaches the residual through its own face
  // (gain 1), the reconstructed face values (gain G, the stencils' row sum) and the viscous
  // gradients (gain G_v), each face flux Lipschitz with a constant <= 2 (a stated hypothesis,
  // supported below from the data side): |R_dev - R_host| <= 64 (1 + L) eps_C A_i
  // [f_k max(1, G) + f^v_k G_v] per cell and component. G and G_v come from the host's weights
  // and geometry and are reported.
  using P = TestType;
  const Source source = GENERATE(from_range(sources));
  const auto flux = GENERATE(Flux::roe, Flux::hllc, Flux::ausm_up);
  const auto fit = GENERATE(GradientFit::linear, GradientFit::quadratic);
  // The first runs' layout as well (zmax a given state), on the mesh and fit where it gives the
  // largest gain measured, G = 4.79 (poly, quadratic): the coverage of W0717 reaches it (#586,
  // run gpunssa-d7-2b32).
  const bool zmax_state = GENERATE(false, true);
  if (zmax_state && !(std::holds_alternative<const char*>(source) &&
                      std::string(std::get<const char*>(source)) == "poly/polyMesh" &&
                      fit == GradientFit::quadratic)) {
    return;
  }
  CAPTURE(name_of(source), static_cast<int>(flux), static_cast<int>(fit), zmax_state);
  Problem pb(source, flux, true, fit, 300.0);
  pb.farfield_and_inlet(!zmax_state);
  const auto n = pb.n;
  const auto& g = pb.g;
  const auto& mesh = pb.mesh;

  // f_k: the face flux magnitude of the reference state (as the Euler device test); f^v_k:
  // mu_max q_ref / l_min for momentum, mu_max q_ref^2 / l_min + k_max dT / l_min for energy,
  // l_min the shortest centroid-to-centroid or centroid-to-face distance, dT the range of T.
  const auto ref = flow(0.5, 0.5, 0.5);
  const double c_ref = std::sqrt(air.gamma * ref[4] / ref[0]);
  const double q = std::hypot(ref[1], ref[2], ref[3]) + c_ref;
  const double h = c_ref * c_ref / (air.gamma - 1.0) + 0.5 * std::pow(q - c_ref, 2);
  std::array<double, 5> f_k{ref[0] * q, ref[0] * q * q, ref[0] * q * q, ref[0] * q * q,
                            ref[0] * h * q};
  double t_min = std::numeric_limits<double>::infinity(), t_max = 0.0, big_l = 0.0;
  auto temperature = [&](const std::array<double, 5>& w) {
    return w[4] / (w[0] * air.gas_constant);
  };
  for (std::size_t c = 0; c < n; ++c) {
    const auto w = flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
    t_min = std::min(t_min, temperature(w));
    t_max = std::max(t_max, temperature(w));
    big_l = std::max(big_l, std::abs(std::log(w[0])));
  }
  for (std::size_t b = 0; b < pb.nb; ++b) {
    const auto f = pb.ni + b;
    const auto w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
    if (pb.kind[b] == BoundaryKind::farfield) big_l = std::max(big_l, std::abs(std::log(w[0])));
    if (pb.kind[b] == BoundaryKind::inlet) {
      big_l = std::max(big_l, std::abs(std::log(temperature(w) / pb.data[pb.nb + b])));
    }
  }
  const auto& tr = pb.t;
  auto mu = [&](double t) {
    return tr.mu_ref * std::pow(t / tr.t_ref, 1.5) * (tr.t_ref + tr.s) / (t + tr.s);
  };
  const double mu_max = std::max(mu(t_min), mu(t_max));
  const double cp = air.gamma * air.gas_constant / (air.gamma - 1.0);
  double l_min = std::numeric_limits<double>::infinity();
  std::vector<std::vector<std::size_t>> faces_of(n);
  for (std::size_t f = 0; f < pb.ni + pb.nb; ++f) {
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    faces_of[o].push_back(f);
    auto dist = [&](std::size_t c) {
      return std::hypot(g.face_centroid_x[f] - g.cell_centroid_x[c],
                        g.face_centroid_y[f] - g.cell_centroid_y[c],
                        g.face_centroid_z[f] - g.cell_centroid_z[c]);
    };
    l_min = std::min(l_min, dist(o));
    if (f < pb.ni) {
      const auto nb_cell = static_cast<std::size_t>(mesh.neighbour()[f]);
      faces_of[nb_cell].push_back(f);
      l_min = std::min(l_min, dist(nb_cell));
    }
  }
  std::array<double, 5> f_v{
      0.0, mu_max * q / l_min, mu_max * q / l_min, mu_max * q / l_min,
      mu_max * q * q / l_min + cp * mu_max / tr.prandtl * (t_max - t_min) / l_min};
  std::vector<double> surface(n, 0.0);
  for (std::size_t f = 0; f < pb.ni + pb.nb; ++f) {
    surface[static_cast<std::size_t>(mesh.owner()[f])] += g.face_area[f];
    if (f < pb.ni) surface[static_cast<std::size_t>(mesh.neighbour()[f])] += g.face_area[f];
  }
  const double eps = std::numeric_limits<typename P::compute>::epsilon();
  double gain = 1.0, gain_v = 0.0;  // max(1, G) and G_v, set below from the host's weights
  // The largest |R - R_ref| of a cell and component, in units of
  // `unit` A_i [f_k max(1, G) + f^v_k G_v].
  auto scaled = [&](const auto& r, const auto& r_ref, double unit) {
    double worst = 0.0;
    for (std::size_t k = 0; k < 5; ++k) {
      for (std::size_t c = 0; c < n; ++c) {
        const double d =
            std::abs(static_cast<double>(r[k * n + c]) - static_cast<double>(r_ref[k * n + c]));
        if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
        worst = std::max(worst, d / (unit * (f_k[k] * gain + f_v[k] * gain_v) * surface[c]));
      }
    }
    return worst;
  };
  const double tolerance = 64.0 * (1.0 + big_l) * eps;

  // One flow for every data set: its boundary fit depends on the conditions only
  // (NavierStokes<P>::set_boundary_data), so it is built once.
  NavierStokes<P> probe(mesh, g, air, pb.s, pb.t, &pb.keys);
  probe.set_boundary(pb.kind, pb.data);
  probe.set_second_order(true);
  std::vector<typename P::state> probe_state(5 * n);
  probe.state(pb.u, probe_state);
  auto host_residual = [&](const std::vector<double>& data) {
    probe.set_boundary_data(data);
    std::vector<typename P::residual> r(5 * n);
    probe.residual(probe_state, r);
    return r;
  };
  NavierStokes<P> host(mesh, g, air, pb.s, pb.t, &pb.keys);
  host.set_boundary(pb.kind, pb.data);
  host.set_second_order(true);
  std::vector<typename P::state> stored(5 * n);
  host.state(pb.u, stored);
  std::vector<typename P::residual> r_host(5 * n);
  host.residual(stored, r_host);
  REQUIRE(finite(r_host));
  REQUIRE(bit_difference(host_residual(pb.data), r_host).first == 0);

  {
    // G_i over the gradient stencils the residual reads: the linear one and, under the
    // quadratic fit, the quadratic one.
    auto amplification = [&](std::span<const std::int64_t> offsets,
                             std::span<const autan::index_t> entries,
                             std::span<const double> weights) {
      const auto ne = entries.size();
      double worst = 0.0;
      for (std::size_t c = 0; c + 1 < offsets.size(); ++c) {
        double reach = 0.0;
        for (const auto f : faces_of[c]) {
          reach = std::max(reach, std::hypot(g.face_centroid_x[f] - g.cell_centroid_x[c],
                                             g.face_centroid_y[f] - g.cell_centroid_y[c],
                                             g.face_centroid_z[f] - g.cell_centroid_z[c]));
        }
        double sum = 0.0;
        for (auto e = static_cast<std::size_t>(offsets[c]);
             e < static_cast<std::size_t>(offsets[c + 1]); ++e) {
          sum += std::hypot(weights[e], weights[ne + e], weights[2 * ne + e]);
        }
        worst = std::max(worst, sum * reach);
      }
      return worst;
    };
    const auto& st = host.euler().stencil();
    const std::vector<std::int64_t> offsets(st.offsets.begin(), st.offsets.end());
    const auto weights = host.euler().gradient_weights();
    double g_max = amplification(offsets, st.entries, weights);
    const double g_linear = g_max;
    double g_quadratic = 0.0;
    if (fit == GradientFit::quadratic) {
      const auto qg = host.euler().quadratic_gradient();
      g_quadratic = amplification(qg.offsets, qg.entries, qg.weights);
      g_max = std::max(g_max, g_quadratic);
    }
    // G_v = max_i H_i l_min, H_i the sum of |w_e| over the linear stencil.
    const auto ne = st.entries.size();
    for (std::size_t c = 0; c + 1 < offsets.size(); ++c) {
      double h_i = 0.0;
      for (auto e = static_cast<std::size_t>(offsets[c]);
           e < static_cast<std::size_t>(offsets[c + 1]); ++e) {
        h_i += std::hypot(weights[e], weights[ne + e], weights[2 * ne + e]);
      }
      gain_v = std::max(gain_v, h_i * l_min);
    }
    gain = std::max(1.0, g_max);
    WARN(name_of(source) << " fit " << static_cast<int>(fit) << ": G = " << g_linear
                         << " (linear), " << g_quadratic << " (quadratic), G_v = " << gain_v);
    // The stated Lipschitz hypothesis, from the data side: each of the 5 data components of each
    // farfield and inlet face perturbed by a relative delta on its own; the absolute row sums
    // sum |dR_ik| / delta stay within 2 A_i [f_k max(1, G) + f^v_k G_v].
    constexpr double delta = 1e-6;
    std::vector<double> rows(5 * n, 0.0);
    for (std::size_t b = 0; b < pb.nb; ++b) {
      if (pb.kind[b] != BoundaryKind::farfield && pb.kind[b] != BoundaryKind::inlet) continue;
      for (std::size_t j = 0; j < 5; ++j) {
        const double v = pb.data[j * pb.nb + b];
        if (v == 0.0) continue;
        auto moved = pb.data;
        moved[j * pb.nb + b] = v * (1.0 + delta);
        const auto r = host_residual(moved);
        for (std::size_t i = 0; i < 5 * n; ++i) {
          rows[i] += std::abs(static_cast<double>(r[i]) - static_cast<double>(r_host[i])) / delta;
        }
      }
    }
    const std::vector<double> zero(5 * n, 0.0);
    const double row_sum = scaled(rows, zero, 1.0);
    INFO("row sums " << row_sum << " A_i [f_k max(1, G) + f^v_k G_v]");
    REQUIRE(row_sum <= 2.0);
    // Tight (#586): the interior's entropy in place of the freestream's on every farfield face,
    // and nothing else: density and pressure at the freestream's p / rho (so its speed of sound,
    // velocity and outgoing invariant are kept), with p / rho^gamma the face cell's.
    auto interior = pb.data;
    for (std::size_t b = 0; b < pb.nb; ++b) {
      if (pb.kind[b] != BoundaryKind::farfield) continue;
      const auto c = static_cast<std::size_t>(mesh.owner()[pb.ni + b]);
      const auto w = flow(g.cell_centroid_x[c], g.cell_centroid_y[c], g.cell_centroid_z[c]);
      const double entropy = w[4] / std::pow(w[0], air.gamma);
      const double rho_f = pb.data[b], p_f = pb.data[4 * pb.nb + b];
      const double rt = p_f / rho_f;
      const double rho = std::pow(entropy / rt, 1.0 / (1.0 - air.gamma));
      interior[b] = rho;
      interior[4 * pb.nb + b] = rho * rt;
    }
    const double tightness = scaled(host_residual(interior), r_host, tolerance);
    INFO("tightness " << tightness);
    CHECK(tightness > 10.0);
  }

  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(mesh, g, host, air.gamma, true, rank, ghosts,
                                                       nullptr, &pb.keys);
  std::vector<double> r_device(5 * n);
  device->residual(pb.u, r_device);
  REQUIRE(finite(r_device));
  const auto [differ, ulps] = bit_difference(r_host, r_device);
  const double distance = scaled(r_device, r_host, tolerance);
  INFO(differ << " of " << 5 * n << " components differ, by at most " << ulps << " ulp; "
              << distance << " times the tolerance");
  CHECK(distance <= 1.0);
}

TEMPLATE_TEST_CASE("an unsteady laminar run on the device repeats the host's bits at every step",
                   "[navier-stokes][gpu]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // #586, criterion 4a: N = 20 SSP-RK3 steps (Shu and Osher 1988, eq. 2.18) of one global time
  // step at CFL 0.5, the host's EulerPhysics::unsteady_step repeated here on the host residual
  // under the mesh's own numbering as keys, against DeviceStepper::step. Every component of the
  // state and the time step have the same bits after every step.
  using P = TestType;
  const Source source = GENERATE(from_range(sources));
  const auto fit = GENERATE(GradientFit::linear, GradientFit::quadratic);
  CAPTURE(name_of(source), static_cast<int>(fit));
  const Problem pb(source, Flux::roe, true, fit, 300.0);
  const auto n = pb.n;
  constexpr int steps = 20;
  constexpr double cfl = 0.5;

  NavierStokes<P> host(pb.mesh, pb.g, air, pb.s, pb.t, &pb.keys);
  host.set_boundary(pb.kind, pb.data);
  host.set_second_order(true);
  autan::runtime::OneRank rank;
  NoGhosts ghosts;
  auto device = autan::physics::make_device_stepper<P>(pb.mesh, pb.g, host, air.gamma, true, rank,
                                                       ghosts, nullptr, &pb.keys);
  device->upload(pb.u);

  std::vector<double> u = pb.u, stage(5 * n), on_device(5 * n);
  std::vector<typename P::state> stored(5 * n), lambda(n);
  std::vector<typename P::residual> res(5 * n);
  auto residual_of = [&](const std::vector<double>& v) {
    host.state(v, stored);
    host.residual(stored, res);
  };
  const std::array<std::array<double, 2>, 3> weights{
      {{0.0, 1.0}, {0.75, 0.25}, {1.0 / 3.0, 2.0 / 3.0}}};
  for (int step = 0; step < steps; ++step) {
    CAPTURE(step);
    residual_of(u);
    host.spectral_radius(lambda);
    double dt = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < n; ++c) {
      dt = std::min(dt, cfl * pb.g.cell_volume[c] / static_cast<double>(lambda[c]));
    }
    stage = u;
    for (std::size_t s = 0; s < 3; ++s) {
      if (s > 0) residual_of(stage);
      for (std::size_t k = 0; k < 5; ++k) {
        for (std::size_t c = 0; c < n; ++c) {
          const auto i = k * n + c;
          const double next = stage[i] - dt / pb.g.cell_volume[c] * static_cast<double>(res[i]);
          stage[i] = weights[s][0] * u[i] + weights[s][1] * next;
        }
      }
    }
    u = stage;
    REQUIRE(finite(u));
    const auto r = device->step(cfl, std::nullopt, false);
    device->download(on_device);
    CHECK(std::bit_cast<std::uint64_t>(r.time_step) == std::bit_cast<std::uint64_t>(dt));
    const auto [differ, ulps] = bit_difference(u, on_device);
    INFO(differ << " of " << 5 * n << " components differ, by at most " << ulps << " ulp");
    REQUIRE(differ == 0);
  }
}
