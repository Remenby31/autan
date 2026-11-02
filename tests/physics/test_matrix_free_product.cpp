// autan::physics::MatrixFreeProduct, the coupled flow and SA product of docs/design/coupled-sa.md,
// against central differences of the coupled residual, on the test meshes of tests/data/mesh.
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
#include "autan/physics/block_matrix.hpp"
#include "autan/physics/matrix_free_product.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"
#include "autan/runtime/exact_sum.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BoundaryKind;
using autan::physics::Gas;
using autan::physics::MatrixFreeProduct;
using autan::physics::NavierStokes;
using autan::physics::NtBounds;
using autan::physics::Scheme;
using autan::physics::SpalartAllmaras;
using autan::physics::Transport;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

constexpr Gas air{1.4, 287.0};
constexpr double mu = 0.5;
constexpr double rho = 1.2;
constexpr double nu = mu / rho;
constexpr double p0 = 1.0e5;

std::array<double, 5> primitive(double x, double y, double z) {
  return {rho, 30.0 * y, 6.0 * y, -10.0 * y, p0 + 2000.0 * x + 500.0 * z};
}

// The flow, SA and the product on one mesh, its conserved state, rho nt and a direction v.
template <class P>
struct Setup {
  using S = typename P::state;
  using R = typename P::residual;

  // `viscosity`: mu; `gauge`: the reference state the stored state is measured from, as a
  // multiple of the freestream's.
  Setup(const char* file, bool limiter, double viscosity = mu, double gauge = 1.0)
      : mesh(autan::io::read_mesh(mesh_dir / file)),
        g(autan::compute_geometry(mesh)),
        n(static_cast<std::size_t>(mesh.n_cells())),
        flow(mesh, g, air, scheme_of(limiter, gauge), transport(viscosity)),
        sa(mesh, g, flow, transport(viscosity)) {
    const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<BoundaryKind> kind(nb, BoundaryKind::state);
    std::vector<double> data(5 * nb), nt_b(nb, 2.0 * nu), wall(n);
    for (std::size_t b = 0; b < nb; ++b) {
      const auto f = ni + b;
      const auto w = primitive(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
      const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
      if (zone == "ymin") {
        kind[b] = BoundaryKind::wall;
      } else {
        for (std::size_t k = 0; k < 5; ++k) data[k * nb + b] = w[k];
      }
    }
    flow.set_boundary(kind, data);
    sa.set_boundary(nt_b);
    for (std::size_t c = 0; c < n; ++c) wall[c] = g.cell_centroid_y[c];
    sa.set_wall_distance(wall);
    u.resize(5 * n);
    v.resize(6 * n);
    shift.resize(n);
    for (std::size_t c = 0; c < n; ++c) {
      const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
      auto w = primitive(x, y, z);
      w[1] += std::sin(7.0 * x + 3.0 * z);
      w[4] += 300.0 * std::cos(5.0 * y);
      u[c] = w[0];
      for (std::size_t k = 0; k < 3; ++k) u[(1 + k) * n + c] = w[0] * w[1 + k];
      u[4 * n + c] =
          w[4] / (air.gamma - 1.0) + 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]);
      const double s = std::sin(2.0 * x + 3.0 * y - z), t = std::cos(x - 4.0 * y + 2.0 * z);
      v[c] = 0.01 * rho * s;
      for (std::size_t k = 1; k < 4; ++k)
        v[k * n + c] = rho * speed * (0.3 * t + 0.1 * static_cast<double>(k) * s);
      v[4 * n + c] = 0.02 * energy * t;
      v[5 * n + c] = rho * nu * (0.5 + s * t);
      shift[c] = 1.0e3 * (1.5 + s);
    }
  }

  static Scheme scheme_of(bool limiter, double gauge) {
    Scheme s;
    s.limiter = limiter;
    s.reference = {gauge * rho, 0.0, 0.0, 0.0, gauge * p0 / (air.gamma - 1.0)};
    return s;
  }
  static Transport transport(double viscosity) {
    Transport t;
    t.sutherland = false;
    t.mu_ref = viscosity;
    return t;
  }

  // The coupled residual at (u, rnu), its switches evaluated there (held: at freeze()'s state).
  std::vector<double> residual(std::span<const double> uu, std::span<const double> rnu, bool held) {
    std::vector<S> s(5 * n), t(n);
    flow.state(uu, s);
    for (std::size_t c = 0; c < n; ++c) t[c] = S(rnu[c]);
    std::vector<R> r(6 * n);
    if (held) {
      flow.residual_frozen(s, std::span(r).first(5 * n), [&] { sa.eddy_viscosity_frozen(t); });
      sa.residual_frozen(t, std::span(r).last(n));
    } else {
      flow.residual(s, std::span(r).first(5 * n));
      sa.eddy_viscosity(t);
      flow.residual(s, std::span(r).first(5 * n));
      sa.residual(t, std::span(r).last(n));
    }
    return {r.begin(), r.end()};
  }

  // freeze()'s order: the unheld residual and eddy viscosity at (u, rnu), then start().
  void start(MatrixFreeProduct<P>& a, std::span<const double> rnu) {
    std::vector<S> s(5 * n);
    flow.state(u, s);
    residual(u, rnu, false);
    a.start(u, s, rnu, nt_inf, shift, {});
  }

  // Root mean square of S^-1 x over the six unknowns, S the product's scales, as the product's
  // contract states eps (an exact sum, as the product sums).
  double scaled_rms(const MatrixFreeProduct<P>& a, std::span<const double> x) const {
    std::array<autan::runtime::ExactSum, 2> sums;
    for (std::size_t k = 0; k < 6; ++k) {
      for (std::size_t c = 0; c < n; ++c) {
        const double q = x[k * n + c] / a.scale()[k];
        sums[0].add(q * q);
      }
    }
    sums[1].add(static_cast<double>(6 * n));
    return std::sqrt(sums[0].value() / sums[1].value());
  }

  // eps = relative (1 + |S^-1 X|) / |S^-1 v|, X = (u, rnu).
  double eps(const MatrixFreeProduct<P>& a, std::span<const double> rnu, double relative) const {
    std::vector<double> x(u);
    x.insert(x.end(), rnu.begin(), rnu.end());
    return relative * (1.0 + scaled_rms(a, x)) / scaled_rms(a, v);
  }

  // [R(X + e v) - R(X)] / e, R the coupled residual with the switches held at start()'s state.
  std::vector<double> one_sided(std::span<const double> rnu, double e) {
    std::vector<double> up(5 * n), np(n);
    for (std::size_t i = 0; i < 5 * n; ++i) up[i] = u[i] + e * v[i];
    for (std::size_t c = 0; c < n; ++c) np[c] = rnu[c] + e * v[5 * n + c];
    const auto r = residual(up, np, true);
    const auto r0 = residual(u, rnu, true);
    std::vector<double> d(6 * n);
    for (std::size_t i = 0; i < 6 * n; ++i) d[i] = (r[i] - r0[i]) / e;
    return d;
  }

  // Central differences along v, the step h = epsilon^(1/3) on the scaled unknowns.
  std::vector<double> central(const MatrixFreeProduct<P>& a, std::span<const double> rnu) {
    const double h = eps(a, rnu, std::cbrt(std::numeric_limits<double>::epsilon()));
    std::vector<double> up(5 * n), um(5 * n), np(n), nm(n);
    for (std::size_t i = 0; i < 5 * n; ++i) {
      up[i] = u[i] + h * v[i];
      um[i] = u[i] - h * v[i];
    }
    for (std::size_t c = 0; c < n; ++c) {
      np[c] = rnu[c] + h * v[5 * n + c];
      nm[c] = rnu[c] - h * v[5 * n + c];
    }
    const auto plus = residual(up, np, true), minus = residual(um, nm, true);
    std::vector<double> d(6 * n);
    for (std::size_t i = 0; i < 6 * n; ++i) d[i] = (plus[i] - minus[i]) / (2.0 * h);
    return d;
  }

  // The product's specification at the state of start(): shift v + one-sided difference at the
  // product's eps (no halving at the test states).
  std::vector<double> expected(const MatrixFreeProduct<P>& a, std::span<const double> rnu) {
    const double e = eps(a, rnu, std::sqrt(static_cast<double>(std::numeric_limits<S>::epsilon())));
    std::vector<double> y = one_sided(rnu, e);
    for (std::size_t k = 0; k < 6; ++k) {
      for (std::size_t c = 0; c < n; ++c) y[k * n + c] = shift[c] * v[k * n + c] + y[k * n + c];
    }
    return y;
  }

  static constexpr double speed = 400.0;
  static constexpr double energy = p0 / (air.gamma - 1.0);
  static constexpr double nt_inf = 3.0 * nu;
  autan::Mesh mesh;
  autan::Geometry g;
  std::size_t n;
  NavierStokes<P> flow;
  SpalartAllmaras<P> sa;
  std::vector<double> u, v, shift;
};

// The flow's own product (no turbulence model).
template <class P>
MatrixFreeProduct<P> flow_product_of(Setup<P>& s) {
  const double c = std::sqrt(air.gamma * p0 / rho);
  return MatrixFreeProduct<P>(s.flow, nullptr, air.gamma,
                              {rho, rho * c, rho * c, rho * c, Setup<P>::energy}, NtBounds{});
}

template <class P>
MatrixFreeProduct<P> product_of(Setup<P>& s) {
  const double c = std::sqrt(air.gamma * p0 / rho);
  return MatrixFreeProduct<P>(s.flow, &s.sa, air.gamma,
                              {rho, rho * c, rho * c, rho * c, Setup<P>::energy}, NtBounds{});
}

}  // namespace

TEMPLATE_TEST_CASE("the coupled product is the one-sided difference of the held coupled residual",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // The product's contract, bit for bit: shift v + [R(X + eps v) - R(X)] / eps with the
  // documented eps, R the coupled residual (the flow's with the eddy viscosity of the perturbed
  // nt, then SA's) held at X. Two states: a smooth one (limiter off, nt > 0), and one where
  // nt runs from -2.5 nu to 3.5 nu with the wall distance y, so that SA-neg's branch on the sign
  // of nt (and f_n), S_tilde's limit, S_tilde <= 0 and r at its cap of 10 occur, the limiter
  // on, every fifth cell with y > 0.3 a quarter of the probe's own step from nt = 0, of either
  // sign: the probe crosses the sign of nt there and the base state's branch must hold.
  using P = TestType;
  const auto* file = GENERATE(from_range(meshes));
  const bool branches = GENERATE(false, true);
  CAPTURE(file, branches);
  Setup<P> s(file, branches);
  auto a = product_of(s);
  std::vector<double> rnu(s.n);
  auto fill = [&](double seed) {
    for (std::size_t c = 0; c < s.n; ++c) {
      const double x = s.g.cell_centroid_x[c], y = s.g.cell_centroid_y[c];
      const double z = s.g.cell_centroid_z[c];
      if (!branches) {
        rnu[c] = rho * nu * (2.0 + std::sin(3.0 * x + z));
      } else if (c % 5 == 0 && y > 0.3) {
        rnu[c] = (c % 10 == 0 ? seed : -seed) * s.v[5 * s.n + c];
      } else {
        rnu[c] = rho * nu * (0.5 + 3.0 * std::sin(4.0 * x + 5.0 * y - 3.0 * z));
      }
    }
  };
  fill(0.0);
  s.start(a, rnu);
  const double root =
      std::sqrt(static_cast<double>(std::numeric_limits<typename P::state>::epsilon()));
  fill(0.25 * s.eps(a, rnu, root));
  s.start(a, rnu);
  std::vector<double> y(6 * s.n);
  a.apply(s.v, y);
  CHECK(a.halvings() == 0);
  CHECK(y == s.expected(a, rnu));
}

TEST_CASE("the held coupled residual's one-sided difference converges at first order",
          "[navier-stokes]") {
  // FP64, the smooth state: errors against central differences at relative steps 1e-5 and
  // 2.5e-6, where truncation outweighs round-off by more than 1e4 and the velocity moves by less
  // than a tenth of the smallest |v.n| of the state (Roe's |v.n| is not held); their ratio is 4
  // at first order, [3, 5] allowing the second-order term (docs/design/coupled-sa.md).
  using P = autan::kernels::fp64;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  Setup<P> s(file, false);
  auto a = product_of(s);
  std::vector<double> rnu(s.n);
  for (std::size_t c = 0; c < s.n; ++c) {
    rnu[c] = rho * nu * (2.0 + std::sin(3.0 * s.g.cell_centroid_x[c] + s.g.cell_centroid_z[c]));
  }
  s.start(a, rnu);
  const auto d = s.central(a, rnu);
  auto error = [&](double relative) {
    const auto f = s.one_sided(rnu, s.eps(a, rnu, relative));
    double e = 0.0;
    for (std::size_t i = 0; i < 6 * s.n; ++i) {
      const double q = (f[i] - d[i]) / a.scale()[i / s.n];
      e += q * q;
    }
    return std::sqrt(e);
  };
  const double ratio = error(1e-5) / error(2.5e-6);
  CAPTURE(ratio);
  CHECK(ratio >= 3.0);
  CHECK(ratio <= 5.0);
}

TEST_CASE("a step that would move rho nt out of its bounds is shortened as a whole",
          "[navier-stokes]") {
  // coupled-sa.md: rho nt changes by at most -0.5 ... +2 times max(|rho nt|, rho nt_inf). Cell 0
  // asks for -1 against a scale of 1 (fraction 0.5), cell 1 for +5 against 2 (4/5 = 0.8), cell 2
  // for +0.5 against rho nt_inf = 0.5 (within bounds): the step is 0.5 for every cell.
  const std::vector<double> density{1.0, 1.0, 1.0}, rnu{1.0, -2.0, 0.0};
  const std::vector<double> update{-1.0, 5.0, 0.5};
  CHECK(autan::physics::nt_step_fraction(density, rnu, update, 0.5, NtBounds{}) == 0.5);
  const std::vector<double> within{-0.25, 1.0, 0.5};
  CHECK(autan::physics::nt_step_fraction(density, rnu, within, 0.5, NtBounds{}) == 1.0);
  const std::vector<double> nan{0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
  CHECK(autan::physics::nt_step_fraction(density, rnu, nan, 0.5, NtBounds{}) == 0.0);
}

TEMPLATE_TEST_CASE("the flow's product is the assembled operator where that is the Jacobian",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // matrix-free-step.md: at first order, with a viscosity too small to count, the assembled
  // operator diag(shift) + J~ is the Jacobian (dF/dW of each face, times dW/dU), so the product
  // equals it to the difference's error, 10 sqrt(epsilon of the stored state) of its magnitude.
  using P = TestType;
  using S = typename P::state;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  Setup<P> s(file, false, 1e-12);
  s.flow.set_second_order(false);
  std::vector<S> state(5 * s.n);
  std::vector<typename P::residual> r(5 * s.n);
  s.flow.state(s.u, state);
  s.flow.residual_and_jacobian(state, r);
  autan::physics::BlockMatrix matrix(s.mesh);
  s.flow.assemble(s.shift, matrix);
  auto a = flow_product_of(s);
  a.start(s.u, state, {}, 0.0, s.shift, {});
  const std::span<const double> v = std::span<const double>(s.v).first(5 * s.n);
  std::vector<double> y(5 * s.n), w(5 * s.n);
  a.apply(v, y);
  matrix.multiply(v, w);
  double error = 0.0, magnitude = 0.0;
  for (std::size_t i = 0; i < 5 * s.n; ++i) {
    error = std::max(error, std::abs(y[i] - w[i]) / a.scale()[i / s.n]);
    magnitude = std::max(magnitude, std::abs(w[i]) / a.scale()[i / s.n]);
  }
  CHECK(a.halvings() == 0);
  CHECK(error <=
        10.0 * std::sqrt(static_cast<double>(std::numeric_limits<S>::epsilon())) * magnitude);
}

TEMPLATE_TEST_CASE("a product whose perturbation rounds away in the stored state fails",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // The stored state is U - U_ref: with a reference 1e12 times the state, a perturbation of
  // sqrt(epsilon) relative to U is below the stored values' spacing. The product fails at once
  // (NaN, no halving: a smaller step only rounds more away), and stays failed.
  using P = TestType;
  using S = typename P::state;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  Setup<P> s(file, false, mu, 1e12);
  std::vector<S> state(5 * s.n);
  std::vector<typename P::residual> r(5 * s.n);
  s.flow.state(s.u, state);
  s.flow.residual(state, r);
  auto a = flow_product_of(s);
  a.start(s.u, state, {}, 0.0, s.shift, {});
  const std::span<const double> v = std::span<const double>(s.v).first(5 * s.n);
  std::vector<double> y(5 * s.n);
  a.apply(v, y);
  CHECK(std::all_of(y.begin(), y.end(), [](double x) { return std::isnan(x); }));
  CHECK(a.halvings() == 0);
  CHECK(a.representation() > 1e-2);
  std::fill(y.begin(), y.end(), 0.0);
  a.apply(v, y);
  CHECK(std::all_of(y.begin(), y.end(), [](double x) { return std::isnan(x); }));
}

TEMPLATE_TEST_CASE("near a shock two products of one direction are bit for bit equal",
                   "[navier-stokes]", autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // matrix-free-step.md: the limiter is held at the state of start(), so a direction's product
  // does not depend on what the solve evaluated before it. A jump of density and pressure at
  // x = 0.5, the limiter on.
  using P = TestType;
  using S = typename P::state;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  Setup<P> s(file, true);
  for (std::size_t c = 0; c < s.n; ++c) {
    if (s.g.cell_centroid_x[c] > 0.5) {
      for (std::size_t k = 0; k < 5; ++k) s.u[k * s.n + c] *= k == 4 ? 2.0 : 1.5;
    }
  }
  std::vector<S> state(5 * s.n);
  std::vector<typename P::residual> r(5 * s.n);
  s.flow.state(s.u, state);
  s.flow.residual(state, r);
  auto a = flow_product_of(s);
  a.start(s.u, state, {}, 0.0, s.shift, {});
  const std::span<const double> v = std::span<const double>(s.v).first(5 * s.n);
  std::vector<double> other(5 * s.n), y1(5 * s.n), y2(5 * s.n);
  for (std::size_t i = 0; i < 5 * s.n; ++i) other[i] = v[i] * std::cos(static_cast<double>(i));
  a.apply(v, y1);
  a.apply(other, y2);
  a.apply(v, y2);
  CHECK(std::all_of(y1.begin(), y1.end(), [](double x) { return std::isfinite(x); }));
  CHECK(y1 == y2);
}

TEMPLATE_TEST_CASE("a probe that switches a face state's fallback is halved", "[navier-stokes]",
                   autan::kernels::fp64, autan::kernels::fp32_acc64) {
  // One cell's conserved state U scaled by f: below a threshold f* a reconstructed face state of
  // a neighbour falls back (NavierStokes::frozen_switches against f = 1). Along v = -U of that
  // cell a probe of step eps moves f to f - eps; the base sits half a step above f*, so the
  // first probe crosses f* (the half step lands on f* itself, either side by round-off). The
  // product halves eps and equals the one-sided difference of the held residual at that eps,
  // bit for bit.
  using P = TestType;
  using S = typename P::state;
  using R = typename P::residual;
  const auto* file = GENERATE(from_range(meshes));
  CAPTURE(file);
  Setup<P> s(file, false);
  const std::size_t n = s.n;
  std::size_t c0 = 0;
  double best = 1e300;
  for (std::size_t c = 0; c < n; ++c) {
    const double dx = s.g.cell_centroid_x[c] - 0.5, dy = s.g.cell_centroid_y[c] - 0.5;
    const double dz = s.g.cell_centroid_z[c] - 0.5;
    const double d = dx * dx + dy * dy + dz * dz;
    if (d < best) best = d, c0 = c;
  }
  const std::vector<double> u1 = s.u;
  auto scaled = [&](double f) {
    std::vector<double> u = u1;
    for (std::size_t k = 0; k < 5; ++k) u[k * n + c0] *= f;
    return u;
  };
  std::vector<S> state(5 * n);
  std::vector<R> r(5 * n);
  s.flow.state(u1, state);
  s.flow.residual(state, r);
  s.flow.freeze(state);
  auto switched = [&](double f) {
    s.flow.state(scaled(f), state);
    s.flow.residual_frozen(state, r);
    return s.flow.frozen_switches() > 0;
  };
  double lo = 1.0;
  while (lo > 1e-12 && !switched(lo)) lo *= 0.1;
  REQUIRE(switched(lo));
  double hi = 1.0;
  for (int i = 0; i < 80; ++i) {
    const double mid = 0.5 * (lo + hi);
    (switched(mid) ? lo : hi) = mid;
  }
  auto a = flow_product_of(s);
  std::vector<double> v(5 * n, 0.0);
  for (std::size_t k = 0; k < 5; ++k) v[k * n + c0] = -u1[k * n + c0];
  // eps by the product's contract; v scales with the base, so f - eps after one step.
  auto eps_at = [&](std::span<const double> u) {
    auto rms = [&](std::span<const double> x) {
      std::array<autan::runtime::ExactSum, 2> sums;
      for (std::size_t i = 0; i < 5 * n; ++i) {
        const double q = x[i] / a.scale()[i / n];
        sums[0].add(q * q);
      }
      sums[1].add(static_cast<double>(5 * n));
      return std::sqrt(sums[0].value() / sums[1].value());
    };
    return std::sqrt(static_cast<double>(std::numeric_limits<S>::epsilon())) * (1.0 + rms(u)) /
           rms(v);
  };
  const double f = hi + 0.5 * eps_at(scaled(hi));
  s.u = scaled(f);
  for (std::size_t k = 0; k < 5; ++k) v[k * n + c0] = -u1[k * n + c0];
  s.flow.state(s.u, state);
  s.flow.residual(state, r);
  a.start(s.u, state, {}, 0.0, s.shift, {});
  std::vector<double> y(5 * n);
  a.apply(v, y);
  CHECK(a.halvings() > 0);
  const double e = std::ldexp(eps_at(s.u), -a.halvings());
  std::vector<double> up(s.u);
  for (std::size_t i = 0; i < 5 * n; ++i) up[i] = s.u[i] + e * v[i];
  std::vector<R> r0(5 * n), r1(5 * n);
  s.flow.residual_frozen(state, r0);
  std::vector<S> probe(5 * n);
  s.flow.state(up, probe);
  s.flow.residual_frozen(probe, r1);
  CHECK(s.flow.frozen_switches() == 0);
  std::vector<double> expected(5 * n);
  for (std::size_t i = 0; i < 5 * n; ++i) {
    expected[i] =
        s.shift[i % n] * v[i] + (static_cast<double>(r1[i]) - static_cast<double>(r0[i])) / e;
  }
  CHECK(y == expected);
}
