// SpalartAllmaras<P>::solve swept by colour (ADR 0005, option B; #163, step 7a): T1, the
// algorithm against a reference written from its definition; T2c, its convergence certified
// with guaranteed (interval) arithmetic, as declared on #163 (T2's dominance bound does not
// apply to these systems: #163, T2's result).
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/kernels/runtime.hpp"
#include "autan/partition/partition.hpp"
#include "autan/physics/navier_stokes.hpp"
#include "autan/physics/spalart_allmaras.hpp"
#include "stretched_slab.hpp"

namespace {

namespace fs = std::filesystem;
using autan::physics::BoundaryKind;
using autan::physics::Gas;
using autan::physics::NavierStokes;
using autan::physics::Scheme;
using autan::physics::SpalartAllmaras;
using autan::physics::Transport;
using P = autan::kernels::fp64;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
constexpr Gas air{1.4, 287.0};
// euler_physics.cpp: the implicit step's first CFL number.
constexpr double cfl_start = 1000.0;

// A smooth subsonic flow over the unit cube: rho, u, v, w, p at (x, y, z).
std::array<double, 5> flow(double x, double y, double z) {
  return {1.2 + 0.1 * std::sin(2.0 * x + y), 80.0 + 15.0 * std::cos(x + 2.0 * z),
          -20.0 + 10.0 * std::sin(y + z), 5.0 * std::cos(3.0 * x),
          1.0e5 + 4.0e3 * std::sin(x + y + z)};
}

double nt_at(double x, double y, double z) {
  return 1.0e-4 * (1.0 + 0.5 * std::sin(2.0 * x + 3.0 * y + z));
}

// The flow and SA of a frozen state, an outlet on xmax, a wall on ymin, slip on ymax and on the
// slab's sides (its one layer of cells needs them for the boundary fit), the flow's state
// elsewhere; diag, upper, lower and b = -R of SA's implicit step at the first CFL number.
struct System {
  explicit System(autan::Mesh m, double cfl = cfl_start)
      : mesh(std::move(m)),
        g(autan::compute_geometry(mesh)),
        n(static_cast<std::size_t>(mesh.n_cells())),
        ni(static_cast<std::size_t>(mesh.n_interior_faces())) {
    const auto nb = static_cast<std::size_t>(mesh.n_boundary_faces());
    std::vector<BoundaryKind> kind(nb);
    std::vector<double> data(5 * nb, 0.0), nt_b(nb), wall(n), u(5 * n);
    for (std::size_t bf = 0; bf < nb; ++bf) {
      const auto f = ni + bf;
      const auto& zone = mesh.face_zones()[static_cast<std::size_t>(mesh.face_zone()[f])].name;
      auto w = flow(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
      nt_b[bf] = nt_at(g.face_centroid_x[f], g.face_centroid_y[f], g.face_centroid_z[f]);
      if (zone == "xmax") {
        kind[bf] = BoundaryKind::outlet;
      } else if (zone == "ymin") {
        kind[bf] = BoundaryKind::wall;
        w = {300.0, 0.0, 0.0, 0.0, 0.0};
      } else if (zone == "ymax" || zone == "sides") {
        kind[bf] = BoundaryKind::slip;
      } else {
        kind[bf] = BoundaryKind::state;
      }
      for (std::size_t k = 0; k < 5; ++k) data[k * nb + bf] = w[k];
    }
    std::vector<P::state> rs(n);
    for (std::size_t c = 0; c < n; ++c) {
      const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c], z = g.cell_centroid_z[c];
      const auto w = flow(x, y, z);
      const double q2 = w[1] * w[1] + w[2] * w[2] + w[3] * w[3];
      const std::array<double, 5> cons = {w[0], w[0] * w[1], w[0] * w[2], w[0] * w[3],
                                          w[4] / (air.gamma - 1.0) + 0.5 * w[0] * q2};
      for (std::size_t k = 0; k < 5; ++k) u[k * n + c] = cons[k];
      rs[c] = w[0] * nt_at(x, y, z);
      wall[c] = y + 0.05;
    }
    Scheme s;
    const auto ref = flow(0.5, 0.5, 0.5);
    s.mach_ref = std::hypot(ref[1], ref[2], ref[3]) / std::sqrt(air.gamma * ref[4] / ref[0]);
    s.scale = {ref[0], 100.0, 100.0, 100.0, ref[4]};
    s.reference = {ref[0], ref[0] * ref[1], ref[0] * ref[2], ref[0] * ref[3],
                   ref[4] / (air.gamma - 1.0) +
                       0.5 * ref[0] * (ref[1] * ref[1] + ref[2] * ref[2] + ref[3] * ref[3])};
    const Transport t;
    ns = std::make_unique<NavierStokes<P>>(mesh, g, air, s, t);
    ns->set_boundary(kind, data);
    ns->set_second_order(true);
    sa = std::make_unique<SpalartAllmaras<P>>(mesh, g, *ns, t);
    sa->set_boundary(nt_b);
    sa->set_wall_distance(wall);
    std::vector<P::state> stored(5 * n), lambda(n);
    std::vector<P::residual> r(5 * n), r_sa(n);
    ns->state(u, stored);
    ns->residual(stored, r);
    sa->eddy_viscosity(rs);
    ns->residual(stored, r);
    ns->spectral_radius(lambda);
    sa->residual_and_jacobian(rs, r_sa);
    std::vector<double> shift(n);
    for (std::size_t c = 0; c < n; ++c) shift[c] = static_cast<double>(lambda[c]) / cfl;
    diag.assign(n, 0.0);
    upper.assign(ni, 0.0);
    lower.assign(ni, 0.0);
    sa->assemble(shift, diag, upper, lower);
    b.resize(n);
    for (std::size_t c = 0; c < n; ++c) b[c] = -static_cast<double>(r_sa[c]);
  }

  autan::Mesh mesh;
  autan::Geometry g;
  std::size_t n, ni;
  std::unique_ptr<NavierStokes<P>> ns;
  std::unique_ptr<SpalartAllmaras<P>> sa;
  std::vector<double> diag, upper, lower, b;
};

// The definition declared on #163 (step 7a): per symmetric pair, colours 0 ... n - 1 then
// n - 1 ... 0; within a colour the cells in `order`; each row b_c minus its interior faces'
// couplings in increasing face index, over its diagonal.
std::vector<double> reference(const autan::Mesh& mesh, const std::vector<double>& diag,
                              const std::vector<double>& upper, const std::vector<double>& lower,
                              const std::vector<double>& b,
                              const std::vector<std::vector<std::size_t>>& colours,
                              std::vector<double> x, int pairs) {
  const auto n = diag.size();
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  std::vector<std::vector<std::size_t>> faces(n);
  for (std::size_t f = 0; f < ni; ++f) {
    faces[static_cast<std::size_t>(mesh.owner()[f])].push_back(f);
    faces[static_cast<std::size_t>(mesh.neighbour()[f])].push_back(f);
  }
  auto relax = [&](std::size_t c) {
    double r = b[c];
    for (const auto f : faces[c]) {
      const auto o = static_cast<std::size_t>(mesh.owner()[f]);
      const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
      r -= o == c ? upper[f] * x[nb] : lower[f] * x[o];
    }
    x[c] = r / diag[c];
  };
  for (int p = 0; p < pairs; ++p) {
    for (const auto& cells : colours) {
      for (const auto c : cells) relax(c);
    }
    for (auto it = colours.rbegin(); it != colours.rend(); ++it) {
      for (const auto c : *it) relax(c);
    }
  }
  return x;
}

std::vector<std::vector<std::size_t>> buckets(const std::vector<std::int32_t>& colours,
                                              std::int32_t n_colours, bool descending) {
  std::vector<std::vector<std::size_t>> out(static_cast<std::size_t>(n_colours));
  for (std::size_t c = 0; c < colours.size(); ++c) {
    out[static_cast<std::size_t>(colours[c])].push_back(c);
  }
  if (descending) {
    for (auto& cells : out) std::ranges::reverse(cells);
  }
  return out;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return std::ranges::equal(a, b, [](double x, double y) {
    return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
  });
}

// Guaranteed arithmetic (T2c): each operation on interval endpoints in round-to-nearest binary64,
// the result widened outward by one representable number at each end, which encloses the exact
// result of every input in the intervals and every round-to-nearest result of such inputs.
struct Iv {
  double lo, hi;
};
double down(double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); }
double up(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
Iv finite(Iv v) {
  if (!std::isfinite(v.lo) || !std::isfinite(v.hi)) throw std::overflow_error("interval overflow");
  return v;
}
Iv point(double v) { return {v, v}; }
Iv sub(Iv a, Iv b) { return finite({down(a.lo - b.hi), up(a.hi - b.lo)}); }
Iv mul(Iv a, Iv b) {
  const std::array<double, 4> p = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  return finite({down(*std::ranges::min_element(p)), up(*std::ranges::max_element(p))});
}
Iv div(Iv a, Iv b) {
  if (b.lo <= 0.0 && b.hi >= 0.0) throw std::domain_error("interval divisor contains 0");
  const std::array<double, 4> p = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
  return finite({down(*std::ranges::min_element(p)), up(*std::ranges::max_element(p))});
}
double mag(Iv a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }      // sup |t|
double mig(Iv a) { return a.lo > 0.0 ? a.lo : a.hi < 0.0 ? -a.hi : 0.0; }  // inf |t|
double add_up(double a, double b) { return up(a + b); }

// The environment T2c's enclosure needs, probed on volatile operands the compiler cannot fold:
// binary64, round to nearest (ties to even), gradual underflow produced and consumed.
bool environment_is_binary64_nearest_with_subnormals() {
  volatile double one = 1.0, half = 0.5, two = 2.0, tie = 0x1p-53, above = 0x3p-54;
  volatile double tiny = DBL_MIN, denorm = 0x1p-1074;
  const double sub_n = tiny * half;
  return std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53 &&
         std::fegetround() == FE_TONEAREST && one + tie == 1.0 && one + above == 1.0 + 0x1p-52 &&
         sub_n != 0.0 && sub_n == 0x1p-1023 && sub_n * two == DBL_MIN &&
         denorm + denorm == 0x1p-1073 && std::nextafter(0.0, 1.0) == 0x1p-1074;
}

// One symmetric pair of the declared sweep in interval arithmetic, with right-hand side `rhs`
// (point intervals; all zero for the linear pairs that give the columns of M).
void interval_pair(const System& s, const std::vector<std::vector<std::size_t>>& colours,
                   const std::vector<std::vector<std::size_t>>& faces, bool with_rhs,
                   std::vector<Iv>& x) {
  auto relax = [&](std::size_t c) {
    Iv r = point(with_rhs ? s.b[c] : 0.0);
    for (const auto f : faces[c]) {
      const auto o = static_cast<std::size_t>(s.mesh.owner()[f]);
      const auto nb = static_cast<std::size_t>(s.mesh.neighbour()[f]);
      r = sub(r, o == c ? mul(point(s.upper[f]), x[nb]) : mul(point(s.lower[f]), x[o]));
    }
    x[c] = div(r, point(s.diag[c]));
  };
  for (const auto& cells : colours) {
    for (const auto c : cells) relax(c);
  }
  for (auto it = colours.rbegin(); it != colours.rend(); ++it) {
    for (const auto c : *it) relax(c);
  }
}

std::vector<std::vector<std::size_t>> interior_faces(const autan::Mesh& mesh, std::size_t n) {
  std::vector<std::vector<std::size_t>> faces(n);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
    faces[static_cast<std::size_t>(mesh.owner()[f])].push_back(f);
    faces[static_cast<std::size_t>(mesh.neighbour()[f])].push_back(f);
  }
  return faces;
}

// C_j >= ||M^j||_inf for j = 1 ... k: each column of M^j the image of a unit vector under j
// linear interval pairs, the absolute row sums accumulated upward.
std::vector<double> certified_powers(const System& s,
                                     const std::vector<std::vector<std::size_t>>& colours,
                                     const std::vector<std::vector<std::size_t>>& faces, int k) {
  const auto n = s.n;
  std::vector<double> rows(static_cast<std::size_t>(k) * n, 0.0);
  std::vector<Iv> x(n);
  for (std::size_t j = 0; j < n; ++j) {
    std::ranges::fill(x, point(0.0));
    x[j] = point(1.0);
    for (int p = 1; p <= k; ++p) {
      interval_pair(s, colours, faces, false, x);
      double* row = &rows[static_cast<std::size_t>(p - 1) * n];
      for (std::size_t i = 0; i < n; ++i) row[i] = add_up(row[i], mag(x[i]));
    }
  }
  std::vector<double> c(static_cast<std::size_t>(k) + 1, 1.0);
  for (int p = 1; p <= k; ++p) {
    const double* row = &rows[static_cast<std::size_t>(p - 1) * n];
    c[static_cast<std::size_t>(p)] = *std::max_element(row, row + n);
  }
  return c;
}

}  // namespace

TEST_CASE("SA's sweeps by colour give the declared symmetric colour Gauss-Seidel bit for bit",
          "[spalart-allmaras]") {
  // T1 (#163, step 7a). The slab's 8 x 12 cells in 3 colours, (i + 2 j) mod 3 (neighbours along x
  // differ by 1, along y by 2), with a system of its own; the hybrid mesh's assembled system in the
  // runtime's colouring. A non-zero initial guess; 1 and 4 symmetric pairs; the reference also in
  // the reverse order within each colour, which the result does not depend on.
  const bool slab = GENERATE(true, false);
  INFO((slab ? "slab, 3 colours" : "hybrid, the runtime's colouring"));
  System s(slab ? autan::test::stretched_slab()
                : autan::io::read_mesh(mesh_dir / "hybrid/hybrid.ascii.msh"));
  std::vector<std::int32_t> colours;
  std::int32_t n_colours = 0;
  if (slab) {
    colours.resize(s.n);
    for (std::size_t c = 0; c < s.n; ++c) {
      colours[c] = static_cast<std::int32_t>((c % 8 + 2 * (c / 8)) % 3);
    }
    n_colours = 3;
    for (std::size_t c = 0; c < s.n; ++c) {
      s.diag[c] = 4.0 + static_cast<double>(c % 5);
      s.b[c] = std::sin(static_cast<double>(c));
    }
    for (std::size_t f = 0; f < s.ni; ++f) {
      s.upper[f] = -(0.3 + 0.1 * static_cast<double>(f % 7));
      s.lower[f] = -(0.2 + 0.1 * static_cast<double>(f % 5));
    }
  } else {
    colours = autan::colour_cells(s.mesh);
    n_colours = *std::ranges::max_element(colours) + 1;
  }
  REQUIRE(n_colours >= 2);
  std::vector<double> x0(s.n);
  for (std::size_t c = 0; c < s.n; ++c) x0[c] = std::cos(static_cast<double>(c));
  const auto groups = autan::physics::colour_groups(s.mesh, colours, n_colours, s.n);
  for (const int pairs : {1, 4}) {
    CAPTURE(pairs);
    auto dx = x0;
    s.sa->solve(s.diag, s.upper, s.lower, s.b, dx, pairs, &groups);
    const auto up = reference(s.mesh, s.diag, s.upper, s.lower, s.b,
                              buckets(colours, n_colours, false), x0, pairs);
    const auto down = reference(s.mesh, s.diag, s.upper, s.lower, s.b,
                                buckets(colours, n_colours, true), x0, pairs);
    CHECK(same_bits(dx, up));
    CHECK(same_bits(up, down));
  }
}

TEST_CASE("SA's colour groups refuse a colouring that couples a colour", "[spalart-allmaras]") {
  const auto mesh = autan::test::stretched_slab();
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  auto code = [&](const std::vector<std::int32_t>& colours, std::int32_t n_colours) {
    try {
      (void)autan::physics::colour_groups(mesh, colours, n_colours, n);
    } catch (const std::invalid_argument& e) {
      return std::string(e.what()).substr(0, 5);
    }
    return std::string();
  };
  // Cells 0 and 1 share a face along x.
  CHECK(code(std::vector<std::int32_t>(n, 0), 1) == "E0720");
  std::vector<std::int32_t> parity(n);
  for (std::size_t c = 0; c < n; ++c) parity[c] = static_cast<std::int32_t>((c % 8 + c / 8) % 2);
  CHECK(code(parity, 1) == "E0720");  // colour 1 outside [0, 1)
  CHECK(code(std::vector<std::int32_t>(n - 1, 0), 2) == "E0720");
  CHECK(code(parity, 2).empty());
}

// Label `certificate`: seconds optimised, minutes in Debug; outside the ordinary ctest, run by
// gpu-tests nightly and on pull requests that touch the SA assembly or sweeps.
TEST_CASE("SA's sweeps by colour converge within bounds certified by interval arithmetic",
          "[spalart-allmaras][.certificate]") {
  // T2c (#163, step 7a, revision 3). On T2's four systems at CFL 1000:
  // - C_j >= ||M^j||_inf, M the exact iteration matrix of one symmetric pair, j = 1 ... 400;
  //   the gate C_400 <= 0.9;
  // - interval sweeps [x]_N from x_0 = 0 enclose the exact iterate x_N and the binary64 one, which
  //   is the production iterate bit for bit (checked at every pair) and lies in [x]_N;
  // - x* lies in [x*] = [x]_2000 +- C_400 / (1 - C_400) ||[x]_2000 - [x]_1600|| (x_N - x* =
  //   M^400 (x_{N-400} - x*)); xi and l bound ||x*|| above and below;
  // - the envelope ||x^_k - x*|| <= w_k + C_k xi at k = 1 ... 4 and 50, checked as
  //   E_k <= w_k + C_k xi + width([x*]); and E_2000 < l.
  // Every derived quantity rounds outward. LU's solution and CFL 1e5 are reported, not gated.
  REQUIRE(environment_is_binary64_nearest_with_subnormals());
  const auto* file = GENERATE(as<const char*>{}, "tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                              "poly/polyMesh", "hybrid/hybrid.ascii.msh");
  CAPTURE(file);
  System s(autan::io::read_mesh(mesh_dir / file));
  const auto n = s.n;
  const auto colours = autan::colour_cells(s.mesh);
  const auto n_colours = *std::ranges::max_element(colours) + 1;
  const auto by_colour = buckets(colours, n_colours, false);
  const auto groups = autan::physics::colour_groups(s.mesh, colours, n_colours, n);
  const auto faces = interior_faces(s.mesh, n);
  {
    // The frozen inputs: the mesh's hash and the SHA-256 of the stored system and colours.
    const auto bytes = fs::temp_directory_path() / "autan_t2c_system.bin";
    {
      std::ofstream out(bytes, std::ios::binary);
      for (const auto* v : {&s.diag, &s.upper, &s.lower, &s.b}) {
        out.write(reinterpret_cast<const char*>(v->data()),
                  static_cast<std::streamsize>(v->size() * sizeof(double)));
      }
      out.write(reinterpret_cast<const char*>(colours.data()),
                static_cast<std::streamsize>(colours.size() * sizeof(std::int32_t)));
    }
    WARN(file << ": mesh " << autan::io::mesh_hash(mesh_dir / file) << ", system and colours "
              << autan::io::mesh_hash(bytes));
    fs::remove(bytes);
  }
  constexpr int k = 400;
  constexpr int last = 2000;

  const auto c = certified_powers(s, by_colour, faces, k);
  WARN(file << ": C_1 = " << c[1] << ", C_2 = " << c[2] << ", C_3 = " << c[3] << ", C_4 = " << c[4]
            << ", C_50 = " << c[50] << ", C_400 = " << c[k]
            << (c[4] >= 1.0 ? " (4 pairs are not certified contractive)" : ""));
  CHECK(c[k] <= 0.9);
  REQUIRE(c[k] < 1.0);

  // The interval iterates, the production iterates (one pair per call) and the reference.
  std::vector<Iv> x(n, point(0.0));
  std::vector<double> production(n, 0.0), ref(n, 0.0);
  std::vector<std::vector<Iv>> kept;  // [x]_N at N = 1, 2, 3, 4, 50, 400, 1600, 2000
  std::vector<std::vector<double>> kept_production;
  const std::array<int, 8> marks = {1, 2, 3, 4, 50, 400, last - k, last};
  bool same = true, inside = true;
  for (int p = 1; p <= last; ++p) {
    interval_pair(s, by_colour, faces, true, x);
    s.sa->solve(s.diag, s.upper, s.lower, s.b, production, 1, &groups);
    ref = reference(s.mesh, s.diag, s.upper, s.lower, s.b, by_colour, ref, 1);
    same = same && same_bits(production, ref);
    if (std::ranges::find(marks, p) != marks.end()) {
      kept.push_back(x);
      kept_production.push_back(production);
      for (std::size_t i = 0; i < n; ++i) {
        inside = inside && x[i].lo <= production[i] && production[i] <= x[i].hi;
      }
    }
  }
  CHECK(same);
  CHECK(inside);

  // [x*] from the a posteriori bound, outward.
  const auto& x_last = kept[7];
  const auto& x_before = kept[6];
  double d = 0.0;
  for (std::size_t i = 0; i < n; ++i) d = std::max(d, mag(sub(x_last[i], x_before[i])));
  const double r = up(up(c[k] * d) / down(1.0 - c[k]));
  std::vector<Iv> star(n);
  double xi = 0.0, ell = 0.0, star_width = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    star[i] = {down(x_last[i].lo - r), up(x_last[i].hi + r)};
    xi = std::max(xi, mag(star[i]));
    ell = std::max(ell, mig(star[i]));
    star_width = std::max(star_width, up(star[i].hi - star[i].lo));
  }
  auto measured = [&](const std::vector<double>& xh) {
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      e = std::max({e, up(std::abs(xh[i] - star[i].lo)), up(std::abs(star[i].hi - xh[i]))});
    }
    return e;
  };
  auto width = [&](const std::vector<Iv>& v) {
    double w = 0.0;
    for (const auto& t : v) w = std::max(w, up(t.hi - t.lo));
    return w;
  };
  // The envelope at k = 1, 2, 3, 4 and 50 (kept[0 ... 4]).
  const std::array<int, 5> steps = {1, 2, 3, 4, 50};
  for (std::size_t m = 0; m < steps.size(); ++m) {
    const int pairs = steps[m];
    const double bound = add_up(width(kept[m]), up(c[static_cast<std::size_t>(pairs)] * xi));
    const double e = measured(kept_production[m]);
    CAPTURE(pairs, e, bound, xi, star_width);
    WARN(file << ": k = " << pairs << ", B_k / xi = " << bound / xi << ", E_k = " << e);
    CHECK(e <= add_up(bound, star_width));
  }
  const double e_last = measured(kept_production[7]);
  CAPTURE(e_last, ell, xi, r);
  WARN(file << ": E_2000 / l = " << e_last / ell << ", width([x*]) = " << star_width);
  CHECK(e_last < ell);

  // Reported only: rho(M) by power iteration, LU's solution against [x*], CFL 1e5's C_400.
  {
    std::vector<double> v(n, 1.0), w(n);
    double rho = 0.0;
    for (int it = 0; it < 200; ++it) {
      w = reference(s.mesh, s.diag, s.upper, s.lower, std::vector<double>(n, 0.0), by_colour, v, 1);
      double norm = 0.0;
      for (const auto t : w) norm = std::max(norm, std::abs(t));
      rho = norm;
      for (std::size_t i = 0; i < n; ++i) v[i] = w[i] / norm;
    }
    WARN(file << ": rho(M) ~ " << rho << " (power iteration, not certified)");
  }
  {
    // Dense LU with partial pivoting: its solution against [x*].
    std::vector<double> m(n * n, 0.0), rhs = s.b, xs(n);
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] = s.diag[i];
    for (std::size_t f = 0; f < s.ni; ++f) {
      const auto o = static_cast<std::size_t>(s.mesh.owner()[f]);
      const auto nb = static_cast<std::size_t>(s.mesh.neighbour()[f]);
      m[o * n + nb] += s.upper[f];
      m[nb * n + o] += s.lower[f];
    }
    for (std::size_t col = 0; col < n; ++col) {
      std::size_t piv = col;
      for (std::size_t i = col + 1; i < n; ++i) {
        if (std::abs(m[i * n + col]) > std::abs(m[piv * n + col])) piv = i;
      }
      if (piv != col) {
        for (std::size_t j = 0; j < n; ++j) std::swap(m[col * n + j], m[piv * n + j]);
        std::swap(rhs[col], rhs[piv]);
      }
      for (std::size_t i = col + 1; i < n; ++i) {
        const double l = m[i * n + col] / m[col * n + col];
        for (std::size_t j = col; j < n; ++j) m[i * n + j] -= l * m[col * n + j];
        rhs[i] -= l * rhs[col];
      }
    }
    for (std::size_t i = n; i-- > 0;) {
      double v = rhs[i];
      for (std::size_t j = i + 1; j < n; ++j) v -= m[i * n + j] * xs[j];
      xs[i] = v / m[i * n + i];
    }
    std::size_t outside = 0;
    for (std::size_t i = 0; i < n; ++i)
      outside += (xs[i] < star[i].lo || xs[i] > star[i].hi) ? 1 : 0;
    WARN(file << ": LU's solution outside [x*] in " << outside << " of " << n
              << " cells (reported)");
  }
  System fast(autan::io::read_mesh(mesh_dir / file), 1.0e5);
  const auto c_fast = certified_powers(fast, by_colour, interior_faces(fast.mesh, n), k);
  WARN(file << ": at CFL 1e5, C_4 = " << c_fast[4] << ", C_400 = " << c_fast[k] << " (reported)");
}
