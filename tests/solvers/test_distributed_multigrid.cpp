#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/distributed_multigrid.hpp"
#include "autan/solvers/multigrid.hpp"
#include "thread_ranks.hpp"

namespace {

namespace fs = std::filesystem;
using autan::solvers::Level;
using autan::solvers::MultigridOptions;

// The multigrid keeps a reference to the hierarchy: no temporary binds to its constructor.
template <class Levels>
constexpr bool multigrid_from =
    std::is_constructible_v<autan::solvers::DistributedMultigrid, Levels,
                            std::span<const std::int32_t>, std::int32_t, std::int32_t,
                            autan::solvers::LevelCollective&>;
static_assert(multigrid_from<const std::vector<Level>&>);
static_assert(!multigrid_from<std::vector<Level>&&>);
static_assert(!multigrid_from<const std::vector<Level>&&>);

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

// A diffusion-like system on the mesh's cells: non-symmetric off-diagonal blocks and dominant
// diagonals, so that every level's smoother and the coarsest LU factor; and a right-hand side.
struct System {
  std::vector<double> diag, upper, lower, b;
};

System make_system(const Level& fine, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  System s;
  const std::size_t n = fine.n_nodes, m = fine.owner.size();
  s.upper.resize(25 * m);
  s.lower.resize(25 * m);
  for (double& x : s.upper) x = 0.1 * u(rng);
  for (double& x : s.lower) x = 0.1 * u(rng);
  s.diag.resize(25 * n);
  for (double& x : s.diag) x = 0.1 * u(rng);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < 5; ++p) s.diag[25 * i + 6 * p] += 8.0;
  }
  s.b.resize(5 * n);
  for (double& x : s.b) x = u(rng);
  return s;
}

struct Outcome {
  std::vector<double> x;
  autan::solvers::KrylovReport report;
};

enum class Action { cycle, solve };

Outcome run(const std::vector<Level>& levels, const System& s,
            const std::vector<std::int32_t>& parts, std::int32_t n_parts,
            const MultigridOptions& options, Action action,
            std::vector<std::exception_ptr>* failures = nullptr,
            const std::vector<double>* start = nullptr) {
  const std::size_t n_cells = levels[0].n_nodes;
  Outcome out{std::vector<double>(5 * n_cells, std::numeric_limits<double>::quiet_NaN()), {}};
  autan::test::Hub hub(n_parts);
  std::vector<std::exception_ptr> failed(at(n_parts));
  std::vector<autan::solvers::KrylovReport> reports(at(n_parts));
  const auto part = [&](std::int32_t rank) {
    try {
      autan::test::ThreadRanks ranks(hub, rank);
      autan::solvers::DistributedMultigrid mg(levels, parts, n_parts, rank, ranks, options);
      const autan::LocalGraph& g = mg.op().graph(0);
      const std::size_t n = mg.op().n_local(0);
      std::vector<double> diag, upper, lower;
      for (std::size_t i = 0; i < g.n_owned; ++i) {
        const auto v = at(g.node_source[i]);
        diag.insert(diag.end(), s.diag.data() + 25 * v, s.diag.data() + 25 * v + 25);
      }
      for (std::size_t k = 0; k < g.n_owned_edges; ++k) {
        const auto e = at(g.edge_global_id[k]);
        upper.insert(upper.end(), s.upper.data() + 25 * e, s.upper.data() + 25 * e + 25);
        lower.insert(lower.end(), s.lower.data() + 25 * e, s.lower.data() + 25 * e + 25);
      }
      mg.set_operator(diag, upper, lower);
      std::vector<double> b(5 * n, std::numeric_limits<double>::quiet_NaN()), x(5 * n, 0.0);
      if (start != nullptr) {
        for (std::size_t i = 0; i < n; ++i) {
          for (std::size_t c = 0; c < 5; ++c)
            x[c * n + i] = (*start)[c * n_cells + at(g.node_source[i])];
        }
      }
      for (std::size_t i = 0; i < g.n_owned; ++i) {
        for (std::size_t c = 0; c < 5; ++c) b[c * n + i] = s.b[c * n_cells + at(g.node_source[i])];
      }
      if (action == Action::cycle) {
        mg.cycle(b, x);
      } else {
        reports[at(rank)] = mg.solve(b, x, 10, 1e-6);
      }
      for (std::size_t i = 0; i < g.n_owned; ++i) {
        for (std::size_t c = 0; c < 5; ++c)
          out.x[c * n_cells + at(g.node_source[i])] = x[c * n + i];
      }
    } catch (...) {
      failed[at(rank)] = std::current_exception();
      hub.failed = true;
      hub.barrier.arrive_and_drop();
    }
  };
  std::vector<std::thread> threads;
  for (std::int32_t r = 0; r < n_parts; ++r) threads.emplace_back(part, r);
  for (auto& t : threads) t.join();
  if (failures != nullptr) {
    *failures = failed;
    return out;
  }
  for (const auto& f : failed) {
    if (f) std::rethrow_exception(f);
  }
  out.report = reports[0];
  for (const auto& r : reports) {
    if (r.cycles != out.report.cycles || std::bit_cast<std::uint64_t>(r.residual_drop) !=
                                             std::bit_cast<std::uint64_t>(out.report.residual_drop))
      throw std::logic_error("the parts' GCR reports differ");
  }
  return out;
}

// The largest |a - b|, infinite when any entry is not finite, so that NaN never passes.
double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double diff = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    const double d = std::abs(a[k] - b[k]);
    if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
    diff = std::max(diff, d);
  }
  return diff;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::ranges::equal(a, b, [](double x, double y) {
           return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
         });
}

// Slabs along x, each line of the finest level on the part of its first cell: the partition
// keeps the smoother's lines whole (euler_groups), as level_parts does above.
std::vector<std::int32_t> slabs(const autan::Geometry& g, const Level& fine, std::int32_t n) {
  const std::size_t n_cells = fine.n_nodes;
  const auto [lo, hi] = std::minmax_element(
      g.cell_centroid_x.begin(), g.cell_centroid_x.begin() + static_cast<std::ptrdiff_t>(n_cells));
  std::vector<std::int32_t> parts(n_cells);
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double t = (g.cell_centroid_x[c] - *lo) / (*hi - *lo);
    parts[c] = std::min(n - 1, static_cast<std::int32_t>(t * n));
  }
  for (std::size_t li = 0; li < fine.n_lines(); ++li) {
    const auto first = at(fine.line_offsets[li]), last = at(fine.line_offsets[li + 1]);
    for (std::size_t k = first + 1; k < last; ++k)
      parts[at(fine.line_nodes[k])] = parts[at(fine.line_nodes[first])];
  }
  return parts;
}

// |b - A x| / |b| from the global blocks, row by row (the definition, not the code's product).
double relative_residual(const Level& fine, const System& s, const std::vector<double>& x) {
  const std::size_t n = fine.n_nodes;
  std::vector<double> r = s.b;
  const auto sub = [&](std::size_t i, const double* a, std::size_t j) {
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) r[p * n + i] -= a[5 * p + q] * x[q * n + j];
    }
  };
  for (std::size_t i = 0; i < n; ++i) sub(i, &s.diag[25 * i], i);
  for (std::size_t e = 0; e < fine.owner.size(); ++e) {
    sub(at(fine.owner[e]), &s.upper[25 * e], at(fine.neighbour[e]));
    sub(at(fine.neighbour[e]), &s.lower[25 * e], at(fine.owner[e]));
  }
  double rr = 0.0, bb = 0.0;
  for (std::size_t k = 0; k < r.size(); ++k) {
    rr += r[k] * r[k];
    bb += s.b[k] * s.b[k];
  }
  return std::sqrt(rr / bb);
}

}  // namespace

TEST_CASE("a distributed V-cycle without scaling is the host multigrid's cycle, bit for bit",
          "[multigrid][distributed]") {
  // #259 D2: a V-cycle with Scaling::none takes no inner product, so on 1 to 4 parts it must give
  // Multigrid::cycle's bits: the sweeps in the global line colours, the couplings and residuals
  // by global edge id, the Galerkin operators, the transfers and the coarsest solve (dense LU, or
  // sweeps when the coarsest level is too large for it).
  const fs::path dir{AUTAN_TEST_MESH_DIR};
  // The hex mesh's 27 cells fit the dense solve at any coarsening, so it runs with it only.
  const auto [file, direct] =
      GENERATE(table<const char*, bool>({{"tet-gmsh/tet-gmsh.ascii.msh", true},
                                         {"tet-gmsh/tet-gmsh.ascii.msh", false},
                                         {"hex/hex.ascii.msh", true},
                                         {"poly/polyMesh", true},
                                         {"poly/polyMesh", false},
                                         {"hybrid/hybrid.ascii.msh", true},
                                         {"hybrid/hybrid.ascii.msh", false}}));
  CAPTURE(file, direct);
  const auto mesh = autan::io::read_mesh(dir / file);
  const auto g = autan::compute_geometry(mesh);
  MultigridOptions options;
  options.cycle = autan::solvers::Cycle::v;
  options.scaling = autan::solvers::Scaling::none;
  options.agglomeration.coarsest_nodes = 4;
  // Without the dense coarsest solve: one level of more than 200 cells, swept as the coarsest
  // (the reference meshes are too small for a coarse level that large).
  if (!direct) options.agglomeration.coarsest_nodes = std::size_t{1} << 30;
  autan::solvers::Multigrid host(mesh, g, options);
  const std::vector<Level>& levels = host.levels();
  REQUIRE(host.direct_coarsest() == direct);
  const System s = make_system(levels[0], 7);
  host.set_operator(s.diag, s.upper, s.lower);
  std::vector<double> want(s.b.size(), 0.0);
  host.cycle(s.b, want);
  for (const std::int32_t n_parts : {1, 2, 3, 4}) {
    CAPTURE(n_parts);
    const auto parts = slabs(g, levels[0], n_parts);
    CHECK(same_bits(run(levels, s, parts, n_parts, options, Action::cycle).x, want));
  }
}

TEST_CASE("a distributed K-cycle and GCR solve repeat their bits on 1 to 4 parts",
          "[multigrid][distributed]") {
  // #259 D2: with the K-cycle and Galerkin scaling (the default), every inner product is a sum
  // over the parts that does not depend on the partition, and every branch is taken from it: one
  // cycle and a GCR solve to 1e-6 give the same x and the same report on 1 to 4 parts. The solve
  // reaches its tolerance, checked by |b - A x| from the global blocks.
  const fs::path dir{AUTAN_TEST_MESH_DIR};
  const char* file =
      GENERATE("tet-gmsh/tet-gmsh.ascii.msh", "poly/polyMesh", "hybrid/hybrid.ascii.msh");
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(dir / file);
  const auto g = autan::compute_geometry(mesh);
  MultigridOptions options;
  options.agglomeration.coarsest_nodes = 4;
  const auto levels = autan::solvers::agglomerate(mesh, g, options.agglomeration);
  REQUIRE(levels.size() >= 3);
  const System s = make_system(levels[0], 11);
  const auto n_cells = static_cast<std::size_t>(mesh.n_cells());
  const std::vector<std::int32_t> whole(n_cells, 0);
  const Outcome one_cycle = run(levels, s, whole, 1, options, Action::cycle);
  const Outcome one_solve = run(levels, s, whole, 1, options, Action::solve);
  CHECK(one_solve.report.residual_drop <= 1e-6);
  CHECK(relative_residual(levels[0], s, one_solve.x) <= 2e-6);
  for (const std::int32_t n_parts : {2, 3, 4}) {
    CAPTURE(n_parts);
    const auto parts = slabs(g, levels[0], n_parts);
    CHECK(same_bits(run(levels, s, parts, n_parts, options, Action::cycle).x, one_cycle.x));
    const Outcome many = run(levels, s, parts, n_parts, options, Action::solve);
    CHECK(same_bits(many.x, one_solve.x));
    CHECK(many.report.cycles == one_solve.report.cycles);
    CHECK(std::bit_cast<std::uint64_t>(many.report.residual_drop) ==
          std::bit_cast<std::uint64_t>(one_solve.report.residual_drop));
  }
  // Two runs at a fixed N = 4 give the same bits.
  const auto four = slabs(g, levels[0], 4);
  const Outcome first = run(levels, s, four, 4, options, Action::solve);
  const Outcome again = run(levels, s, four, 4, options, Action::solve);
  CHECK(same_bits(first.x, again.x));
  CHECK(first.report.cycles == again.report.cycles);
  CHECK(std::bit_cast<std::uint64_t>(first.report.residual_drop) ==
        std::bit_cast<std::uint64_t>(again.report.residual_drop));
  // The same algorithm as Multigrid's K-cycle, whose inner products are sums in index order:
  // one cycle within 1e-10 of its size (a branch taken differently would leave far more).
  autan::solvers::Multigrid host(mesh, g, options);
  host.set_operator(s.diag, s.upper, s.lower);
  std::vector<double> want(s.b.size(), 0.0);
  host.cycle(s.b, want);
  double size = 0.0;
  for (const double v : want) size = std::max(size, std::abs(v));
  REQUIRE(std::isfinite(size));
  CHECK(max_diff(one_cycle.x, want) <= 1e-10 * size);
}

TEST_CASE("a distributed two-level cycle without smoothing applies P (P^T A P)^-1 P^T",
          "[multigrid][distributed]") {
  // #259 D2: with no sweep, no scaling and a dense coarse solve, one cycle from x0 is x0 plus the
  // Galerkin coarse correction, x0 + P (P^T A P)^-1 P^T (b - A x0). The oracle builds P^T A P and
  // P^T (b - A x0) from the global blocks and solves it with a dense LU of its own: within 1e-10
  // of |x|, on 1 to 4 parts, from x0 = 0 and from a random x0.
  const fs::path dir{AUTAN_TEST_MESH_DIR};
  const char* file =
      GENERATE("tet-gmsh/tet-gmsh.ascii.msh", "poly/polyMesh", "hybrid/hybrid.ascii.msh");
  CAPTURE(file);
  const auto mesh = autan::io::read_mesh(dir / file);
  const auto g = autan::compute_geometry(mesh);
  MultigridOptions options;
  options.cycle = autan::solvers::Cycle::v;
  options.scaling = autan::solvers::Scaling::none;
  options.pre_sweeps = 0;
  options.post_sweeps = 0;
  options.agglomeration.max_levels = 2;
  const auto levels = autan::solvers::agglomerate(mesh, g, options.agglomeration);
  REQUIRE(levels.size() == 2);
  REQUIRE(5 * levels[1].n_nodes <= 1000);
  const System s = make_system(levels[0], 5);
  const Level& f = levels[0];
  const std::size_t n = f.n_nodes, nc = levels[1].n_nodes, m = 5 * nc;
  const bool from_zero = GENERATE(true, false);
  CAPTURE(from_zero);
  std::vector<double> x0(5 * n, 0.0);
  if (!from_zero) {
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (double& v : x0) v = u(rng);
  }
  // r0 = b - A x0, A's block row of node i: its diagonal block, the upper block of each edge it
  // owns (towards the neighbour), the lower block of each edge it neighbours (towards the owner).
  std::vector<double> r0 = s.b;
  const auto apply = [&](std::size_t i, std::size_t j, const double* a) {
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) r0[p * n + i] -= a[5 * p + q] * x0[q * n + j];
    }
  };
  for (std::size_t i = 0; i < n; ++i) apply(i, i, &s.diag[25 * i]);
  for (std::size_t e = 0; e < f.owner.size(); ++e) {
    apply(at(f.owner[e]), at(f.neighbour[e]), &s.upper[25 * e]);
    apply(at(f.neighbour[e]), at(f.owner[e]), &s.lower[25 * e]);
  }
  std::vector<double> ac(m * m, 0.0), bc(m, 0.0);
  const auto put = [&](std::size_t i, std::size_t j, const double* a) {
    const auto I = at(f.coarse_node[i]), J = at(f.coarse_node[j]);
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) ac[(p * nc + I) * m + q * nc + J] += a[5 * p + q];
    }
  };
  for (std::size_t i = 0; i < n; ++i) put(i, i, &s.diag[25 * i]);
  for (std::size_t e = 0; e < f.owner.size(); ++e) {
    put(at(f.owner[e]), at(f.neighbour[e]), &s.upper[25 * e]);
    put(at(f.neighbour[e]), at(f.owner[e]), &s.lower[25 * e]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c) bc[c * nc + at(f.coarse_node[i])] += r0[c * n + i];
  }
  // Gaussian elimination with partial pivoting, written here.
  std::vector<std::size_t> perm(m);
  for (std::size_t k = 0; k < m; ++k) perm[k] = k;
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t p = k;
    for (std::size_t r = k + 1; r < m; ++r) {
      if (std::abs(ac[r * m + k]) > std::abs(ac[p * m + k])) p = r;
    }
    if (p != k) {
      for (std::size_t c = 0; c < m; ++c) std::swap(ac[k * m + c], ac[p * m + c]);
      std::swap(bc[k], bc[p]);
    }
    for (std::size_t r = k + 1; r < m; ++r) {
      const double factor = ac[r * m + k] / ac[k * m + k];
      for (std::size_t c = k; c < m; ++c) ac[r * m + c] -= factor * ac[k * m + c];
      bc[r] -= factor * bc[k];
    }
  }
  std::vector<double> xc(m);
  for (std::size_t k = m; k-- > 0;) {
    double v = bc[k];
    for (std::size_t c = k + 1; c < m; ++c) v -= ac[k * m + c] * xc[c];
    xc[k] = v / ac[k * m + k];
  }
  std::vector<double> want(5 * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c)
      want[c * n + i] = x0[c * n + i] + xc[c * nc + at(f.coarse_node[i])];
  }
  double size = 0.0;
  for (const double v : want) size = std::max(size, std::abs(v));
  REQUIRE(std::isfinite(size));
  for (const std::int32_t n_parts : {1, 2, 3, 4}) {
    CAPTURE(n_parts);
    const auto x =
        run(levels, s, slabs(g, levels[0], n_parts), n_parts, options, Action::cycle, nullptr, &x0)
            .x;
    CHECK(max_diff(x, want) <= 1e-10 * size);
  }
}

TEST_CASE("a singular block on one part stops every part", "[multigrid][distributed]") {
  // #259 D2: the factorisation's failure on one rank is agreed by every rank, which all throw,
  // instead of the others waiting in an exchange.
  const fs::path dir{AUTAN_TEST_MESH_DIR};
  const auto mesh = autan::io::read_mesh(dir / "hex/hex.ascii.msh");
  const auto g = autan::compute_geometry(mesh);
  MultigridOptions options;
  options.agglomeration.coarsest_nodes = 4;
  const auto levels = autan::solvers::agglomerate(mesh, g, options.agglomeration);
  System s = make_system(levels[0], 3);
  const auto parts = slabs(g, levels[0], 2);
  // The last cell's diagonal block zero: singular on its part only. Every part reports the
  // singular block itself, none merely another part's failure.
  std::fill_n(s.diag.end() - 25, 25, 0.0);
  std::vector<std::exception_ptr> failures;
  (void)run(levels, s, parts, 2, options, Action::cycle, &failures);
  REQUIRE(failures.size() == 2);
  for (const auto& e : failures) {
    REQUIRE(e);
    CHECK_THROWS_AS(std::rethrow_exception(e), std::domain_error);
  }
}
