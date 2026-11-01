// Agglomeration hierarchy and multigrid solver of solvers/.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/core/renumber.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/multigrid.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::Mesh;
using autan::solvers::Level;
using autan::solvers::Multigrid;
using autan::solvers::MultigridOptions;

const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

enum class Fill { hexa, prism };

// Box [0,1]^2 x [0,h] of nx x nx x nz cells: hexahedra, or prisms extruded along z. The
// spacing in z grows geometrically from `first` at the wall z = 0 (zone "wall") up to
// `largest` (the in-plane spacing by default), then stays uniform up to z = 1/2. Cells with
// x < 1/2 are in a second cell zone when `two_zones`.
Mesh layered_box(int nx, double first, Fill fill, bool two_zones = false, double largest = 0.0) {
  autan::CellwiseMesh m;
  for (const char* name : {"a", "b"}) {
    autan::CellZone z;
    z.name = name;
    z.kind = autan::CellKind::fluid;
    m.cell_zones.push_back(z);
  }
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  const double dx = 1.0 / nx;
  const double top = largest > 0.0 ? largest : dx;
  std::vector<double> z{0.0};
  for (double d = first; z.size() < 400; d = std::min(top, 1.2 * d)) {
    z.push_back(z.back() + d);
    if (d == top && z.back() >= 0.5) break;
  }
  const int nz = static_cast<int>(z.size()) - 1;
  const int p = nx + 1;
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= nx; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.x.push_back(i * dx);
        m.y.push_back(j * dx);
        m.z.push_back(z[static_cast<std::size_t>(k)]);
      }
    }
  }
  auto v = [&](int i, int j, int k) { return static_cast<index_t>(i + p * (j + p * k)); };
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nx; ++i) {
        const index_t zone = two_zones && 2 * i < nx ? 1 : 0;
        if (fill == Fill::hexa) {
          m.add_cell(autan::CellShape::hexa,
                     std::array<index_t, 8>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k),
                                            v(i, j + 1, k), v(i, j, k + 1), v(i + 1, j, k + 1),
                                            v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
                     zone);
        } else {
          m.add_cell(
              autan::CellShape::prism,
              std::array<index_t, 6>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k), v(i, j, k + 1),
                                     v(i + 1, j, k + 1), v(i + 1, j + 1, k + 1)},
              zone);
          m.add_cell(
              autan::CellShape::prism,
              std::array<index_t, 6>{v(i, j, k), v(i + 1, j + 1, k), v(i, j + 1, k), v(i, j, k + 1),
                                     v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
              zone);
        }
        if (k == 0) {
          // The wall faces, seen from outside the cells.
          if (fill == Fill::hexa) {
            m.add_labelled_face(std::array<index_t, 4>{v(i, j, 0), v(i, j + 1, 0),
                                                       v(i + 1, j + 1, 0), v(i + 1, j, 0)},
                                0);
          } else {
            m.add_labelled_face(
                std::array<index_t, 3>{v(i, j, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0)}, 0);
            m.add_labelled_face(
                std::array<index_t, 3>{v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0)}, 0);
          }
        }
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

// Operator of the tests: a diffusion stencil (V shift + sum over faces of A_f / d_f) on every
// component, coupled through a random face perturbation, so that the 5 x 5 blocks are full.
struct System {
  std::vector<double> diag, upper, lower;
};

System diffusion(const Mesh& mesh, const autan::Geometry& g, double shift, double coupling,
                 bool directional = false, unsigned seed = 7) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  System s;
  s.diag.assign(25 * n, 0.0);
  s.upper.assign(25 * ni, 0.0);
  s.lower.assign(25 * ni, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t i = 0; i < 5; ++i) s.diag[25 * c + 6 * i] = shift * g.cell_volume[c];
  }
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
    const double dx = g.cell_centroid_x[nb] - g.cell_centroid_x[o];
    const double dy = g.cell_centroid_y[nb] - g.cell_centroid_y[o];
    const double dz = g.cell_centroid_z[nb] - g.cell_centroid_z[o];
    const double w = g.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
    // Row o takes w a (x_o - x_n), row n takes w b (x_n - x_o); `directional`: b != a.
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        const double a = (i == j ? 1.0 : 0.0) + coupling * u(rng);
        const double b = directional ? (i == j ? 1.0 : 0.0) + coupling * u(rng) : a;
        s.upper[25 * f + 5 * i + j] = -w * a;
        s.lower[25 * f + 5 * i + j] = -w * b;
        s.diag[25 * o + 5 * i + j] += w * a;
        s.diag[25 * nb + 5 * i + j] += w * b;
      }
    }
  }
  // Boundary faces: a Dirichlet-like A_f / d_f on the diagonal.
  for (auto f = static_cast<std::size_t>(mesh.n_interior_faces());
       f < static_cast<std::size_t>(mesh.n_faces()); ++f) {
    const auto c = static_cast<std::size_t>(mesh.owner()[f]);
    const double dx = g.face_centroid_x[f] - g.cell_centroid_x[c];
    const double dy = g.face_centroid_y[f] - g.cell_centroid_y[c];
    const double dz = g.face_centroid_z[f] - g.cell_centroid_z[c];
    const double w = g.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
    for (std::size_t i = 0; i < 5; ++i) s.diag[25 * c + 6 * i] += w;
  }
  return s;
}

// y = A x for the operator layout of Multigrid.
std::vector<double> apply(const Mesh& mesh, const System& s, const std::vector<double>& x) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> y(5 * n, 0.0);
  auto add = [&](const double* a, std::size_t row, std::size_t col) {
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) y[i * n + row] += a[5 * i + j] * x[j * n + col];
    }
  };
  for (std::size_t c = 0; c < n; ++c) add(&s.diag[25 * c], c, c);
  for (std::size_t f = 0; f < static_cast<std::size_t>(mesh.n_interior_faces()); ++f) {
    const auto o = static_cast<std::size_t>(mesh.owner()[f]);
    const auto nb = static_cast<std::size_t>(mesh.neighbour()[f]);
    add(&s.upper[25 * f], o, nb);
    add(&s.lower[25 * f], nb, o);
  }
  return y;
}

double norm(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s);
}

void check_hierarchy(const Mesh& mesh, const std::vector<Level>& levels) {
  REQUIRE(!levels.empty());
  REQUIRE(levels[0].n_nodes == static_cast<std::size_t>(mesh.n_cells()));
  std::vector<index_t> zone(mesh.cell_zone().begin(), mesh.cell_zone().end());
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const Level& lv = levels[l];
    INFO("level " << l);
    // Lines partition the nodes, consecutive nodes adjacent.
    std::set<std::pair<index_t, index_t>> edges;
    for (std::size_t e = 0; e < lv.owner.size(); ++e) {
      edges.emplace(std::min(lv.owner[e], lv.neighbour[e]), std::max(lv.owner[e], lv.neighbour[e]));
    }
    std::vector<int> seen(lv.n_nodes, 0);
    for (std::size_t li = 0; li < lv.n_lines(); ++li) {
      for (auto k = lv.line_offsets[li]; k < lv.line_offsets[li + 1]; ++k) {
        const index_t a = lv.line_nodes[static_cast<std::size_t>(k)];
        ++seen[static_cast<std::size_t>(a)];
        if (k > lv.line_offsets[li]) {
          const index_t b = lv.line_nodes[static_cast<std::size_t>(k - 1)];
          CHECK(edges.count({std::min(a, b), std::max(a, b)}) == 1);
        }
      }
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int s) { return s == 1; }));
    if (l + 1 == levels.size()) {
      CHECK(lv.coarse_node.empty());
      break;
    }
    const Level& c = levels[l + 1];
    // Every coarse node holds fine nodes of one zone, connected through its inner edges.
    REQUIRE(lv.coarse_node.size() == lv.n_nodes);
    std::vector<std::vector<index_t>> members(c.n_nodes);
    for (std::size_t i = 0; i < lv.n_nodes; ++i) {
      members[static_cast<std::size_t>(lv.coarse_node[i])].push_back(static_cast<index_t>(i));
    }
    std::vector<index_t> coarse_zone(c.n_nodes);
    for (std::size_t C = 0; C < c.n_nodes; ++C) {
      REQUIRE(!members[C].empty());
      coarse_zone[C] = zone[static_cast<std::size_t>(members[C][0])];
      for (const index_t i : members[C]) CHECK(zone[static_cast<std::size_t>(i)] == coarse_zone[C]);
      std::set<index_t> reached{members[C][0]};
      for (bool grew = true; grew;) {
        grew = false;
        for (std::size_t e = 0; e < lv.owner.size(); ++e) {
          if (lv.coarse_edge[e] >= 0 ||
              lv.coarse_node[static_cast<std::size_t>(lv.owner[e])] != static_cast<index_t>(C)) {
            continue;
          }
          const bool a = reached.count(lv.owner[e]) != 0, b = reached.count(lv.neighbour[e]) != 0;
          if (a != b) {
            reached.insert(a ? lv.neighbour[e] : lv.owner[e]);
            grew = true;
          }
        }
      }
      CHECK(reached.size() == members[C].size());
    }
    // Coarse edges: one per adjacent pair, owner < neighbour, each fine edge mapped to its pair.
    for (std::size_t e = 0; e < c.owner.size(); ++e) CHECK(c.owner[e] < c.neighbour[e]);
    for (std::size_t e = 0; e < lv.owner.size(); ++e) {
      const index_t a = lv.coarse_node[static_cast<std::size_t>(lv.owner[e])];
      const index_t b = lv.coarse_node[static_cast<std::size_t>(lv.neighbour[e])];
      if (a == b) {
        CHECK(lv.coarse_edge[e] == -1);
        continue;
      }
      const auto E = static_cast<std::size_t>(lv.coarse_edge[e]);
      CHECK(c.owner[E] == (lv.coarse_flipped[e] != 0 ? b : a));
      CHECK(c.neighbour[E] == (lv.coarse_flipped[e] != 0 ? a : b));
    }
    zone = coarse_zone;
  }
}

}  // namespace

TEST_CASE("agglomerates are connected, stay in one cell zone and lines join adjacent nodes",
          "[multigrid]") {
  SECTION("reference meshes") {
    const char* file = GENERATE(from_range(meshes));
    INFO(file);
    const auto mesh = autan::io::read_mesh(mesh_dir / file);
    const auto g = autan::compute_geometry(mesh);
    autan::solvers::AgglomerationOptions opt;
    opt.coarsest_nodes = 4;
    const auto levels = autan::solvers::agglomerate(mesh, g, opt);
    CHECK(levels.size() >= 2);
    check_hierarchy(mesh, levels);
  }
  SECTION("wall layers in two cell zones") {
    const Fill fill = GENERATE(Fill::hexa, Fill::prism);
    const auto mesh = layered_box(8, 1e-4, fill, true);
    const auto g = autan::compute_geometry(mesh);
    const auto levels = autan::solvers::agglomerate(mesh, g);
    check_hierarchy(mesh, levels);
  }
}

TEST_CASE("wall layers form one line per wall cell, kept through the coarse levels",
          "[multigrid]") {
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  const auto mesh = layered_box(8, 1e-4, fill);
  const auto g = autan::compute_geometry(mesh);
  const auto levels = autan::solvers::agglomerate(mesh, g);
  const std::size_t wall_cells = fill == Fill::hexa ? 64 : 128;
  // Layers whose aspect ratio is at least 2 (line_ratio 4 = aspect ratio squared).
  std::size_t lines = 0, first_layer = 0;
  for (std::size_t li = 0; li < levels[0].n_lines(); ++li) {
    const auto len = levels[0].line_offsets[li + 1] - levels[0].line_offsets[li];
    if (len < 2) continue;
    ++lines;
    const auto c = static_cast<std::size_t>(
        levels[0].line_nodes[static_cast<std::size_t>(levels[0].line_offsets[li])]);
    if (g.cell_centroid_z[c] < 1e-4) {
      ++first_layer;
      CHECK(len >= 20);
    }
  }
  CHECK(first_layer == wall_cells);
  CHECK(lines >= wall_cells);
  // The coarse levels agglomerate in the wall plane: the line length is kept.
  REQUIRE(levels.size() >= 3);
  for (std::size_t l = 1; l < levels.size(); ++l) {
    std::size_t longest = 0;
    for (std::size_t li = 0; li < levels[l].n_lines(); ++li) {
      longest = std::max(longest, static_cast<std::size_t>(levels[l].line_offsets[li + 1] -
                                                           levels[l].line_offsets[li]));
    }
    CHECK(longest >= 20);
  }
}

TEST_CASE("layers coarsen in the plane whichever way their lines run", "[multigrid]") {
  // Uniform cells of aspect ratio 16: every cell is a line seed, so the cell numbering decides
  // where lines start. Reversing the numbering along every other column makes neighbouring
  // lines run opposite ways; their layers must still be matched.
  const int nx = 8;
  const auto mesh = layered_box(nx, 1.0 / 128, Fill::hexa, false, 1.0 / 128);
  const auto nz = static_cast<int>(mesh.n_cells()) / (nx * nx);
  std::vector<index_t> new_to_old(static_cast<std::size_t>(mesh.n_cells()));
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nx; ++i) {
        const int kk = (i + j) % 2 == 0 ? k : nz - 1 - k;
        new_to_old[static_cast<std::size_t>(i + nx * (j + nx * k))] =
            static_cast<index_t>(i + nx * (j + nx * kk));
      }
    }
  }
  const auto reversed = autan::renumber_cells(mesh, new_to_old);
  for (const Mesh* m : {&mesh, &reversed}) {
    const auto g = autan::compute_geometry(*m);
    const auto levels = autan::solvers::agglomerate(*m, g);
    REQUIRE(levels.size() >= 2);
    CHECK(2 * levels[1].n_nodes <= levels[0].n_nodes);
    check_hierarchy(*m, levels);
  }
}

TEST_CASE("one line sweep solves a block-tridiagonal system exactly", "[multigrid]") {
  // One column of cells of aspect ratio 4 and more: a single line holding every cell.
  const auto mesh = layered_box(1, 1e-5, Fill::hexa, false, 0.25);
  const auto g = autan::compute_geometry(mesh);
  Multigrid mg(mesh, g);
  REQUIRE(mg.levels()[0].n_lines() == 1);
  const auto s = diffusion(mesh, g, 1e-3, 0.3);
  mg.set_operator(s.diag, s.upper, s.lower);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> x_exact(5 * n);
  for (std::size_t k = 0; k < x_exact.size(); ++k)
    x_exact[k] = std::sin(0.37 * static_cast<double>(k));
  const auto b = apply(mesh, s, x_exact);
  std::vector<double> x(5 * n, 0.0);
  mg.smooth(b, x, 1);
  CHECK(std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); }));
  double err = 0.0, scale = 0.0;
  for (std::size_t k = 0; k < x.size(); ++k) {
    err = std::max(err, std::abs(x[k] - x_exact[k]));
    scale = std::max(scale, std::abs(x_exact[k]));
  }
  CHECK(err <= 1e3 * std::numeric_limits<double>::epsilon() * scale);
}

TEST_CASE("a two-level cycle applies the Galerkin coarse correction P (P^T A P)^-1 P^T",
          "[multigrid]") {
  // Meshes in RCM order, so that coarse edges of both orientations occur.
  const char* file = GENERATE("hybrid/hybrid.ascii.msh", "poly/polyMesh");
  INFO(file);
  const auto read = autan::io::read_mesh(mesh_dir / file);
  const auto mesh = autan::renumber_cells(read, autan::rcm_order(read));
  const auto g = autan::compute_geometry(mesh);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  MultigridOptions opt;
  opt.pre_sweeps = 0;
  opt.post_sweeps = 0;
  opt.scaling = autan::solvers::Scaling::none;
  opt.agglomeration.coarsest_nodes = n - 1;
  Multigrid mg(mesh, g, opt);
  REQUIRE(mg.levels().size() == 2);
  const std::size_t nc = mg.levels()[1].n_nodes;
  REQUIRE(5 * nc <= 1000);
  const auto s = diffusion(mesh, g, 1.0, 0.3, true);
  mg.set_operator(s.diag, s.upper, s.lower);
  std::vector<double> b(5 * n);
  for (std::size_t k = 0; k < b.size(); ++k) b[k] = std::cos(0.11 * static_cast<double>(k));
  std::vector<double> x(5 * n, 0.0);
  mg.cycle(b, x);
  CHECK(std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); }));
  // The hierarchy has edges whose owner lies in the coarse neighbour: both block orders are
  // exercised.
  const auto& flipped = mg.levels()[0].coarse_flipped;
  CHECK(std::count(flipped.begin(), flipped.end(), std::uint8_t{1}) > 0);
  // Dense P^T A P from the columns A P e_j, then x = P (P^T A P)^-1 P^T b by elimination.
  const auto& parent = mg.levels()[0].coarse_node;
  const std::size_t m = 5 * nc;
  std::vector<double> ac(m * m, 0.0), bc(m, 0.0);
  for (std::size_t j = 0; j < m; ++j) {
    std::vector<double> pe(5 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (static_cast<std::size_t>(parent[i]) == j % nc) pe[(j / nc) * n + i] = 1.0;
    }
    const auto ape = apply(mesh, s, pe);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < 5; ++c) {
        ac[(c * nc + static_cast<std::size_t>(parent[i])) * m + j] += ape[c * n + i];
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c)
      bc[c * nc + static_cast<std::size_t>(parent[i])] += b[c * n + i];
  }
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < m; ++i) {
      if (std::abs(ac[i * m + k]) > std::abs(ac[p * m + k])) p = i;
    }
    std::swap_ranges(&ac[k * m], &ac[k * m] + m, &ac[p * m]);
    std::swap(bc[k], bc[p]);
    for (std::size_t i = k + 1; i < m; ++i) {
      const double f = ac[i * m + k] / ac[k * m + k];
      for (std::size_t j = k; j < m; ++j) ac[i * m + j] -= f * ac[k * m + j];
      bc[i] -= f * bc[k];
    }
  }
  for (std::size_t i = m; i-- > 0;) {
    for (std::size_t j = i + 1; j < m; ++j) bc[i] -= ac[i * m + j] * bc[j];
    bc[i] /= ac[i * m + i];
  }
  double err = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < 5; ++c) {
      const double want = bc[c * nc + static_cast<std::size_t>(parent[i])];
      err = std::max(err, std::abs(x[c * n + i] - want));
      scale = std::max(scale, std::abs(want));
    }
  }
  CHECK(err <= 1e6 * std::numeric_limits<double>::epsilon() * scale);
}

// Iterations of the multigrid-preconditioned solve to a 1e-6 residual drop.
int iterations(const Mesh& mesh, const autan::Geometry& g, const MultigridOptions& options) {
  const auto s = diffusion(mesh, g, 1e-3, 0.1);
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  std::vector<double> b(5 * n), x(5 * n, 0.0);
  for (std::size_t k = 0; k < b.size(); ++k) b[k] = std::sin(0.7 * static_cast<double>(k));
  Multigrid mg(mesh, g, options);
  mg.set_operator(s.diag, s.upper, s.lower);
  const auto report = mg.solve(b, x, 60, 1e-6);
  CHECK(report.residual_drop <= 1e-6);
  const auto r = apply(mesh, s, x);
  double rr = 0.0;
  for (std::size_t k = 0; k < r.size(); ++k) rr += (b[k] - r[k]) * (b[k] - r[k]);
  CHECK(std::sqrt(rr) <= 1.01e-6 * norm(b));
  return report.cycles;
}

TEST_CASE("multigrid iterations do not grow with refinement", "[multigrid]") {
  // Diffusion, the stiff part of the viscous Jacobian, with a small pseudo-time shift on
  // isotropic cells; Gauss-Seidel alone needs O(n^2) sweeps for the same drop.
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  std::vector<int> counts;
  for (const int nx : {8, 16, 32}) {
    const auto mesh = layered_box(nx, 1.0 / nx, fill);
    const auto g = autan::compute_geometry(mesh);
    counts.push_back(iterations(mesh, g, {}));
    INFO("nx " << nx << ": " << counts.back() << " iterations");
    CHECK(counts.back() <= 9);
  }
  INFO("fill " << static_cast<int>(fill) << ", iterations " << counts[0] << ", " << counts[1]
               << ", " << counts[2]);
  CHECK(counts[2] <= counts[0] + 2);
}

TEST_CASE("line smoothing keeps the iterations bounded on wall layers of aspect ratio 1e4",
          "[multigrid]") {
  const Fill fill = GENERATE(Fill::hexa, Fill::prism);
  for (const int nx : {8, 16}) {
    const auto mesh = layered_box(nx, 1e-5, fill);
    const auto g = autan::compute_geometry(mesh);
    const int with_lines = iterations(mesh, g, {});
    MultigridOptions point;
    point.agglomeration.line_ratio = std::numeric_limits<double>::infinity();
    const int without = iterations(mesh, g, point);
    INFO("nx " << nx << ": " << with_lines << " iterations with lines, " << without << " without");
    CHECK(with_lines <= 9);
    CHECK(without >= 2 * with_lines);
  }
}
