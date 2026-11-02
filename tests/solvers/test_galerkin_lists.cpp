// Gather lists of the device Galerkin product, against the host multigrid's summation order and
// against a dense Pᵀ A P (docs/design/multigrid-gpu.md#galerkin-coarse-operator).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/device_order.hpp"
#include "autan/solvers/galerkin_lists.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::offset_t;
using autan::solvers::Level;
using autan::solvers::gpu::EdgeSlots;
using autan::solvers::gpu::GalerkinLists;

constexpr std::size_t b2 = 25;
const fs::path mesh_dir{AUTAN_TEST_MESH_DIR};
const std::array<const char*, 4> meshes = {"tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh",
                                           "poly/polyMesh", "hybrid/hybrid.ascii.msh"};

// Hexahedra on [0,1]^2 x [0,1/2], nx x nx in the plane, layers growing by 1.2 from 1e-4 at the
// wall z = 0: anisotropic wall cells, so the hierarchy holds lines.
autan::Mesh wall_layers(int nx) {
  autan::CellwiseMesh m;
  autan::CellZone zone;
  zone.name = "fluid";
  zone.kind = autan::CellKind::fluid;
  m.cell_zones.push_back(zone);
  autan::FaceZone wall;
  wall.name = "wall";
  wall.kind = autan::BcKind::wall;
  m.face_zones.push_back(wall);
  const double dx = 1.0 / nx;
  std::vector<double> z{0.0};
  for (double d = 1e-4; z.back() < 0.5; d = std::min(dx, 1.2 * d)) z.push_back(z.back() + d);
  const int nz = static_cast<int>(z.size()) - 1, p = nx + 1;
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
        m.add_cell(autan::CellShape::hexa,
                   std::array<index_t, 8>{v(i, j, k), v(i + 1, j, k), v(i + 1, j + 1, k),
                                          v(i, j + 1, k), v(i, j, k + 1), v(i + 1, j, k + 1),
                                          v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)},
                   0);
        if (k == 0) {
          m.add_labelled_face(std::array<index_t, 4>{v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0),
                                                     v(i + 1, j, 0)},
                              0);
        }
      }
    }
  }
  return autan::assemble(autan::to_face_based(std::move(m)));
}

std::vector<Level> hierarchy(const autan::Mesh& mesh) {
  autan::solvers::AgglomerationOptions opt;
  opt.coarsest_nodes = 4;
  return autan::solvers::agglomerate(mesh, autan::compute_geometry(mesh), opt);
}

// Slots of a level laid out as a CSR over its device nodes, each node's edges in increasing
// edge index, followed by `padding` slots no edge uses.
EdgeSlots csr_slots(const Level& l, const std::vector<index_t>& order, offset_t padding,
                    offset_t& n_slots) {
  const std::size_t n = l.n_nodes, ne = l.owner.size();
  std::vector<index_t> rank(n);
  for (std::size_t j = 0; j < n; ++j)
    rank[static_cast<std::size_t>(order[j])] = static_cast<index_t>(j);
  std::vector<offset_t> start(n + 1, 0);
  for (std::size_t e = 0; e < ne; ++e) {
    ++start[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.owner[e])]) + 1];
    ++start[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.neighbour[e])]) + 1];
  }
  for (std::size_t j = 0; j < n; ++j) start[j + 1] += start[j];
  EdgeSlots s;
  s.owner.resize(ne);
  s.neighbour.resize(ne);
  std::vector<offset_t> fill(start.begin(), start.end() - 1);
  for (std::size_t e = 0; e < ne; ++e) {
    s.owner[e] = fill[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.owner[e])])]++;
    s.neighbour[e] =
        fill[static_cast<std::size_t>(rank[static_cast<std::size_t>(l.neighbour[e])])]++;
  }
  n_slots = start.back() + padding;
  return s;
}

// A level's operator in host form: diagonal per node, upper = A(owner, neighbour) and
// lower = A(neighbour, owner) per edge, row-major 5 x 5.
struct Blocks {
  std::vector<double> diag, upper, lower;
};

// Values over 120 binades with both signs, so that any other summation order rounds otherwise.
Blocks spread(const Level& l, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  std::uniform_int_distribution<int> exponent(-60, 60), sign(0, 1);
  auto fill = [&](std::size_t n) {
    std::vector<double> v(n);
    for (double& x : v) x = (sign(rng) ? -1.0 : 1.0) * std::ldexp(mantissa(rng), exponent(rng));
    return v;
  };
  return {fill(b2 * l.n_nodes), fill(b2 * l.owner.size()), fill(b2 * l.owner.size())};
}

// The host multigrid's coarse operator (Multigrid::set_operator): diagonals first, then the
// edges inside agglomerates (upper, then lower), then the coarse edges, each fine edge's blocks
// in increasing edge index, a flipped edge's swapped.
Blocks host_galerkin(const Level& f, const Level& c, const Blocks& a) {
  Blocks g;
  g.diag.assign(b2 * c.n_nodes, 0.0);
  g.upper.assign(b2 * c.owner.size(), 0.0);
  g.lower.assign(b2 * c.owner.size(), 0.0);
  auto add = [](double* to, const double* from) {
    for (std::size_t k = 0; k < b2; ++k) to[k] += from[k];
  };
  for (std::size_t i = 0; i < f.n_nodes; ++i) {
    add(&g.diag[b2 * static_cast<std::size_t>(f.coarse_node[i])], &a.diag[b2 * i]);
  }
  for (std::size_t e = 0; e < f.owner.size(); ++e) {
    const index_t E = f.coarse_edge[e];
    if (E < 0) {
      double* d = &g.diag[b2 * static_cast<std::size_t>(
                                   f.coarse_node[static_cast<std::size_t>(f.owner[e])])];
      add(d, &a.upper[b2 * e]);
      add(d, &a.lower[b2 * e]);
      continue;
    }
    const bool flip = f.coarse_flipped[e] != 0;
    add(&g.upper[b2 * static_cast<std::size_t>(E)], flip ? &a.lower[b2 * e] : &a.upper[b2 * e]);
    add(&g.lower[b2 * static_cast<std::size_t>(E)], flip ? &a.upper[b2 * e] : &a.lower[b2 * e]);
  }
  return g;
}

// The lists applied to the fine level in its device layout: each block the sum, from +0.0, of
// its list's blocks in list order.
struct Device {
  std::vector<double> diag, off;
};

Device to_device(const Blocks& a, const std::vector<index_t>& order, const EdgeSlots& s,
                 offset_t n_slots) {
  Device d;
  d.diag.resize(b2 * order.size());
  for (std::size_t j = 0; j < order.size(); ++j) {
    std::copy_n(&a.diag[b2 * static_cast<std::size_t>(order[j])], b2, &d.diag[b2 * j]);
  }
  d.off.assign(b2 * static_cast<std::size_t>(n_slots), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t e = 0; e < s.owner.size(); ++e) {
    std::copy_n(&a.upper[b2 * e], b2, &d.off[b2 * static_cast<std::size_t>(s.owner[e])]);
    std::copy_n(&a.lower[b2 * e], b2, &d.off[b2 * static_cast<std::size_t>(s.neighbour[e])]);
  }
  return d;
}

Device apply(const GalerkinLists& g, const Device& fine) {
  Device c;
  const std::size_t nc = g.diag_offsets.size() - 1, ns = g.off_offsets.size() - 1;
  c.diag.assign(b2 * nc, 0.0);
  c.off.assign(b2 * ns, 0.0);
  for (std::size_t J = 0; J < nc; ++J) {
    for (auto k = g.diag_offsets[J]; k < g.diag_offsets[J + 1]; ++k) {
      const std::int64_t t = g.diag[static_cast<std::size_t>(k)];
      const double* b = t >= 0 ? &fine.diag[b2 * static_cast<std::size_t>(t)]
                               : &fine.off[b2 * static_cast<std::size_t>(-(t + 1))];
      for (std::size_t q = 0; q < b2; ++q) c.diag[b2 * J + q] += b[q];
    }
  }
  for (std::size_t s = 0; s < ns; ++s) {
    for (auto k = g.off_offsets[s]; k < g.off_offsets[s + 1]; ++k) {
      const double* b =
          &fine.off[b2 * static_cast<std::size_t>(g.off[static_cast<std::size_t>(k)])];
      for (std::size_t q = 0; q < b2; ++q) c.off[b2 * s + q] += b[q];
    }
  }
  return c;
}

bool same_bits(const double* a, const double* b) {
  for (std::size_t q = 0; q < b2; ++q) {
    if (std::bit_cast<std::uint64_t>(a[q]) != std::bit_cast<std::uint64_t>(b[q])) return false;
  }
  return true;
}

// Every level of a hierarchy: the lists give the host's coarse operator bit for bit, at the
// device position of each coarse block, and no list reads a padding slot.
void check_hierarchy(const std::vector<Level>& levels) {
  REQUIRE(levels.size() >= 2);
  const auto orders = autan::solvers::gpu::device_orders(levels);
  Blocks a = spread(levels[0], 3);
  offset_t n_fine_slots = 0;
  EdgeSlots fine_slots = csr_slots(levels[0], orders[0], 0, n_fine_slots);
  for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
    INFO("level " << l);
    const Level &f = levels[l], &c = levels[l + 1];
    offset_t n_coarse_slots = 0;
    const EdgeSlots coarse_slots = csr_slots(c, orders[l + 1], 3, n_coarse_slots);
    const GalerkinLists g = autan::solvers::gpu::galerkin_lists(
        f, c, orders[l], orders[l + 1], fine_slots, coarse_slots, n_coarse_slots);
    const Blocks expected = host_galerkin(f, c, a);
    const Device got = apply(g, to_device(a, orders[l], fine_slots, n_fine_slots));
    for (std::size_t J = 0; J < c.n_nodes; ++J) {
      const auto H = static_cast<std::size_t>(orders[l + 1][J]);
      CHECK(same_bits(&got.diag[b2 * J], &expected.diag[b2 * H]));
    }
    for (std::size_t E = 0; E < c.owner.size(); ++E) {
      CHECK(same_bits(&got.off[b2 * static_cast<std::size_t>(coarse_slots.owner[E])],
                      &expected.upper[b2 * E]));
      CHECK(same_bits(&got.off[b2 * static_cast<std::size_t>(coarse_slots.neighbour[E])],
                      &expected.lower[b2 * E]));
    }
    for (auto s = n_coarse_slots - 3; s < n_coarse_slots; ++s) {
      CHECK(g.off_offsets[static_cast<std::size_t>(s)] ==
            g.off_offsets[static_cast<std::size_t>(s) + 1]);
    }
    a = expected;
    fine_slots = coarse_slots;
    n_fine_slots = n_coarse_slots;
  }
}

}  // namespace

TEST_CASE("Galerkin lists give the host multigrid's coarse operators bit for bit", "[multigrid]") {
  SECTION("reference meshes") {
    const char* file = GENERATE(from_range(meshes));
    INFO(file);
    check_hierarchy(hierarchy(autan::io::read_mesh(mesh_dir / file)));
  }
  SECTION("wall layers, with lines") {
    const auto levels = hierarchy(wall_layers(8));
    bool lines = false;
    for (const Level& l : levels) lines = lines || l.n_lines() < l.n_nodes;
    REQUIRE(lines);
    check_hierarchy(levels);
  }
}

TEST_CASE("Galerkin lists sum to P^T A P within the rounding of two summation orders",
          "[multigrid]") {
  const auto levels = hierarchy(wall_layers(8));
  const Level &f = levels[0], &c = levels[1];
  const std::size_t n = f.n_nodes, nc = c.n_nodes;
  const auto orders = autan::solvers::gpu::device_orders(levels);
  offset_t nf_slots = 0, nc_slots = 0;
  const EdgeSlots fs = csr_slots(f, orders[0], 0, nf_slots);
  const EdgeSlots cs = csr_slots(c, orders[1], 0, nc_slots);
  const Blocks a = spread(f, 11);
  const Device got =
      apply(autan::solvers::gpu::galerkin_lists(f, c, orders[0], orders[1], fs, cs, nc_slots),
            to_device(a, orders[0], fs, nf_slots));

  // (Pᵀ A P)(I, J) is the sum of the fine blocks A(i, j) with i in I and j in J. Summed here per
  // fine row i, then its edges in reverse, an order unrelated to the lists'; each sum rounds at
  // most k − 1 times over k terms, so the two differ by at most 2 k u Σ |terms|.
  struct Entry {
    std::array<double, b2> sum{}, magnitude{};
    std::size_t terms = 0;
  };
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> row_edges(n);
  for (std::size_t e = 0; e < f.owner.size(); ++e) {
    row_edges[static_cast<std::size_t>(f.owner[e])].push_back({e, 0});
    row_edges[static_cast<std::size_t>(f.neighbour[e])].push_back({e, 1});
  }
  std::vector<Entry> diag(nc);
  std::vector<std::array<Entry, 2>> edge(c.owner.size());  // [0] A(owner, neighbour), [1] reverse
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> coarse_edges(nc);
  for (std::size_t E = 0; E < c.owner.size(); ++E) {
    coarse_edges[static_cast<std::size_t>(c.owner[E])].push_back(
        {static_cast<std::size_t>(c.neighbour[E]), E});
  }
  auto target = [&](std::size_t I, std::size_t J) -> Entry& {
    if (I == J) return diag[I];
    const std::size_t lo = std::min(I, J), hi = std::max(I, J);
    for (const auto& [other, E] : coarse_edges[lo]) {
      if (other == hi) return edge[E][I == lo ? 0 : 1];
    }
    throw std::logic_error("no coarse edge joins two coupled agglomerates");
  };
  auto add = [](Entry& t, const double* b) {
    for (std::size_t q = 0; q < b2; ++q) {
      t.sum[q] += b[q];
      t.magnitude[q] += std::abs(b[q]);
    }
    ++t.terms;
  };
  for (std::size_t i = n; i-- > 0;) {
    const auto I = static_cast<std::size_t>(f.coarse_node[i]);
    add(target(I, I), &a.diag[b2 * i]);
    for (auto it = row_edges[i].rbegin(); it != row_edges[i].rend(); ++it) {
      const auto [e, side] = *it;
      const auto j = static_cast<std::size_t>(side == 0 ? f.neighbour[e] : f.owner[e]);
      add(target(I, static_cast<std::size_t>(f.coarse_node[j])),
          side == 0 ? &a.upper[b2 * e] : &a.lower[b2 * e]);
    }
  }
  const double u = std::numeric_limits<double>::epsilon() / 2;
  auto check = [&](const double* block, const Entry& t) {
    for (std::size_t q = 0; q < b2; ++q) {
      CHECK(std::abs(block[q] - t.sum[q]) <=
            2.0 * static_cast<double>(t.terms) * u * t.magnitude[q]);
    }
  };
  for (std::size_t J = 0; J < nc; ++J) {
    check(&got.diag[b2 * J], diag[static_cast<std::size_t>(orders[1][J])]);
  }
  for (std::size_t E = 0; E < c.owner.size(); ++E) {
    check(&got.off[b2 * static_cast<std::size_t>(cs.owner[E])], edge[E][0]);
    check(&got.off[b2 * static_cast<std::size_t>(cs.neighbour[E])], edge[E][1]);
  }
}

TEST_CASE("Galerkin lists refuse inputs that are not consecutive levels", "[multigrid]") {
  const auto levels = hierarchy(autan::io::read_mesh(mesh_dir / "hex/hex.ascii.msh"));
  const auto orders = autan::solvers::gpu::device_orders(levels);
  offset_t nf = 0, nc = 0;
  const EdgeSlots fs = csr_slots(levels[0], orders[0], 0, nf);
  const EdgeSlots cs = csr_slots(levels[1], orders[1], 0, nc);
  auto lists = [&](const Level& f, const Level& c, std::vector<index_t> fine_order,
                   std::vector<index_t> coarse_order, const EdgeSlots& s, offset_t n) {
    return autan::solvers::gpu::galerkin_lists(f, c, fine_order, coarse_order, fs, s, n);
  };
  CHECK_NOTHROW(lists(levels[0], levels[1], orders[0], orders[1], cs, nc));
  CHECK_THROWS_AS(lists(levels.back(), levels[1], orders.back(), orders[1], cs, nc),
                  std::invalid_argument);
  auto repeated = orders[1];
  repeated[0] = repeated[1];
  CHECK_THROWS_AS(lists(levels[0], levels[1], orders[0], repeated, cs, nc), std::invalid_argument);
  CHECK_THROWS_AS(lists(levels[0], levels[1], orders[0], orders[1], cs, nc - 1),
                  std::invalid_argument);
}
