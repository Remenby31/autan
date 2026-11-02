// Gather lists of the device Galerkin product, against the host multigrid's summation order and
// against a dense Pᵀ A P (docs/design/multigrid-gpu.md#galerkin-coarse-operator).
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
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
#include "galerkin_fixture.hpp"

namespace {

using namespace galerkin_fixture;

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
