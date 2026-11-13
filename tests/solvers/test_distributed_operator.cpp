#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/io/read_mesh.hpp"
#include "autan/partition/partition.hpp"
#include "autan/solvers/agglomeration.hpp"
#include "autan/solvers/distributed_operator.hpp"
#include "autan/solvers/multigrid.hpp"
#include "thread_ranks.hpp"

namespace {

namespace fs = std::filesystem;
using autan::index_t;
using autan::solvers::Level;

// The operator keeps a reference to the hierarchy: no temporary binds to its constructor.
template <class Levels>
constexpr bool operator_from =
    std::is_constructible_v<autan::solvers::DistributedOperator, Levels,
                            std::span<const std::int32_t>, std::int32_t, std::int32_t,
                            autan::solvers::LevelExchange&>;
static_assert(operator_from<const std::vector<Level>&>);
static_assert(!operator_from<std::vector<Level>&&>);
static_assert(!operator_from<const std::vector<Level>&&>);

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

// Global blocks of level 0 (one per cell, two per interior face, in the face's orientation) and
// global vectors per level, non-symmetric and deterministic.
struct Data {
  std::vector<double> diag, upper, lower;
  std::vector<std::vector<double>> x, r, xc;  // per level: multiply, restrict, prolong inputs
};

Data make_data(const std::vector<Level>& levels) {
  std::mt19937 rng(2026);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Data d;
  const auto fill = [&](std::size_t n) {
    std::vector<double> v(n);
    for (double& x : v) x = u(rng);
    return v;
  };
  d.diag = fill(25 * levels[0].n_nodes);
  // Dominant diagonals, so that the host multigrid factors every level.
  for (std::size_t i = 0; i < levels[0].n_nodes; ++i) {
    for (std::size_t p = 0; p < 5; ++p) d.diag[25 * i + 6 * p] += 64.0;
  }
  d.upper = fill(25 * levels[0].owner.size());
  d.lower = fill(25 * levels[0].owner.size());
  for (const Level& l : levels) {
    d.x.push_back(fill(5 * l.n_nodes));
    d.r.push_back(fill(5 * l.n_nodes));
    d.xc.push_back(fill(5 * l.n_nodes));
  }
  return d;
}

// A ghost node's diagonal block as one part holds it.
struct GhostBlock {
  std::size_t level, node;
  std::array<double, 25> block;
};

// What every part computes, gathered by global id: each level's blocks (owned nodes, owned
// edges), A x, the restriction of r and the prolongation of xc; the part that owns each node;
// the diagonal blocks of the ghosts.
struct Result {
  std::vector<std::vector<double>> diag, upper, lower, y, coarse, e;
  std::vector<std::vector<std::int32_t>> owner;
  std::vector<GhostBlock> ghost_diag;
  std::size_t reversed = 0;  // local edges whose local orientation runs against the level's
};

Result run(const std::vector<Level>& levels, const Data& d, const std::vector<std::int32_t>& parts,
           std::int32_t n_parts) {
  const std::size_t n_levels = levels.size();
  Result out;
  for (const Level& l : levels) {
    out.diag.emplace_back(25 * l.n_nodes, std::numeric_limits<double>::quiet_NaN());
    out.upper.emplace_back(25 * l.owner.size(), std::numeric_limits<double>::quiet_NaN());
    out.lower.emplace_back(25 * l.owner.size(), std::numeric_limits<double>::quiet_NaN());
    out.y.emplace_back(5 * l.n_nodes, std::numeric_limits<double>::quiet_NaN());
    out.coarse.emplace_back(5 * l.n_nodes, std::numeric_limits<double>::quiet_NaN());
    out.e.emplace_back(5 * l.n_nodes, std::numeric_limits<double>::quiet_NaN());
    out.owner.emplace_back(l.n_nodes, -1);
  }
  std::vector<std::vector<GhostBlock>> ghosts(at(n_parts));
  autan::test::Hub hub(n_parts);
  std::vector<std::exception_ptr> failed(at(n_parts));
  std::vector<std::size_t> reversed(at(n_parts), 0);
  const auto part = [&](std::int32_t rank) {
    try {
      autan::test::ThreadRanks ex(hub, rank);
      autan::solvers::DistributedOperator op(levels, parts, n_parts, rank, ex);
      const autan::LocalGraph& g0 = op.graph(0);
      std::vector<double> diag, upper, lower;
      for (std::size_t i = 0; i < g0.n_owned; ++i) {
        const auto v = at(g0.node_source[i]);
        diag.insert(diag.end(), d.diag.data() + 25 * v, d.diag.data() + 25 * v + 25);
      }
      for (std::size_t k = 0; k < g0.n_owned_edges; ++k) {
        const auto e = at(g0.edge_global_id[k]);
        upper.insert(upper.end(), d.upper.data() + 25 * e, d.upper.data() + 25 * e + 25);
        lower.insert(lower.end(), d.lower.data() + 25 * e, d.lower.data() + 25 * e + 25);
      }
      op.set_operator(diag, upper, lower);
      for (std::size_t l = 0; l < n_levels; ++l) {
        const autan::LocalGraph& g = op.graph(l);
        reversed[at(rank)] += static_cast<std::size_t>(std::ranges::count(g.edge_reversed, 1));
        const std::size_t n = op.n_local(l);
        for (std::size_t i = 0; i < g.n_owned; ++i) {
          const auto v = at(g.node_source[i]);
          std::copy_n(&op.diagonal(l)[25 * i], 25, &out.diag[l][25 * v]);
          out.owner[l][v] = rank;
        }
        for (std::size_t i = g.n_owned; i < n; ++i) {
          GhostBlock b{.level = l, .node = at(g.node_source[i]), .block = {}};
          std::copy_n(&op.diagonal(l)[25 * i], 25, b.block.begin());
          ghosts[at(rank)].push_back(b);
        }
        for (std::size_t k = 0; k < g.n_owned_edges; ++k) {
          const auto e = at(g.edge_global_id[k]);
          std::copy_n(&op.upper(l)[25 * k], 25, &out.upper[l][25 * e]);
          std::copy_n(&op.lower(l)[25 * k], 25, &out.lower[l][25 * e]);
        }
        // Owned values only; ghosts start as NaN, so a missing exchange shows.
        const auto local = [&](const std::vector<double>& global) {
          std::vector<double> v(5 * n, std::numeric_limits<double>::quiet_NaN());
          for (std::size_t i = 0; i < g.n_owned; ++i) {
            for (std::size_t c = 0; c < 5; ++c)
              v[c * n + i] = global[c * levels[l].n_nodes + at(g.node_source[i])];
          }
          return v;
        };
        auto x = local(d.x[l]);
        // Outputs start as NaN, so an entry left unwritten or accumulated into shows.
        std::vector<double> y(5 * n, std::numeric_limits<double>::quiet_NaN());
        op.multiply(l, x, y);
        for (std::size_t i = 0; i < g.n_owned; ++i) {
          for (std::size_t c = 0; c < 5; ++c)
            out.y[l][c * levels[l].n_nodes + at(g.node_source[i])] = y[c * n + i];
        }
        if (l + 1 == n_levels) continue;
        const autan::LocalGraph& gc = op.graph(l + 1);
        const std::size_t nc = op.n_local(l + 1), nglob = levels[l + 1].n_nodes;
        auto r = local(d.r[l]);
        std::vector<double> coarse(5 * nc, std::numeric_limits<double>::quiet_NaN());
        op.restrict_to(l, r, coarse);
        std::vector<double> xc(5 * nc, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t k = 0; k < gc.n_owned; ++k) {
          const auto v = at(gc.node_source[k]);
          for (std::size_t c = 0; c < 5; ++c) {
            out.coarse[l + 1][c * nglob + v] = coarse[c * nc + k];
            xc[c * nc + k] = d.xc[l + 1][c * nglob + v];
          }
        }
        std::vector<double> e(5 * n, std::numeric_limits<double>::quiet_NaN());
        op.prolong_from(l, xc, e);
        for (std::size_t i = 0; i < g.n_owned; ++i) {
          for (std::size_t c = 0; c < 5; ++c)
            out.e[l][c * levels[l].n_nodes + at(g.node_source[i])] = e[c * n + i];
        }
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
  for (const auto& f : failed) {
    if (f) std::rethrow_exception(f);
  }
  for (const std::size_t k : reversed) out.reversed += k;
  for (const auto& blocks : ghosts)
    out.ghost_diag.insert(out.ghost_diag.end(), blocks.begin(), blocks.end());
  return out;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k])) return false;
  }
  return true;
}

// The definitions, summed in another order than the code's, against the one-part result: each
// coarse block of P^T A P as the sum of A(i, j) over the children i of I and j of J; A x row by
// row from every block; the restriction as the sum of the children; the prolongation as a copy.
// Within 2 k u of the sum of the terms' magnitudes, k the number of terms and u the unit
// round-off (the bound of multigrid-gpu.md's Galerkin row).
void check_definitions(const std::vector<Level>& levels, const Data& d, const Result& one) {
  constexpr double u = 0x1p-53;
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const Level& f = levels[l];
    const std::size_t n = f.n_nodes;
    const auto& dg = l == 0 ? d.diag : one.diag[l];
    const auto& up = l == 0 ? d.upper : one.upper[l];
    const auto& lo = l == 0 ? d.lower : one.lower[l];
    if (l == 0) {
      CHECK(same_bits(one.diag[0], d.diag));
      CHECK(same_bits(one.upper[0], d.upper));
      CHECK(same_bits(one.lower[0], d.lower));
    }
    // A x, row by row.
    std::vector<double> y(5 * n, 0.0), mag(5 * n, 0.0), terms(5 * n, 0.0);
    const auto add = [&](std::size_t i, const double* a, std::size_t j) {
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) {
          y[p * n + i] += a[5 * p + q] * d.x[l][q * n + j];
          mag[p * n + i] += std::abs(a[5 * p + q] * d.x[l][q * n + j]);
          terms[p * n + i] += 1.0;
        }
      }
    };
    for (std::size_t e = f.owner.size(); e-- > 0;) {
      add(at(f.owner[e]), &up[25 * e], at(f.neighbour[e]));
      add(at(f.neighbour[e]), &lo[25 * e], at(f.owner[e]));
    }
    for (std::size_t i = n; i-- > 0;) add(i, &dg[25 * i], i);
    for (std::size_t k = 0; k < 5 * n; ++k)
      CHECK(std::abs(one.y[l][k] - y[k]) <= 2 * terms[k] * u * mag[k]);
    if (l + 1 == levels.size()) continue;
    // P^T A P.
    const Level& c = levels[l + 1];
    // Per coarse pair: the sum, the sum of magnitudes, and the number of terms.
    std::map<std::pair<std::size_t, std::size_t>, std::array<double, 51>> blocks;
    const auto put = [&](std::size_t i, std::size_t j, const double* a) {
      auto& b = blocks[{at(f.coarse_node[i]), at(f.coarse_node[j])}];
      for (std::size_t q = 0; q < 25; ++q) {
        b[q] += a[q];
        b[25 + q] += std::abs(a[q]);
      }
      b[50] += 1.0;
    };
    for (std::size_t i = 0; i < n; ++i) put(i, i, &dg[25 * i]);
    for (std::size_t e = 0; e < f.owner.size(); ++e) {
      put(at(f.owner[e]), at(f.neighbour[e]), &up[25 * e]);
      put(at(f.neighbour[e]), at(f.owner[e]), &lo[25 * e]);
    }
    const auto close = [&](const double* got, const std::array<double, 51>& want) {
      for (std::size_t q = 0; q < 25; ++q)
        CHECK(std::abs(got[q] - want[q]) <= 2 * want[50] * u * want[25 + q]);
    };
    for (std::size_t i = 0; i < c.n_nodes; ++i) close(&one.diag[l + 1][25 * i], blocks[{i, i}]);
    for (std::size_t e = 0; e < c.owner.size(); ++e) {
      const auto o = at(c.owner[e]), m = at(c.neighbour[e]);
      close(&one.upper[l + 1][25 * e], blocks[{o, m}]);
      close(&one.lower[l + 1][25 * e], blocks[{m, o}]);
    }
    // Restriction and prolongation.
    const std::size_t nc = c.n_nodes;
    std::vector<double> sum(5 * nc, 0.0), size(5 * nc, 0.0), count(5 * nc, 0.0);
    for (std::size_t i = n; i-- > 0;) {
      for (std::size_t q = 0; q < 5; ++q) {
        sum[q * nc + at(f.coarse_node[i])] += d.r[l][q * n + i];
        size[q * nc + at(f.coarse_node[i])] += std::abs(d.r[l][q * n + i]);
        count[q * nc + at(f.coarse_node[i])] += 1.0;
      }
    }
    for (std::size_t k = 0; k < 5 * nc; ++k)
      CHECK(std::abs(one.coarse[l + 1][k] - sum[k]) <= 2 * count[k] * u * size[k]);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t q = 0; q < 5; ++q)
        CHECK(one.e[l][q * n + i] == d.xc[l + 1][q * nc + at(f.coarse_node[i])]);
    }
  }
}

// Returns whether a level-1 line of at least two nodes exercised level_parts' line rule.
bool check_parts(const autan::Mesh& mesh, const autan::Geometry& g,
                 const autan::solvers::MultigridOptions& options,
                 std::vector<std::vector<std::int32_t>> cuts) {
  autan::solvers::Multigrid host(mesh, g, options);
  const std::vector<Level>& levels = host.levels();
  const Data d = make_data(levels);
  const std::vector<std::int32_t> whole(levels[0].n_nodes, 0);
  const Result one = run(levels, d, whole, 1);
  check_definitions(levels, d, one);
  // The host multigrid's Galerkin operators, bit for bit: the same sums in the same order.
  host.set_operator(d.diag, d.upper, d.lower);
  for (std::size_t l = 1; l < levels.size(); ++l) {
    CAPTURE(l);
    const auto span_bits = [](std::span<const double> a, const std::vector<double>& b) {
      return same_bits(std::vector<double>(a.begin(), a.end()), b);
    };
    CHECK(span_bits(host.diagonal(l), one.diag[l]));
    CHECK(span_bits(host.upper(l), one.upper[l]));
    CHECK(span_bits(host.lower(l), one.lower[l]));
  }
  // A cut through a level-1 line of at least two nodes: the children of one of its nodes on part
  // 1, every other cell on part 0. That node is not the parent of the line's least child (the
  // least fine id), so level_parts gives the whole line to part 0 (partition.hpp, level_parts),
  // and the node moved holds none of its children there; coarse_parts alone splits the line.
  const Level& fine = levels[0];
  std::optional<std::pair<std::size_t, std::size_t>> line_cut;  // the line, the node moved
  if (levels.size() > 1) {
    const Level& c = levels[1];
    for (std::size_t k = 0; k < c.n_lines() && !line_cut; ++k) {
      const auto begin = at(c.line_offsets[k]), end = at(c.line_offsets[k + 1]);
      if (end - begin < 2) continue;
      std::vector<std::uint8_t> on_line(c.n_nodes, 0);
      for (std::size_t j = begin; j < end; ++j) on_line[at(c.line_nodes[j])] = 1;
      std::size_t least = c.n_nodes;
      for (std::size_t i = 0; i < fine.n_nodes && least == c.n_nodes; ++i) {
        if (on_line[at(fine.coarse_node[i])] != 0) least = at(fine.coarse_node[i]);
      }
      const std::size_t first = at(c.line_nodes[begin]);
      const std::size_t moved = first == least ? at(c.line_nodes[begin + 1]) : first;
      std::vector<std::int32_t> cut(fine.n_nodes, 0);
      for (std::size_t i = 0; i < fine.n_nodes; ++i) {
        if (at(fine.coarse_node[i]) == moved) cut[i] = 1;
      }
      cuts.push_back(cut);
      line_cut = std::pair{k, moved};
    }
  }
  std::size_t reversed = 0, ghost_blocks = 0;
  for (std::size_t q = 0; q < cuts.size(); ++q) {
    const auto& parts = cuts[q];
    const bool constructed = line_cut && q + 1 == cuts.size();
    std::size_t split_by_coarse_parts = 0, no_local_child = 0;
    const std::int32_t n_parts = *std::ranges::max_element(parts) + 1;
    CAPTURE(n_parts);
    const Result many = run(levels, d, parts, n_parts);
    reversed += many.reversed;
    // DistributedOperator::diagonal: ghosts filled, with the owner's block.
    for (const GhostBlock& b : many.ghost_diag) {
      CAPTURE(b.level, b.node);
      CHECK(
          same_bits(std::vector<double>(b.block.begin(), b.block.end()),
                    std::vector<double>(
                        one.diag[b.level].begin() + 25 * static_cast<std::ptrdiff_t>(b.node),
                        one.diag[b.level].begin() + 25 * static_cast<std::ptrdiff_t>(b.node + 1))));
    }
    ghost_blocks += many.ghost_diag.size();
    // partitioning.md, level_parts: every line of a coarse level lies on one part. The same parts
    // taken by coarse_parts alone (each node to its least child's part) split some lines.
    std::vector<std::int32_t> finer = parts;
    for (std::size_t l = 1; l < levels.size(); ++l) {
      CAPTURE(l);
      const Level& c = levels[l];
      std::vector<std::int64_t> ids(levels[l - 1].n_nodes);
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int64_t>(i);
      const auto alone = autan::coarse_parts(finer, levels[l - 1].coarse_node, ids, c.n_nodes);
      for (std::size_t k = 0; k < c.n_lines(); ++k) {
        const auto begin = at(c.line_offsets[k]), end = at(c.line_offsets[k + 1]);
        if (end - begin < 2) continue;
        bool one_part = true, split = false;
        for (std::size_t j = begin + 1; j < end; ++j) {
          one_part = one_part &&
                     many.owner[l][at(c.line_nodes[j])] == many.owner[l][at(c.line_nodes[begin])];
          split = split || alone[at(c.line_nodes[j])] != alone[at(c.line_nodes[begin])];
        }
        CHECK(one_part);
        if (split) ++split_by_coarse_parts;
      }
      // A node the line moved to a part that holds none of its children: its restriction and
      // prolongation go through remote children and parent imports only.
      std::vector<std::uint8_t> local_child(c.n_nodes, 0);
      for (std::size_t i = 0; i < levels[l - 1].n_nodes; ++i) {
        const auto v = at(levels[l - 1].coarse_node[i]);
        if (finer[i] == many.owner[l][v]) local_child[v] = 1;
      }
      no_local_child += static_cast<std::size_t>(std::ranges::count(local_child, 0));
      finer = many.owner[l];
    }
    for (std::size_t l = 0; l < levels.size(); ++l) {
      CAPTURE(l);
      CHECK(same_bits(many.diag[l], one.diag[l]));
      CHECK(same_bits(many.upper[l], one.upper[l]));
      CHECK(same_bits(many.lower[l], one.lower[l]));
      CHECK(same_bits(many.y[l], one.y[l]));
      if (l > 0) CHECK(same_bits(many.coarse[l], one.coarse[l]));
      if (l + 1 < levels.size()) CHECK(same_bits(many.e[l], one.e[l]));
    }
    if (constructed) {
      const Level& c = levels[1];
      for (auto j = at(c.line_offsets[line_cut->first]);
           j < at(c.line_offsets[line_cut->first + 1]); ++j)
        CHECK(many.owner[1][at(c.line_nodes[j])] == 0);
      CHECK(split_by_coarse_parts > 0);
      CHECK(no_local_child > 0);
    }
  }
  // Some local edges run against their level's orientation, so the global orientation of the
  // blocks is exercised; some ghosts hold diagonal blocks.
  CHECK(reversed > 0);
  CHECK(ghost_blocks > 0);
  return line_cut.has_value();
}

std::vector<std::vector<std::int32_t>> slabs(const autan::Geometry& g, std::size_t n_cells) {
  double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
  for (std::size_t c = 0; c < n_cells; ++c) {
    lo = std::min(lo, g.cell_centroid_x[c]);
    hi = std::max(hi, g.cell_centroid_x[c]);
  }
  std::vector<std::vector<std::int32_t>> cuts;
  for (const std::int32_t n : {2, 3, 4}) {
    std::vector<std::int32_t> parts(n_cells);
    for (std::size_t c = 0; c < n_cells; ++c) {
      const double t = (g.cell_centroid_x[c] - lo) / (hi - lo);
      parts[c] = std::min(n - 1, static_cast<std::int32_t>(t * n));
    }
    cuts.push_back(parts);
  }
  return cuts;
}

}  // namespace

TEST_CASE(
    "the distributed Galerkin operators, products and transfers repeat their bits on 1 to 4 parts",
    "[multigrid][distributed]") {
  // #259 D2: on 1, 2, 3 and 4 parts every owned block of every level, A x, the restriction and
  // the prolongation have the same bits, gathered by global id; on 1 part they match the
  // definitions (P^T A P, A x, P^T r, P x) summed in another order, and the coarse blocks are the
  // host multigrid's bit for bit.
  const fs::path dir{AUTAN_TEST_MESH_DIR};
  const char* file = GENERATE("tet-gmsh/tet-gmsh.ascii.msh", "hex/hex.ascii.msh", "poly/polyMesh",
                              "hybrid/hybrid.ascii.msh");
  INFO(file);
  const auto mesh = autan::io::read_mesh(dir / file);
  const auto g = autan::compute_geometry(mesh);
  autan::solvers::MultigridOptions options;
  options.agglomeration.coarsest_nodes = 4;
  REQUIRE(autan::solvers::agglomerate(mesh, g, options.agglomeration).size() >= 2);
  const bool line_rule =
      check_parts(mesh, g, options, slabs(g, static_cast<std::size_t>(mesh.n_cells())));
  // The polyhedral mesh has lines on level 1, so the line rule is never left untested; the others
  // may have none (the uniform hex mesh's couplings are equal).
  if (std::string_view(file) == "poly/polyMesh") REQUIRE(line_rule);
}
