#include "autan/solvers/distributed_multigrid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "autan/solvers/dense.hpp"
#include "blocks.hpp"

namespace autan::solvers {

namespace {

using detail::add5;
using detail::b2;
using detail::invert5;
using detail::multiply5;

// The coarsest level is factored densely up to this many unknowns, as in Multigrid.
constexpr std::size_t dense_unknowns = 1000;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

}  // namespace

// Multigrid::Smoother on one rank's lines: the lines of its owned nodes, in the global colours.
struct DistributedMultigrid::Smoother {
  std::vector<std::vector<std::vector<std::size_t>>> colours;  // colour -> lines -> local nodes
  std::vector<index_t> prev, next;
  std::vector<std::size_t> slot;
  std::vector<double> dinv, lower, gain, work;

  Smoother(const Level& level, const LocalGraph& g) {
    std::vector<index_t> local(level.n_nodes, -1);
    for (std::size_t i = 0; i < g.n_owned; ++i)
      local[at(g.node_source[i])] = static_cast<index_t>(i);
    prev.assign(g.n_owned, -1);
    next.assign(g.n_owned, -1);
    slot.assign(g.n_owned, 0);
    std::size_t n_slots = 0, longest = 1;
    for (const auto& colour : line_colours(level)) {
      auto& mine = colours.emplace_back();
      for (const std::size_t li : colour) {
        const auto first = at(level.line_offsets[li]), last = at(level.line_offsets[li + 1]);
        if (local[at(level.line_nodes[first])] < 0) continue;
        std::vector<std::size_t> nodes;
        for (std::size_t k = first; k < last; ++k) {
          const index_t i = local[at(level.line_nodes[k])];
          if (i < 0) throw std::logic_error("a line spans two parts");
          nodes.push_back(at(i));
        }
        longest = std::max(longest, nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) {
          if (k > 0) prev[nodes[k]] = static_cast<index_t>(nodes[k - 1]);
          if (k + 1 < nodes.size()) next[nodes[k]] = static_cast<index_t>(nodes[k + 1]);
          if (nodes.size() > 1) slot[nodes[k]] = n_slots++;
        }
        mine.push_back(std::move(nodes));
      }
    }
    lower.assign(b2 * n_slots, 0.0);
    gain.assign(b2 * n_slots, 0.0);
    dinv.assign(b2 * g.n_owned, 0.0);
    work.assign(5 * longest, 0.0);
  }

  // The coupling A(i, j) summed over the edges joining them, by increasing global edge id.
  static void coupling(const DistributedOperator& op, std::size_t l, std::size_t i, std::size_t j,
                       double* a) {
    std::fill(a, a + b2, 0.0);
    for (const index_t k : op.incident(l, i)) {
      const auto [blk, o] = op.off(l, at(k), i);
      if (o == j) add5(a, blk);
    }
  }

  // Block LU of every line, as Multigrid::Smoother::factor.
  void factor(const DistributedOperator& op, std::size_t l) {
    std::array<double, b2> d{}, t{}, u{};
    const auto diag = op.diagonal(l);
    for (const auto& colour : colours) {
      for (const auto& line : colour) {
        for (const std::size_t i : line) {
          std::copy_n(&diag[b2 * i], b2, d.data());
          if (prev[i] >= 0) {
            const auto p = at(prev[i]);
            double* lo = &lower[b2 * slot[i]];
            coupling(op, l, i, p, lo);
            multiply5(lo, &gain[b2 * slot[p]], t.data());
            for (std::size_t m = 0; m < b2; ++m) d[m] -= t[m];
          }
          invert5(d.data(), &dinv[b2 * i]);
          if (next[i] >= 0) {
            coupling(op, l, i, at(next[i]), u.data());
            multiply5(&dinv[b2 * i], u.data(), &gain[b2 * slot[i]]);
          }
        }
      }
    }
  }

  // Multigrid::Smoother::relax_line on local nodes: b - (couplings to nodes off the line) x, by
  // increasing global edge id, then the line's block-tridiagonal solve.
  void relax_line(const DistributedOperator& op, std::size_t l,
                  const std::vector<std::size_t>& line, std::span<const double> b,
                  std::span<double> x) {
    const std::size_t n = op.n_local(l);
    for (std::size_t k = 0; k < line.size(); ++k) {
      const std::size_t i = line[k];
      double* r = &work[5 * k];
      for (std::size_t c = 0; c < 5; ++c) r[c] = b[c * n + i];
      for (const index_t e : op.incident(l, i)) {
        const auto [a, j] = op.off(l, at(e), i);
        if (static_cast<index_t>(j) == prev[i] || static_cast<index_t>(j) == next[i]) continue;
        for (std::size_t p = 0; p < 5; ++p) {
          for (std::size_t q = 0; q < 5; ++q) r[p] -= a[5 * p + q] * x[q * n + j];
        }
      }
    }
    std::array<double, 5> s{};
    for (std::size_t k = 0; k < line.size(); ++k) {
      const std::size_t i = line[k];
      double* r = &work[5 * k];
      if (k > 0) {
        const double* lo = &lower[b2 * slot[i]];
        const double* y = &work[5 * (k - 1)];
        for (std::size_t a = 0; a < 5; ++a) {
          for (std::size_t c = 0; c < 5; ++c) r[a] -= lo[5 * a + c] * y[c];
        }
      }
      const double* inv = &dinv[b2 * i];
      for (std::size_t a = 0; a < 5; ++a) {
        s[a] = 0.0;
        for (std::size_t c = 0; c < 5; ++c) s[a] += inv[5 * a + c] * r[c];
      }
      std::copy_n(s.data(), 5, r);
    }
    for (std::size_t k = line.size(); k-- > 0;) {
      const std::size_t i = line[k];
      double* y = &work[5 * k];
      if (k + 1 < line.size()) {
        const double* g = &gain[b2 * slot[i]];
        const double* xn = &work[5 * (k + 1)];
        for (std::size_t a = 0; a < 5; ++a) {
          for (std::size_t c = 0; c < 5; ++c) y[a] -= g[5 * a + c] * xn[c];
        }
      }
      for (std::size_t a = 0; a < 5; ++a) x[a * n + i] = y[a];
    }
  }
};

DistributedMultigrid::DistributedMultigrid(const std::vector<Level>& levels,
                                           std::span<const std::int32_t> cell_parts,
                                           std::int32_t n_parts, std::int32_t rank,
                                           LevelCollective& collective,
                                           const MultigridOptions& options)
    : levels_(levels),
      options_(options),
      collective_(collective),
      op_(levels, cell_parts, n_parts, rank, collective) {
  const std::size_t n_levels = levels.size();
  for (std::size_t l = 0; l < n_levels; ++l) {
    smoothers_.emplace_back(levels[l], op_.graph(l));
    const std::size_t n = 5 * op_.n_local(l);
    const bool above = l + 1 < n_levels, below = l > 0;
    r_.emplace_back(above ? n : 0);
    e_.emplace_back(above ? n : 0);
    q_.emplace_back(above && options_.scaling != Scaling::none ? n : 0);
    bc_.emplace_back(below ? n : 0);
    xc_.emplace_back(below ? n : 0);
    krylov_.emplace_back(options_.cycle == Cycle::k && above && below ? 5 : 0,
                         std::vector<double>(n));
  }
  direct_ = 5 * levels.back().n_nodes <= dense_unknowns;
}

DistributedMultigrid::~DistributedMultigrid() = default;

void DistributedMultigrid::set_operator(std::span<const double> diagonal,
                                        std::span<const double> upper,
                                        std::span<const double> lower) {
  op_.set_operator(diagonal, upper, lower);
  const std::size_t n_levels = levels_.size();
  double singular = 0.0;
  for (std::size_t l = 0; l < n_levels; ++l) {
    const bool coarsest = l + 1 == n_levels;
    if (coarsest && direct_ && l > 0) continue;
    try {
      smoothers_[l].factor(op_, l);
    } catch (const std::domain_error&) {
      singular = 1.0;
    }
  }
  if (direct_) {
    // Every rank assembles the coarsest level's dense matrix as Multigrid::Smoother::factor_dense
    // does, from its blocks gathered in global order.
    const std::size_t lc = n_levels - 1;
    const Level& c = levels_[lc];
    const LocalGraph& g = op_.graph(lc);
    std::vector<double> mine;
    for (std::size_t i = 0; i < g.n_owned; ++i) {
      mine.push_back(0.0);
      mine.push_back(static_cast<double>(g.node_global_id[i]));
      const auto d = op_.diagonal(lc);
      mine.insert(mine.end(), d.begin() + static_cast<std::ptrdiff_t>(b2 * i),
                  d.begin() + static_cast<std::ptrdiff_t>(b2 * i + b2));
    }
    for (std::size_t k = 0; k < g.n_owned_edges; ++k) {
      mine.push_back(1.0);
      mine.push_back(static_cast<double>(g.edge_global_id[k]));
      const auto u = op_.upper(lc), w = op_.lower(lc);
      mine.insert(mine.end(), u.begin() + static_cast<std::ptrdiff_t>(b2 * k),
                  u.begin() + static_cast<std::ptrdiff_t>(b2 * k + b2));
      mine.insert(mine.end(), w.begin() + static_cast<std::ptrdiff_t>(b2 * k),
                  w.begin() + static_cast<std::ptrdiff_t>(b2 * k + b2));
    }
    const std::vector<double> all = collective_.all_gather(mine);
    std::vector<double> diag(b2 * c.n_nodes), up(b2 * c.owner.size()), lo(b2 * c.owner.size());
    for (std::size_t p = 0; p < all.size();) {
      const auto id = static_cast<std::size_t>(all[p + 1]);
      if (all[p] == 0.0) {
        std::copy_n(&all[p + 2], b2, &diag[b2 * id]);
        p += 2 + b2;
      } else {
        std::copy_n(&all[p + 2], b2, &up[b2 * id]);
        std::copy_n(&all[p + 2 + b2], b2, &lo[b2 * id]);
        p += 2 + 2 * b2;
      }
    }
    std::vector<std::vector<std::size_t>> incident(c.n_nodes);
    for (std::size_t e = 0; e < c.owner.size(); ++e) {
      incident[at(c.owner[e])].push_back(e);
      incident[at(c.neighbour[e])].push_back(e);
    }
    const std::size_t n = c.n_nodes, m = 5 * n;
    dense_.assign(m * m, 0.0);
    const auto put = [&](std::size_t i, std::size_t j, const double* a) {
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) dense_[(p * n + i) * m + q * n + j] += a[5 * p + q];
      }
    };
    for (std::size_t i = 0; i < n; ++i) {
      put(i, i, &diag[b2 * i]);
      for (const std::size_t e : incident[i]) {
        const bool owner = at(c.owner[e]) == i;
        put(i, at(owner ? c.neighbour[e] : c.owner[e]), owner ? &up[b2 * e] : &lo[b2 * e]);
      }
    }
    pivot_.resize(m);
    try {
      factor_dense(dense_, m, pivot_);
    } catch (const std::domain_error&) {
      singular = 1.0;
    }
  }
  if (collective_.max(singular) > 0.0) throw std::domain_error("singular block");
}

double DistributedMultigrid::dot(std::size_t l, std::span<const double> u,
                                 std::span<const double> v) {
  const std::size_t n = op_.n_local(l), owned = op_.graph(l).n_owned;
  std::vector<double> terms;
  terms.reserve(5 * owned);
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < owned; ++i) terms.push_back(u[c * n + i] * v[c * n + i]);
  }
  return collective_.sum(terms);
}

void DistributedMultigrid::sweep(std::size_t l, std::span<const double> b, std::span<double> x,
                                 int sweeps) {
  const LocalGraph& g = op_.graph(l);
  const std::size_t n = op_.n_local(l);
  const auto exchange = [&] {
    std::array<double*, 5> components{};
    for (std::size_t c = 0; c < 5; ++c) components[c] = x.data() + c * n;
    collective_.forward(g.node_halo, g.n_owned, components);
  };
  Smoother& s = smoothers_[l];
  for (int k = 0; k < sweeps; ++k) {
    exchange();
    for (const auto& colour : s.colours) {
      for (const auto& line : colour) s.relax_line(op_, l, line, b, x);
      exchange();
    }
    for (auto it = s.colours.rbegin(); it != s.colours.rend(); ++it) {
      for (const auto& line : *it) s.relax_line(op_, l, line, b, x);
      exchange();
    }
  }
}

// r = b - A x on the owned nodes, as Multigrid's: b, then each edge by increasing global id, then
// the diagonal.
void DistributedMultigrid::residual(std::size_t l, std::span<const double> b, std::span<double> x,
                                    std::span<double> r) {
  const LocalGraph& g = op_.graph(l);
  const std::size_t n = op_.n_local(l);
  std::array<double*, 5> components{};
  for (std::size_t c = 0; c < 5; ++c) components[c] = x.data() + c * n;
  collective_.forward(g.node_halo, g.n_owned, components);
  const auto diag = op_.diagonal(l);
  std::array<double, 5> s{};
  for (std::size_t i = 0; i < g.n_owned; ++i) {
    for (std::size_t c = 0; c < 5; ++c) s[c] = b[c * n + i];
    for (const index_t e : op_.incident(l, i)) {
      const auto [a, j] = op_.off(l, at(e), i);
      for (std::size_t p = 0; p < 5; ++p) {
        for (std::size_t q = 0; q < 5; ++q) s[p] -= a[5 * p + q] * x[q * n + j];
      }
    }
    const double* d = &diag[b2 * i];
    for (std::size_t p = 0; p < 5; ++p) {
      for (std::size_t q = 0; q < 5; ++q) s[p] -= d[5 * p + q] * x[q * n + i];
      r[p * n + i] = s[p];
    }
  }
}

void DistributedMultigrid::solve_coarsest(std::span<const double> b, std::span<double> x) {
  const std::size_t lc = levels_.size() - 1;
  const LocalGraph& g = op_.graph(lc);
  const std::size_t n_local = op_.n_local(lc), n = levels_[lc].n_nodes;
  std::vector<double> mine;
  for (std::size_t i = 0; i < g.n_owned; ++i) {
    mine.push_back(static_cast<double>(g.node_global_id[i]));
    for (std::size_t c = 0; c < 5; ++c) mine.push_back(b[c * n_local + i]);
  }
  const std::vector<double> all = collective_.all_gather(mine);
  std::vector<double> rhs(5 * n, 0.0);
  for (std::size_t p = 0; p < all.size(); p += 6) {
    const auto id = static_cast<std::size_t>(all[p]);
    for (std::size_t c = 0; c < 5; ++c) rhs[c * n + id] = all[p + 1 + c];
  }
  solve_dense(dense_, pivot_, rhs);
  for (std::size_t i = 0; i < g.n_owned; ++i) {
    const auto id = at(g.node_global_id[i]);
    for (std::size_t c = 0; c < 5; ++c) x[c * n_local + i] = rhs[c * n + id];
  }
}

void DistributedMultigrid::cycle(std::span<const double> b, std::span<double> x) {
  cycle_level(0, b, x);
}

void DistributedMultigrid::cycle_level(std::size_t l, std::span<const double> b,
                                       std::span<double> x) {
  if (l + 1 == levels_.size()) {
    if (direct_) {
      solve_coarsest(b, x);
    } else {
      sweep(l, b, x, options_.coarsest_sweeps);
    }
    return;
  }
  sweep(l, b, x, options_.pre_sweeps);
  const std::size_t n = op_.n_local(l), owned = op_.graph(l).n_owned;
  auto& r = r_[l];
  auto& bc = bc_[l + 1];
  auto& xc = xc_[l + 1];
  residual(l, b, x, r);
  op_.restrict_to(l, r, bc);
  coarse_solve(l + 1, bc, xc);
  auto& e = e_[l];
  op_.prolong_from(l, xc, e);
  double alpha = 1.0;
  if (options_.scaling != Scaling::none) {
    auto& q = q_[l];
    op_.multiply(l, e, q);
    const bool residual_scaling = options_.scaling == Scaling::residual;
    const double num = residual_scaling ? dot(l, r, q) : dot(l, r, e);
    const double den = residual_scaling ? dot(l, q, q) : dot(l, q, e);
    if (den > 0.0) alpha = num / den;
  }
  for (std::size_t c = 0; c < 5; ++c) {
    for (std::size_t i = 0; i < owned; ++i) x[c * n + i] += alpha * e[c * n + i];
  }
  sweep(l, b, x, options_.post_sweeps);
}

// Multigrid::coarse_solve, its inner products reduced over the ranks; every branch is taken from
// the reduced values, so every rank takes it.
void DistributedMultigrid::coarse_solve(std::size_t m, std::span<const double> b,
                                        std::span<double> x) {
  std::fill(x.begin(), x.end(), 0.0);
  const bool coarsest = m + 1 == levels_.size();
  if (options_.cycle != Cycle::k || coarsest) {
    const int visits = options_.cycle != Cycle::v && !coarsest ? 2 : 1;
    for (int k = 0; k < visits; ++k) cycle_level(m, b, x);
    return;
  }
  const std::size_t n = op_.n_local(m), owned = op_.graph(m).n_owned;
  auto& kv = krylov_[m];
  std::vector<double>&v1 = kv[0], &w1 = kv[1], &v2 = kv[2], &w2 = kv[3], &r1 = kv[4];
  const auto each = [&](auto&& f) {
    for (std::size_t c = 0; c < 5; ++c) {
      for (std::size_t i = 0; i < owned; ++i) f(c * n + i);
    }
  };
  std::fill(v1.begin(), v1.end(), 0.0);
  cycle_level(m, b, v1);
  op_.multiply(m, v1, w1);
  const double rho1 = dot(m, w1, w1);
  if (!(rho1 > 0.0)) return;
  const double a1 = dot(m, w1, b) / rho1;
  each([&](std::size_t k) {
    x[k] = a1 * v1[k];
    r1[k] = b[k] - a1 * w1[k];
  });
  if (dot(m, r1, r1) <= 0.0625 * dot(m, b, b)) return;
  std::fill(v2.begin(), v2.end(), 0.0);
  cycle_level(m, r1, v2);
  op_.multiply(m, v2, w2);
  const double beta = dot(m, w2, w1) / rho1;
  each([&](std::size_t k) {
    v2[k] -= beta * v1[k];
    w2[k] -= beta * w1[k];
  });
  const double rho2 = dot(m, w2, w2);
  if (!(rho2 > 0.0)) return;
  const double a2 = dot(m, w2, r1) / rho2;
  each([&](std::size_t k) { x[k] += a2 * v2[k]; });
}

KrylovReport DistributedMultigrid::solve(std::span<const double> b, std::span<double> x,
                                         int max_cycles, double tolerance) {
  const std::size_t n = op_.n_local(0), owned = op_.graph(0).n_owned;
  residual_.resize(5 * n);
  std::vector<double> scratch(5 * n);
  std::copy(x.begin(), x.end(), scratch.begin());
  residual(0, b, scratch, residual_);
  Gcr::Reductions reductions{
      .dot = [&](std::span<const double> u, std::span<const double> v) { return dot(0, u, v); },
      .max_abs =
          [&](std::span<const double> u) {
            double m = 0.0;
            for (std::size_t c = 0; c < 5; ++c) {
              for (std::size_t i = 0; i < owned; ++i) {
                const double a = std::abs(u[c * n + i]);
                m = std::isnan(a) ? std::numeric_limits<double>::infinity() : std::max(m, a);
              }
            }
            return collective_.max(m);
          },
  };
  return gcr_.solve(
      [&](std::span<const double> v, std::span<double> y) {
        std::copy(v.begin(), v.end(), scratch.begin());
        op_.multiply(0, scratch, y);
      },
      [&](std::span<const double> r, std::span<double> z) { cycle_level(0, r, z); }, residual_, x,
      max_cycles, tolerance, reductions);
}

}  // namespace autan::solvers
