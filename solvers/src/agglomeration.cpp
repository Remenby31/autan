#include "autan/solvers/agglomeration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <picosha2.h>

#include "autan/core/lines.hpp"
#include "autan/core/renumber.hpp"

namespace autan::solvers {

namespace {

using Normal = std::array<double, 3>;

// The coupling graph agglomeration works on: edge weights, and what a node's cells touch.
struct Graph {
  std::size_t n = 0;
  std::vector<index_t> owner, neighbour;
  std::vector<double> weight;
  std::vector<offset_t> node_offsets;  // node -> edges, CSR
  std::vector<index_t> node_edges;
  std::vector<index_t> zone;
  std::vector<std::vector<index_t>> boundary;  // sorted boundary zones touched
  // Per boundary zone touched (the order of `boundary`), the sum of its faces' area vectors and
  // of their magnitudes.
  std::vector<std::vector<Normal>> zone_normal;
  std::vector<std::vector<double>> zone_area;
  std::vector<Normal> centroid;
  std::vector<double> volume;
  std::vector<offset_t> line_offsets{0};
  std::vector<index_t> line_nodes;

  [[nodiscard]] index_t other(std::size_t e, std::size_t node) const {
    return static_cast<std::size_t>(owner[e]) == node ? neighbour[e] : owner[e];
  }

  void build_adjacency() {
    node_offsets.assign(n + 1, 0);
    for (std::size_t e = 0; e < owner.size(); ++e) {
      ++node_offsets[static_cast<std::size_t>(owner[e]) + 1];
      ++node_offsets[static_cast<std::size_t>(neighbour[e]) + 1];
    }
    for (std::size_t i = 0; i < n; ++i) node_offsets[i + 1] += node_offsets[i];
    node_edges.resize(static_cast<std::size_t>(node_offsets.back()));
    std::vector<offset_t> fill(node_offsets.begin(), node_offsets.end() - 1);
    for (std::size_t e = 0; e < owner.size(); ++e) {
      node_edges[static_cast<std::size_t>(fill[static_cast<std::size_t>(owner[e])]++)] =
          static_cast<index_t>(e);
      node_edges[static_cast<std::size_t>(fill[static_cast<std::size_t>(neighbour[e])]++)] =
          static_cast<index_t>(e);
    }
  }

  // Each neighbour of a node with the sum of the couplings of the edges joining them, in the
  // order the neighbours first appear.
  [[nodiscard]] std::vector<std::pair<std::size_t, double>> couplings(std::size_t node) const {
    std::vector<std::pair<std::size_t, double>> out;
    for_edges(node, [&](std::size_t e, std::size_t m) {
      auto it = std::find_if(out.begin(), out.end(), [&](const auto& p) { return p.first == m; });
      if (it == out.end()) {
        out.emplace_back(m, weight[e]);
      } else {
        it->second += weight[e];
      }
    });
    return out;
  }

  template <class F>
  void for_edges(std::size_t node, F&& f) const {
    for (auto k = node_offsets[node]; k < node_offsets[node + 1]; ++k) {
      const auto e = static_cast<std::size_t>(node_edges[static_cast<std::size_t>(k)]);
      f(e, static_cast<std::size_t>(other(e, node)));
    }
  }
};

double dot(const Normal& a, const Normal& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Whether node b may join an agglomerate seeded by node a: same cell zone, no boundary zone the
// seed does not touch (so that coarse boundary faces stay on the seed's patches), and boundary
// faces facing the same side (so that an agglomerate never wraps around a thin wall).
// Whether nodes a and b may share an agglomerate at all: same cell zone, and no shared boundary
// zone whose faces face opposite sides in them (a thin wall between them).
bool joinable(const Graph& g, std::size_t a, std::size_t b);

bool compatible(const Graph& g, std::size_t a, std::size_t b) {
  if (!std::includes(g.boundary[a].begin(), g.boundary[a].end(), g.boundary[b].begin(),
                     g.boundary[b].end())) {
    return false;
  }
  return joinable(g, a, b);
}

bool joinable(const Graph& g, std::size_t a, std::size_t b) {
  if (g.zone[a] != g.zone[b]) return false;
  // Per zone, and only where a zone's faces face one side in both nodes: the two side planes of
  // a one-cell-thick mesh, in one zone or two, cancel out and give no direction.
  auto directed = [&](std::size_t c, std::size_t j) {
    const Normal& n = g.zone_normal[c][j];
    return std::sqrt(dot(n, n)) > 0.5 * g.zone_area[c][j];
  };
  for (std::size_t j = 0; j < g.boundary[b].size(); ++j) {
    const auto it = std::lower_bound(g.boundary[a].begin(), g.boundary[a].end(), g.boundary[b][j]);
    if (it == g.boundary[a].end() || *it != g.boundary[b][j]) continue;
    const auto i = static_cast<std::size_t>(it - g.boundary[a].begin());
    if (directed(a, i) && directed(b, j) && dot(g.zone_normal[a][i], g.zone_normal[b][j]) <= 0.0) {
      return false;
    }
  }
  return true;
}

Graph fine_graph(const Mesh& mesh, const Geometry& geometry) {
  Graph g;
  g.n = static_cast<std::size_t>(mesh.n_cells());
  const auto ni = static_cast<std::size_t>(mesh.n_interior_faces());
  g.owner.assign(mesh.owner().begin(), mesh.owner().begin() + static_cast<std::ptrdiff_t>(ni));
  g.neighbour.assign(mesh.neighbour().begin(), mesh.neighbour().end());
  g.weight.resize(ni);
  const auto& gm = geometry;
  for (std::size_t f = 0; f < ni; ++f) {
    const auto o = static_cast<std::size_t>(g.owner[f]);
    const auto nb = static_cast<std::size_t>(g.neighbour[f]);
    const double dx = gm.cell_centroid_x[nb] - gm.cell_centroid_x[o];
    const double dy = gm.cell_centroid_y[nb] - gm.cell_centroid_y[o];
    const double dz = gm.cell_centroid_z[nb] - gm.cell_centroid_z[o];
    g.weight[f] = gm.face_area[f] / std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  g.zone.assign(mesh.cell_zone().begin(), mesh.cell_zone().end());
  g.volume = gm.cell_volume;
  g.centroid.resize(g.n);
  for (std::size_t c = 0; c < g.n; ++c) {
    g.centroid[c] = {gm.cell_centroid_x[c], gm.cell_centroid_y[c], gm.cell_centroid_z[c]};
  }
  g.boundary.assign(g.n, {});
  g.zone_normal.assign(g.n, {});
  g.zone_area.assign(g.n, {});
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  struct Sum {
    Normal normal{};
    double area = 0.0;
  };
  std::vector<std::vector<std::pair<index_t, Sum>>> per_zone(g.n);
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(mesh.owner()[f]);
    const index_t z = mesh.face_zone()[f];
    auto it = std::find_if(per_zone[c].begin(), per_zone[c].end(),
                           [&](const auto& p) { return p.first == z; });
    if (it == per_zone[c].end()) it = per_zone[c].insert(per_zone[c].end(), {z, Sum{}});
    it->second.normal[0] += gm.face_area_x[f];
    it->second.normal[1] += gm.face_area_y[f];
    it->second.normal[2] += gm.face_area_z[f];
    it->second.area += gm.face_area[f];
  }
  for (std::size_t c = 0; c < g.n; ++c) {
    std::sort(per_zone[c].begin(), per_zone[c].end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    for (const auto& [z, sum] : per_zone[c]) {
      g.boundary[c].push_back(z);
      g.zone_normal[c].push_back(sum.normal);
      g.zone_area[c].push_back(sum.area);
    }
  }
  g.build_adjacency();
  return g;
}

// Level 0's lines: the mesh's lines of strong coupling, then each remaining cell alone.
void fine_lines(Graph& g, const Mesh& mesh, const Geometry& geometry, double ratio) {
  const Lines lines = smoother_lines(mesh, geometry, ratio);
  g.line_offsets = lines.offsets;
  g.line_nodes = lines.cells;
  std::vector<char> in_line(g.n, 0);
  for (const index_t c : lines.cells) in_line[static_cast<std::size_t>(c)] = 1;
  for (std::size_t c = 0; c < g.n; ++c) {
    if (in_line[c] == 0) {
      g.line_nodes.push_back(static_cast<index_t>(c));
      g.line_offsets.push_back(static_cast<offset_t>(g.line_nodes.size()));
    }
  }
}

// One coarsening step. Lines are agglomerated with their neighbouring lines layer by layer, so
// that anisotropic layers coarsen in the wall plane and coarse lines stay lines; the other nodes
// form compact isotropic agglomerates grown from a seed, seeds taken on an advancing front from
// the boundary corners inwards (Mavriplis and Venkatakrishnan 1996, Comput. Fluids
// 25(4):361-378). Fills the level's maps to the coarse graph and returns it.
Graph coarsen(const Graph& g, const AgglomerationOptions& opt, Level& level) {
  const std::size_t n_lines = g.line_offsets.size() - 1;
  auto line_size = [&](std::size_t l) {
    return static_cast<std::size_t>(g.line_offsets[l + 1] - g.line_offsets[l]);
  };
  auto line_node = [&](std::size_t l, std::size_t k) {
    return static_cast<std::size_t>(g.line_nodes[static_cast<std::size_t>(g.line_offsets[l]) + k]);
  };
  std::vector<std::size_t> line_of(g.n), position(g.n);
  for (std::size_t l = 0; l < n_lines; ++l) {
    for (std::size_t k = 0; k < line_size(l); ++k) {
      line_of[line_node(l, k)] = l;
      position[line_node(l, k)] = k;
    }
  }
  constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> agg(g.n, none);
  std::vector<std::vector<index_t>> members;     // coarse node -> fine nodes
  std::vector<std::vector<std::size_t>> clines;  // coarse lines

  // Lines with their neighbouring lines, layer by layer: a group grows from a seed line by the
  // free line touching the most lines of the group, at the most layers, the nearest among equals
  // (a 2 x 2 block of wall lines). A neighbouring line may run the other way and start at another
  // layer (lines that do not start on a wall): its layers are matched to the seed's in the
  // orientation and shift that pair the most adjacent nodes.
  std::vector<int> grouped(n_lines, 0);
  struct Member {
    std::size_t line;
    bool reversed;
    std::ptrdiff_t shift;  // index along the line of the seed's layer k: k + shift
  };
  auto layer_node = [&](const Member& m, std::size_t k) {
    const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(k) + m.shift;
    const auto size = static_cast<std::ptrdiff_t>(line_size(m.line));
    if (j < 0 || j >= size) return none;
    return line_node(m.line, static_cast<std::size_t>(m.reversed ? size - 1 - j : j));
  };
  auto separation = [&](std::size_t a, std::size_t b) {
    double d = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
      d += (g.centroid[a][i] - g.centroid[b][i]) * (g.centroid[a][i] - g.centroid[b][i]);
    return d;
  };
  // Semi-coarsening along a run of consecutive line nodes, two per coarse node (Mavriplis and
  // Pirzadeh 1999, J. Aircraft 36(6):987-998): the coarse nodes form a coarse line.
  auto pair_along = [&](const std::vector<std::size_t>& run) {
    std::vector<std::size_t> coarse_line;
    bool open = false;  // the last coarse node holds one node
    for (std::size_t k = 0; k < run.size(); ++k) {
      if (open && compatible(g, run[k - 1], run[k])) {
        members[coarse_line.back()].push_back(static_cast<index_t>(run[k]));
        agg[run[k]] = coarse_line.back();
        open = false;
        continue;
      }
      agg[run[k]] = members.size();
      coarse_line.push_back(members.size());
      members.push_back({static_cast<index_t>(run[k])});
      open = true;
    }
    clines.push_back(std::move(coarse_line));
  };
  for (std::size_t s = 0; s < n_lines; ++s) {
    if (line_size(s) < opt.min_line || grouped[s] != 0) continue;
    grouped[s] = 1;
    std::vector<Member> group{{s, false, 0}};
    while (group.size() < opt.max_lines) {
      using Key = std::pair<int, std::ptrdiff_t>;  // (reversed, shift)
      struct Candidate {
        std::size_t line, touched = 0;
        std::vector<Key> votes;  // one entry per vote, counted once all are cast
        std::pair<Key, std::size_t> best{};
      };
      std::vector<Candidate> candidates;
      for (const Member& member : group) {
        std::vector<std::size_t> seen;  // candidate lines this member touches
        for (std::size_t k = 0; k < line_size(s); ++k) {
          const std::size_t a = layer_node(member, k);
          if (a == none) continue;
          // One vote per neighbouring cell, however many faces they share.
          for (const auto& [m, w] : g.couplings(a)) {
            const std::size_t l = line_of[m];
            if (grouped[l] != 0 || line_size(l) < opt.min_line) continue;
            // Only a layer the seed line's node there would accept counts (else the line would
            // join the group at no layer and keep the seed line from coarsening along itself).
            if (!compatible(g, line_node(s, k), m)) continue;
            auto it = std::find_if(candidates.begin(), candidates.end(),
                                   [&](const Candidate& c) { return c.line == l; });
            if (it == candidates.end())
              it = candidates.insert(candidates.end(), Candidate{l, 0, {}});
            const auto p = static_cast<std::ptrdiff_t>(position[m]);
            const auto last = static_cast<std::ptrdiff_t>(line_size(l)) - 1;
            const auto kk = static_cast<std::ptrdiff_t>(k);
            it->votes.push_back({0, p - kk});
            it->votes.push_back({1, last - p - kk});
            if (std::find(seen.begin(), seen.end(), l) == seen.end()) {
              seen.push_back(l);
              ++it->touched;
            }
          }
        }
      }
      if (candidates.empty()) break;
      // The orientation and shift of a candidate: the most layers, then forward, then the
      // smallest shift.
      auto better = [](const std::pair<Key, std::size_t>& x, const std::pair<Key, std::size_t>& y) {
        if (x.second != y.second) return x.second > y.second;
        if (x.first.first != y.first.first) return x.first.first < y.first.first;
        const auto ax = x.first.second < 0 ? -x.first.second : x.first.second;
        const auto ay = y.first.second < 0 ? -y.first.second : y.first.second;
        return ax != ay ? ax < ay : x.first.second < y.first.second;
      };
      for (Candidate& c : candidates) {
        std::sort(c.votes.begin(), c.votes.end());
        for (std::size_t i = 0; i < c.votes.size();) {
          std::size_t j = i;
          while (j < c.votes.size() && c.votes[j] == c.votes[i]) ++j;
          const std::pair<Key, std::size_t> run{c.votes[i], j - i};
          if (i == 0 || better(run, c.best)) c.best = run;
          i = j;
        }
        c.votes = std::vector<Key>();
      }
      auto hypothesis = [](const Candidate& c) { return c.best; };
      const std::size_t origin = line_node(s, 0);
      const auto best = std::min_element(
          candidates.begin(), candidates.end(), [&](const Candidate& x, const Candidate& y) {
            if (x.touched != y.touched) return x.touched > y.touched;
            const std::size_t lx = hypothesis(x).second, ly = hypothesis(y).second;
            if (lx != ly) return lx > ly;
            const double dx = separation(line_node(x.line, 0), origin);
            const double dy = separation(line_node(y.line, 0), origin);
            return dx != dy ? dx < dy : x.line < y.line;
          });
      grouped[best->line] = 1;
      const auto h = hypothesis(*best);
      group.push_back({best->line, h.first.first == 1, h.first.second});
    }
    if (group.size() == 1) {
      // No free neighbouring line: the line coarsens along itself, so that coarse levels keep
      // shrinking once the wall plane is exhausted.
      std::vector<std::size_t> run;
      for (std::size_t k = 0; k < line_size(s); ++k) run.push_back(line_node(s, k));
      pair_along(run);
      continue;
    }
    // Each of the seed's layers with the member nodes adjacent to it; a layer no member joined
    // pairs with the next such layer along the seed line, so that a group touching the seed at a
    // few layers still coarsens it.
    constexpr std::size_t pending = none - 1;
    std::vector<std::vector<std::size_t>> layers(line_size(s));
    std::vector<std::size_t> joined(group.size(), 0);
    for (std::size_t k = 0; k < line_size(s); ++k) {
      const std::size_t seed = line_node(s, k);
      layers[k].push_back(seed);
      agg[seed] = pending;
      for (std::size_t i = 1; i < group.size(); ++i) {
        const std::size_t m = layer_node(group[i], k);
        if (m == none || agg[m] != none) continue;
        bool adjacent = false;
        g.for_edges(m, [&](std::size_t, std::size_t o) {
          adjacent =
              adjacent || std::find(layers[k].begin(), layers[k].end(), o) != layers[k].end();
        });
        if (adjacent && compatible(g, seed, m)) {
          layers[k].push_back(m);
          agg[m] = pending;
          ++joined[i];
        }
      }
    }
    std::vector<std::size_t> coarse_line;
    bool open = false;  // the last coarse node holds one unjoined seed layer
    for (std::size_t k = 0; k < line_size(s); ++k) {
      const std::size_t seed = layers[k][0];
      if (layers[k].size() == 1 && open && compatible(g, layers[k - 1][0], seed)) {
        members[coarse_line.back()].push_back(static_cast<index_t>(seed));
        agg[seed] = coarse_line.back();
        open = false;
        continue;
      }
      const std::size_t c = members.size();
      members.emplace_back();
      for (const std::size_t m : layers[k]) {
        members[c].push_back(static_cast<index_t>(m));
        agg[m] = c;
      }
      coarse_line.push_back(c);
      open = layers[k].size() == 1;
    }
    clines.push_back(std::move(coarse_line));
    // A line that joined at some layer coarsens the rest of its nodes along itself, in runs that
    // remain lines; a line that joined nowhere is free again for another group.
    for (std::size_t i = 1; i < group.size(); ++i) {
      const std::size_t l = group[i].line;
      if (joined[i] == 0) {
        grouped[l] = 0;
        continue;
      }
      std::vector<std::size_t> run;
      for (std::size_t k = 0; k <= line_size(l); ++k) {
        if (k < line_size(l) && agg[line_node(l, k)] == none) {
          run.push_back(line_node(l, k));
        } else if (!run.empty()) {
          pair_along(run);
          run.clear();
        }
      }
    }
  }

  // Isotropic agglomerates of the nodes outside lines.
  auto is_point = [&](std::size_t c) { return line_size(line_of[c]) < opt.min_line; };
  std::vector<std::size_t> order;
  for (std::size_t c = 0; c < g.n; ++c) {
    if (is_point(c)) order.push_back(c);
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return g.boundary[a].size() > g.boundary[b].size();
  });
  std::set<std::tuple<std::size_t, std::size_t, std::size_t>> front;  // (-zones, arrival, node)
  std::size_t arrival = 0, next = 0;
  const std::size_t first_point = members.size();
  for (;;) {
    std::size_t seed = none;
    while (!front.empty() && seed == none) {
      const std::size_t c = std::get<2>(*front.begin());
      front.erase(front.begin());
      if (agg[c] == none) seed = c;
    }
    while (seed == none && next < order.size()) {
      if (agg[order[next]] == none) seed = order[next];
      ++next;
    }
    if (seed == none) break;
    const std::size_t c = members.size();
    members.emplace_back();
    // Grow the agglomerate by the free neighbour with the most faces on it, the nearest to the
    // seed among equals: compact agglomerates (2 x 2 x 2 blocks of hexahedra). A node brings the
    // other nodes of its short line, so that a strong coupling never joins two agglomerates.
    std::vector<std::pair<std::size_t, int>> inside;  // candidate, faces on the agglomerate
    // A node brings its whole short line: the node must suit the seed, and every node of its line
    // be joinable (same cell zone, no thin wall between); the seed's patches are not imposed on
    // the rest of the line, a quality rule that would split strongly coupled pairs.
    auto line_compatible = [&](std::size_t a, std::size_t m) {
      if (!compatible(g, a, m)) return false;
      const std::size_t l = line_of[m];
      for (std::size_t k = 0; k < line_size(l); ++k) {
        if (!joinable(g, a, line_node(l, k))) return false;
      }
      return true;
    };
    auto admit = [&](std::size_t node) {
      g.for_edges(node, [&](std::size_t, std::size_t m) {
        if (agg[m] != none || !is_point(m) || !line_compatible(seed, m)) return;
        auto it =
            std::find_if(inside.begin(), inside.end(), [&](const auto& x) { return x.first == m; });
        if (it == inside.end()) {
          inside.emplace_back(m, 1);
        } else {
          ++it->second;
        }
      });
    };
    auto distance = [&](std::size_t m) {
      double d = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        d += (g.centroid[m][k] - g.centroid[seed][k]) * (g.centroid[m][k] - g.centroid[seed][k]);
      }
      return d;
    };
    auto take = [&](std::size_t node) {
      const std::size_t l = line_of[node];
      for (std::size_t k = 0; k < line_size(l); ++k) {
        const std::size_t q = line_node(l, k);
        if (agg[q] != none) continue;
        members[c].push_back(static_cast<index_t>(q));
        agg[q] = c;
      }
      for (std::size_t k = 0; k < line_size(l); ++k) admit(line_node(l, k));
      std::erase_if(inside, [&](const auto& x) { return agg[x.first] != none; });
    };
    take(seed);
    while (members[c].size() < opt.max_agglomerate && !inside.empty()) {
      std::size_t best = 0;
      for (std::size_t k = 1; k < inside.size(); ++k) {
        const auto& [m, faces] = inside[k];
        const auto& [mb, fb] = inside[best];
        const double d = distance(m), db = distance(mb);
        if (faces > fb || (faces == fb && (d < db || (d == db && m < mb)))) best = k;
      }
      take(inside[best].first);
    }
    for (const index_t f : members[c]) {
      g.for_edges(static_cast<std::size_t>(f), [&](std::size_t, std::size_t m) {
        if (agg[m] == none && is_point(m)) {
          front.emplace(std::numeric_limits<std::size_t>::max() - g.boundary[m].size(), arrival++,
                        m);
        }
      });
    }
  }
  // Quality: a node left alone joins the neighbouring isotropic agglomerate it is most strongly
  // coupled to, when that agglomerate's seed accepts it.
  for (std::size_t c = first_point; c < members.size(); ++c) {
    if (members[c].size() != 1) continue;
    const auto node = static_cast<std::size_t>(members[c][0]);
    std::vector<std::pair<double, std::size_t>> targets;  // (coupling, coarse node)
    g.for_edges(node, [&](std::size_t e, std::size_t m) {
      const std::size_t t = agg[m];
      if (t < first_point || t == c || members[t].size() < 2 ||
          !compatible(g, static_cast<std::size_t>(members[t][0]), node)) {
        return;
      }
      auto it = std::find_if(targets.begin(), targets.end(),
                             [&](const auto& x) { return x.second == t; });
      if (it == targets.end()) {
        targets.emplace_back(g.weight[e], t);
      } else {
        it->first += g.weight[e];
      }
    });
    if (targets.empty()) continue;
    const auto best =
        *std::min_element(targets.begin(), targets.end(), [](const auto& a, const auto& b) {
          return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
    members[best.second].push_back(static_cast<index_t>(node));
    agg[node] = best.second;
    members[c].clear();
  }
  // Number the coarse nodes: line agglomerates as created, then the isotropic ones.
  std::vector<std::size_t> renumber(members.size(), none);
  std::size_t nc = 0;
  for (std::size_t c = 0; c < members.size(); ++c) {
    if (!members[c].empty()) renumber[c] = nc++;
  }
  for (std::size_t c = first_point; c < members.size(); ++c) {
    if (!members[c].empty()) clines.push_back({c});
  }

  Graph coarse;
  coarse.n = nc;
  coarse.zone.assign(nc, 0);
  coarse.boundary.assign(nc, {});
  coarse.zone_normal.assign(nc, {});
  coarse.zone_area.assign(nc, {});
  coarse.centroid.assign(nc, Normal{0.0, 0.0, 0.0});
  coarse.volume.assign(nc, 0.0);
  level.coarse_node.assign(g.n, 0);
  for (std::size_t c = 0; c < members.size(); ++c) {
    if (members[c].empty()) continue;
    const std::size_t C = renumber[c];
    coarse.zone[C] = g.zone[static_cast<std::size_t>(members[c][0])];
    std::vector<std::pair<index_t, std::pair<Normal, double>>> zones;  // sorted by zone
    for (const index_t f : members[c]) {
      const auto fu = static_cast<std::size_t>(f);
      level.coarse_node[fu] = static_cast<index_t>(C);
      for (std::size_t j = 0; j < g.boundary[fu].size(); ++j) {
        const index_t z = g.boundary[fu][j];
        auto it = std::lower_bound(zones.begin(), zones.end(), z,
                                   [](const auto& p, index_t v) { return p.first < v; });
        if (it == zones.end() || it->first != z) it = zones.insert(it, {z, {Normal{}, 0.0}});
        for (std::size_t d = 0; d < 3; ++d) it->second.first[d] += g.zone_normal[fu][j][d];
        it->second.second += g.zone_area[fu][j];
      }
      coarse.volume[C] += g.volume[fu];
      for (std::size_t d = 0; d < 3; ++d) {
        coarse.centroid[C][d] += g.volume[fu] * g.centroid[fu][d];
      }
    }
    for (std::size_t d = 0; d < 3; ++d) coarse.centroid[C][d] /= coarse.volume[C];
    for (const auto& [z, sum] : zones) {
      coarse.boundary[C].push_back(z);
      coarse.zone_normal[C].push_back(sum.first);
      coarse.zone_area[C].push_back(sum.second);
    }
  }
  for (const auto& line : clines) {
    for (const std::size_t c : line) coarse.line_nodes.push_back(static_cast<index_t>(renumber[c]));
    coarse.line_offsets.push_back(static_cast<offset_t>(coarse.line_nodes.size()));
  }
  // Coarse edges: one per pair of adjacent coarse nodes, in (owner, neighbour) order.
  std::vector<std::tuple<index_t, index_t, std::size_t>> pairs;
  level.coarse_edge.assign(g.owner.size(), -1);
  level.coarse_flipped.assign(g.owner.size(), 0);
  for (std::size_t e = 0; e < g.owner.size(); ++e) {
    const index_t a = level.coarse_node[static_cast<std::size_t>(g.owner[e])];
    const index_t b = level.coarse_node[static_cast<std::size_t>(g.neighbour[e])];
    if (a == b) continue;
    pairs.emplace_back(std::min(a, b), std::max(a, b), e);
    level.coarse_flipped[e] = a > b ? 1 : 0;
  }
  std::sort(pairs.begin(), pairs.end());
  for (const auto& [a, b, e] : pairs) {
    if (coarse.owner.empty() || coarse.owner.back() != a || coarse.neighbour.back() != b) {
      coarse.owner.push_back(a);
      coarse.neighbour.push_back(b);
      coarse.weight.push_back(0.0);
    }
    coarse.weight.back() += g.weight[e];
    level.coarse_edge[e] = static_cast<index_t>(coarse.owner.size() - 1);
  }
  coarse.build_adjacency();
  return coarse;
}

Level as_level(const Graph& g) {
  Level l;
  l.n_nodes = g.n;
  l.owner = g.owner;
  l.neighbour = g.neighbour;
  l.line_offsets = g.line_offsets;
  l.line_nodes = g.line_nodes;
  return l;
}

}  // namespace

std::vector<std::vector<std::size_t>> line_colours(const Level& l) {
  std::vector<std::size_t> line_of(l.n_nodes);
  for (std::size_t li = 0; li < l.n_lines(); ++li) {
    for (auto k = l.line_offsets[li]; k < l.line_offsets[li + 1]; ++k) {
      line_of[static_cast<std::size_t>(l.line_nodes[static_cast<std::size_t>(k)])] = li;
    }
  }
  // Greedy colouring of the lines in line order: lines of a colour share no edge.
  std::vector<std::vector<std::size_t>> adjacent(l.n_lines());
  for (std::size_t e = 0; e < l.owner.size(); ++e) {
    const std::size_t a = line_of[static_cast<std::size_t>(l.owner[e])];
    const std::size_t b = line_of[static_cast<std::size_t>(l.neighbour[e])];
    if (a == b) continue;
    adjacent[a].push_back(b);
    adjacent[b].push_back(a);
  }
  std::vector<int> colour(l.n_lines(), -1);
  std::vector<int> seen;
  int n_colours = 0;
  for (std::size_t li = 0; li < l.n_lines(); ++li) {
    seen.assign(static_cast<std::size_t>(n_colours) + 1, 0);
    for (const std::size_t o : adjacent[li]) {
      if (colour[o] >= 0) seen[static_cast<std::size_t>(colour[o])] = 1;
    }
    int k = 0;
    while (seen[static_cast<std::size_t>(k)] != 0) ++k;
    colour[li] = k;
    n_colours = std::max(n_colours, k + 1);
  }
  std::vector<std::vector<std::size_t>> colours(static_cast<std::size_t>(n_colours));
  for (std::size_t li = 0; li < l.n_lines(); ++li) {
    colours[static_cast<std::size_t>(colour[li])].push_back(li);
  }
  return colours;
}

namespace {

std::vector<Level> build(const Mesh& mesh, const Geometry& geometry,
                         const AgglomerationOptions& options) {
  Graph g = fine_graph(mesh, geometry);
  fine_lines(g, mesh, geometry, options.line_ratio);
  std::vector<Level> levels;
  levels.push_back(as_level(g));
  while (g.n > options.coarsest_nodes && static_cast<int>(levels.size()) < options.max_levels) {
    Level& fine = levels.back();
    Graph coarse = coarsen(g, options, fine);
    if (static_cast<double>(coarse.n) > options.min_coarsening * static_cast<double>(g.n) ||
        static_cast<double>(coarse.owner.size()) >
            options.min_coarsening * static_cast<double>(g.owner.size())) {
      fine.coarse_node = std::vector<index_t>();
      fine.coarse_edge = std::vector<index_t>();
      fine.coarse_flipped = std::vector<std::uint8_t>();
      break;
    }
    g = std::move(coarse);
    levels.push_back(as_level(g));
  }
  return levels;
}

// The canonical cell order: by centroid, lexicographic on its exact FP64 values, then by global id
// for cells whose centroids coincide. A permutation of the cells of one mesh, with its geometry
// carried rather than recomputed, leaves this order unchanged (#365).
std::vector<index_t> canonical_order(const Mesh& mesh, const Geometry& geometry) {
  const auto n = static_cast<std::size_t>(mesh.n_cells());
  const auto ids = mesh.cell_global_id();
  auto id = [&](std::size_t c) { return ids.empty() ? static_cast<std::int64_t>(c) : ids[c]; };
  std::vector<index_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
    const auto ua = static_cast<std::size_t>(a), ub = static_cast<std::size_t>(b);
    const std::array<double, 3> xa{geometry.cell_centroid_x[ua], geometry.cell_centroid_y[ua],
                                   geometry.cell_centroid_z[ua]};
    const std::array<double, 3> xb{geometry.cell_centroid_x[ub], geometry.cell_centroid_y[ub],
                                   geometry.cell_centroid_z[ub]};
    return xa != xb ? xa < xb : id(ua) < id(ub);
  });
  return order;
}

void put(picosha2::hash256_one_by_one& h, std::uint64_t v) {
  std::array<unsigned char, 8> b{};
  for (std::size_t k = 0; k < 8; ++k) b[k] = static_cast<unsigned char>(v >> (8 * k));
  h.process(b.begin(), b.end());
}

// A vector as its length (u64) then its values: indices and offsets as i64, flags as u8.
template <class T>
void put(picosha2::hash256_one_by_one& h, const std::vector<T>& values) {
  put(h, static_cast<std::uint64_t>(values.size()));
  if constexpr (std::is_same_v<T, std::uint8_t>) {
    h.process(values.begin(), values.end());
  } else {
    for (const T v : values) put(h, static_cast<std::uint64_t>(static_cast<std::int64_t>(v)));
  }
}

}  // namespace

std::vector<Level> agglomerate(const Mesh& mesh, const Geometry& geometry,
                               const AgglomerationOptions& options) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw std::invalid_argument("agglomeration of a mesh with ghost cells");
  }
  const std::vector<index_t> new_to_old = canonical_order(mesh, geometry);
  bool identity = true;
  for (std::size_t i = 0; i < new_to_old.size() && identity; ++i) {
    identity = static_cast<std::size_t>(new_to_old[i]) == i;
  }
  if (identity) return build(mesh, geometry, options);

  AssemblyMap map;
  std::vector<Level> levels;
  {
    const Mesh canonical = renumber_cells(mesh, new_to_old, &map);
    levels = build(canonical, renumber_geometry(geometry, new_to_old, map), options);
  }
  // Level 0 back in the caller's cells and faces. A face the renumbering reversed has its owner
  // on the other side, so a retained edge's flip changes with it.
  Level& fine = levels.front();
  const auto n = new_to_old.size();
  const std::size_t ni = fine.owner.size();
  fine.owner.assign(mesh.owner().begin(), mesh.owner().begin() + static_cast<std::ptrdiff_t>(ni));
  fine.neighbour.assign(mesh.neighbour().begin(), mesh.neighbour().end());
  for (index_t& c : fine.line_nodes) c = new_to_old[static_cast<std::size_t>(c)];
  if (!fine.coarse_node.empty()) {
    std::vector<index_t> coarse_node(n), coarse_edge(ni);
    std::vector<std::uint8_t> coarse_flipped(ni);
    for (std::size_t i = 0; i < n; ++i) {
      coarse_node[static_cast<std::size_t>(new_to_old[i])] = fine.coarse_node[i];
    }
    for (std::size_t k = 0; k < ni; ++k) {
      const auto f = static_cast<std::size_t>(map.face_source[k]);
      coarse_edge[f] = fine.coarse_edge[k];
      coarse_flipped[f] =
          fine.coarse_edge[k] < 0
              ? std::uint8_t{0}
              : static_cast<std::uint8_t>(fine.coarse_flipped[k] ^ map.face_reversed[k]);
    }
    fine.coarse_node = std::move(coarse_node);
    fine.coarse_edge = std::move(coarse_edge);
    fine.coarse_flipped = std::move(coarse_flipped);
  }
  return levels;
}

std::string hierarchy_sha256(const std::vector<Level>& levels, const Mesh& mesh,
                             const Geometry& geometry) {
  const std::vector<index_t> new_to_old = canonical_order(mesh, geometry);
  std::vector<index_t> old_to_new(new_to_old.size());
  for (std::size_t i = 0; i < new_to_old.size(); ++i) {
    old_to_new[static_cast<std::size_t>(new_to_old[i])] = static_cast<index_t>(i);
  }
  picosha2::hash256_one_by_one h;
  put(h, std::uint64_t{1});  // format version
  put(h, static_cast<std::uint64_t>(levels.size()));
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const Level& level = levels[l];
    put(h, static_cast<std::uint64_t>(level.n_nodes));
    if (l > 0) {
      put(h, level.owner);
      put(h, level.neighbour);
      put(h, level.line_offsets);
      put(h, level.line_nodes);
      put(h, level.coarse_node);
      put(h, level.coarse_edge);
      put(h, level.coarse_flipped);
      continue;
    }
    // Level 0 in canonical cells; its edges, oriented from the lower canonical cell, in
    // increasing (owner, neighbour, coarse edge, flip): the canonical face order up to faces
    // that no field tells apart.
    const bool coarse = !level.coarse_node.empty();
    std::vector<std::array<std::int64_t, 4>> edges(level.owner.size());
    for (std::size_t e = 0; e < edges.size(); ++e) {
      const index_t a = old_to_new[static_cast<std::size_t>(level.owner[e])];
      const index_t b = old_to_new[static_cast<std::size_t>(level.neighbour[e])];
      const std::int64_t ce = coarse ? level.coarse_edge[e] : 0;
      const int flip = coarse && ce >= 0 ? (level.coarse_flipped[e] ^ (a > b ? 1 : 0)) : 0;
      edges[e] = {std::min(a, b), std::max(a, b), ce, flip};
    }
    std::sort(edges.begin(), edges.end());
    std::vector<index_t> owner(edges.size()), neighbour(edges.size());
    std::vector<std::int64_t> coarse_edge;
    std::vector<std::uint8_t> coarse_flipped;
    for (std::size_t e = 0; e < edges.size(); ++e) {
      owner[e] = static_cast<index_t>(edges[e][0]);
      neighbour[e] = static_cast<index_t>(edges[e][1]);
      if (coarse) {
        coarse_edge.push_back(edges[e][2]);
        coarse_flipped.push_back(static_cast<std::uint8_t>(edges[e][3]));
      }
    }
    std::vector<index_t> line_nodes(level.line_nodes.size()), coarse_node;
    for (std::size_t k = 0; k < line_nodes.size(); ++k) {
      line_nodes[k] = old_to_new[static_cast<std::size_t>(level.line_nodes[k])];
    }
    if (coarse) {
      coarse_node.resize(level.n_nodes);
      for (std::size_t i = 0; i < level.n_nodes; ++i) {
        coarse_node[i] = level.coarse_node[static_cast<std::size_t>(new_to_old[i])];
      }
    }
    put(h, owner);
    put(h, neighbour);
    put(h, level.line_offsets);
    put(h, line_nodes);
    put(h, coarse_node);
    put(h, coarse_edge);
    put(h, coarse_flipped);
  }
  h.finish();
  return picosha2::get_hash_hex_string(h);
}

}  // namespace autan::solvers
