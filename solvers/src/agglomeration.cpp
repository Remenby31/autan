#include "autan/solvers/agglomeration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

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
  std::vector<Normal> normal;                  // sum of the boundary faces' area vectors
  std::vector<Normal> centroid;
  std::vector<double> volume;
  // Level 0: weakest and strongest coupling of each cell through its faces, boundary faces
  // included (A_f over twice the distance to the face: the distance to the mirror cell).
  std::vector<double> weakest, strongest;
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
bool compatible(const Graph& g, std::size_t a, std::size_t b) {
  if (g.zone[a] != g.zone[b]) return false;
  if (!std::includes(g.boundary[a].begin(), g.boundary[a].end(), g.boundary[b].begin(),
                     g.boundary[b].end())) {
    return false;
  }
  return g.boundary[a].empty() || g.boundary[b].empty() || dot(g.normal[a], g.normal[b]) > 0.0;
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
  g.normal.assign(g.n, Normal{0.0, 0.0, 0.0});
  const auto nf = static_cast<std::size_t>(mesh.n_faces());
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(mesh.owner()[f]);
    auto& zones = g.boundary[c];
    const index_t z = mesh.face_zone()[f];
    if (std::find(zones.begin(), zones.end(), z) == zones.end()) zones.push_back(z);
    g.normal[c][0] += gm.face_area_x[f];
    g.normal[c][1] += gm.face_area_y[f];
    g.normal[c][2] += gm.face_area_z[f];
  }
  for (auto& zones : g.boundary) std::sort(zones.begin(), zones.end());
  g.weakest.assign(g.n, std::numeric_limits<double>::infinity());
  g.strongest.assign(g.n, 0.0);
  auto couple = [&](std::size_t c, double w) {
    g.weakest[c] = std::min(g.weakest[c], w);
    g.strongest[c] = std::max(g.strongest[c], w);
  };
  for (std::size_t f = 0; f < ni; ++f) {
    couple(static_cast<std::size_t>(g.owner[f]), g.weight[f]);
    couple(static_cast<std::size_t>(g.neighbour[f]), g.weight[f]);
  }
  for (std::size_t f = ni; f < nf; ++f) {
    const auto c = static_cast<std::size_t>(mesh.owner()[f]);
    const double dx = gm.face_centroid_x[f] - gm.cell_centroid_x[c];
    const double dy = gm.face_centroid_y[f] - gm.cell_centroid_y[c];
    const double dz = gm.face_centroid_z[f] - gm.cell_centroid_z[c];
    couple(c, gm.face_area[f] / (2.0 * std::sqrt(dx * dx + dy * dy + dz * dz)));
  }
  g.build_adjacency();
  return g;
}

// Lines of strong coupling (Mavriplis 1998, J. Comput. Phys. 145:141-165, sec. 4): seeds are
// the anisotropic cells, those on the boundary first, then by decreasing anisotropy; a line
// grows from its seed through the strongest face whose coupling is at least `ratio` times the
// weakest coupling of both cells, in both directions.
void make_lines(Graph& g, double ratio) {
  const std::vector<double>& weakest = g.weakest;
  const std::vector<double>& strongest = g.strongest;
  std::vector<std::size_t> seeds;
  for (std::size_t c = 0; c < g.n; ++c) {
    if (strongest[c] >= ratio * weakest[c]) seeds.push_back(c);
  }
  std::stable_sort(seeds.begin(), seeds.end(), [&](std::size_t a, std::size_t b) {
    const bool wa = !g.boundary[a].empty(), wb = !g.boundary[b].empty();
    if (wa != wb) return wa;
    return strongest[a] / weakest[a] > strongest[b] / weakest[b];
  });
  std::vector<int> in_line(g.n, 0);
  auto grow = [&](std::size_t start, std::vector<index_t>& out) {
    std::size_t cur = start;
    for (;;) {
      std::size_t best = g.n;
      double best_w = 0.0;
      g.for_edges(cur, [&](std::size_t e, std::size_t m) {
        if (in_line[m] != 0 || g.zone[m] != g.zone[cur]) return;
        const double w = g.weight[e];
        if (w < ratio * weakest[cur] || w < ratio * weakest[m]) return;
        if (w > best_w || (w == best_w && m < best)) {
          best_w = w;
          best = m;
        }
      });
      if (best == g.n) return;
      in_line[best] = 1;
      out.push_back(static_cast<index_t>(best));
      cur = best;
    }
  };
  std::vector<std::vector<index_t>> lines;
  for (const std::size_t s : seeds) {
    if (in_line[s] != 0) continue;
    in_line[s] = 1;
    std::vector<index_t> forward{static_cast<index_t>(s)}, backward;
    grow(s, forward);
    grow(s, backward);
    if (forward.size() + backward.size() < 2) {
      in_line[s] = 0;
      continue;
    }
    std::vector<index_t> line(backward.rbegin(), backward.rend());
    line.insert(line.end(), forward.begin(), forward.end());
    lines.push_back(std::move(line));
  }
  g.line_offsets.assign(1, 0);
  g.line_nodes.clear();
  for (const auto& line : lines) {
    g.line_nodes.insert(g.line_nodes.end(), line.begin(), line.end());
    g.line_offsets.push_back(static_cast<offset_t>(g.line_nodes.size()));
  }
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
  std::vector<std::vector<index_t>> members;      // coarse node -> fine nodes
  std::vector<std::vector<std::size_t>> clines;  // coarse lines

  // Lines with their neighbouring lines.
  std::vector<int> grouped(n_lines, 0);
  for (std::size_t s = 0; s < n_lines; ++s) {
    if (line_size(s) < 2 || grouped[s] != 0) continue;
    grouped[s] = 1;
    // Neighbouring lines with the number of layers at which they touch line s.
    std::vector<std::pair<std::size_t, std::size_t>> touching;  // (line, layers)
    for (std::size_t k = 0; k < line_size(s); ++k) {
      g.for_edges(line_node(s, k), [&](std::size_t, std::size_t m) {
        const std::size_t l = line_of[m];
        if (l == s || grouped[l] != 0 || line_size(l) < 2 || position[m] != k) return;
        auto it = std::find_if(touching.begin(), touching.end(),
                               [&](const auto& t) { return t.first == l; });
        if (it == touching.end()) {
          touching.emplace_back(l, 1);
        } else {
          ++it->second;
        }
      });
    }
    std::stable_sort(touching.begin(), touching.end(), [](const auto& a, const auto& b) {
      return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (touching.size() + 1 > opt.max_agglomerate) touching.resize(opt.max_agglomerate - 1);
    std::vector<std::size_t> group{s};
    for (const auto& t : touching) {
      grouped[t.first] = 1;
      group.push_back(t.first);
    }
    std::vector<std::size_t> coarse_line;
    for (std::size_t k = 0; k < line_size(s); ++k) {
      const std::size_t seed = line_node(s, k);
      const std::size_t c = members.size();
      members.push_back({static_cast<index_t>(seed)});
      agg[seed] = c;
      for (std::size_t i = 1; i < group.size(); ++i) {
        const std::size_t l = group[i];
        if (k >= line_size(l)) continue;
        const std::size_t m = line_node(l, k);
        bool adjacent = false;
        g.for_edges(seed, [&](std::size_t, std::size_t o) { adjacent = adjacent || o == m; });
        if (adjacent && compatible(g, seed, m)) {
          members[c].push_back(static_cast<index_t>(m));
          agg[m] = c;
        }
      }
      coarse_line.push_back(c);
    }
    clines.push_back(std::move(coarse_line));
    // What did not join line s keeps its own coarse nodes, in runs that remain lines.
    for (std::size_t i = 1; i < group.size(); ++i) {
      const std::size_t l = group[i];
      std::vector<std::size_t> run;
      for (std::size_t k = 0; k <= line_size(l); ++k) {
        if (k < line_size(l) && agg[line_node(l, k)] == none) {
          const std::size_t c = members.size();
          members.push_back({static_cast<index_t>(line_node(l, k))});
          agg[line_node(l, k)] = c;
          run.push_back(c);
        } else if (!run.empty()) {
          clines.push_back(std::move(run));
          run.clear();
        }
      }
    }
  }

  // Isotropic agglomerates of the nodes outside lines.
  auto is_point = [&](std::size_t c) { return line_size(line_of[c]) == 1; };
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
    members.push_back({static_cast<index_t>(seed)});
    agg[seed] = c;
    // Grow the agglomerate by the free neighbour with the most faces on it, the nearest to the
    // seed among equals: compact agglomerates (2 x 2 x 2 blocks of hexahedra).
    std::vector<std::pair<std::size_t, int>> inside;  // candidate, faces on the agglomerate
    auto admit = [&](std::size_t node) {
      g.for_edges(node, [&](std::size_t, std::size_t m) {
        if (agg[m] != none || !is_point(m) || !compatible(g, seed, m)) return;
        auto it = std::find_if(inside.begin(), inside.end(),
                               [&](const auto& x) { return x.first == m; });
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
    admit(seed);
    while (members[c].size() < opt.max_agglomerate && !inside.empty()) {
      std::size_t best = 0;
      for (std::size_t k = 1; k < inside.size(); ++k) {
        const auto& [m, faces] = inside[k];
        const auto& [mb, fb] = inside[best];
        const double d = distance(m), db = distance(mb);
        if (faces > fb || (faces == fb && (d < db || (d == db && m < mb)))) best = k;
      }
      const std::size_t m = inside[best].first;
      inside.erase(inside.begin() + static_cast<std::ptrdiff_t>(best));
      members[c].push_back(static_cast<index_t>(m));
      agg[m] = c;
      admit(m);
    }
    for (const index_t f : members[c]) {
      g.for_edges(static_cast<std::size_t>(f), [&](std::size_t, std::size_t m) {
        if (agg[m] == none && is_point(m)) {
          front.emplace(std::numeric_limits<std::size_t>::max() - g.boundary[m].size(),
                        arrival++, m);
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
    const auto best = *std::min_element(targets.begin(), targets.end(), [](const auto& a,
                                                                           const auto& b) {
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
  coarse.normal.assign(nc, Normal{0.0, 0.0, 0.0});
  coarse.centroid.assign(nc, Normal{0.0, 0.0, 0.0});
  coarse.volume.assign(nc, 0.0);
  level.coarse_node.assign(g.n, 0);
  for (std::size_t c = 0; c < members.size(); ++c) {
    if (members[c].empty()) continue;
    const std::size_t C = renumber[c];
    coarse.zone[C] = g.zone[static_cast<std::size_t>(members[c][0])];
    std::vector<index_t> zones;
    for (const index_t f : members[c]) {
      const auto fu = static_cast<std::size_t>(f);
      level.coarse_node[fu] = static_cast<index_t>(C);
      std::vector<index_t> merged;
      std::set_union(zones.begin(), zones.end(), g.boundary[fu].begin(), g.boundary[fu].end(),
                     std::back_inserter(merged));
      zones = std::move(merged);
      coarse.volume[C] += g.volume[fu];
      for (std::size_t d = 0; d < 3; ++d) {
        coarse.normal[C][d] += g.normal[fu][d];
        coarse.centroid[C][d] += g.volume[fu] * g.centroid[fu][d];
      }
    }
    for (std::size_t d = 0; d < 3; ++d) coarse.centroid[C][d] /= coarse.volume[C];
    coarse.boundary[C] = std::move(zones);
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

std::vector<Level> agglomerate(const Mesh& mesh, const Geometry& geometry,
                               const AgglomerationOptions& options) {
  if (mesh.n_owned_cells() != mesh.n_cells()) {
    throw std::invalid_argument("agglomeration of a mesh with ghost cells");
  }
  Graph g = fine_graph(mesh, geometry);
  make_lines(g, options.line_ratio);
  std::vector<Level> levels;
  levels.push_back(as_level(g));
  while (g.n > options.coarsest_nodes && static_cast<int>(levels.size()) < options.max_levels) {
    Level& fine = levels.back();
    Graph coarse = coarsen(g, options, fine);
    if (static_cast<double>(coarse.n) > options.min_coarsening * static_cast<double>(g.n)) {
      fine.coarse_node.clear();
      fine.coarse_edge.clear();
      fine.coarse_flipped.clear();
      break;
    }
    g = std::move(coarse);
    levels.push_back(as_level(g));
  }
  return levels;
}

}  // namespace autan::solvers
