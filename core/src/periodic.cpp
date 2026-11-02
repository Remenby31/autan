#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/mesh.hpp"

namespace autan {
namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

template <typename... Parts>
std::string concat(const Parts&... parts) {
  std::string out;
  ((out += parts), ...);
  return out;
}

[[noreturn]] void invalid(const std::string& message) {
  throw Error(ErrorCode::mesh_periodic_invalid, message);
}

[[noreturn]] void unmatched(const std::string& message) {
  throw Error(ErrorCode::mesh_periodic_unmatched, message);
}

void check_declaration(const Mesh& mesh, std::span<const PeriodicPair> pairs) {
  const auto zones = mesh.face_zones();
  const bool partitioned = mesh.n_owned_cells() != mesh.n_cells();
  if (partitioned && !pairs.empty()) invalid("a partition carries no periodic pairs");
  std::vector<int> uses(zones.size(), 0);
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const auto& pair = pairs[p];
    const std::string which = "periodic pair " + std::to_string(p);
    for (const index_t z : {pair.zone_a, pair.zone_b}) {
      if (z < 0 || at(z) >= zones.size()) invalid(which + " names no face zone");
      const auto off = mesh.boundary_zone_offsets();
      if (off[at(z) + 1] == off[at(z)])
        invalid(which + ": zone '" + zones[at(z)].name + "' has no face");
      if (!zones[at(z)].boundary || zones[at(z)].kind != BcKind::periodic) {
        invalid(which + ": zone '" + zones[at(z)].name +
                "' is not a boundary zone of kind periodic");
      }
      if (++uses[at(z)] > 1)
        invalid("zone '" + zones[at(z)].name + "' is in several periodic pairs");
    }
    if (pair.zone_a == pair.zone_b)
      invalid(which + " pairs zone '" + zones[at(pair.zone_a)].name + "' with itself");
    const auto& t = pair.translation;
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2]) ||
        (t[0] == 0.0 && t[1] == 0.0 && t[2] == 0.0)) {
      invalid(which + " has no finite, non-zero translation");
    }
  }
  for (std::size_t z = 0; z < zones.size(); ++z) {
    // A partition turns periodic faces into interior faces to image ghosts:
    // only a boundary zone needs its pair.
    if (uses[z] == 0 && zones[z].kind == BcKind::periodic && zones[z].boundary) {
      invalid("zone '" + zones[z].name + "' is periodic but in no pair");
    }
  }
}

// Shortest edge of the zone's faces at each of their nodes (infinity elsewhere).
std::vector<double> shortest_edges(const Mesh& mesh, index_t zone) {
  const auto x = mesh.x(), y = mesh.y(), z = mesh.z();
  std::vector<double> h(at(mesh.n_points()), std::numeric_limits<double>::infinity());
  const auto off = mesh.boundary_zone_offsets();
  for (index_t f = off[at(zone)]; f < off[at(zone) + 1]; ++f) {
    const auto nodes = mesh.face_nodes(f);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const auto a = at(nodes[i]), b = at(nodes[(i + 1) % nodes.size()]);
      const double length = std::hypot(x[a] - x[b], y[a] - y[b], z[a] - z[b]);
      h[a] = std::min(h[a], length);
      h[b] = std::min(h[b], length);
    }
  }
  return h;
}

struct PairMatch {
  std::vector<index_t> partners;
  std::vector<std::array<index_t, 2>> nodes;
  double max_relative = 0.0;
};

PairMatch match_pair(const Mesh& mesh, const PeriodicPair& pair, std::size_t p) {
  const auto off = mesh.boundary_zone_offsets();
  const auto zones = mesh.face_zones();
  const std::string name_a = zones[at(pair.zone_a)].name, name_b = zones[at(pair.zone_b)].name;
  const index_t a0 = off[at(pair.zone_a)], a1 = off[at(pair.zone_a) + 1];
  const index_t b0 = off[at(pair.zone_b)], b1 = off[at(pair.zone_b) + 1];
  if (a1 - a0 != b1 - b0) {
    unmatched("periodic pair " + std::to_string(p) + ": zone '" + name_a + "' has " +
              std::to_string(a1 - a0) + " faces, zone '" + name_b + "' " + std::to_string(b1 - b0));
  }
  const auto x = mesh.x(), y = mesh.y(), z = mesh.z();
  const auto& t = pair.translation;

  // Two nodes coincide when closer than 1e-4 of the shortest edge at the
  // node of zone a: far below any mesh spacing, far above coordinates
  // written in single precision. Partners have the same shortest edge (the
  // surfaces are translates), so zone b's nodes are binned by the octave of
  // theirs, each octave in a grid at its own scale: a query looks at the
  // octaves next to its own, whatever the grading.
  constexpr double relative_tolerance = 1e-4;
  constexpr double max_bins = 4.5e15;  // bin indices stay exact in double and int64
  const auto where = [&](const std::string& what) {
    return "periodic pair " + std::to_string(p) + ": " + what;
  };
  const std::vector<double> ha = shortest_edges(mesh, pair.zone_a);
  const std::vector<double> hb = shortest_edges(mesh, pair.zone_b);
  std::array<double, 3> origin{std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
  for (index_t f = b0; f < b1; ++f) {
    for (const index_t n : mesh.face_nodes(f)) {
      origin = {std::min(origin[0], x[at(n)]), std::min(origin[1], y[at(n)]),
                std::min(origin[2], z[at(n)])};
    }
  }
  using Key = std::array<std::int64_t, 4>;  // octave, then bin indices
  const auto bin_size = [](int octave) { return std::ldexp(relative_tolerance, octave + 1); };
  const auto bin = [&](double v, double origin_v, double size, std::int64_t& out) {
    const double b = std::floor((v - origin_v) / size);
    if (!(std::abs(b) <= max_bins)) return false;
    out = static_cast<std::int64_t>(b);
    return true;
  };
  const auto octave_of = [&](double h, index_t node, const std::string& zone) {
    if (!(h > 0.0) || !std::isfinite(h)) {
      unmatched(where(
          concat("node ", std::to_string(node), " of zone '", zone, "' has a degenerate edge")));
    }
    return std::ilogb(h);
  };
  std::map<Key, std::vector<index_t>> grid;
  std::vector<char> seen(at(mesh.n_points()), 0);
  for (index_t f = b0; f < b1; ++f) {
    for (const index_t n : mesh.face_nodes(f)) {
      if (seen[at(n)] != 0) continue;
      seen[at(n)] = 1;
      const int octave = octave_of(hb[at(n)], n, name_b);
      const double size = bin_size(octave);
      Key k{octave, 0, 0, 0};
      if (!bin(x[at(n)], origin[0], size, k[1]) || !bin(y[at(n)], origin[1], size, k[2]) ||
          !bin(z[at(n)], origin[2], size, k[3])) {
        unmatched(where("zone '" + name_b + "' is too large for its spacing"));
      }
      grid[k].push_back(n);
    }
  }

  PairMatch out;
  std::vector<index_t> image(at(mesh.n_points()), -1);
  for (index_t f = a0; f < a1; ++f) {
    for (const index_t n : mesh.face_nodes(f)) {
      if (image[at(n)] >= 0) continue;
      const std::array<double, 3> q{x[at(n)] + t[0], y[at(n)] + t[1], z[at(n)] + t[2]};
      const int own = octave_of(ha[at(n)], n, name_a);
      const double tolerance = relative_tolerance * ha[at(n)];
      index_t found = -1;
      int count = 0;
      for (int octave = own - 1; octave <= own + 1; ++octave) {
        const double size = bin_size(octave);
        if (!(size > 0.0) || !std::isfinite(size)) continue;
        // tolerance < 2 size for normal sizes (1 or 2 bins); subnormal
        // rounding may need a few more. Clamped before the conversion.
        const auto reach = static_cast<std::int64_t>(std::min(std::ceil(tolerance / size), 8.0));
        std::array<std::int64_t, 3> c{};
        bool inside = true;
        for (std::size_t d = 0; d < 3; ++d) inside = inside && bin(q[d], origin[d], size, c[d]);
        if (!inside) continue;
        for (std::int64_t i = -reach; i <= reach; ++i) {
          for (std::int64_t j = -reach; j <= reach; ++j) {
            for (std::int64_t l = -reach; l <= reach; ++l) {
              const auto it = grid.find(Key{octave, c[0] + i, c[1] + j, c[2] + l});
              if (it == grid.end()) continue;
              for (const index_t m : it->second) {
                if (std::hypot(x[at(m)] - q[0], y[at(m)] - q[1], z[at(m)] - q[2]) <= tolerance) {
                  found = m;
                  ++count;
                }
              }
            }
          }
        }
      }
      if (count != 1) {
        unmatched(where(concat("node ", std::to_string(n), " of zone '", name_a,
                               "', translated, coincides with ", std::to_string(count),
                               " nodes of zone '", name_b, "'")));
      }
      image[at(n)] = found;
      out.nodes.push_back({n, found});
      const double d = std::hypot(x[at(found)] - q[0], y[at(found)] - q[1], z[at(found)] - q[2]);
      out.max_relative = std::max(out.max_relative, d / tolerance);
    }
  }

  std::map<std::vector<index_t>, index_t> faces_b;
  for (index_t f = b0; f < b1; ++f) {
    std::vector<index_t> nodes(mesh.face_nodes(f).begin(), mesh.face_nodes(f).end());
    std::ranges::sort(nodes);
    faces_b.emplace(std::move(nodes), f);
  }
  std::vector<index_t> partner(at(a1 - a0), -1);
  std::vector<char> used(at(b1 - b0), 0);
  for (index_t f = a0; f < a1; ++f) {
    std::vector<index_t> mapped;
    for (const index_t n : mesh.face_nodes(f)) mapped.push_back(image[at(n)]);
    std::vector<index_t> sorted = mapped;
    std::ranges::sort(sorted);
    const auto it = faces_b.find(sorted);
    if (it == faces_b.end() || used[at(it->second - b0)] != 0) {
      unmatched(where(concat("face ", std::to_string(f), " of zone '", name_a,
                             "', translated, is no face of zone '", name_b, "'")));
    }
    // Both faces point out of the domain, so the translated face of zone a
    // runs through its nodes in the opposite direction to its partner.
    const auto partner_nodes = mesh.face_nodes(it->second);
    std::ranges::reverse(mapped);
    std::ranges::rotate(mapped, std::ranges::find(mapped, partner_nodes[0]));
    if (!std::ranges::equal(mapped, partner_nodes)) {
      unmatched(where("face " + std::to_string(f) + " of zone '" + name_a +
                      "' and its partner point the same way"));
    }
    used[at(it->second - b0)] = 1;
    partner[at(f - a0)] = it->second;
  }
  out.partners = std::move(partner);
  return out;
}

}  // namespace

PeriodicMatch match_periodic_faces(const Mesh& mesh, std::span<const PeriodicPair> pairs) {
  check_declaration(mesh, pairs);
  PeriodicMatch out;
  out.partners.reserve(pairs.size());
  out.nodes.reserve(pairs.size());
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    PairMatch m = match_pair(mesh, pairs[p], p);
    out.partners.push_back(std::move(m.partners));
    out.nodes.push_back(std::move(m.nodes));
    out.max_relative = std::max(out.max_relative, m.max_relative);
  }
  return out;
}

PeriodicMoves place_periodic_nodes(std::span<const double> x, std::span<const double> y,
                                   std::span<const double> z, std::span<const PeriodicPair> pairs,
                                   const PeriodicMatch& match) {
  const std::size_t n = x.size();
  std::vector<char> zone_b(n, 0), periodic(n, 0);
  std::vector<std::vector<std::pair<index_t, std::size_t>>> forward(n);
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    for (const auto& [a, b] : match.nodes[p]) {
      forward[at(a)].emplace_back(b, p);
      zone_b[at(b)] = 1;
      periodic[at(a)] = periodic[at(b)] = 1;
    }
  }
  // Classes: the nodes joined by matched pairs, either way.
  std::vector<std::vector<index_t>> joined(n);
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    for (const auto& [a, b] : match.nodes[p]) {
      joined[at(a)].push_back(b);
      joined[at(b)].push_back(a);
    }
  }
  std::vector<std::array<double, 3>> position(n);
  std::vector<char> placed(n, 0), classed(n, 0);
  std::vector<index_t> members, queue;
  std::vector<char> reached(n, 0);
  // The nodes forward steps from `root` reach, placed at root + the translations on the way.
  const auto walk = [&](index_t root) {
    queue.assign(1, root);
    reached[at(root)] = 1;
    position[at(root)] = {x[at(root)], y[at(root)], z[at(root)]};
    for (std::size_t next = 0; next < queue.size(); ++next) {
      const auto from = at(queue[next]);
      for (const auto& [to, p] : forward[from]) {
        if (reached[at(to)] != 0) continue;
        reached[at(to)] = 1;
        const auto& t = pairs[p].translation;
        position[at(to)] = {position[from][0] + t[0], position[from][1] + t[1],
                            position[from][2] + t[2]};
        queue.push_back(to);
      }
    }
    for (const index_t q : queue) reached[at(q)] = 0;
    return queue.size();
  };
  for (std::size_t first = 0; first < n; ++first) {
    if (periodic[first] == 0 || classed[first] != 0) continue;
    members.assign(1, static_cast<index_t>(first));
    classed[first] = 1;
    for (std::size_t next = 0; next < members.size(); ++next) {
      for (const index_t q : joined[at(members[next])]) {
        if (classed[at(q)] != 0) continue;
        classed[at(q)] = 1;
        members.push_back(q);
      }
    }
    std::ranges::sort(members);
    // The root: the smallest node that is zone b of no pair, else the smallest node, from which
    // forward steps reach the whole class.
    index_t root = -1;
    for (const bool zone_a_only : {true, false}) {
      for (const index_t q : members) {
        if (root >= 0 || (zone_a_only && zone_b[at(q)] != 0)) continue;
        if (walk(q) == members.size()) root = q;
      }
    }
    if (root < 0) {
      unmatched(concat("periodic node ", std::to_string(first),
                       ": no node of its class reaches the others by the pairs' translations"));
    }
    (void)walk(root);
    for (const index_t q : members) placed[at(q)] = 1;
  }
  const auto ulp = [](double v) {
    const double a = std::abs(v);
    return std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
  };
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const auto& t = pairs[p].translation;
    for (const auto& [a, b] : match.nodes[p]) {
      for (std::size_t d = 0; d < 3; ++d) {
        const double expected = position[at(a)][d] + t[d];
        if (std::abs(position[at(b)][d] - expected) >
            4.0 * ulp(std::max(std::abs(position[at(a)][d]), std::abs(t[d])))) {
          unmatched(concat("periodic pair ", std::to_string(p), ": node ", std::to_string(b),
                           " cannot be node ", std::to_string(a),
                           " translated, as the other pairs that join them place it"));
        }
      }
    }
  }
  PeriodicMoves out;
  for (std::size_t i = 0; i < n; ++i) {
    if (placed[i] == 0) continue;
    const std::array<double, 3> from{x[i], y[i], z[i]};
    if (position[i] == from) continue;
    out.nodes.push_back(static_cast<index_t>(i));
    out.to.push_back(position[i]);
    bool significant = false;
    for (std::size_t d = 0; d < 3; ++d) {
      significant =
          significant || std::abs(position[i][d] - from[d]) >
                             4.0 * ulp(std::max(std::abs(from[d]), std::abs(position[i][d])));
    }
    if (significant) {
      ++out.significant;
      out.max_distance = std::max(
          out.max_distance,
          std::hypot(position[i][0] - from[0], position[i][1] - from[1], position[i][2] - from[2]));
    }
  }
  return out;
}

}  // namespace autan
