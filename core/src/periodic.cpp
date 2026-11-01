#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/mesh.hpp"

namespace autan {
namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

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
      if (!zones[at(z)].boundary || zones[at(z)].kind != BcKind::periodic) {
        invalid(which + ": zone '" + zones[at(z)].name +
                "' is not a boundary zone of kind periodic");
      }
      ++uses[at(z)];
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
    if (uses[z] > 1) invalid("zone '" + zones[z].name + "' is in several periodic pairs");
    if (uses[z] == 0 && zones[z].kind == BcKind::periodic && !partitioned) {
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

std::vector<index_t> match_pair(const Mesh& mesh, const PeriodicPair& pair, std::size_t p) {
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
  // written in single precision.
  constexpr double relative_tolerance = 1e-4;
  const std::vector<double> h = shortest_edges(mesh, pair.zone_a);
  double cell = 0.0;
  for (const double v : h) {
    if (std::isfinite(v)) cell = std::max(cell, relative_tolerance * v);
  }
  if (!(cell > 0.0))
    unmatched("periodic pair " + std::to_string(p) + ": zone '" + name_a + "' has no extent");
  using Key = std::array<std::int64_t, 3>;
  const auto key = [cell](double px, double py, double pz) {
    return Key{static_cast<std::int64_t>(std::floor(px / cell)),
               static_cast<std::int64_t>(std::floor(py / cell)),
               static_cast<std::int64_t>(std::floor(pz / cell))};
  };
  std::map<Key, std::vector<index_t>> grid;
  std::vector<char> seen(at(mesh.n_points()), 0);
  for (index_t f = b0; f < b1; ++f) {
    for (const index_t n : mesh.face_nodes(f)) {
      if (seen[at(n)] != 0) continue;
      seen[at(n)] = 1;
      grid[key(x[at(n)], y[at(n)], z[at(n)])].push_back(n);
    }
  }

  std::vector<index_t> image(at(mesh.n_points()), -1);
  for (index_t f = a0; f < a1; ++f) {
    for (const index_t n : mesh.face_nodes(f)) {
      if (image[at(n)] >= 0) continue;
      const double qx = x[at(n)] + t[0], qy = y[at(n)] + t[1], qz = z[at(n)] + t[2];
      const double tolerance = relative_tolerance * h[at(n)];
      const Key k = key(qx, qy, qz);
      index_t found = -1;
      int count = 0;
      for (std::int64_t i = -1; i <= 1; ++i) {
        for (std::int64_t j = -1; j <= 1; ++j) {
          for (std::int64_t l = -1; l <= 1; ++l) {
            const auto it = grid.find(Key{k[0] + i, k[1] + j, k[2] + l});
            if (it == grid.end()) continue;
            for (const index_t m : it->second) {
              const double d = std::hypot(x[at(m)] - qx, y[at(m)] - qy, z[at(m)] - qz);
              if (d <= tolerance) {
                found = m;
                ++count;
              }
            }
          }
        }
      }
      if (count != 1) {
        unmatched("periodic pair " + std::to_string(p) + ": node " + std::to_string(n) +
                  " of zone '" + name_a + "', translated, coincides with " + std::to_string(count) +
                  " nodes of zone '" + name_b + "'");
      }
      image[at(n)] = found;
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
      unmatched("periodic pair " + std::to_string(p) + ": face " + std::to_string(f) +
                " of zone '" + name_a + "', translated, is no face of zone '" + name_b + "'");
    }
    used[at(it->second - b0)] = 1;
    partner[at(f - a0)] = it->second;
  }
  return partner;
}

}  // namespace

std::vector<std::vector<index_t>> match_periodic_faces(const Mesh& mesh,
                                                       std::span<const PeriodicPair> pairs) {
  check_declaration(mesh, pairs);
  std::vector<std::vector<index_t>> partners;
  partners.reserve(pairs.size());
  for (std::size_t p = 0; p < pairs.size(); ++p) partners.push_back(match_pair(mesh, pairs[p], p));
  return partners;
}

}  // namespace autan
