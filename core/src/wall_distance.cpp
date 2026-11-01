#include "autan/core/wall_distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "autan/core/error.hpp"

namespace autan {
namespace {

using Vec = std::array<double, 3>;

std::size_t at(std::int64_t i) { return static_cast<std::size_t>(i); }

Vec operator-(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec operator+(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec operator*(double s, const Vec& a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Closest point of triangle abc to p: Ericson, Real-Time Collision
// Detection (2005), section 5.1.5, by Voronoi regions of the vertices,
// edges and face.
Vec closest_on_triangle(const Vec& p, const Vec& a, const Vec& b, const Vec& c) {
  const Vec ab = b - a, ac = c - a, ap = p - a;
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) return a;
  const Vec bp = p - b;
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + (d1 / (d1 - d3)) * ab;
  const Vec cp = p - c;
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + (d2 / (d2 - d6)) * ac;
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
  const double denom = 1.0 / (va + vb + vc);
  return a + (vb * denom) * ab + (vc * denom) * ac;
}

double box_distance2(const Vec& p, const Vec& lo, const Vec& hi) {
  double d2 = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    const double e = p[k] < lo[k] ? lo[k] - p[k] : (p[k] > hi[k] ? p[k] - hi[k] : 0.0);
    d2 += e * e;
  }
  return d2;
}

constexpr std::int32_t leaf_size = 4;

}  // namespace

WallSurface::WallSurface(const Mesh& mesh, std::span<const index_t> wall_zones) {
  const auto zones = mesh.face_zones();
  const auto off = mesh.boundary_zone_offsets();
  const auto x = mesh.x(), y = mesh.y(), z = mesh.z();

  // Periodic copies: every combination of -1, 0, +1 times each distinct
  // translation (opposite ones count once).
  std::vector<Vec> directions;
  for (const auto& pair : mesh.periodic_pairs()) {
    const Vec& t = pair.translation;
    const Vec minus{-t[0], -t[1], -t[2]};
    if (std::ranges::find(directions, t) == directions.end() &&
        std::ranges::find(directions, minus) == directions.end()) {
      directions.push_back(t);
    }
  }
  std::vector<Vec> shifts{Vec{}};
  for (const Vec& d : directions) {
    std::vector<Vec> more;
    for (const Vec& s : shifts) {
      for (const double k : {-1.0, 0.0, 1.0}) more.push_back(s + k * d);
    }
    shifts = std::move(more);
  }

  for (const index_t wz : wall_zones) {
    if (wz < 0 || at(wz) >= zones.size()) {
      throw Error(ErrorCode::mesh_index_out_of_range,
                  "wall zone " + std::to_string(wz) + " is not a face zone");
    }
    if (!zones[at(wz)].boundary) {
      throw Error(ErrorCode::mesh_index_out_of_range,
                  "wall zone '" + zones[at(wz)].name + "' holds interior faces");
    }
    for (index_t f = off[at(wz)]; f < off[at(wz) + 1]; ++f) {
      const auto nodes = mesh.face_nodes(f);
      Vec m{};
      for (const index_t n : nodes) m = m + Vec{x[at(n)], y[at(n)], z[at(n)]};
      m = (1.0 / static_cast<double>(nodes.size())) * m;
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto p = at(nodes[i]), q = at(nodes[(i + 1) % nodes.size()]);
        for (const Vec& s : shifts) {
          a_.push_back(m + s);
          b_.push_back(Vec{x[p], y[p], z[p]} + s);
          c_.push_back(Vec{x[q], y[q], z[q]} + s);
        }
      }
    }
  }
  if (!a_.empty()) {
    nodes_.push_back({});
    build(0, 0, static_cast<std::int32_t>(a_.size()));
  }
}

// Builds node `self` over triangles [begin, end): a median split along the
// longest axis of the triangle centroids, children appended to nodes_.
void WallSurface::build(std::int32_t self, std::int32_t begin, std::int32_t end) {
  Node node{
      .lo = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()},
      .hi = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()},
      .first = begin,
      .count = end - begin};
  Vec clo = node.lo, chi = node.hi;
  for (std::int32_t t = begin; t < end; ++t) {
    for (const Vec* v : {&a_[at(t)], &b_[at(t)], &c_[at(t)]}) {
      for (std::size_t k = 0; k < 3; ++k) {
        node.lo[k] = std::min(node.lo[k], (*v)[k]);
        node.hi[k] = std::max(node.hi[k], (*v)[k]);
      }
    }
    const Vec centroid = (1.0 / 3.0) * (a_[at(t)] + b_[at(t)] + c_[at(t)]);
    for (std::size_t k = 0; k < 3; ++k) {
      clo[k] = std::min(clo[k], centroid[k]);
      chi[k] = std::max(chi[k], centroid[k]);
    }
  }
  nodes_[at(self)] = node;
  if (end - begin <= leaf_size) return;

  std::size_t axis = 0;
  for (std::size_t k = 1; k < 3; ++k) {
    if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
  }
  // Reorder the three vertex arrays together through a permutation.
  std::vector<std::int32_t> order(at(end - begin));
  std::iota(order.begin(), order.end(), begin);
  const auto key = [&](std::int32_t t) {
    return a_[at(t)][axis] + b_[at(t)][axis] + c_[at(t)][axis];
  };
  const auto middle = order.begin() + (end - begin) / 2;
  std::nth_element(order.begin(), middle, order.end(), [&](std::int32_t l, std::int32_t r) {
    return key(l) != key(r) ? key(l) < key(r) : l < r;
  });
  std::vector<Vec> a, b, c;
  for (const std::int32_t t : order) {
    a.push_back(a_[at(t)]);
    b.push_back(b_[at(t)]);
    c.push_back(c_[at(t)]);
  }
  std::ranges::copy(a, a_.begin() + begin);
  std::ranges::copy(b, b_.begin() + begin);
  std::ranges::copy(c, c_.begin() + begin);

  const std::int32_t mid = begin + (end - begin) / 2;
  const auto left = static_cast<std::int32_t>(nodes_.size());
  nodes_.push_back({});
  nodes_.push_back({});
  nodes_[at(self)].first = left;
  nodes_[at(self)].count = 0;
  build(left, begin, mid);
  build(left + 1, mid, end);
}

double WallSurface::distance(const std::array<double, 3>& point) const {
  if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
    return std::numeric_limits<double>::quiet_NaN();
  double best = std::numeric_limits<double>::infinity();
  if (nodes_.empty()) return best;
  std::vector<std::int32_t> stack{0};
  while (!stack.empty()) {
    const Node& node = nodes_[at(stack.back())];
    stack.pop_back();
    if (box_distance2(point, node.lo, node.hi) >= best) continue;
    if (node.count > 0) {
      for (std::int32_t t = node.first; t < node.first + node.count; ++t) {
        const Vec d = point - closest_on_triangle(point, a_[at(t)], b_[at(t)], c_[at(t)]);
        best = std::min(best, dot(d, d));
      }
      continue;
    }
    // Visit the nearer child first.
    const Node& l = nodes_[at(node.first)];
    const Node& r = nodes_[at(node.first + 1)];
    const bool left_first = box_distance2(point, l.lo, l.hi) <= box_distance2(point, r.lo, r.hi);
    stack.push_back(left_first ? node.first + 1 : node.first);
    stack.push_back(left_first ? node.first : node.first + 1);
  }
  return std::sqrt(best);
}

std::vector<double> wall_distance(const Mesh& mesh, const Geometry& geometry,
                                  std::span<const index_t> wall_zones) {
  const WallSurface surface(mesh, wall_zones);
  std::vector<double> d(at(mesh.n_cells()));
  for (std::size_t c = 0; c < d.size(); ++c) {
    d[c] = surface.distance(
        {geometry.cell_centroid_x[c], geometry.cell_centroid_y[c], geometry.cell_centroid_z[c]});
  }
  return d;
}

}  // namespace autan
