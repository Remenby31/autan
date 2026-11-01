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
Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec& a) { return std::hypot(a[0], a[1], a[2]); }

// For O(1) coordinates: plain products cannot overflow there.
double segment_distance(const Vec& p, const Vec& u, const Vec& v) {
  const Vec d = v - u;
  const double dd = dot(d, d);
  const double t = dd > 0.0 ? std::clamp(dot(p - u, d) / dd, 0.0, 1.0) : 0.0;
  const Vec e = p - (u + t * d);
  return std::sqrt(dot(e, e));
}

double max_abs(const Vec& a) { return std::max({std::abs(a[0]), std::abs(a[1]), std::abs(a[2])}); }

// Distance from p to triangle abc. In coordinates relative to a and scaled
// by the largest component of the edges and of p - a, so that every term is
// O(1) and no intermediate overflows or underflows whatever the sizes: the
// plane distance when p projects inside (sign of each edge's cross product
// along the normal), else the nearest of the three edges. A triangle of zero
// area reduces to its edges.
double triangle_distance(const Vec& p, const Vec& a, const Vec& b, const Vec& c) {
  const double scale = std::max({max_abs(b - a), max_abs(c - a), max_abs(c - b), max_abs(p - a)});
  if (!(scale > 0.0)) return 0.0;
  const double s = 1.0 / scale;
  const Vec B = s * (b - a), C = s * (c - a), P = s * (p - a);
  const Vec A{};
  const Vec n = cross(B, C);
  const double length = std::sqrt(dot(n, n));
  // A projection inside the triangle is the nearest point.
  if (length > 0.0 && dot(cross(B, P), n) >= 0.0 && dot(cross(C - B, P - B), n) >= 0.0 &&
      dot(cross(A - C, P - C), n) >= 0.0) {
    return std::abs(dot(P, n)) / length * scale;
  }
  return std::min(
             {segment_distance(P, A, B), segment_distance(P, B, C), segment_distance(P, C, A)}) *
         scale;
}

double box_distance(const Vec& p, const Vec& lo, const Vec& hi) {
  Vec gap{};
  for (std::size_t k = 0; k < 3; ++k)
    gap[k] = p[k] < lo[k] ? lo[k] - p[k] : (p[k] > hi[k] ? p[k] - hi[k] : 0.0);
  return norm(gap);
}

constexpr std::int64_t leaf_size = 4;
constexpr double max_reach = 1e5;
constexpr double max_combinations = 1e6;

// Inverse of the leading n x n block of a symmetric positive definite matrix.
std::array<std::array<double, 3>, 3> invert(const std::array<std::array<double, 3>, 3>& m,
                                            std::size_t n) {
  std::array<std::array<double, 3>, 3> inv{};
  if (n == 1) {
    inv[0][0] = 1.0 / m[0][0];
  } else if (n == 2) {
    const double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    inv[0][0] = m[1][1] / det;
    inv[1][1] = m[0][0] / det;
    inv[0][1] = -m[0][1] / det;
    inv[1][0] = -m[1][0] / det;
  } else {
    const Vec c0 = cross(Vec{m[1][0], m[1][1], m[1][2]}, Vec{m[2][0], m[2][1], m[2][2]});
    const Vec c1 = cross(Vec{m[2][0], m[2][1], m[2][2]}, Vec{m[0][0], m[0][1], m[0][2]});
    const Vec c2 = cross(Vec{m[0][0], m[0][1], m[0][2]}, Vec{m[1][0], m[1][1], m[1][2]});
    const double det = dot(Vec{m[0][0], m[0][1], m[0][2]}, c0);
    for (std::size_t j = 0; j < 3; ++j) {
      inv[j][0] = c0[j] / det;
      inv[j][1] = c1[j] / det;
      inv[j][2] = c2[j] / det;
    }
  }
  return inv;
}

// The lattice of periodic translations as at most three independent
// directions: translations on one line reduce to the shortest, of which the
// others must be integer multiples.
std::vector<Vec> periodic_directions(const Mesh& mesh) {
  // Group the translations by line, keeping the shortest of each.
  std::vector<std::vector<Vec>> lines;
  for (const auto& pair : mesh.periodic_pairs()) {
    const Vec& t = pair.translation;
    const auto line = std::ranges::find_if(lines, [&](const std::vector<Vec>& l) {
      return norm(cross(l[0], t)) <= 1e-12 * norm(l[0]) * norm(t);
    });
    if (line == lines.end())
      lines.push_back({t});
    else
      line->push_back(t);
  }
  std::vector<Vec> directions;
  for (const auto& line : lines) {
    const Vec shortest = *std::ranges::min_element(
        line, [](const Vec& l, const Vec& r) { return norm(l) < norm(r); });
    for (const Vec& t : line) {
      const double ratio = norm(t) / norm(shortest);
      if (std::abs(ratio - std::round(ratio)) > 1e-9 * ratio) {
        throw Error(ErrorCode::mesh_periodic_invalid,
                    "periodic translations on one line that are not multiples of the shortest");
      }
    }
    directions.push_back(shortest);
  }
  if (directions.size() > 3) {
    throw Error(ErrorCode::mesh_periodic_invalid,
                "periodic translations in more than three directions");
  }
  return directions;
}

}  // namespace

WallSurface::WallSurface(const Mesh& mesh, std::span<const index_t> wall_zones) {
  const auto zones = mesh.face_zones();
  const auto off = mesh.boundary_zone_offsets();
  const auto px = mesh.x(), py = mesh.y(), pz = mesh.z();
  const auto point = [&](index_t n) { return Vec{px[at(n)], py[at(n)], pz[at(n)]}; };

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
      // The fan apex as compute_geometry takes it: the node mean,
      // accumulated relative to the first node.
      const Vec origin = point(nodes[0]);
      Vec sum{};
      for (const index_t n : nodes) {
        const Vec q = point(n);
        if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) {
          throw Error(ErrorCode::geometry_degenerate_face,
                      "wall face " + std::to_string(f) + " has a node that is not finite");
        }
        sum = sum + (q - origin);
      }
      const Vec m = origin + (1.0 / static_cast<double>(nodes.size())) * sum;
      if (!std::isfinite(m[0]) || !std::isfinite(m[1]) || !std::isfinite(m[2])) {
        throw Error(ErrorCode::geometry_degenerate_face,
                    "the node mean of wall face " + std::to_string(f) + " is not finite");
      }
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        a_.push_back(m);
        b_.push_back(point(nodes[i]));
        c_.push_back(point(nodes[(i + 1) % nodes.size()]));
      }
    }
  }
  if (a_.empty()) return;

  // The tree is built over a permutation of the triangles, applied once.
  std::vector<std::int64_t> order(a_.size());
  std::iota(order.begin(), order.end(), std::int64_t{0});
  build(order);
  std::vector<Vec> a(a_.size()), b(a_.size()), c(a_.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    a[i] = a_[at(order[i])];
    b[i] = b_[at(order[i])];
    c[i] = c_[at(order[i])];
  }
  a_ = std::move(a);
  b_ = std::move(b);
  c_ = std::move(c);

  // Periodic copies are visited by shifting the query point by every
  // combination of k times each direction, |k| up to the number of periods
  // the bounding box of the mesh spans along it, plus one.
  directions_ = periodic_directions(mesh);
  Vec lo = nodes_[0].lo, hi = nodes_[0].hi;
  for (index_t n = 0; n < mesh.n_points(); ++n) {
    const Vec q = point(n);
    for (std::size_t k = 0; k < 3; ++k) {
      if (!std::isfinite(q[k])) continue;
      lo[k] = std::min(lo[k], q[k]);
      hi[k] = std::max(hi[k], q[k]);
    }
  }
  // Coefficients of a displacement v within the box along the directions D
  // (independent columns): k = (D^T D)^-1 D^T v, so |k_j| <= sum_i |row j|_i
  // times the box extent along axis i.
  const std::size_t n = directions_.size();
  if (n > 0) {
    std::array<std::array<double, 3>, 3> gram{};
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) gram[i][j] = dot(directions_[i], directions_[j]);
    }
    const auto inverse = invert(gram, n);
    double combinations = 1.0;
    for (std::size_t j = 0; j < n; ++j) {
      double bound = 0.0;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        double row = 0.0;  // ((D^T D)^-1 D^T)_{j, axis}
        for (std::size_t i = 0; i < n; ++i) row += inverse[j][i] * directions_[i][axis];
        bound += std::abs(row) * (hi[axis] - lo[axis]);
      }
      const double reach = std::ceil(bound) + 1.0;
      combinations *= 2.0 * reach + 1.0;
      if (!(reach <= max_reach) || !(combinations <= max_combinations)) {
        throw Error(ErrorCode::mesh_periodic_invalid,
                    "periodic translations too small or too oblique for the mesh extent");
      }
      reach_.push_back(static_cast<std::int64_t>(reach));
    }
  }
}

// Builds the tree over `order`: each node splits its triangles at the
// median along the longest axis of their centroids, children appended to
// nodes_. Iterative: a work list of (node, begin, end).
void WallSurface::build(std::vector<std::int64_t>& order) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  struct Work {
    std::int64_t self, begin, end;
  };
  nodes_.push_back({});
  std::vector<Work> work{{0, 0, static_cast<std::int64_t>(order.size())}};
  while (!work.empty()) {
    const auto [self, begin, end] = work.back();
    work.pop_back();
    Node node{
        .lo = {inf, inf, inf}, .hi = {-inf, -inf, -inf}, .first = begin, .count = end - begin};
    Vec clo = node.lo, chi = node.hi;
    for (std::int64_t i = begin; i < end; ++i) {
      const auto t = at(order[at(i)]);
      for (const Vec* v : {&a_[t], &b_[t], &c_[t]}) {
        for (std::size_t k = 0; k < 3; ++k) {
          node.lo[k] = std::min(node.lo[k], (*v)[k]);
          node.hi[k] = std::max(node.hi[k], (*v)[k]);
        }
      }
      const Vec centroid = (1.0 / 3.0) * (a_[t] + b_[t] + c_[t]);
      for (std::size_t k = 0; k < 3; ++k) {
        clo[k] = std::min(clo[k], centroid[k]);
        chi[k] = std::max(chi[k], centroid[k]);
      }
    }
    nodes_[at(self)] = node;
    if (end - begin <= leaf_size) continue;

    std::size_t axis = 0;
    for (std::size_t k = 1; k < 3; ++k) {
      if (chi[k] - clo[k] > chi[axis] - clo[axis]) axis = k;
    }
    const auto key = [&](std::int64_t t) {
      return a_[at(t)][axis] + b_[at(t)][axis] + c_[at(t)][axis];
    };
    const std::int64_t mid = begin + (end - begin) / 2;
    std::nth_element(
        order.begin() + begin, order.begin() + mid, order.begin() + end,
        [&](std::int64_t l, std::int64_t r) { return key(l) != key(r) ? key(l) < key(r) : l < r; });
    const auto left = static_cast<std::int64_t>(nodes_.size());
    nodes_.push_back({});
    nodes_.push_back({});
    nodes_[at(self)].first = left;
    nodes_[at(self)].count = 0;
    work.push_back({left + 1, mid, end});
    work.push_back({left, begin, mid});
  }
}

double WallSurface::nearest(const std::array<double, 3>& p, double best) const {
  // A median split halves every level: 128 entries cover any tree.
  std::array<std::int64_t, 128> stack{};
  std::size_t top = 0;
  stack[top++] = 0;
  while (top > 0) {
    const Node& node = nodes_[at(stack[--top])];
    if (box_distance(p, node.lo, node.hi) >= best) continue;
    if (node.count > 0) {
      for (std::int64_t t = node.first; t < node.first + node.count; ++t)
        best = std::min(best, triangle_distance(p, a_[at(t)], b_[at(t)], c_[at(t)]));
      continue;
    }
    const Node& l = nodes_[at(node.first)];
    const Node& r = nodes_[at(node.first + 1)];
    const bool left_first = box_distance(p, l.lo, l.hi) <= box_distance(p, r.lo, r.hi);
    stack[top++] = left_first ? node.first + 1 : node.first;
    stack[top++] = left_first ? node.first : node.first + 1;
  }
  return best;
}

double WallSurface::distance(const std::array<double, 3>& point) const {
  if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
    return std::numeric_limits<double>::quiet_NaN();
  if (nodes_.empty()) return std::numeric_limits<double>::infinity();
  // The unshifted copy first, for a tight bound; then the others, skipped
  // when the shifted point is farther from the whole surface's box.
  double best = nearest(point, std::numeric_limits<double>::infinity());
  const std::size_t n = directions_.size();
  std::array<std::int64_t, 3> k{};
  for (std::size_t i = 0; i < n; ++i) k[i] = -reach_[i];
  while (n > 0) {
    bool zero = true;
    Vec q = point;
    for (std::size_t i = 0; i < n; ++i) {
      zero = zero && k[i] == 0;
      q = q - static_cast<double>(k[i]) * directions_[i];
    }
    if (!zero && box_distance(q, nodes_[0].lo, nodes_[0].hi) < best) best = nearest(q, best);
    std::size_t i = 0;
    while (i < n && ++k[i] > reach_[i]) {
      k[i] = -reach_[i];
      ++i;
    }
    if (i == n) break;
  }
  return best;
}

std::vector<double> wall_distance(const Mesh& mesh, const Geometry& geometry,
                                  std::span<const index_t> wall_zones) {
  const auto n = at(mesh.n_cells());
  if (geometry.cell_centroid_x.size() != n || geometry.cell_centroid_y.size() != n ||
      geometry.cell_centroid_z.size() != n) {
    throw Error(ErrorCode::mesh_inconsistent_sizes, "the geometry does not belong to the mesh");
  }
  const WallSurface surface(mesh, wall_zones);
  std::vector<double> d(n);
  for (std::size_t c = 0; c < n; ++c) {
    d[c] = surface.distance(
        {geometry.cell_centroid_x[c], geometry.cell_centroid_y[c], geometry.cell_centroid_z[c]});
  }
  return d;
}

}  // namespace autan
