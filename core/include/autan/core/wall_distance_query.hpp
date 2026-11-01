#pragma once

// The query of a WallSurface: one implementation, compiled for the host and,
// by nvcc, for CUDA devices, so that both give the same bits (with FMA
// contraction off on both sides: -ffp-contract=off, -fmad=false).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define AUTAN_WALL_HD __host__ __device__
#else
#define AUTAN_WALL_HD
#endif

namespace autan::wall {

using Vec = std::array<double, 3>;

/// A node of the tree: its box, and for a leaf (count > 0) its triangles
/// [first, first + count), for an inner node (count = 0) its children
/// first and first + 1.
struct Node {
  Vec lo, hi;
  std::int64_t first;
  std::int64_t count;
};

/// Arrays of a WallSurface, in host or device memory.
struct View {
  const Node* nodes = nullptr;
  std::int64_t n_nodes = 0;
  const Vec* a = nullptr;  // triangle t: a[t], b[t], c[t], in tree order
  const Vec* b = nullptr;
  const Vec* c = nullptr;
  const Vec* directions = nullptr;  // periodic directions, at most 3
  const std::int64_t* reach = nullptr;
  int n_directions = 0;
};

AUTAN_WALL_HD inline Vec sub(const Vec& a, const Vec& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
AUTAN_WALL_HD inline Vec add(const Vec& a, const Vec& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
AUTAN_WALL_HD inline Vec scale(double s, const Vec& a) { return {s * a[0], s * a[1], s * a[2]}; }
AUTAN_WALL_HD inline double dot(const Vec& a, const Vec& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
AUTAN_WALL_HD inline Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
AUTAN_WALL_HD inline double max_abs(const Vec& a) {
  const double x = std::fabs(a[0]), y = std::fabs(a[1]), z = std::fabs(a[2]);
  return x > y ? (x > z ? x : z) : (y > z ? y : z);
}

// For O(1) coordinates: plain products cannot overflow there.
AUTAN_WALL_HD inline double segment_distance(const Vec& p, const Vec& u, const Vec& v) {
  const Vec d = sub(v, u);
  const double dd = dot(d, d);
  double t = dd > 0.0 ? dot(sub(p, u), d) / dd : 0.0;
  t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  const Vec e = sub(p, add(u, scale(t, d)));
  return std::sqrt(dot(e, e));
}

// Distance from p to triangle abc. In coordinates relative to a and scaled
// by the largest component of the edges and of p - a, so that every term is
// O(1) and no intermediate overflows or underflows whatever the sizes: the
// plane distance when p projects inside (sign of each edge's cross product
// along the normal), else the nearest of the three edges. A triangle of zero
// area reduces to its edges. Only correctly rounded operations: the same bits
// on every IEEE 754 target without contraction.
AUTAN_WALL_HD inline double triangle_distance(const Vec& p, const Vec& a, const Vec& b,
                                              const Vec& c) {
  double scale_ = max_abs(sub(b, a));
  for (const double m : {max_abs(sub(c, a)), max_abs(sub(c, b)), max_abs(sub(p, a))})
    scale_ = m > scale_ ? m : scale_;
  if (!(scale_ > 0.0)) return 0.0;
  const double s = 1.0 / scale_;
  const Vec B = scale(s, sub(b, a)), C = scale(s, sub(c, a)), P = scale(s, sub(p, a));
  const Vec A{};
  const Vec n = cross(B, C);
  const double length = std::sqrt(dot(n, n));
  // A projection inside the triangle is the nearest point.
  if (length > 0.0 && dot(cross(B, P), n) >= 0.0 && dot(cross(sub(C, B), sub(P, B)), n) >= 0.0 &&
      dot(cross(sub(A, C), sub(P, C)), n) >= 0.0) {
    return std::fabs(dot(P, n)) / length * scale_;
  }
  double d = segment_distance(P, A, B);
  const double e = segment_distance(P, B, C), f = segment_distance(P, C, A);
  d = e < d ? e : d;
  d = f < d ? f : d;
  return d * scale_;
}

// Distance from p to a box. It decides which subtrees the traversal skips,
// and two triangles sharing an edge give that edge's distance with different
// roundings, so the result depends on it: scaled by the largest gap (no
// overflow at any size) and built from correctly rounded operations only,
// for the same bits on every target.
AUTAN_WALL_HD inline double box_distance(const Vec& p, const Vec& lo, const Vec& hi) {
  Vec gap{};
  for (std::size_t k = 0; k < 3; ++k)
    gap[k] = p[k] < lo[k] ? lo[k] - p[k] : (p[k] > hi[k] ? p[k] - hi[k] : 0.0);
  const double largest = max_abs(gap);
  if (!(largest > 0.0)) return largest;
  const Vec g = scale(1.0 / largest, gap);
  return std::sqrt(dot(g, g)) * largest;
}

/// The nearest triangle of the tree to p, if nearer than `best`.
AUTAN_WALL_HD inline double nearest(const View& s, const Vec& p, double best) {
  // A median split halves every level: 128 entries cover any tree.
  std::int64_t stack[128];
  int top = 0;
  stack[top++] = 0;
  while (top > 0) {
    const Node& node = s.nodes[stack[--top]];
    if (box_distance(p, node.lo, node.hi) >= best) continue;
    if (node.count > 0) {
      for (std::int64_t t = node.first; t < node.first + node.count; ++t) {
        const double d = triangle_distance(p, s.a[t], s.b[t], s.c[t]);
        best = d < best ? d : best;
      }
      continue;
    }
    const Node& l = s.nodes[node.first];
    const Node& r = s.nodes[node.first + 1];
    const bool left_first = box_distance(p, l.lo, l.hi) <= box_distance(p, r.lo, r.hi);
    stack[top++] = left_first ? node.first + 1 : node.first;
    stack[top++] = left_first ? node.first : node.first + 1;
  }
  return best;
}

/// Distance from p to the surface and its periodic copies: +inf without
/// triangles, NaN for a point that is not finite.
AUTAN_WALL_HD inline double distance(const View& s, const Vec& point) {
  if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))
    return std::numeric_limits<double>::quiet_NaN();
  if (s.n_nodes == 0) return std::numeric_limits<double>::infinity();
  // The unshifted copy first, for a tight bound; then the others, skipped
  // when the shifted point is farther from the whole surface's box.
  double best = nearest(s, point, std::numeric_limits<double>::infinity());
  const int n = s.n_directions;
  std::int64_t k[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) k[i] = -s.reach[i];
  while (n > 0) {
    bool zero = true;
    Vec q = point;
    for (int i = 0; i < n; ++i) {
      zero = zero && k[i] == 0;
      q = sub(q, scale(static_cast<double>(k[i]), s.directions[i]));
    }
    if (!zero && box_distance(q, s.nodes[0].lo, s.nodes[0].hi) < best) best = nearest(s, q, best);
    int i = 0;
    while (i < n && ++k[i] > s.reach[i]) {
      k[i] = -s.reach[i];
      ++i;
    }
    if (i == n) break;
  }
  return best;
}

}  // namespace autan::wall
