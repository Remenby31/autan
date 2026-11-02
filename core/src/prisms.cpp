#include "autan/core/prisms.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"

namespace autan {
namespace {

using Vec = std::array<double, 3>;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

Vec sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double length(const Vec& a) { return std::sqrt(dot(a, a)); }

// A triangle of a tetrahedron, its nodes ordered so that its normal points out of the tetrahedron.
struct Tri {
  std::array<index_t, 3> n{};
  index_t face = 0;
};

// Three tetrahedra filling a prism: the wall-side cap `bottom` (oriented out of the prism), the
// other cap `top` with top[i] the pillar partner of bottom[i], and per side i (the quadrilateral
// bottom[i] bottom[i+1] top[i+1] top[i]) its two triangles.
struct Prism {
  std::array<index_t, 3> cells{};
  std::array<index_t, 3> bottom{}, top{};
  std::array<Tri, 3> side_low{}, side_high{};  // two bottom nodes, two top nodes
  index_t top_face = 0;
};

enum class Fail : std::uint8_t {
  none,
  no_candidate,
  ambiguous,
  not_planar,
  not_aligned,
  not_layer,
  invalid,
  across_zones,
  not_tetrahedron
};

class Merger {
 public:
  Merger(const Mesh& mesh, const PrismOptions& options) : m_(mesh), o_(options) {}

  [[nodiscard]] bool is_tet(index_t c) const {
    const auto faces = m_.cell_faces(c);
    return faces.size() == 4 &&
           std::ranges::all_of(faces, [&](index_t f) { return m_.face_nodes(f).size() == 3; });
  }

  [[nodiscard]] Vec point(index_t p) const { return {m_.x()[at(p)], m_.y()[at(p)], m_.z()[at(p)]}; }

  [[nodiscard]] std::array<Tri, 4> outward(index_t c) const {
    std::array<Tri, 4> out{};
    std::size_t k = 0;
    for (const index_t f : m_.cell_faces(c)) {
      const auto n = m_.face_nodes(f);
      out[k] = m_.owner()[at(f)] == c ? Tri{.n = {n[0], n[1], n[2]}, .face = f}
                                      : Tri{.n = {n[0], n[2], n[1]}, .face = f};
      ++k;
    }
    return out;
  }

  // The cell across face f from c, or no_cell on the boundary.
  [[nodiscard]] index_t across(index_t f, index_t c) const {
    if (f >= m_.n_interior_faces()) return no_cell;
    return m_.owner()[at(f)] == c ? m_.neighbour()[at(f)] : m_.owner()[at(f)];
  }

  // The prism the tetrahedra {c0, c1, c2} fill with `cap` (an outer face of c0) as its wall-side
  // cap, if their outer faces are exactly the prism's pattern.
  [[nodiscard]] std::optional<Prism> pattern(const std::array<index_t, 3>& cells,
                                             const Tri& cap) const {
    std::vector<Tri> faces;
    std::vector<index_t> nodes;
    for (const index_t c : cells) {
      for (const Tri& t : outward(c)) {
        faces.push_back(t);
        nodes.insert(nodes.end(), t.n.begin(), t.n.end());
      }
    }
    std::ranges::sort(nodes);
    if (std::distance(nodes.begin(), std::ranges::unique(nodes).begin()) != 6) return std::nullopt;
    std::vector<Tri> outer;
    for (const Tri& t : faces) {
      const auto twice =
          std::ranges::count_if(faces, [&](const Tri& u) { return u.face == t.face; });
      if (twice == 1) outer.push_back(t);
    }
    if (outer.size() != 8) return std::nullopt;
    const auto in_cap = [&](index_t p) { return std::ranges::find(cap.n, p) != cap.n.end(); };
    const auto bottoms = [&](const Tri& t) { return std::ranges::count_if(t.n, in_cap); };
    Prism p{.cells = cells, .bottom = cap.n};
    std::vector<Tri> low, high;
    bool has_cap = false, has_top = false;
    for (const Tri& t : outer) {
      const auto b = bottoms(t);
      if (b == 3) {
        has_cap = has_cap || t.face == cap.face;
      } else if (b == 0) {
        if (has_top) return std::nullopt;
        has_top = true;
        p.top_face = t.face;
      } else {
        (b == 2 ? low : high).push_back(t);
      }
    }
    if (!has_cap || !has_top || low.size() != 3 || high.size() != 3) return std::nullopt;
    // Per side uv: its low triangle (u, v, x) and a high triangle (w, x, y), w in {u, v}, sharing
    // the diagonal w x; then the other of u, v has its pillar at x, and w at y. The high triangles
    // are assigned to the sides in the one way whose pillars agree: a high triangle of the next
    // side shares the pillar w x too, so the first match need not be the side's.
    std::array<std::optional<Tri>, 3> lows;
    for (std::size_t s = 0; s < 3; ++s) {
      const index_t u = p.bottom[s], v = p.bottom[(s + 1) % 3];
      for (const Tri& t : low) {
        if (std::ranges::find(t.n, u) != t.n.end() && std::ranges::find(t.n, v) != t.n.end())
          lows[s] = t;
      }
      if (!lows[s]) return std::nullopt;
    }
    std::array<std::size_t, 3> order{0, 1, 2};
    std::optional<Prism> found;
    do {
      std::array<index_t, 3> partner{no_cell, no_cell, no_cell};
      const auto set_partner = [&](index_t bottom, index_t top) {
        const auto i =
            static_cast<std::size_t>(std::ranges::find(p.bottom, bottom) - p.bottom.begin());
        if (partner[i] != no_cell && partner[i] != top) return false;
        partner[i] = top;
        return true;
      };
      bool fits = true;
      for (std::size_t s = 0; s < 3 && fits; ++s) {
        const index_t u = p.bottom[s], v = p.bottom[(s + 1) % 3];
        const Tri& l = *lows[s];
        const Tri& h = high[order[s]];
        const index_t x = *std::ranges::find_if(l.n, [&](index_t q) { return !in_cap(q); });
        const index_t w = *std::ranges::find_if(h.n, in_cap);
        if ((w != u && w != v) || std::ranges::find(h.n, x) == h.n.end()) {
          fits = false;
          break;
        }
        const index_t y = *std::ranges::find_if(h.n, [&](index_t q) { return q != w && q != x; });
        fits = set_partner(w == u ? v : u, x) && set_partner(w, y);
      }
      if (!fits) continue;
      std::array<index_t, 3> tops = partner;
      std::ranges::sort(tops);
      if (tops[0] == no_cell || std::ranges::adjacent_find(tops) != tops.end()) continue;
      if (found) return std::nullopt;  // two assignments: not a prism's pattern
      Prism q = p;
      q.top = partner;
      for (std::size_t s = 0; s < 3; ++s) {
        q.side_low[s] = *lows[s];
        q.side_high[s] = high[order[s]];
      }
      found = q;
    } while (std::ranges::next_permutation(order).found);
    if (!found) return std::nullopt;
    p = *found;
    return p;
  }

  // The geometry checks of a prism (PrismOptions; valid corners).
  [[nodiscard]] Fail geometry(const Prism& p) const {
    const double cos_tol = std::cos(o_.planarity_deg * std::numbers::pi / 180.0);
    for (std::size_t s = 0; s < 3; ++s) {
      const auto normal = [&](const Tri& t) {
        return cross(sub(point(t.n[1]), point(t.n[0])), sub(point(t.n[2]), point(t.n[0])));
      };
      const Vec a = normal(p.side_low[s]), b = normal(p.side_high[s]);
      if (dot(a, b) < cos_tol * length(a) * length(b)) return Fail::not_planar;
      // Convex: every corner turns the same way about the quadrilateral's normal.
      const std::array<index_t, 4> q{p.bottom[s], p.bottom[(s + 1) % 3], p.top[(s + 1) % 3],
                                     p.top[s]};
      const Vec n{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
      int positive = 0;
      for (std::size_t k = 0; k < 4; ++k) {
        const Vec e0 = sub(point(q[k]), point(q[(k + 3) % 4]));
        const Vec e1 = sub(point(q[(k + 1) % 4]), point(q[k]));
        positive += dot(cross(e0, e1), n) > 0.0 ? 1 : 0;
      }
      if (positive != 0 && positive != 4) return Fail::not_planar;
    }
    // The pillars point into the prism, against the bottom cap's outward normal.
    const Vec down = cross(sub(point(p.bottom[1]), point(p.bottom[0])),
                           sub(point(p.bottom[2]), point(p.bottom[0])));
    const double cos_pillar = std::cos(o_.pillar_deg * std::numbers::pi / 180.0);
    for (std::size_t i = 0; i < 3; ++i) {
      const Vec pillar = sub(point(p.top[i]), point(p.bottom[i]));
      if (-dot(pillar, down) < cos_pillar * length(pillar) * length(down)) return Fail::not_aligned;
    }
    double height = 0.0, shortest = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < 3; ++i) {
      height += length(sub(point(p.top[i]), point(p.bottom[i]))) / 3.0;
      shortest = std::min(shortest, length(sub(point(p.bottom[(i + 1) % 3]), point(p.bottom[i]))));
    }
    if (height > o_.aspect * shortest) return Fail::not_layer;
    // The bottom cap's normal points out of the prism: its corners' edge triple products are
    // negative there and positive at the top cap's (same cyclic order through the pillars).
    for (std::size_t i = 0; i < 3; ++i) {
      const std::size_t j = (i + 1) % 3, k = (i + 2) % 3;
      const auto corner = [&](const std::array<index_t, 3>& cap,
                              const std::array<index_t, 3>& other) {
        return dot(cross(sub(point(cap[j]), point(cap[i])), sub(point(cap[k]), point(cap[i]))),
                   sub(point(other[i]), point(cap[i])));
      };
      if (!(corner(p.bottom, p.top) < 0.0) || !(corner(p.top, p.bottom) > 0.0))
        return Fail::invalid;
    }
    return Fail::none;
  }

  // The prism on front face f, its wall-side cell c0, or why there is none.
  std::pair<std::optional<Prism>, Fail> prism_on(index_t f, index_t c0) const {
    if (c0 == no_cell || !is_tet(c0) || merged_[at(c0)] >= 0)
      return {std::nullopt, Fail::not_tetrahedron};
    Tri cap{};
    for (const Tri& t : outward(c0)) {
      if (t.face == f) cap = t;
    }
    // Neighbouring tetrahedra of c0, and of each of them.
    const auto tet_neighbours = [&](index_t c) {
      std::vector<index_t> out;
      for (const index_t g : m_.cell_faces(c)) {
        const index_t n = across(g, c);
        if (n != no_cell && is_tet(n)) out.push_back(n);
      }
      return out;
    };
    std::set<std::array<index_t, 3>> seen;
    std::vector<Prism> found;
    const auto first = tet_neighbours(c0);
    for (const index_t c1 : first) {
      std::vector<index_t> second = first;
      for (const index_t n : tet_neighbours(c1)) second.push_back(n);
      for (const index_t c2 : second) {
        if (c2 == c0 || c2 == c1) continue;
        std::array<index_t, 3> key{c0, std::min(c1, c2), std::max(c1, c2)};
        if (!seen.insert(key).second) continue;
        if (auto p = pattern({c0, c1, c2}, cap)) found.push_back(*p);
      }
    }
    if (found.empty()) return {std::nullopt, Fail::no_candidate};
    // Of the sets the pattern admits (a set leaning into a neighbouring column can have it too),
    // exactly one must be a valid layer cell; when none is, the reason is that of the set that
    // came furthest through the checks (zones, planarity, pillars, height, corners).
    std::vector<Prism> valid;
    Fail furthest = Fail::none;
    const auto stage = [](Fail why) {
      return why == Fail::across_zones  ? 0
             : why == Fail::not_planar  ? 1
             : why == Fail::not_aligned ? 2
             : why == Fail::not_layer   ? 3
                                        : 4;
    };
    for (const Prism& p : found) {
      const index_t zone = m_.cell_zone()[at(p.cells[0])];
      const Fail why =
          m_.cell_zone()[at(p.cells[1])] != zone || m_.cell_zone()[at(p.cells[2])] != zone
              ? Fail::across_zones
              : geometry(p);
      if (why == Fail::none) {
        valid.push_back(p);
      } else if (furthest == Fail::none || stage(why) > stage(furthest)) {
        furthest = why;
      }
    }
    if (valid.empty()) return {std::nullopt, furthest};
    if (valid.size() > 1) return {std::nullopt, Fail::ambiguous};
    for (const index_t c : valid.front().cells) {
      if (merged_[at(c)] >= 0) return {std::nullopt, Fail::not_tetrahedron};
    }
    return {valid.front(), Fail::none};
  }

  // Advances the layers from the walls, zones in order, faces in order, layer by layer.
  void advance(PrismReport& report) {
    merged_.assign(at(m_.n_cells()), -1);
    struct Front {
      index_t face, cell;
      std::int64_t layer;
    };
    std::deque<Front> front;
    const auto offsets = m_.boundary_zone_offsets();
    for (std::size_t z = 0; z < m_.face_zones().size(); ++z) {
      const FaceZone& zone = m_.face_zones()[z];
      if (!zone.boundary || (zone.kind != BcKind::wall && zone.kind != BcKind::thermal_wall))
        continue;
      for (index_t f = offsets[z]; f < offsets[z + 1]; ++f)
        front.push_back({f, m_.owner()[at(f)], 1});
    }
    while (!front.empty()) {
      const Front at_front = front.front();
      front.pop_front();
      auto [prism, fail] = prism_on(at_front.face, at_front.cell);
      if (!prism) {
        count(report, fail);
        continue;
      }
      const auto index = static_cast<index_t>(prisms_.size());
      for (const index_t c : prism->cells) merged_[at(c)] = index;
      report.layers = std::max(report.layers, at_front.layer);
      // The cell across the top cap, away from the prism: the next layer's.
      const index_t top = prism->top_face;
      index_t next = no_cell;
      for (const index_t c :
           {m_.owner()[at(top)], top < m_.n_interior_faces() ? m_.neighbour()[at(top)] : no_cell}) {
        if (c != no_cell && std::ranges::find(prism->cells, c) == prism->cells.end()) next = c;
      }
      prisms_.push_back(*prism);
      if (next != no_cell) front.push_back({top, next, at_front.layer + 1});
    }
  }

  static void count(PrismReport& r, Fail f) {
    switch (f) {
      case Fail::no_candidate:
        ++r.no_candidate;
        break;
      case Fail::ambiguous:
        ++r.ambiguous;
        break;
      case Fail::not_planar:
        ++r.not_planar;
        break;
      case Fail::not_aligned:
        ++r.not_aligned;
        break;
      case Fail::not_layer:
        ++r.not_layer;
        break;
      case Fail::invalid:
        ++r.invalid;
        break;
      case Fail::across_zones:
        ++r.across_zones;
        break;
      case Fail::not_tetrahedron:
        ++r.not_tetrahedron;
        break;
      case Fail::none:
        break;
    }
  }

  // The merged mesh: each prism one cell (numbered at its first tetrahedron), the faces inside a
  // prism gone, every other face kept: a side stays two triangles, whose union is exactly the
  // tetrahedra's (one quadrilateral face, the fan around its node mean, would not be).
  MeshInput result(PrismReport& report, std::vector<index_t>& old_to_new) const {
    MeshInput in;
    in.x.assign(m_.x().begin(), m_.x().end());
    in.y.assign(m_.y().begin(), m_.y().end());
    in.z.assign(m_.z().begin(), m_.z().end());
    in.face_zones.assign(m_.face_zones().begin(), m_.face_zones().end());
    in.cell_zones.assign(m_.cell_zones().begin(), m_.cell_zones().end());
    in.source = m_.source();
    in.face_node_offsets.push_back(0);
    old_to_new.assign(at(m_.n_cells()), no_cell);
    std::vector<index_t> prism_cell(prisms_.size(), no_cell);
    index_t n = 0;
    for (index_t c = 0; c < m_.n_cells(); ++c) {
      const index_t p = merged_[at(c)];
      if (p >= 0 && prism_cell[at(p)] != no_cell) {
        old_to_new[at(c)] = prism_cell[at(p)];
        continue;
      }
      if (p >= 0) prism_cell[at(p)] = n;
      old_to_new[at(c)] = n++;
      in.cell_zone.push_back(m_.cell_zone()[at(c)]);
      if (p < 0) ++(is_tet(c) ? report.tetrahedra : report.other_cells);
    }
    const auto new_cell = [&](index_t c) { return c == no_cell ? no_cell : old_to_new[at(c)]; };
    report.prisms = static_cast<std::int64_t>(prisms_.size());
    for (index_t f = 0; f < m_.n_faces(); ++f) {
      const index_t c0 = new_cell(m_.owner()[at(f)]);
      const index_t c1 = f < m_.n_interior_faces() ? new_cell(m_.neighbour()[at(f)]) : no_cell;
      if (c0 == c1) continue;
      const auto nodes = m_.face_nodes(f);
      in.face_nodes.insert(in.face_nodes.end(), nodes.begin(), nodes.end());
      in.face_node_offsets.push_back(static_cast<offset_t>(in.face_nodes.size()));
      in.face_cell0.push_back(c0);
      in.face_cell1.push_back(c1);
      in.face_zone.push_back(m_.face_zone()[at(f)]);
    }
    return in;
  }

  [[nodiscard]] const std::vector<Prism>& prisms() const { return prisms_; }

 private:
  const Mesh& m_;
  const PrismOptions& o_;
  std::vector<index_t> merged_;
  std::vector<Prism> prisms_;
};

// A boundary face as its zone and its nodes from the least, in order: the same face whatever
// node it starts from.
std::vector<index_t> boundary_key(const Mesh& m, index_t f) {
  const auto n = m.face_nodes(f);
  const auto first = std::ranges::min_element(n) - n.begin();
  std::vector<index_t> key{m.face_zone()[at(f)]};
  for (std::size_t k = 0; k < n.size(); ++k)
    key.push_back(n[(static_cast<std::size_t>(first) + k) % n.size()]);
  return key;
}

std::vector<std::vector<index_t>> boundary_keys(const Mesh& m) {
  std::vector<std::vector<index_t>> keys;
  for (index_t f = m.n_interior_faces(); f < m.n_faces(); ++f) keys.push_back(boundary_key(m, f));
  std::ranges::sort(keys);
  return keys;
}

}  // namespace

PrismMerge merge_layer_prisms(const Mesh& mesh, const PrismOptions& options) {
  if (mesh.n_owned_cells() != mesh.n_cells() || !mesh.periodic_pairs().empty()) {
    throw Error(ErrorCode::io_unsupported,
                "merging prisms needs an unpartitioned mesh without periodic pairs");
  }
  PrismMerge out{.mesh = {}, .report = {}};
  Merger merger(mesh, options);
  merger.advance(out.report);
  std::vector<index_t> old_to_new;
  out.mesh = assemble(merger.result(out.report, old_to_new));

  // Invariants: the same nodes, the same boundary faces in the same zones, and each cell the
  // volume of the cells it merges, within 1e-12 relative (so the total too).
  const Mesh& r = out.mesh;
  if (!std::ranges::equal(r.x(), mesh.x()) || !std::ranges::equal(r.y(), mesh.y()) ||
      !std::ranges::equal(r.z(), mesh.z()))
    throw std::logic_error("merging prisms changed the nodes");
  if (boundary_keys(r) != boundary_keys(mesh))
    throw std::logic_error("merging prisms changed the boundary faces");
  const Geometry before = compute_geometry(mesh);
  const Geometry after = compute_geometry(r);  // E0120, E0121
  std::vector<double> merged(at(r.n_cells()), 0.0);
  for (index_t c = 0; c < mesh.n_cells(); ++c)
    merged[at(old_to_new[at(c)])] += before.cell_volume[at(c)];
  double difference = 0.0, total = 0.0;
  for (index_t c = 0; c < r.n_cells(); ++c) {
    const double d = std::abs(after.cell_volume[at(c)] - merged[at(c)]);
    if (d > 1e-12 * merged[at(c)])
      throw std::logic_error("merged cell " + std::to_string(c) + " has another volume");
    difference += d;
    total += merged[at(c)];
  }
  if (difference > 1e-12 * total) throw std::logic_error("merging prisms changed the volume");
  const auto closure = cell_closure_error(r, after);
  if (!closure.empty() && *std::ranges::max_element(closure) > 1e-12)
    throw std::logic_error("a merged cell is not closed");
  return out;
}

}  // namespace autan
