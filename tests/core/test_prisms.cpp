#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "autan/core/cellwise.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/prisms.hpp"

using autan::index_t;

namespace {

// A boundary layer as VGRID builds one (Pirzadeh 1996): the unit square at z = 0, a wall, cut
// into n^2 squares of two triangles each (diagonal from (i, j) to (i + 1, j + 1)), extruded in
// `layers` thin layers growing by 1.2 from 0.01 / n, then one layer 2 / n high, as wide as tall
// cells are not. Each prism is split into three tetrahedra with the diagonal of each side
// quadrilateral at its lowest node (Dompierre et al., IJNME 45, 1999), so neighbours agree.
// `moved`, when set, shifts the node (i, j) of z-level q by a tenth of a square along x.
struct Layered {
  int n = 4;
  int layers = 5;
  struct Moved {
    int i, j, q;
  };
  std::optional<Moved> moved;
};

autan::CellwiseMesh layered(const Layered& g) {
  autan::CellwiseMesh mesh;
  const int side = g.n + 1;
  std::vector<double> z{0.0};
  double h = 0.01 / g.n;
  for (int k = 0; k < g.layers; ++k, h *= 1.2) z.push_back(z.back() + h);
  z.push_back(z.back() + 2.0 / g.n);
  const auto node = [&](int i, int j, int q) {
    return static_cast<index_t>((q * side + j) * side + i);
  };
  for (std::size_t q = 0; q < z.size(); ++q) {
    for (int j = 0; j < side; ++j) {
      for (int i = 0; i < side; ++i) {
        const bool shifted =
            g.moved && g.moved->i == i && g.moved->j == j && g.moved->q == static_cast<int>(q);
        mesh.x.push_back(static_cast<double>(i) / g.n + (shifted ? 0.1 / g.n : 0.0));
        mesh.y.push_back(static_cast<double>(j) / g.n);
        mesh.z.push_back(z[q]);
      }
    }
  }
  mesh.face_zones.push_back(autan::FaceZone{.name = "wall",
                                            .bc_type = "",
                                            .kind = autan::BcKind::wall,
                                            .boundary = true,
                                            .provenance = {}});
  mesh.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  const auto point = [&](index_t p) {
    const auto s = static_cast<std::size_t>(p);
    return std::array<double, 3>{mesh.x[s], mesh.y[s], mesh.z[s]};
  };
  const auto tet = [&](std::array<index_t, 4> t) {
    const auto a = point(t[0]), b = point(t[1]), c = point(t[2]), d = point(t[3]);
    const std::array<double, 3> u{b[0] - a[0], b[1] - a[1], b[2] - a[2]},
        v{c[0] - a[0], c[1] - a[1], c[2] - a[2]}, w{d[0] - a[0], d[1] - a[1], d[2] - a[2]};
    const double volume = u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
                          u[2] * (v[0] * w[1] - v[1] * w[0]);
    if (volume < 0.0) std::swap(t[2], t[3]);
    mesh.add_cell(autan::CellShape::tetra, t, 0);
  };
  for (std::size_t q = 0; q + 1 < z.size(); ++q) {
    for (int j = 0; j < g.n; ++j) {
      for (int i = 0; i < g.n; ++i) {
        const int k = static_cast<int>(q);
        for (const auto& tri :
             {std::array<std::array<int, 2>, 3>{{{i, j}, {i + 1, j}, {i + 1, j + 1}}},
              std::array<std::array<int, 2>, 3>{{{i, j}, {i + 1, j + 1}, {i, j + 1}}}}) {
          std::array<index_t, 6> p{};
          for (std::size_t s = 0; s < 3; ++s) {
            p[s] = node(tri[s][0], tri[s][1], k);
            p[s + 3] = node(tri[s][0], tri[s][1], k + 1);
          }
          if (q == 0) mesh.add_labelled_face(std::array<index_t, 3>{p[0], p[1], p[2]}, 0);
          // Dompierre et al. 1999: the least node at position 0, by flipping and rotating.
          auto m = static_cast<std::size_t>(std::ranges::min_element(p) - p.begin());
          if (m >= 3) {
            p = {p[3], p[4], p[5], p[0], p[1], p[2]};
            m -= 3;
          }
          std::array<index_t, 6> v{};
          for (std::size_t s = 0; s < 3; ++s) {
            v[s] = p[(s + m) % 3];
            v[s + 3] = p[(s + m) % 3 + 3];
          }
          if (std::min(v[1], v[5]) < std::min(v[2], v[4])) {
            tet({v[0], v[1], v[2], v[5]});
            tet({v[0], v[1], v[5], v[4]});
          } else {
            tet({v[0], v[1], v[2], v[4]});
            tet({v[0], v[4], v[2], v[5]});
          }
          tet({v[0], v[4], v[5], v[3]});
        }
      }
    }
  }
  return mesh;
}

autan::Mesh assembled(const Layered& g) {
  return autan::assemble(autan::to_face_based(layered(g)));
}

// Wall triangles with an edge on the square's border: their prisms keep a boundary side split.
std::int64_t border_triangles(int n) {
  std::int64_t count = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      count += (j == 0 || i == n - 1) ? 1 : 0;  // (i, j) (i+1, j) (i+1, j+1)
      count += (i == 0 || j == n - 1) ? 1 : 0;  // (i, j) (i+1, j+1) (i, j+1)
    }
  }
  return count;
}

}  // namespace

TEST_CASE("Boundary-layer tetrahedra merge into prisms, layer by layer from the wall", "[prisms]") {
  const Layered g{.n = 4, .layers = 5, .moved = std::nullopt};
  const autan::Mesh mesh = assembled(g);
  const auto merged = autan::merge_layer_prisms(mesh);
  const auto& r = merged.report;
  const std::int64_t columns = 2 * g.n * g.n;
  CAPTURE(r.no_candidate, r.ambiguous, r.not_planar, r.not_layer, r.invalid, r.across_zones,
          r.not_tetrahedron);
  // Every thin layer merges; the top layer, twice as high as wide, is left as tetrahedra.
  CHECK(r.prisms + r.split_prisms == columns * g.layers);
  CHECK(r.split_prisms == border_triangles(g.n) * g.layers);
  CHECK(r.tetrahedra == 3 * columns);
  CHECK(r.other_cells == 0);
  CHECK(r.layers == g.layers);
  CHECK(r.not_layer == columns);
  CHECK(r.no_candidate + r.ambiguous + r.not_planar + r.not_aligned + r.invalid + r.across_zones +
            r.not_tetrahedron ==
        0);
  CHECK(merged.mesh.n_cells() == mesh.n_cells() - 2 * columns * g.layers);
  // Merged cells are prisms: two caps and three sides, quadrilaterals between two prisms.
  std::int64_t five = 0;
  for (index_t c = 0; c < merged.mesh.n_cells(); ++c) five += merged.mesh.cell_faces(c).size() == 5;
  CHECK(five == r.prisms);
  // The volume of the box, before and after.
  const auto total = [](const autan::Mesh& m) {
    double t = 0.0;
    for (const double v : autan::compute_geometry(m).cell_volume) t += v;
    return t;
  };
  CHECK(std::abs(total(merged.mesh) - total(mesh)) <= 1e-12 * total(mesh));
}

TEST_CASE("A column stops at its first prism beyond the planarity tolerance", "[prisms]") {
  // The node (2, 2) of z-level 3 moved along x: the prisms around it in layer 2, the first with
  // that node, have a side far from planar; their columns keep layers 0 and 1, the others all.
  const Layered g{.n = 4, .layers = 5, .moved = Layered::Moved{.i = 2, .j = 2, .q = 3}};
  const auto merged = autan::merge_layer_prisms(assembled(g));
  const auto& r = merged.report;
  // Wall triangles holding (2, 2), with the diagonal (i, j)-(i+1, j+1): both of squares (1, 1)
  // and (2, 2), one of (1, 2) and one of (2, 1).
  const std::int64_t around = 6, columns = 2 * g.n * g.n;
  CHECK(r.prisms + r.split_prisms == columns * g.layers - around * (g.layers - 2));
  // Every column stops once: the six at layer 2, the others at the top layer. (Which check a
  // column fails is reported for the set that came furthest, informational here.)
  CHECK(r.not_planar + r.not_aligned + r.not_layer + r.invalid == columns);
  CHECK(r.no_candidate + r.ambiguous + r.across_zones + r.not_tetrahedron == 0);
  CHECK(r.layers == g.layers);
}

TEST_CASE("Tolerances decide which layers merge", "[prisms]") {
  // An aspect of 4 lets the top layer (height 2 / n against edges of 1 / n) merge too, with
  // pillars within 20 degrees of the normal: an oblique prism of that layer leaning into the
  // next column leans atan(1 / 2) = 26.6 degrees.
  const Layered g{.n = 3, .layers = 2, .moved = std::nullopt};
  const autan::Mesh mesh = assembled(g);
  const auto wide =
      autan::merge_layer_prisms(mesh, {.planarity_deg = 5.0, .aspect = 4.0, .pillar_deg = 20.0});
  CHECK(wide.report.prisms + wide.report.split_prisms == 2 * g.n * g.n * (g.layers + 1));
  CHECK(wide.report.tetrahedra == 0);
  // A mesh without wall zone merges nothing.
  auto bare = layered(g);
  bare.face_zones[0].kind = autan::BcKind::unspecified;
  const auto none = autan::merge_layer_prisms(autan::assemble(autan::to_face_based(bare)));
  CHECK(none.report.prisms + none.report.split_prisms == 0);
  CHECK(none.mesh.n_cells() == mesh.n_cells());
}
