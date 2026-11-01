#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/wall_distance.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Vec = std::array<double, 3>;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Distance from p to triangle abc, computed independently of the library:
// the plane distance when the projection falls inside, else the nearest of
// the three edges as clamped segments.
double triangle_distance(const Vec& p, const Vec& a, const Vec& b, const Vec& c) {
  const auto sub = [](const Vec& u, const Vec& v) {
    return Vec{u[0] - v[0], u[1] - v[1], u[2] - v[2]};
  };
  const auto dot = [](const Vec& u, const Vec& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const auto cross = [](const Vec& u, const Vec& v) {
    return Vec{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
  };
  const auto segment = [&](const Vec& u, const Vec& v) {
    const Vec d = sub(v, u);
    const double t = std::clamp(dot(sub(p, u), d) / dot(d, d), 0.0, 1.0);
    const Vec q{u[0] + t * d[0], u[1] + t * d[1], u[2] + t * d[2]};
    const Vec e = sub(p, q);
    return std::sqrt(dot(e, e));
  };
  const Vec n = cross(sub(b, a), sub(c, a));
  const double nn = std::sqrt(dot(n, n));
  const bool inside = dot(cross(sub(b, a), sub(p, a)), n) >= 0.0 &&
                      dot(cross(sub(c, b), sub(p, b)), n) >= 0.0 &&
                      dot(cross(sub(a, c), sub(p, c)), n) >= 0.0;
  if (inside) return std::abs(dot(sub(p, a), n)) / nn;
  return std::min({segment(a, b), segment(b, c), segment(c, a)});
}

index_t zone(const autan::Mesh& mesh, const char* name) {
  for (std::size_t z = 0; z < mesh.face_zones().size(); ++z) {
    if (mesh.face_zones()[z].name == name) return static_cast<index_t>(z);
  }
  FAIL("no zone " << name);
  return -1;
}

}  // namespace

TEST_CASE("Wall distance in a box with two walls is the distance to the nearer plane", "[wall]") {
  // Walls ymin (y = 0) and zmax (z = 1): every centroid projects inside both
  // squares, so d = min(y, 1 - z).
  using autan::test::BoxFill;
  const BoxFill fill = GENERATE(BoxFill::hexa, BoxFill::tetra, BoxFill::hybrid);
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::box(4, fill, 0.2)));
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::array<index_t, 2> walls{zone(mesh, "ymin"), zone(mesh, "zmax")};
  const auto d = autan::wall_distance(mesh, g, walls);
  REQUIRE(d.size() == at(mesh.n_cells()));
  for (std::size_t c = 0; c < d.size(); ++c) {
    CHECK_THAT(d[c], WithinAbs(std::min(g.cell_centroid_y[c], 1.0 - g.cell_centroid_z[c]), 1e-14));
  }
}

TEST_CASE("Wall distance sees the nearest wall across a periodic boundary", "[wall][periodic]") {
  // Unit box in 4^3 hexahedra, periodic in x. Only the ymin faces with
  // x <= 1/4 are wall: its copies are [k, k + 1/4] x {0} x [0, 1], so
  // d = min over k of hypot(distance from x to [k, k + 1/4], y).
  autan::CellwiseMesh cells = autan::test::box(4, autan::test::BoxFill::hexa, 0.2);
  cells.face_zones.push_back(
      {.name = "wall", .bc_type = "", .kind = autan::BcKind::wall, .provenance = {}});
  for (std::size_t f = 0; f < cells.labelled_face_zone.size(); ++f) {
    if (cells.labelled_face_zone[f] != 2) continue;  // ymin
    bool low = true;
    for (auto k = cells.labelled_face_node_offsets[f]; k < cells.labelled_face_node_offsets[f + 1];
         ++k)
      low = low && cells.x[at(cells.labelled_face_nodes[static_cast<std::size_t>(k)])] <= 0.25;
    if (low) cells.labelled_face_zone[f] = 6;
  }
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(std::move(cells)));
  const autan::Geometry g = autan::compute_geometry(mesh);
  const std::array<index_t, 1> walls{zone(mesh, "wall")};
  const auto d = autan::wall_distance(mesh, g, walls);
  for (std::size_t c = 0; c < d.size(); ++c) {
    const double x = g.cell_centroid_x[c], y = g.cell_centroid_y[c];
    double expected = std::numeric_limits<double>::infinity();
    for (const double k : {-1.0, 0.0, 1.0}) {
      const double dx = x < k ? k - x : (x > k + 0.25 ? x - k - 0.25 : 0.0);
      expected = std::min(expected, std::hypot(dx, y));
    }
    CHECK_THAT(d[c], WithinAbs(expected, 1e-14));
  }
}

TEST_CASE("The nearest wall point may be a vertex, an edge or a face", "[wall]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(2, autan::test::BoxFill::tetra)));
  const std::array<index_t, 1> walls{zone(mesh, "ymin")};  // the square [0,1] x {0} x [0,1]
  const autan::WallSurface surface(mesh, walls);
  CHECK_THAT(surface.distance({0.3, 0.7, 0.4}), WithinAbs(0.7, 1e-15));                // face
  CHECK_THAT(surface.distance({-1.0, -1.0, 0.5}), WithinAbs(std::sqrt(2.0), 1e-15));   // edge
  CHECK_THAT(surface.distance({-1.0, -1.0, -1.0}), WithinAbs(std::sqrt(3.0), 1e-15));  // corner
  CHECK(std::isnan(surface.distance({std::nan(""), 0.0, 0.0})));
  CHECK(autan::WallSurface(mesh, std::array<index_t, 0>{}).distance({0.5, 0.5, 0.5}) ==
        std::numeric_limits<double>::infinity());
  for (const index_t bad : {index_t{-1}, static_cast<index_t>(mesh.face_zones().size())}) {
    try {
      (void)autan::WallSurface(mesh, std::array<index_t, 1>{bad});
      FAIL("expected E0102");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::mesh_index_out_of_range);
    }
  }
}

TEST_CASE("The search tree returns the brute-force nearest distance", "[wall]") {
  // All six sides of a perturbed hybrid box are wall; points inside and
  // outside it, against every fan triangle checked one by one.
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(4, autan::test::BoxFill::hybrid, 0.2)));
  std::vector<index_t> walls;
  for (const char* side : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    walls.push_back(zone(mesh, side));
  const autan::WallSurface surface(mesh, walls);
  std::vector<std::array<Vec, 3>> triangles;
  const auto off = mesh.boundary_zone_offsets();
  for (const index_t z : walls) {
    for (index_t f = off[at(z)]; f < off[at(z) + 1]; ++f) {
      const auto nodes = mesh.face_nodes(f);
      Vec m{};
      for (const index_t n : nodes) {
        m[0] += mesh.x()[at(n)] / static_cast<double>(nodes.size());
        m[1] += mesh.y()[at(n)] / static_cast<double>(nodes.size());
        m[2] += mesh.z()[at(n)] / static_cast<double>(nodes.size());
      }
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto p = at(nodes[i]), q = at(nodes[(i + 1) % nodes.size()]);
        triangles.push_back({m, Vec{mesh.x()[p], mesh.y()[p], mesh.z()[p]},
                             Vec{mesh.x()[q], mesh.y()[q], mesh.z()[q]}});
      }
    }
  }
  CHECK(surface.triangles() == triangles.size());
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-0.5, 1.5);
  for (int i = 0; i < 2000; ++i) {
    const Vec p{u(rng), u(rng), u(rng)};
    double expected = std::numeric_limits<double>::infinity();
    for (const auto& t : triangles)
      expected = std::min(expected, triangle_distance(p, t[0], t[1], t[2]));
    CHECK_THAT(surface.distance(p), WithinAbs(expected, 1e-13));
  }
}
