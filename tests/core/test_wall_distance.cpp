#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "autan/core/wall_distance.hpp"
#include "fixtures.hpp"

#ifdef AUTAN_HAVE_CUDA
#include "wall_distance_device_test.hpp"
#endif

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Vec = std::array<double, 3>;

namespace {

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

// Distance from p to triangle abc, computed independently of the library:
// Ericson, Real-Time Collision Detection (2005), section 5.1.5, closest point
// by Voronoi regions (well conditioned for the triangles of these tests).
double triangle_distance(const Vec& p, const Vec& a, const Vec& b, const Vec& c) {
  const auto sub = [](const Vec& u, const Vec& v) {
    return Vec{u[0] - v[0], u[1] - v[1], u[2] - v[2]};
  };
  const auto dot = [](const Vec& u, const Vec& v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  };
  const auto along = [](const Vec& o, double t, const Vec& d) {
    return Vec{o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
  };
  const auto dist = [&](const Vec& q) {
    const Vec e = sub(p, q);
    return std::sqrt(dot(e, e));
  };
  const Vec ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return dist(a);
  const Vec bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return dist(b);
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return dist(along(a, d1 / (d1 - d3), ab));
  const Vec cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return dist(c);
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return dist(along(a, d2 / (d2 - d6), ac));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    return dist(along(b, (d4 - d3) / ((d4 - d3) + (d5 - d6)), sub(c, b)));
  const double v = vb / (va + vb + vc), w = vc / (va + vb + vc);
  return dist(Vec{a[0] + v * ab[0] + w * ac[0], a[1] + v * ab[1] + w * ac[1],
                  a[2] + v * ab[2] + w * ac[2]});
}

// The box of `cells` with every node mapped by f.
autan::Mesh mapped(autan::CellwiseMesh cells, const std::function<Vec(const Vec&)>& f) {
  for (std::size_t i = 0; i < cells.x.size(); ++i) {
    const Vec q = f({cells.x[i], cells.y[i], cells.z[i]});
    cells.x[i] = q[0];
    cells.y[i] = q[1];
    cells.z[i] = q[2];
  }
  return autan::assemble(autan::to_face_based(std::move(cells)));
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

TEST_CASE("Wall distance holds at any scale and position", "[wall]") {
  // The unit box mapped by x -> s x + o, wall ymin: d = (y - o_y) exactly.
  struct Case {
    double scale;
    Vec origin;
  };
  const Case c = GENERATE(Case{1e-9, {0, 0, 0}}, Case{1e80, {0, 0, 0}}, Case{1e-80, {0, 0, 0}},
                          Case{1.0, {0, 1e7, 0}});
  CAPTURE(c.scale, c.origin[1]);
  const autan::Mesh mesh =
      mapped(autan::test::box(3, autan::test::BoxFill::tetra, 0.2), [&](const Vec& q) {
        return Vec{c.scale * q[0] + c.origin[0], c.scale * q[1] + c.origin[1],
                   c.scale * q[2] + c.origin[2]};
      });
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto d = autan::wall_distance(mesh, g, std::array<index_t, 1>{zone(mesh, "ymin")});
  for (std::size_t i = 0; i < d.size(); ++i) {
    const double expected = g.cell_centroid_y[i] - c.origin[1];
    CHECK_THAT(d[i], WithinAbs(expected, 1e-12 * c.scale + 1e-15 * std::abs(c.origin[1])));
  }
  // A thin box, 1 x 2e-9 x 1e-8: its wall triangles are slivers.
  const autan::Mesh thin = mapped(autan::test::box(2, autan::test::BoxFill::hexa),
                                  [](const Vec& q) { return Vec{q[0], 2e-9 * q[1], 1e-8 * q[2]}; });
  const autan::WallSurface surface(thin, std::array<index_t, 1>{zone(thin, "ymin")});
  CHECK_THAT(surface.distance({0.4, 1e-9, 3e-9}), WithinAbs(1e-9, 1e-21));
  // A tiny wall seen from far away, a huge one from very close.
  for (const auto& [length, height] : {std::pair{1e-80, 1e80}, std::pair{1e80, 1e-100}}) {
    const autan::Mesh box =
        mapped(autan::test::box(1, autan::test::BoxFill::hexa),
               [&](const Vec& q) { return Vec{length * q[0], length * q[1], length * q[2]}; });
    const autan::WallSurface wall(box, std::array<index_t, 1>{zone(box, "ymin")});
    CHECK_THAT(wall.distance({0.4 * length, height, 0.3 * length}),
               Catch::Matchers::WithinRel(height, 1e-12));
  }
}

TEST_CASE("Wall faces with hanging nodes have degenerate fan triangles", "[wall]") {
  // hanging_node_box: [0,2] x [0,1]^2, its unassigned sides are y = 0, 1,
  // z = 0, 1 and x = 2; the big cell's side faces carry collinear hanging
  // nodes, so some fan triangles have zero area.
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto d = autan::wall_distance(mesh, g, std::array<index_t, 1>{zone(mesh, "unassigned")});
  for (std::size_t i = 0; i < d.size(); ++i) {
    const double y = g.cell_centroid_y[i], z = g.cell_centroid_z[i], x = g.cell_centroid_x[i];
    CHECK_THAT(d[i], WithinAbs(std::min({y, 1 - y, z, 1 - z, 2 - x}), 1e-14));
  }
}

TEST_CASE("A periodic wall is found however far the domain shears", "[wall][periodic]") {
  // The unit box sheared by x -> x + 10 y, periodic in x by (1, 0, 0), wall
  // ymin: the nearest wall point lies straight below, up to 10 periods away.
  autan::CellwiseMesh cells = autan::test::box(3, autan::test::BoxFill::hexa);
  cells.face_zones[0].kind = autan::BcKind::periodic;
  cells.face_zones[1].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  const autan::Mesh mesh =
      mapped(std::move(cells), [](const Vec& q) { return Vec{q[0] + 10 * q[1], q[1], q[2]}; });
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto d = autan::wall_distance(mesh, g, std::array<index_t, 1>{zone(mesh, "ymin")});
  for (std::size_t i = 0; i < d.size(); ++i)
    CHECK_THAT(d[i], WithinAbs(g.cell_centroid_y[i], 1e-14));
}

TEST_CASE("A periodic wall is found along oblique periodic directions", "[wall][periodic]") {
  // One hexahedron mapped by (x + y, 0.01 y + z, z): periodic by (1, 0, 0)
  // and (1, 0.01, 0), whose copies of the zmin wall tile the plane z = 0.
  // Reaching it from the centroid takes about 50 of each period.
  autan::CellwiseMesh cells = autan::test::box(1, autan::test::BoxFill::hexa);
  for (const int z : {0, 1, 2, 3}) cells.face_zones[at(z)].kind = autan::BcKind::periodic;
  cells.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
  cells.periodic.push_back({.zone_a = 2, .zone_b = 3, .translation = {1.0, 0.01, 0.0}});
  const autan::Mesh mesh = mapped(
      std::move(cells), [](const Vec& q) { return Vec{q[0] + q[1], 0.01 * q[1] + q[2], q[2]}; });
  const autan::Geometry g = autan::compute_geometry(mesh);
  const auto d = autan::wall_distance(mesh, g, std::array<index_t, 1>{zone(mesh, "zmin")});
  CHECK_THAT(d[0], WithinAbs(g.cell_centroid_z[0], 1e-14));
}

TEST_CASE("Wall inputs the surface cannot use are coded errors", "[wall]") {
  autan::CellwiseMesh cells = autan::test::box(2, autan::test::BoxFill::hexa);
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cells));
  const auto require_code = [](autan::ErrorCode code, const std::function<void()>& f) {
    try {
      f();
      FAIL("expected " << autan::error_code_string(code));
    } catch (const autan::Error& e) {
      CHECK(e.code() == code);
    }
  };
  require_code(autan::ErrorCode::mesh_inconsistent_sizes, [&] {
    (void)autan::wall_distance(mesh, autan::Geometry{}, std::array<index_t, 1>{zone(mesh, "ymin")});
  });
  for (std::size_t i = 0; i < cells.y.size(); ++i) {
    if (cells.y[i] == 0.0 && cells.x[i] == 0.5)
      cells.y[i] = std::numeric_limits<double>::infinity();
  }
  const autan::Mesh broken = autan::assemble(autan::to_face_based(cells));
  require_code(autan::ErrorCode::geometry_degenerate_face, [&] {
    (void)autan::WallSurface(broken, std::array<index_t, 1>{zone(broken, "ymin")});
  });
  // A wall so wide that its node mean overflows.
  const autan::Mesh wide = mapped(autan::test::box(1, autan::test::BoxFill::hexa),
                                  [](const Vec& q) { return Vec{1e308 * q[0], q[1], q[2]}; });
  require_code(autan::ErrorCode::geometry_degenerate_face,
               [&] { (void)autan::WallSurface(wide, std::array<index_t, 1>{zone(wide, "ymin")}); });
}

#ifdef AUTAN_HAVE_CUDA
TEST_CASE("The device computes the host's wall distances bit for bit", "[wall][gpu]") {
  // The same query compiled for the device (wall_distance_query.hpp), FMA
  // contraction off on both sides: every distance has the host's bits, on
  // surfaces that exercise each path of the query.
  const auto same = [](const autan::WallSurface& surface, const std::vector<Vec>& points) {
    const auto device = autan::test::device_distances(surface, points);
    REQUIRE(device.size() == points.size());
    std::size_t differ = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double host = surface.distance(points[i]);
      const bool equal = std::isnan(host) ? std::isnan(device[i])
                                          : std::bit_cast<std::uint64_t>(host) ==
                                                std::bit_cast<std::uint64_t>(device[i]);
      if (!equal) ++differ;
    }
    CHECK(differ == 0);
  };
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-0.5, 1.5);
  const auto random_points = [&](std::size_t n, double s = 1.0) {
    std::vector<Vec> p(n);
    for (auto& q : p) q = {s * u(rng), s * u(rng), s * u(rng)};
    return p;
  };
  SECTION("every side of a hybrid box, points inside and out, and points not finite") {
    const autan::Mesh mesh = autan::assemble(
        autan::to_face_based(autan::test::box(4, autan::test::BoxFill::hybrid, 0.2)));
    std::vector<index_t> walls;
    for (const char* side : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
      walls.push_back(zone(mesh, side));
    auto points = random_points(5000);
    constexpr double inf = std::numeric_limits<double>::infinity();
    points.push_back({std::nan(""), 0.0, 0.0});
    points.push_back({inf, 0.5, 0.5});
    points.push_back({0.5, -inf, 0.5});
    same(autan::WallSurface(mesh, walls), points);
    same(autan::WallSurface(mesh, std::array<index_t, 0>{}), random_points(10));
  }
  SECTION("faces with hanging nodes, so fan triangles of zero area") {
    const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
    same(autan::WallSurface(mesh, std::array<index_t, 1>{zone(mesh, "unassigned")}),
         random_points(2000, 2.0));
  }
  SECTION("extreme scales") {
    for (const double scale : {1e-80, 1e80}) {
      const autan::Mesh box =
          mapped(autan::test::box(2, autan::test::BoxFill::tetra, 0.2),
                 [&](const Vec& q) { return Vec{scale * q[0], scale * q[1], scale * q[2]}; });
      same(autan::WallSurface(box, std::array<index_t, 1>{zone(box, "ymin")}),
           random_points(500, scale));
    }
  }
  SECTION("periodic walls, sheared and along oblique directions") {
    autan::CellwiseMesh sheared = autan::test::box(3, autan::test::BoxFill::hexa);
    sheared.face_zones[0].kind = autan::BcKind::periodic;
    sheared.face_zones[1].kind = autan::BcKind::periodic;
    sheared.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
    const autan::Mesh a =
        mapped(std::move(sheared), [](const Vec& q) { return Vec{q[0] + 10 * q[1], q[1], q[2]}; });
    same(autan::WallSurface(a, std::array<index_t, 1>{zone(a, "ymin")}), random_points(500, 4.0));
    autan::CellwiseMesh oblique = autan::test::box(1, autan::test::BoxFill::hexa);
    for (const int z : {0, 1, 2, 3}) oblique.face_zones[at(z)].kind = autan::BcKind::periodic;
    oblique.periodic.push_back({.zone_a = 0, .zone_b = 1, .translation = {1.0, 0.0, 0.0}});
    oblique.periodic.push_back({.zone_a = 2, .zone_b = 3, .translation = {1.0, 0.01, 0.0}});
    const autan::Mesh b = mapped(std::move(oblique), [](const Vec& q) {
      return Vec{q[0] + q[1], 0.01 * q[1] + q[2], q[2]};
    });
    same(autan::WallSurface(b, std::array<index_t, 1>{zone(b, "zmin")}), random_points(200));
  }
}
#endif
