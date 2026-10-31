#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "autan/core/error.hpp"
#include "autan/core/geometry.hpp"
#include "fixtures.hpp"

using autan::index_t;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double closure_tolerance = 1e-12;

std::size_t at(index_t i) { return static_cast<std::size_t>(i); }

double max_closure(const autan::Mesh& mesh, const autan::Geometry& g) {
  const auto error = autan::cell_closure_error(mesh, g);
  return *std::max_element(error.begin(), error.end());
}

double total_volume(const autan::Geometry& g) {
  return std::accumulate(g.cell_volume.begin(), g.cell_volume.end(), 0.0);
}

}  // namespace

TEST_CASE("Reference cells have the analytic volume and centroid", "[geometry]") {
  const auto shape = GENERATE(autan::CellShape::tetra, autan::CellShape::pyramid,
                              autan::CellShape::prism, autan::CellShape::hexa);
  const std::array<std::array<double, 3>, 3> a{
      {{1.7, 0.3, -0.2}, {0.1, 0.9, 0.4}, {-0.3, 0.2, 1.3}}};
  const std::array<double, 3> b{0.25, -1.5, 3.0};
  const auto ref = autan::test::affine_cell(shape, a, b);
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(ref.mesh));
  const autan::Geometry g = autan::compute_geometry(mesh);

  REQUIRE(mesh.n_cells() == 1);
  CHECK_THAT(g.cell_volume[0], WithinRel(ref.volume, 1e-14));
  CHECK_THAT(g.cell_centroid_x[0], WithinAbs(ref.centroid[0], 1e-14));
  CHECK_THAT(g.cell_centroid_y[0], WithinAbs(ref.centroid[1], 1e-14));
  CHECK_THAT(g.cell_centroid_z[0], WithinAbs(ref.centroid[2], 1e-14));
  CHECK(max_closure(mesh, g) < closure_tolerance);
}

TEST_CASE("A warped face's area vector is the loop integral of its boundary", "[geometry]") {
  // Any surface spanning a closed loop has vector area 1/2 sum p_i x p_{i+1};
  // lifting one hex corner makes three of its faces non-planar.
  auto ref = autan::test::affine_cell(autan::CellShape::hexa, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
                                      {0, 0, 0});
  ref.mesh.z[6] = 1.4;
  ref.mesh.x[6] = 1.2;
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(ref.mesh));
  const autan::Geometry g = autan::compute_geometry(mesh);

  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    const auto nodes = mesh.face_nodes(f);
    std::array<double, 3> s{0, 0, 0};
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const std::size_t p = at(nodes[i]);
      const std::size_t q = at(nodes[(i + 1) % nodes.size()]);
      s[0] += 0.5 * (mesh.y()[p] * mesh.z()[q] - mesh.z()[p] * mesh.y()[q]);
      s[1] += 0.5 * (mesh.z()[p] * mesh.x()[q] - mesh.x()[p] * mesh.z()[q]);
      s[2] += 0.5 * (mesh.x()[p] * mesh.y()[q] - mesh.y()[p] * mesh.x()[q]);
    }
    CHECK_THAT(g.face_area_x[at(f)], WithinAbs(s[0], 1e-15));
    CHECK_THAT(g.face_area_y[at(f)], WithinAbs(s[1], 1e-15));
    CHECK_THAT(g.face_area_z[at(f)], WithinAbs(s[2], 1e-15));
  }
  CHECK(max_closure(mesh, g) < closure_tolerance);
}

TEST_CASE("Box meshes close every cell and fill the box exactly", "[geometry]") {
  const auto fill =
      GENERATE(autan::test::BoxFill::hexa, autan::test::BoxFill::tetra, autan::test::BoxFill::prism,
               autan::test::BoxFill::pyramid, autan::test::BoxFill::hybrid);
  const double perturbation = GENERATE(0.0, 0.3);
  CAPTURE(autan::test::to_string(fill), perturbation);
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(4, fill, perturbation)));
  const autan::Geometry g = autan::compute_geometry(mesh);

  CHECK(max_closure(mesh, g) < closure_tolerance);
  CHECK_THAT(total_volume(g), WithinRel(1.0, 1e-13));

  // Each side of the unit box has area 1 and outward unit normal e_k.
  const auto offsets = mesh.boundary_zone_offsets();
  for (std::size_t zone = 0; zone < 6; ++zone) {
    std::array<double, 3> s{0, 0, 0};
    for (index_t f = offsets[zone]; f < offsets[zone + 1]; ++f) {
      s[0] += g.face_area_x[at(f)];
      s[1] += g.face_area_y[at(f)];
      s[2] += g.face_area_z[at(f)];
    }
    std::array<double, 3> expected{0, 0, 0};
    expected[zone / 2] = zone % 2 == 0 ? -1.0 : 1.0;
    CAPTURE(zone);
    for (std::size_t k = 0; k < 3; ++k) CHECK_THAT(s[k], WithinAbs(expected[k], 1e-14));
  }
}

TEST_CASE("Geometry is invariant under translation far from the origin", "[geometry]") {
  auto input = autan::test::box(4, autan::test::BoxFill::hybrid, 0.3);
  const autan::Mesh near = autan::assemble(autan::to_face_based(input));
  for (auto& x : input.x) x += 1e6;
  for (auto& y : input.y) y -= 3e6;
  for (auto& z : input.z) z += 7e5;
  const autan::Mesh far = autan::assemble(autan::to_face_based(input));
  const autan::Geometry g_near = autan::compute_geometry(near);
  const autan::Geometry g_far = autan::compute_geometry(far);

  // Coordinates near 1e6 carry an absolute rounding of ~1e-10 on a spacing
  // of 0.25, which bounds the attainable relative accuracy at ~1e-9.
  CHECK(max_closure(far, g_far) < closure_tolerance);
  CHECK_THAT(total_volume(g_far), WithinRel(1.0, 1e-9));
  for (std::size_t c = 0; c < g_near.cell_volume.size(); ++c) {
    CHECK_THAT(g_far.cell_volume[c], WithinRel(g_near.cell_volume[c], 1e-8));
  }
}

TEST_CASE("A 105-face cell with hanging nodes has the volume and centroid of its cube",
          "[geometry]") {
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(autan::test::hanging_node_box()));
  const autan::Geometry g = autan::compute_geometry(mesh);

  REQUIRE(mesh.cell_faces(0).size() == 105);
  CHECK_THAT(g.cell_volume[0], WithinRel(1.0, 1e-14));
  CHECK_THAT(g.cell_centroid_x[0], WithinAbs(0.5, 1e-14));
  CHECK_THAT(g.cell_centroid_y[0], WithinAbs(0.5, 1e-14));
  CHECK_THAT(g.cell_centroid_z[0], WithinAbs(0.5, 1e-14));
  CHECK(max_closure(mesh, g) < closure_tolerance);
  CHECK_THAT(total_volume(g), WithinRel(2.0, 1e-14));

  // Collinear hanging nodes leave the polygon's area and centroid unchanged.
  std::size_t polygons = 0;
  for (const index_t f : mesh.cell_faces(0)) {
    if (mesh.face_nodes(f).size() != 13) continue;
    ++polygons;
    CHECK_THAT(g.face_area[at(f)], WithinRel(1.0, 1e-14));
    CHECK_THAT(g.face_centroid_x[at(f)], WithinAbs(0.5, 1e-14));
  }
  CHECK(polygons == 4);
}

TEST_CASE("Degenerate cells are rejected with a geometry error code", "[geometry]") {
  SECTION("zero-area face") {
    // Nodes 0, 1, 2 collinear: face (0, 2, 1) has no area.
    auto ref = autan::test::affine_cell(autan::CellShape::tetra,
                                        {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {0, 0, 0});
    ref.mesh.x[2] = 2.0;
    ref.mesh.y[2] = 0.0;
    const autan::Mesh mesh = autan::assemble(autan::to_face_based(ref.mesh));
    try {
      (void)autan::compute_geometry(mesh);
      FAIL("expected an error");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::geometry_degenerate_face);
    }
  }
  SECTION("inverted cell") {
    auto ref = autan::test::affine_cell(autan::CellShape::tetra,
                                        {{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}}}, {0, 0, 0});
    const autan::Mesh mesh = autan::assemble(autan::to_face_based(ref.mesh));
    try {
      (void)autan::compute_geometry(mesh);
      FAIL("expected an error");
    } catch (const autan::Error& e) {
      CHECK(e.code() == autan::ErrorCode::geometry_non_positive_volume);
    }
  }
}

TEST_CASE("A prism on a concave polygon has its shoelace area, volume and centroid", "[geometry]") {
  // C-shaped octagon whose node mean (2.25, 2) lies in the notch, outside the
  // polygon, so part of the fan folds back. Area 16 - 6 = 10, centroid
  // ((16 * 2 - 6 * 2.5) / 10, 2) = (1.7, 2).
  const std::array<std::array<double, 2>, 8> outline{
      {{0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 3}, {4, 3}, {4, 4}, {0, 4}}};
  autan::CellwiseMesh cw;
  cw.cell_zones.push_back(autan::CellZone{
      .name = "fluid", .type = "", .kind = autan::CellKind::unspecified, .provenance = {}});
  for (const double z : {0.0, 1.0}) {
    for (const auto& p : outline) {
      cw.x.push_back(p[0]);
      cw.y.push_back(p[1]);
      cw.z.push_back(z);
    }
  }
  std::vector<autan::offset_t> offsets{0};
  std::vector<index_t> nodes;
  for (index_t i = 7; i >= 0; --i) nodes.push_back(i);  // bottom, normal -z
  offsets.push_back(static_cast<autan::offset_t>(nodes.size()));
  for (index_t i = 0; i < 8; ++i) nodes.push_back(8 + i);  // top, normal +z
  offsets.push_back(static_cast<autan::offset_t>(nodes.size()));
  for (index_t i = 0; i < 8; ++i) {
    const index_t j = (i + 1) % 8;
    nodes.insert(nodes.end(), {i, j, static_cast<index_t>(8 + j), static_cast<index_t>(8 + i)});
    offsets.push_back(static_cast<autan::offset_t>(nodes.size()));
  }
  cw.add_polyhedron(offsets, nodes, 0);
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(cw));
  const autan::Geometry g = autan::compute_geometry(mesh);

  CHECK_THAT(g.cell_volume[0], WithinRel(10.0, 1e-14));
  CHECK_THAT(g.cell_centroid_x[0], WithinAbs(1.7, 1e-14));
  CHECK_THAT(g.cell_centroid_y[0], WithinAbs(2.0, 1e-14));
  CHECK_THAT(g.cell_centroid_z[0], WithinAbs(0.5, 1e-14));
  std::size_t caps = 0;
  for (index_t f = 0; f < mesh.n_faces(); ++f) {
    if (mesh.face_nodes(f).size() != 8) continue;
    ++caps;
    CHECK_THAT(g.face_area[at(f)], WithinRel(10.0, 1e-14));
    CHECK_THAT(g.face_centroid_x[at(f)], WithinAbs(1.7, 1e-14));
    CHECK_THAT(g.face_centroid_y[at(f)], WithinAbs(2.0, 1e-14));
  }
  CHECK(caps == 2);
  CHECK(max_closure(mesh, g) < closure_tolerance);
}

TEST_CASE("Face centroids and area vectors integrate linear fields exactly on planar faces",
          "[geometry]") {
  // Divergence theorem: sum_f (c_f)_i (S_f)_j = V delta_ij for every cell
  // whose faces are planar, so Green-Gauss is exact for linear fields. Warped
  // faces break this at the order of the warp (documented in the contract).
  auto check = [](const autan::Mesh& mesh) {
    const autan::Geometry g = autan::compute_geometry(mesh);
    for (index_t c = 0; c < mesh.n_owned_cells(); ++c) {
      std::array<std::array<double, 3>, 3> m{};
      for (const index_t f : mesh.cell_faces(c)) {
        const std::size_t i = at(f);
        const double sign = mesh.owner()[i] == c ? 1.0 : -1.0;
        const std::array<double, 3> cf{g.face_centroid_x[i], g.face_centroid_y[i],
                                       g.face_centroid_z[i]};
        const std::array<double, 3> s{g.face_area_x[i], g.face_area_y[i], g.face_area_z[i]};
        for (std::size_t a = 0; a < 3; ++a)
          for (std::size_t b = 0; b < 3; ++b) m[a][b] += sign * cf[a] * s[b];
      }
      const double v = g.cell_volume[at(c)];
      for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = 0; b < 3; ++b)
          CHECK_THAT(m[a][b] / v, WithinAbs(a == b ? 1.0 : 0.0, 1e-12));
    }
  };
  SECTION("boxes") {
    const auto fill = GENERATE(autan::test::BoxFill::hexa, autan::test::BoxFill::tetra,
                               autan::test::BoxFill::prism, autan::test::BoxFill::pyramid,
                               autan::test::BoxFill::hybrid);
    CAPTURE(autan::test::to_string(fill));
    check(autan::assemble(autan::to_face_based(autan::test::box(3, fill))));
  }
  SECTION("105-face cell with hanging nodes") {
    check(autan::assemble(autan::to_face_based(autan::test::hanging_node_box())));
  }
}

TEST_CASE("Geometry holds at extreme length scales", "[geometry]") {
  const double scale = GENERATE(1e-90, 1e80);
  const auto shape = GENERATE(autan::CellShape::tetra, autan::CellShape::pyramid,
                              autan::CellShape::prism, autan::CellShape::hexa);
  CAPTURE(scale);
  const std::array<std::array<double, 3>, 3> a{{{scale, 0, 0}, {0, scale, 0}, {0, 0, scale}}};
  const auto ref = autan::test::affine_cell(shape, a, {0, 0, 0});
  const autan::Mesh mesh = autan::assemble(autan::to_face_based(ref.mesh));
  const autan::Geometry g = autan::compute_geometry(mesh);
  CHECK_THAT(g.cell_volume[0], WithinRel(ref.volume, 1e-14));
  CHECK_THAT(g.cell_centroid_x[0], WithinRel(ref.centroid[0], 1e-14));
  CHECK_THAT(g.cell_centroid_y[0], WithinRel(ref.centroid[1], 1e-14));
  CHECK_THAT(g.cell_centroid_z[0], WithinRel(ref.centroid[2], 1e-14));
}

TEST_CASE("Closure of a geometry computed for another mesh is rejected", "[geometry]") {
  const autan::Mesh mesh =
      autan::assemble(autan::to_face_based(autan::test::box(1, autan::test::BoxFill::hexa)));
  try {
    (void)autan::cell_closure_error(mesh, autan::Geometry{});
    FAIL("expected an error");
  } catch (const autan::Error& e) {
    CHECK(e.code() == autan::ErrorCode::mesh_inconsistent_sizes);
  }
}
